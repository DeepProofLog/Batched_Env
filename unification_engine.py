# unification_engine.py
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
from torch import Tensor

from debug_helper import DebugHelper




@torch.no_grad()
def deduplicate_states_packed(
    states: Tensor,
    state_counts: Tensor,
    padding_idx: int,
    hash_cache: Optional[GPUHashCache] = None
) -> Tuple[Tensor, Tensor]:
    """
    GPU-vectorized deduplication - NO LOOPS AT ALL!
    Uses batched sort + unique operations entirely on GPU.
    
    Args:
        states: [B, max_states, max_atoms, 3] - packed states from all branches
        state_counts: [B] - number of valid states per batch element
        padding_idx: padding value
        hash_cache: optional GPUHashCache object for prime powers
        
    Returns:
        unique_states: [B, max_unique, max_atoms, 3] - deduplicated states  
        unique_counts: [B] - number of unique states per batch element
    """
    if states.numel() == 0:
        return states, state_counts
    
    B, max_states, max_atoms, arity = states.shape
    device = states.device
    
    # Flatten each state to [B, max_states, max_atoms * 3]
    flat_states = states.reshape(B, max_states, -1).long()
    
    # Valid state mask
    valid_mask = state_counts.unsqueeze(1) > torch.arange(max_states, device=device).unsqueeze(0)  # [B, max_states]
    
    # Compute polynomial hash
    prime = 31
    mod_val = 2**31 - 1
    max_len = max_atoms * arity
    
    if hash_cache is None:
        powers = torch.arange(max_len, device=device, dtype=torch.long)
        prime_powers = torch.pow(torch.tensor(prime, device=device, dtype=torch.long), powers) % mod_val
    else:
        prime_powers = hash_cache.get_powers(max_len)
    
    # Hash computation: [B, max_states]
    hashes = (flat_states * prime_powers.unsqueeze(0).unsqueeze(0)).sum(dim=2) % mod_val
    hashes = torch.where(valid_mask, hashes, torch.tensor(mod_val, device=device, dtype=torch.long))  # Max hash for invalid
    
    # Batched sort: [B, max_states]
    sorted_hashes, sort_indices = torch.sort(hashes, dim=1)
    
    # Gather sorted states: [B, max_states, max_atoms, 3]
    batch_indices = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, max_states, max_atoms, arity)
    state_indices = sort_indices.view(B, max_states, 1, 1).expand(B, max_states, max_atoms, arity)
    sorted_states = torch.gather(states, 1, state_indices)
    
    # Find unique mask: consecutive elements with different hashes OR different content
    # [B, max_states]
    unique_mask = torch.ones((B, max_states), dtype=torch.bool, device=device)
    
    # First check hash differences
    hash_diff = sorted_hashes[:, 1:] != sorted_hashes[:, :-1]
    
    # For elements with same hash, check actual content
    same_hash = ~hash_diff  # [B, max_states-1]
    if same_hash.any():
        # Compare actual state content for same-hash pairs
        # Compare sorted_states[:, 1:] with sorted_states[:, :-1]
        states_curr = sorted_states[:, 1:]  # [B, max_states-1, max_atoms, 3]
        states_prev = sorted_states[:, :-1]  # [B, max_states-1, max_atoms, 3]
        # Check if all elements match
        content_same = (states_curr == states_prev).all(dim=(2, 3))  # [B, max_states-1]
        # Unique if hash differs OR content differs
        unique_mask[:, 1:] = hash_diff | ~content_same
    else:
        unique_mask[:, 1:] = hash_diff
    
    # Also mask out invalid entries (those past state_counts)
    valid_after_sort = sort_indices < state_counts.unsqueeze(1)
    unique_mask = unique_mask & valid_after_sort
    
    # Count uniques per batch
    unique_counts = unique_mask.sum(dim=1)  # [B]
    # OPTIMIZED: Use max_states as upper bound instead of .item() call
    # In worst case, all states are unique, so max_unique <= max_states
    max_unique_val = max_states
    
    # Early return check using tensor operations
    if (unique_counts.max() if unique_counts.numel() > 0 else torch.tensor(0, device=device)) == 0:
        return torch.full((B, 0, max_atoms, arity), padding_idx, dtype=states.dtype, device=device), unique_counts
    
    # Compact using cumsum to create packed indices (NO LOOP!)
    # Create position indices for each unique element
    # cumsum gives us the position within the unique subset
    positions = torch.cumsum(unique_mask.long(), dim=1) - 1  # [B, max_states] - position in output (0-indexed)
    
    # Mask to valid range
    valid_positions = positions < max_unique_val
    final_mask = unique_mask & valid_positions
    
    # Create scatter indices: [B, max_unique, max_atoms, 3]
    # We'll gather from sorted_states into unique_states
    # Use masked_scatter or advanced indexing
    
    # Simpler approach: use boolean indexing with reshape
    # For each batch, select the unique states
    # Since we need to pad to max_unique, we'll use a different approach:
    
    # Create output tensor
    unique_states = torch.full((B, max_unique_val, max_atoms, arity), padding_idx, dtype=states.dtype, device=device)
    
    # Use scatter: for each valid unique position, copy the corresponding state
    # Build indices tensor [num_valid, 4] for batch, unique_idx, atom, arity
    batch_idx_flat = torch.arange(B, device=device).view(B, 1).expand(B, max_states)[final_mask]  # [num_valid]
    unique_idx_flat = positions[final_mask]  # [num_valid]
    
    # Expand to get states to copy
    states_to_copy = sorted_states[final_mask.unsqueeze(-1).unsqueeze(-1).expand_as(sorted_states)].reshape(-1, max_atoms, arity)
    
    # Scatter into output
    unique_states[batch_idx_flat, unique_idx_flat] = states_to_copy
    
    return unique_states, unique_counts



# =========================================================
# Utilities
# =========================================================

@torch.no_grad()
def _pack_triples_64(atoms: Tensor, base: int) -> Tensor:
    """
    Perfect 64-bit packing for [N,3] atoms with indices in [0 .. base-1].
    hash = ((p * base) + a) * base + b
    """
    if atoms.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=atoms.device)
    p = atoms[:, 0].long()
    a = atoms[:, 1].long()
    b = atoms[:, 2].long()
    base = torch.as_tensor(base, dtype=torch.int64, device=atoms.device)
    return ((p * base) + a) * base + b


class GPUHashCache:
    """Small cache for state hashing (unchanged interface, sturdier defaults)."""
    def __init__(self, device: torch.device, max_len: int = 4096):
        self.device = device
        self.prime = 31
        self.mod_val = 2**61 - 1  # larger prime to reduce collisions
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        powers = torch.arange(max_len, device=self.device, dtype=torch.int64)
        self.prime_powers = torch.pow(torch.tensor(self.prime, device=self.device, dtype=torch.int64), powers) % self.mod_val
        self.max_len = max_len

    def get_powers(self, length: int) -> Tensor:
        if length > self.max_len:
            self._build_cache(length)
        return self.prime_powers[:length]


class GPUFactIndex:
    """
    GPU-accelerated fact membership with searchsorted over packed 64-bit keys.
    """
    def __init__(self, facts: Tensor, pack_base: int):
        self.device = facts.device
        self.pack_base = pack_base
        if facts.numel() == 0:
            self.fact_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            h = _pack_triples_64(facts.long(), pack_base)
            self.fact_hashes = h.sort()[0]

    @torch.no_grad()
    def contains(self, atoms: Tensor) -> Tensor:
        if atoms.numel() == 0 or self.fact_hashes.numel() == 0:
            return torch.zeros((atoms.shape[0],), dtype=torch.bool, device=atoms.device)
        keys = _pack_triples_64(atoms, self.pack_base)
        idx = torch.searchsorted(self.fact_hashes, keys)
        mask = torch.zeros_like(keys, dtype=torch.bool)
        valid = idx < self.fact_hashes.shape[0]
        mask[valid] = (self.fact_hashes[idx[valid]] == keys[valid])
        return mask

# =========================================================
# Substitution & Unification
# =========================================================

@torch.no_grad()
def _apply_substitutions(goals_batch: Tensor,
                                 substitutions_batch: Tensor,
                                 padding_idx: int) -> Tensor:
    """
    goals_batch:         [B, G, 3]  (int32)
    substitutions_batch: [B, S, 2]  (pairs: [from, to]); rows may contain padding pairs
    Applies only to argument cols (1,2).
    """
    if goals_batch.numel() == 0:
        return goals_batch
    B, G, _ = goals_batch.shape
    preds = goals_batch[:, :, 0:1]
    args = goals_batch[:, :, 1:].clone()

    valid = substitutions_batch[..., 0] != padding_idx
    if not valid.any():
        return goals_batch

    subs = substitutions_batch.long()
    for slot in range(subs.shape[1]):
        slot_mask = valid[:, slot]
        if not slot_mask.any():
            continue
        rows = torch.nonzero(slot_mask, as_tuple=False).squeeze(1)
        subs_rows = subs.index_select(0, rows)
        frm = subs_rows[:, slot, 0].view(-1, 1, 1)
        to = subs_rows[:, slot, 1].view(-1, 1, 1)
        rows_args = args.index_select(0, rows)
        rows_args = torch.where(rows_args == frm, to, rows_args)
        args.index_copy_(0, rows, rows_args)

    return torch.cat([preds, args], dim=2)


@torch.no_grad()
def _unify_one_to_one(queries: Tensor,
                                terms: Tensor,
                                constant_no: int,
                                padding_idx: int) -> Tuple[Tensor, Tensor]:
    """
    Pairwise unify N query atoms with N term atoms.
    Returns:
      mask: [N]       (unifiable)
      subs: [N,2,2]   (two pairs: [from, to], padding-filled)
    """
    device = queries.device
    N = queries.shape[0]
    if N == 0:
        return (torch.empty(0, dtype=torch.bool, device=device),
                torch.full((0, 2, 2), padding_idx, dtype=torch.long, device=device))

    var_start = constant_no + 1

    # Predicates must match
    pred_ok = queries[:, 0] == terms[:, 0]

    # Constant-constant compatibility per arg
    q_args = queries[:, 1:]
    t_args = terms[:, 1:]
    q_const = q_args <= constant_no
    t_const = t_args <= constant_no
    const_conflict = (q_const & t_const & (q_args != t_args)).any(dim=1)

    # Provisional feasibility
    mask = pred_ok & ~const_conflict

    # Early exit
    subs = torch.full((N, 2, 2), padding_idx, dtype=torch.long, device=device)
    if not mask.any():
        return mask, subs

    # Work only on feasible rows to avoid pointless tensor ops
    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    q = q_args.index_select(0, valid_idx)
    t = t_args.index_select(0, valid_idx)
    qv = (q >= var_start) & (q != padding_idx)
    tv = (t >= var_start) & (t != padding_idx)

    subs_sel = torch.full((valid_idx.shape[0], 2, 2), padding_idx, dtype=torch.long, device=device)
    from_slot = subs_sel[:, :, 0]
    to_slot = subs_sel[:, :, 1]

    # Case 1: qVar - tConst (non-padding const)
    case1 = qv & (~tv) & (t != 0)
    if case1.any():
        from_slot = torch.where(case1, q.long(), from_slot)
        to_slot = torch.where(case1, t.long(), to_slot)

    # Case 2: tVar - qConst (non-padding const)
    case2 = (~qv) & (q != 0) & tv
    if case2.any():
        from_slot = torch.where(case2, t.long(), from_slot)
        to_slot = torch.where(case2, q.long(), to_slot)

    # Case 3: both variables
    case3 = qv & tv
    if case3.any():
        from_slot = torch.where(case3, t.long(), from_slot)
        to_slot = torch.where(case3, q.long(), to_slot)

    subs_sel[:, :, 0] = from_slot
    subs_sel[:, :, 1] = to_slot

    # ---- Consistency check: same variable must map to single target
    same_var = (from_slot[:, 0] == from_slot[:, 1]) & (from_slot[:, 0] != padding_idx)
    diff_target = to_slot[:, 0] != to_slot[:, 1]
    conflict_rows = same_var & diff_target
    if conflict_rows.any():
        bad_rows = valid_idx[conflict_rows]
        mask[bad_rows] = False
        subs_sel[conflict_rows] = padding_idx

    subs.index_copy_(0, valid_idx, subs_sel)

    return mask, subs


# =========================================================
# Unification wrt rules/facts (loop-free)
# =========================================================
# === Paste these functions into your current `unification_engine.py` ===
# They replace the previous predicate matching kernels with a ragged, range-based, loop-free implementation.
# Keep the same imports and helper functions you already have.

# === Paste these functions into your current `unification_engine.py` ===
# They replace the previous predicate matching kernels with a ragged, range-based, loop-free implementation.
# Keep the same imports and helper functions you already have.


@torch.no_grad()
def _pairs_via_predicate_ranges(query_preds: Tensor, seg_starts: Tensor, seg_lens: Tensor) -> Tuple[Tensor, Tensor]:
    """Build *all* (query_idx, item_idx) pairs for queries grouped by predicate ranges.
    Args:
        query_preds: [A] predicate ids for each query (queries[:,0])
        seg_starts:  [P+1] start index for each predicate segment in the *items* array (facts or rules)
        seg_lens:    [P+1] length for each predicate segment in the *items* array
    Returns:
        qi: [L] indices into queries
        ii: [L] indices into items (facts or rules)
    """
    device = query_preds.device
    # OPTIMIZATION: Avoid redundant .to(torch.long) if already long
    lens = seg_lens[query_preds.long()]
    starts = seg_starts[query_preds.long()]
    if lens.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z
    keep = lens > 0
    kept_qidx = torch.arange(query_preds.shape[0], device=device)[keep]
    if kept_qidx.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z
    lens = lens[kept_qidx]
    starts = starts[kept_qidx]
    # OPTIMIZED: Use repeat_interleave first, then use row_ids.numel() to avoid .item() call
    # row_ids.numel() == total_items, but we get it without synchronization
    row_ids = torch.repeat_interleave(torch.arange(lens.numel(), device=device), lens)
    if row_ids.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z
    prefix = torch.cumsum(lens, dim=0) - lens
    # Use row_ids.numel() instead of total_items.item() - same value, no sync
    pos_in_row = torch.arange(row_ids.numel(), device=device, dtype=torch.long) - prefix[row_ids]
    item_idx = starts[row_ids] + pos_in_row
    query_idx = kept_qidx[row_ids]
    return query_idx, item_idx

@torch.no_grad()
def _unify_with_facts(
    engine,
    queries: Tensor,
    remaining_goals: Tensor,
    remaining_counts: Tensor,
    pred_indices: Tensor,
    excluded_queries: Optional[Tensor] = None,
    max_output_per_query: int = 500,
    verbose: int = 0,
) -> Tuple[Tensor, Tensor]:
    """One-step *fact* unification using predicate ranges (loop-free & vectorised).

    Supports optional filtering to *exclude trivial self-matches* purely via
    `excluded_queries` (no labels required). If `excluded_queries[i]` is True,
    and the query at row *i* is a *ground* atom equal to a KB fact, we drop that
    specific (query,fact) pair. This preserves other fact matches for the same
    query and keeps branching predictable.

    Args:
        engine: exposes `.facts_idx` (sorted by predicate), `.predicate_range_map`,
                `.padding_idx`, `.constant_no`.
        queries:          [A, 3] leftmost goal per active state.
        remaining_goals:  [A, G, 3] other goals after removing `queries`.
        remaining_counts: [A] number of valid goals in `remaining_goals` per row.
        pred_indices:     [A] predicate id for each query.
        excluded_queries: [A, max_atoms, 3] excluded query states; when unifying, drop facts
                          that match the first atom of the excluded query.
        labels:           (ignored) kept only for API compatibility.
        max_output_per_query: per-owner cap.
        verbose:          verbosity level for debugging.

    Returns:
        states: [A, K, M, 3] packed successors.
        counts: [A]          valid successors per input row (≤ K).
    """
    device = queries.device
    pad = engine.padding_idx
    A, G = remaining_goals.shape[:2]

    # Fast exits --------------------------------------------------------------
    if A == 0 or engine.facts_idx.numel() == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    q_args = queries[:, 1:]
    ground_mask = (q_args <= engine.constant_no).all(dim=1)
    drop_ground_mask = None
    if excluded_queries is not None:
        # Extract first atom from excluded_queries (shape [A, max_atoms, 3] -> [A, 3])
        excluded_first_atoms = excluded_queries[:, 0, :]
        same_atom = (excluded_first_atoms == queries).all(dim=1)
        drop_ground_mask = same_atom & ground_mask
        
        if verbose > 1 and engine.debug_helper:
            print(f"\n_unify_with_facts: A={A} queries")
            print(f"  Ground queries: {ground_mask.sum().item()}")
            print(f"  Non-ground queries: {(~ground_mask).sum().item()}")
            if drop_ground_mask is not None and drop_ground_mask.any():
                print(f"  Ground queries to exclude: {drop_ground_mask.sum().item()}")

    qi_accum: List[Tensor] = []
    inst_accum: List[Tensor] = []
    remcount_accum: List[Tensor] = []

    # Fast O(1) lookup for ground queries ------------------------------------
    ground_idx = torch.nonzero(ground_mask, as_tuple=False).view(-1)
    if ground_idx.numel() > 0:
        ground_queries = queries.index_select(0, ground_idx).long()
        ground_hits = engine.fact_index.contains(ground_queries)
        if drop_ground_mask is not None:
            ground_hits = ground_hits & ~drop_ground_mask.index_select(0, ground_idx)
        if ground_hits.any():
            emit_idx = ground_idx[ground_hits]
            qi_accum.append(emit_idx)
            inst_accum.append(remaining_goals.index_select(0, emit_idx))
            remcount_accum.append(remaining_counts.index_select(0, emit_idx))

    # Ragged pairing for non-ground queries ----------------------------------
    non_ground_mask = ~ground_mask
    if non_ground_mask.any():
        ng_idx = torch.nonzero(non_ground_mask, as_tuple=False).view(-1)
        queries_ng = queries.index_select(0, ng_idx)
        preds_ng = pred_indices.index_select(0, ng_idx)

        prm = engine.predicate_range_map
        seg_starts = prm[:, 0].long()
        seg_lens = (prm[:, 1] - prm[:, 0]).long()

        qi_local, fi = _pairs_via_predicate_ranges(preds_ng, seg_starts, seg_lens)
        if qi_local.numel() > 0:
            q_pairs = queries_ng.index_select(0, qi_local)
            f_pairs = engine.facts_idx.index_select(0, fi)
            
            # FILTER: For non-ground queries with constant args, only keep facts that match those constants
            # Check which args in queries are constants (≤ constant_no)
            q_is_const = q_pairs[:, 1:] <= engine.constant_no
            f_args = f_pairs[:, 1:]
            q_args = q_pairs[:, 1:]
            
            # For positions where query has a constant, fact must match that constant
            # matches[i] = True if all constant positions in query match the fact
            matches = (~q_is_const | (q_args == f_args)).all(dim=1)
            
            # Filter to only matching pairs
            if matches.any():
                qi_local = qi_local[matches]
                fi = fi[matches]
                q_pairs = q_pairs[matches]
                f_pairs = f_pairs[matches]
            else:
                # No matches after filtering
                qi_local = torch.empty(0, dtype=torch.long, device=device)
                
        if qi_local.numel() > 0:
            ok, subs = _unify_one_to_one(q_pairs, f_pairs, engine.constant_no, pad)
            if ok.any():
                qi_orig = ng_idx.index_select(0, qi_local)
                qi_ok = qi_orig[ok]
                
                if verbose > 1:
                    print(f"  Non-ground unifications: {ok.sum().item()} successful")
                
                # Filter out excluded facts for non-ground unification
                # If a fact matches the excluded query, don't include it
                if excluded_queries is not None and ok.sum() > 0:
                    f_ok_indices = fi[ok]
                    f_ok = engine.facts_idx[f_ok_indices]
                    excl_atoms = excluded_queries[qi_ok, 0, :]
                    
                    if verbose > 1 and engine.debug_helper:
                        print(f"  Checking exclusions: {f_ok.shape[0]} facts to check")
                        print(f"  Excluded atom: {engine.debug_helper.atom_to_str(excl_atoms[0])}")
                    
                    # Check if each unified fact equals the excluded atom for that query
                    not_excluded = (f_ok != excl_atoms).any(dim=1)
                    num_excluded = (~not_excluded).sum().item()
                    
                    if verbose > 1:
                        print(f"  Number excluded: {num_excluded}")
                    
                    qi_ok = qi_ok[not_excluded]
                    ok_indices = torch.nonzero(ok, as_tuple=False).view(-1)[not_excluded]
                    if qi_ok.numel() == 0:
                        # All were excluded, skip
                        qi_ok = torch.empty(0, dtype=torch.long, device=device)
                    else:
                        subs_ok = subs[ok_indices].view(-1, 2, 2)
                else:
                    subs_ok = subs[ok].view(-1, 2, 2)
                
                if qi_ok.numel() > 0:
                    remaining_sel = remaining_goals.index_select(0, qi_ok)
                    remain_inst = _apply_substitutions(
                        remaining_sel, subs_ok.view(subs_ok.shape[0], -1, 2), pad
                    )

                    qi_accum.append(qi_ok)
                    inst_accum.append(remain_inst)
                    remcount_accum.append(remaining_counts.index_select(0, qi_ok))

    if not qi_accum:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    qi = torch.cat(qi_accum, dim=0)
    remain_inst = torch.cat(inst_accum, dim=0)
    rem_counts = torch.cat(remcount_accum, dim=0)
    if rem_counts.numel() == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)
    
    # NOTE: No capping here! To match string engine behavior, we return ALL fact unifications
    # and let get_derived_states cap the combined (facts + rules) result.
    # The string engine generates all facts and rules, then caps the combined result.
    
    # EARLY EXIT OPTIMIZATION (matching string engine behavior):
    # Check if all remaining goals in each derived state are facts.
    # CRITICAL: This check happens AFTER capping! The string engine checks during
    # iteration over the FIRST K matches only. We cap first, then check those K matches for proofs.
    
    # For each match, check if ALL remaining atoms are ground facts (vectorized)
    N_matches = remain_inst.shape[0]
    
    # Create valid mask: atoms with count within rem_counts
    # Shape: [N_matches, G]
    atom_idx_grid = torch.arange(G, device=device).unsqueeze(0).expand(N_matches, -1)
    valid_atom_mask = atom_idx_grid < rem_counts.unsqueeze(1)
    
    # Check if atoms are ground (all args <= constant_no)
    # Shape: [N_matches, G, 2] for the two arguments
    args = remain_inst[:, :, 1:3]
    is_ground = (args <= engine.constant_no).all(dim=2)  # [N_matches, G]
    
    # Only check valid atoms
    is_ground = is_ground & valid_atom_mask
    
    # For ground atoms, check if they're facts (batch operation)
    # Flatten to get all ground atoms
    ground_atoms_mask = is_ground.flatten()  # [N_matches * G]
    if ground_atoms_mask.any():
        all_atoms_flat = remain_inst.view(-1, 3)  # [N_matches * G, 3]
        ground_atoms = all_atoms_flat[ground_atoms_mask]  # [num_ground, 3]
        
        # Check which are facts (batched)
        are_facts = engine.fact_index.contains(ground_atoms)  # [num_ground]
        
        # Handle excluded queries (avoid circular proofs)
        if excluded_queries is not None:
            # Match each ground atom to its owner
            match_indices = torch.arange(N_matches, device=device).unsqueeze(1).expand(-1, G).flatten()[ground_atoms_mask]
            owners = qi[match_indices]  # [num_ground]
            
            # Get excluded atoms for each owner
            excl_atoms = excluded_queries[owners, 0, :]  # [num_ground, 3]
            
            # Mark as non-fact if it matches excluded query
            matches_excluded = (ground_atoms == excl_atoms).all(dim=1)
            are_facts = are_facts & ~matches_excluded
        
        # Scatter results back to grid
        is_fact_grid = torch.zeros(N_matches, G, dtype=torch.bool, device=device)
        is_fact_grid.view(-1)[ground_atoms_mask] = are_facts
    else:
        is_fact_grid = torch.zeros(N_matches, G, dtype=torch.bool, device=device)
    
    # For each match: all valid atoms must be ground AND facts
    # If any valid atom is non-ground OR (ground but not a fact), the match is not a proof
    all_valid_are_ground_facts = valid_atom_mask.clone()
    all_valid_are_ground_facts[valid_atom_mask] = is_ground[valid_atom_mask] & is_fact_grid[valid_atom_mask]
    
    # A match is a proof if all its valid atoms are ground facts
    all_atoms_are_facts = (all_valid_are_ground_facts | ~valid_atom_mask).all(dim=1)  # [N_matches]
    

    
    # If ANY unification leads to all facts, return True() for that owner immediately
    # (matching string engine's behavior: return [[Term('True', ())]] on first proof found)
    if all_atoms_are_facts.any():
        # Group by owner to find which owners have proofs
        proof_match_indices = torch.nonzero(all_atoms_are_facts, as_tuple=False).view(-1)
        proof_owners = qi[proof_match_indices].unique()
        
        if verbose > 1:
            print(f"  EARLY EXIT: Found {len(proof_owners)} owners with immediate proofs from fact unification")
            print(f"    Proof owners: {proof_owners.tolist()}")
        
        # Return True() states for owners with proofs, empty for others
        # Create output with one True() state per proof owner
        true_tensor = torch.tensor([engine.true_pred_idx, 0, 0], dtype=torch.long, device=device)
        out_true = torch.full((A, 1, 1, 3), pad, dtype=torch.long, device=device)
        counts_true = torch.zeros(A, dtype=torch.long, device=device)
        
        out_true[proof_owners, 0, 0] = true_tensor
        counts_true[proof_owners] = 1
        
        return out_true, counts_true
    
    # No immediate proofs found - pack and return ALL states (no capping)
    # Pack states into output tensor
    max_cols_val = G if G > 0 else 1
    states_per_match = remain_inst[:, :max_cols_val]
    
    # Determine maximum states per owner for output allocation
    states_per_owner = torch.bincount(qi, minlength=A)
    max_states_any_owner = states_per_owner.max().item() if states_per_owner.numel() > 0 else 1
    
    out = torch.full((A, max_states_any_owner, max_cols_val, 3), pad, dtype=torch.long, device=device)
    counts_per_q = torch.bincount(qi, minlength=A)
    
    if qi.numel() > 0:
        # Compute positions again after capping
        sort_vals, sort_idx = torch.sort(qi, stable=True)
        states_sorted = states_per_match[sort_idx]
        
        seg_start = torch.ones_like(sort_vals, dtype=torch.bool, device=device)
        seg_start[1:] = sort_vals[1:] != sort_vals[:-1]
        seg_indices = torch.arange(sort_vals.shape[0], device=device)
        seg_first_idx = torch.zeros_like(sort_vals, dtype=torch.long)
        seg_first_idx[seg_start] = seg_indices[seg_start]
        seg_first_idx = torch.cummax(seg_first_idx, dim=0)[0]
        pos_in_owner = seg_indices - seg_first_idx
        
        db = sort_vals
        ds = pos_in_owner
        out[db, ds] = states_sorted

    return out, counts_per_q

@torch.no_grad()
def _unify_with_rules(
    engine,
    queries: Tensor,
    remaining_goals: Tensor,
    remaining_counts: Tensor,
    pred_indices: Tensor,
    next_var_indices: Tensor,
    max_output_per_query: int = 100,
) -> Tuple[Tensor, Tensor]:
    """One-step *rule* unification using per-call rule-head ranges (vectorised).

    We sort rule heads by predicate *once per call* (cheap and stable on GPU) and
    then apply the same ragged, range-based pairing strategy used for facts. If you
    prefer, you can precompute and store the rule ranges at engine construction.

    Args:
        engine: object that exposes `.rules_heads_idx`, `.rules_idx`, `.rule_lens`,
                `.padding_idx`, `.constant_no`.
        queries:          [A, 3]  atoms picked as the leftmost goal in each active state.
        remaining_goals:  [A, G, 3] the other goals after removing `queries`.
        remaining_counts: [A] number of *valid* goals in `remaining_goals` per row.
        pred_indices:     [A] predicate id for each `queries[i]`.
        next_var_indices: [A] next available runtime variable index for each query.
        max_output_per_query: cap on branching factor per active state.

    Returns:
        states: [A, K, M, 3] packed successor states (rule body ⊕ remaining goals), padded.
        counts: [A]          number of valid successors per input row (≤ K).
    """
    device = queries.device
    pad = engine.padding_idx
    A, G = remaining_goals.shape[:2]

    # Fast exits --------------------------------------------------------------
    if A == 0 or engine.rules_heads_sorted.numel() == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    heads_sorted = engine.rules_heads_sorted
    bodies_sorted = engine.rules_idx_sorted
    lens_sorted = engine.rule_lens_sorted

    # 2) Ragged pairing (query, rule) via ranges -----------------------------
    qi, ri = _pairs_via_predicate_ranges(
        pred_indices, engine.rule_seg_starts, engine.rule_seg_lens
    )
    if qi.numel() == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    q_pairs = queries.index_select(0, qi)                 # [L, 3]
    h_pairs_template = heads_sorted.index_select(0, ri)   # [L, 3] with template vars
    b_pairs_template = bodies_sorted.index_select(0, ri)  # [L, Bmax, 3] with template vars
    
    # Rename template variables to fresh runtime variables for each match
    # Template variables start at constant_no + 1
    # For each match i, rename template vars to next_var_indices[qi[i]] + offset
    template_start = engine.constant_no + 1
    
    # Get next_var for each match
    next_var_for_matches = next_var_indices.index_select(0, qi)  # [L]
    
    # Rename head arguments
    h_pairs = h_pairs_template.clone()
    for arg_idx in [1, 2]:  # Process both arguments
        args = h_pairs[:, arg_idx]
        is_template = (args >= template_start) & (args != pad)
        if is_template.any():
            # Compute offset: (template_var - template_start)
            offsets = args - template_start
            # New var: next_var + offset
            h_pairs[:, arg_idx] = torch.where(is_template, next_var_for_matches + offsets, args)
    
    # Rename body arguments
    b_pairs = b_pairs_template.clone()
    for arg_idx in [1, 2]:  # Process both arguments  
        args = b_pairs[:, :, arg_idx]  # [L, Bmax]
        is_template = (args >= template_start) & (args != pad)
        if is_template.any():
            offsets = args - template_start
            new_vars = next_var_for_matches.unsqueeze(1) + offsets
            b_pairs[:, :, arg_idx] = torch.where(is_template, new_vars, args)

    ok, subs = _unify_one_to_one(q_pairs, h_pairs, engine.constant_no, pad)
    if not ok.any():
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    # Keep successful matches only -------------------------------------------
    qi = qi[ok]
    ri = ri[ok]
    b_pairs = b_pairs[ok]  # Keep renamed bodies for successful matches
    h_pairs = h_pairs[ok]  # Keep renamed heads for successful matches
    subs = subs[ok].view(-1, 2, 2)

    # 3) Instantiate (rule body ⊕ remaining goals) ---------------------------
    bodies_all = b_pairs                                  # [M', Bmax, 3] already renamed
    body_lens  = lens_sorted.index_select(0, ri)          # [M']
    remaining_sel = remaining_goals.index_select(0, qi)   # [M', G, 3]

    subs_b = subs.view(subs.shape[0], -1, 2)
    bodies_inst = _apply_substitutions(bodies_all, subs_b, pad)
    remain_inst = _apply_substitutions(remaining_sel, subs_b, pad)

    # Keep active atoms only (drop padded tail for both parts)
    Bmax = bodies_all.shape[1]
    pos_b = torch.arange(Bmax, device=device).unsqueeze(0).expand_as(bodies_inst[:, :, 0])
    take_b = pos_b < body_lens.unsqueeze(1)

    pos_g = torch.arange(G, device=device).unsqueeze(0).expand_as(remain_inst[:, :, 0])
    take_g = pos_g < remaining_counts.index_select(0, qi).unsqueeze(1)

    counts = take_b.sum(1) + take_g.sum(1)                # [M']
    # OPTIMIZED: Use Bmax + G as fixed maximum to avoid .item() call
    max_cols_val = Bmax + G

    # Concatenate body and remaining, then compact to `max_cols`
    cat_full = torch.full((counts.shape[0], Bmax + G, 3), pad, dtype=torch.long, device=device)
    cat_full[:, :Bmax] = bodies_inst
    cat_full[:, Bmax:] = remain_inst
    keep_mask = torch.cat([take_b, take_g], dim=1)

    pos = torch.cumsum(keep_mask.long(), dim=1) - 1
    if keep_mask.any():
        cat_compact = torch.full((counts.shape[0], max_cols_val, 3), pad, dtype=torch.long, device=device)
        src = torch.arange(keep_mask.shape[1], device=device).unsqueeze(0).expand_as(keep_mask)
        row_ids = torch.arange(keep_mask.shape[0], device=device).unsqueeze(1).expand_as(keep_mask)
        sel_rows = row_ids[keep_mask]
        sel_cols = src[keep_mask]
        cat_compact[sel_rows, pos[keep_mask]] = cat_full[sel_rows, sel_cols]
    else:
        cat_compact = torch.full((counts.shape[0], 1, 3), pad, dtype=torch.long, device=device)

    # 4) Pack all rule states (no capping) to match string engine behavior ----------
    # The string engine generates all rules and all facts, then caps the combined result.
    K_per_owner = torch.bincount(qi, minlength=A)
    
    max_k = K_per_owner.max() if K_per_owner.numel() > 0 else torch.tensor(0, device=device)
    
    # Check if we have no matches at all
    if max_k == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.long, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    # Allocate output for maximum states per owner (no artificial cap)
    max_states_per_owner = max_k.item() if isinstance(max_k, torch.Tensor) else int(max_k)
    out = torch.full((A, max_states_per_owner, cat_compact.shape[1], 3), pad, dtype=torch.long, device=device)
    counts_per_q = K_per_owner

    # Pack all states per owner (loop-free) --------------------------
    owner_sorted, perm = torch.sort(qi)
    cat_sorted = cat_compact.index_select(0, perm)
    seg_start = torch.ones_like(owner_sorted, dtype=torch.bool, device=device)
    seg_start[1:] = owner_sorted[1:] != owner_sorted[:-1]
    
    # Compute position within each segment
    seg_indices = torch.arange(owner_sorted.shape[0], device=device)
    seg_first_idx = torch.zeros_like(owner_sorted, dtype=torch.long)
    seg_first_idx[seg_start] = seg_indices[seg_start]
    seg_first_idx = torch.cummax(seg_first_idx, dim=0)[0]
    pos_in_owner = seg_indices - seg_first_idx
    
    # Pack all states (no filtering)
    db = owner_sorted
    ds = pos_in_owner
    out[db, ds] = cat_sorted
    
    return out, counts_per_q



# =========================================================
# Engine
# =========================================================

class UnificationEngine:
    """Encapsulates facts/rules and batched single-step unification."""

    def __init__(self,
                 facts_idx: Tensor,
                 rules_idx: Tensor,
                 rule_lens: Tensor,
                 rules_heads_idx: Tensor,
                 padding_idx: int,
                 constant_no: int,
                 runtime_var_end_index: int,
                 true_pred_idx: Optional[int],
                 false_pred_idx: Optional[int],
                 max_arity: int,
                 predicate_range_map: Optional[Tensor],
                 device: torch.device,
                 pack_base: Optional[int] = None,
                 stringifier_params: Optional[dict] = None):
        self.device = device
        self.padding_idx = int(padding_idx)
        self.constant_no = int(constant_no)
        self.runtime_var_end_index = int(runtime_var_end_index)
        self.max_arity = int(max_arity)
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx

        # Handle None properly for tensors
        self.facts_idx = facts_idx.to(device=device, dtype=torch.long)
        self.rules_idx = rules_idx.to(device=device, dtype=torch.long)
        self.rule_lens = rule_lens.to(device=device, dtype=torch.long)
        self.rules_heads_idx = rules_heads_idx.to(device=device, dtype=torch.long)

        # canonical pack base (>= max index + 1)
        # OPTIMIZED: Minimize .item() calls in initialization
        max_idx = 1
        tensors_to_check = [self.facts_idx, self.rules_idx, self.rules_heads_idx]
        non_empty_tensors = [t for t in tensors_to_check if t.numel() > 0]
        if non_empty_tensors:
            # Compute all maxes in one go
            max_vals = [t.max() for t in non_empty_tensors]
            max_idx = max(max_idx, int(torch.stack(max_vals).max().item()))
        self.pack_base = int(pack_base if pack_base is not None else (max(max_idx, self.runtime_var_end_index) + 2))

        # predicate ranges (GPU copy if provided)
        self.predicate_range_map = None
        if predicate_range_map is not None and predicate_range_map.numel() > 0:
            self.predicate_range_map = predicate_range_map.to(device=device, dtype=torch.long)

        # helper tensors for TRUE/FALSE atoms
        pad = self.padding_idx
        if self.true_pred_idx is not None:
            self.true_tensor = torch.tensor([[self.true_pred_idx, pad, pad]], dtype=torch.long, device=device)
        else:
            self.true_tensor = None
        if self.false_pred_idx is not None:
            self.false_tensor = torch.tensor([[self.false_pred_idx, pad, pad]], dtype=torch.long, device=device)
        else:
            self.false_tensor = None

        # indices for fast membership & dedup
        self.fact_index = GPUFactIndex(self.facts_idx, self.pack_base)
        if self.facts_idx.numel() > 0:
            self.facts_packed = _pack_triples_64(self.facts_idx.long(), self.pack_base)
        else:
            self.facts_packed = torch.empty((0,), dtype=torch.long, device=device)
        self.hash_cache = GPUHashCache(device)

        # Pre-sort rules by predicate once and build predicate ranges
        if self.rules_heads_idx.numel() > 0:
            order = torch.argsort(self.rules_heads_idx[:, 0])
            self.rules_heads_sorted = self.rules_heads_idx.index_select(0, order)
            self.rules_idx_sorted = self.rules_idx.index_select(0, order)
            self.rule_lens_sorted = self.rule_lens.index_select(0, order)

            preds = self.rules_heads_sorted[:, 0]
            uniq, counts = torch.unique_consecutive(preds, return_counts=True)
            starts = torch.cumsum(
                torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts[:-1]]),
                dim=0,
            )
            # Determine predicate table size (reuse fact map if available)
            if self.predicate_range_map is not None and self.predicate_range_map.numel() > 0:
                num_predicates = max(
                    self.predicate_range_map.shape[0],
                    int(preds.max().item()) + 2,
                )
            else:
                num_predicates = int(preds.max().item()) + 2
            self.rule_seg_starts = torch.zeros((num_predicates,), dtype=torch.long, device=device)
            self.rule_seg_lens = torch.zeros((num_predicates,), dtype=torch.long, device=device)
            self.rule_seg_starts[uniq] = starts
            self.rule_seg_lens[uniq] = counts
        else:
            self.rules_heads_sorted = self.rules_heads_idx
            self.rules_idx_sorted = self.rules_idx
            self.rule_lens_sorted = self.rule_lens
            self.rule_seg_starts = torch.zeros((1,), dtype=torch.long, device=device)
            self.rule_seg_lens = torch.zeros((1,), dtype=torch.long, device=device)

        # Initialize DebugHelper for verbose output
        self.debug_helper = DebugHelper(**stringifier_params) if stringifier_params else None

    # ---- factory ----
    @classmethod
    def from_index_manager(cls, im, take_ownership: bool = False, stringifier_params: Optional[dict] = None) -> 'UnificationEngine':
        engine = cls(
            facts_idx=getattr(im, 'facts_idx', None),
            rules_idx=getattr(im, 'rules_idx', None),
            rule_lens=getattr(im, 'rule_lens', None),
            rules_heads_idx=getattr(im, 'rules_heads_idx', None),
            padding_idx=im.padding_idx,
            constant_no=im.constant_no,
            runtime_var_end_index=im.runtime_var_end_index,
            true_pred_idx=im.true_pred_idx,
            false_pred_idx=im.false_pred_idx,
            max_arity=im.max_arity,
            predicate_range_map=getattr(im, 'predicate_range_map', None),
            device=im.device,
            pack_base=getattr(im, 'total_vocab_size', None),
            stringifier_params=stringifier_params
        )
        if take_ownership:
            im.facts_idx = None
            im.rules_idx = None
            im.rule_lens = None
            im.rules_heads_idx = None
        return engine

    # ---- core ----
    @torch.no_grad()
    def get_derived_states(self,
                           current_states: Tensor,
                           next_var_indices: Tensor,
                           excluded_queries: Optional[Tensor] = None,
                           verbose: int = 0,
                           debug: bool = False,
                           max_derived_per_state: int = 500,
                           deduplicate: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        One-step expansion, strictly ordered:

        1) preprocessing
        2) rule unification
        3) fact unification
        4) prune ground facts & detect immediate proofs
        5) combine
        6) canonicalise + dedup (optional)
        7) pad back
        
        Args:
            current_states: [B, max_atoms, 3] tensor of current states
            next_var_indices: [B] tensor of next available variable indices
            excluded_queries: Optional [B, max_atoms, 3] tensor of queries to exclude from facts
            verbose: If > 0, print debug information with atom/state strings.
            debug: If True, print detailed debug information for unifications, canonicalization, and dedup.
            max_derived_per_state: Maximum number of derived states per input state. Default 500.
            deduplicate: If True, remove duplicate states after canonicalization. Default False to match string engine.
        """
        device = current_states.device
        pad = self.padding_idx
        B, max_atoms, _ = current_states.shape

        # Helper for verbose printing - now uses DebugHelper
        def print_states(title: str, states_tensor: Tensor, counts: Tensor = None):
            """Print states in human-readable format."""
            if self.debug_helper:
                self.debug_helper.print_states(title, states_tensor, counts, pad)

        # i) Print current states
        if verbose > 0:
            print_states("i) CURRENT STATES", current_states)

        # ---- 1) preprocessing
        valid_mask = current_states[:, :, 0] != pad
        has_any = valid_mask.any(dim=1)
        empty_states = ~has_any

        has_false = (current_states[:, :, 0] == self.false_pred_idx).any(dim=1) if self.false_pred_idx is not None \
                    else torch.zeros(B, dtype=torch.bool, device=device)
        only_true = has_any & ((current_states[:, :, 0] == self.true_pred_idx) | ~valid_mask).all(dim=1) \
                    if self.true_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=device)

        terminal = empty_states | has_false | only_true

        # max_derived_per_state is now passed as a parameter (default 500)
        # Output states can have more atoms than input (e.g., rule body + remaining goals)
        # Use a reasonable maximum based on max rule body size
        max_atoms_output = max(max_atoms, 10)  # At least 10 atoms for derived states
        all_derived = torch.full((B, max_derived_per_state, max_atoms_output, 3), pad, dtype=torch.long, device=device)
        derived_counts = torch.zeros(B, dtype=torch.long, device=device)
        updated_next_var_indices = next_var_indices.clone()

        # emit trivials
        only_true_idx = torch.arange(B, device=device)[only_true]
        if only_true_idx.numel() > 0:
            all_derived[only_true_idx, 0, 0] = self.true_tensor.squeeze(0)
            derived_counts[only_true_idx] = 1

        false_mask = (has_false | empty_states) & ~only_true
        false_idx = torch.arange(B, device=device)[false_mask]
        if false_idx.numel() > 0:
            if self.false_tensor is None:
                raise ValueError("False predicate not defined")
            all_derived[false_idx, 0, 0] = self.false_tensor.squeeze(0)
            derived_counts[false_idx] = 1

        active = ~terminal
        active_idx = torch.arange(B, device=device)[active]
        if active_idx.numel() == 0:
            return (all_derived, derived_counts, updated_next_var_indices)
        A = active_idx.shape[0]
        active_states = current_states[active_idx]
        active_valid  = valid_mask[active_idx]

        # Ensure next_var_indices are higher than any variable in the current states
        # to prevent variable collisions when introducing fresh variables from rules
        # Extract only the arguments (columns 1 and 2), not predicates (column 0)
        args_in_states = active_states[:, :, 1:][active_valid]  # Get args where valid
        max_arg_in_states = args_in_states.max() if args_in_states.numel() > 0 else self.constant_no
        if max_arg_in_states > self.constant_no:
            # Update next_var to be at least max_var + 1
            min_next_var = max_arg_in_states + 1
            updated_next_var_indices[active_idx] = torch.maximum(
                updated_next_var_indices[active_idx],
                torch.full_like(updated_next_var_indices[active_idx], min_next_var)
            )
        active_next_var = updated_next_var_indices[active_idx]



        # split into (query, remaining)
        first_pos = active_valid.long().argmax(dim=1)
        arangeA = torch.arange(A, device=device)
        queries = active_states[arangeA, first_pos]

        remaining_goals = torch.full_like(active_states, pad)
        remaining_counts = (active_valid.sum(dim=1) - 1).clamp(min=0)
        if A > 0:
            pos_grid = torch.arange(max_atoms, device=device).unsqueeze(0).expand(A, -1)
            before = active_valid & (pos_grid < first_pos.unsqueeze(1))
            after  = active_valid & (pos_grid > first_pos.unsqueeze(1))
            scatter = before | after
            if scatter.any():
                row_ids = torch.arange(A, device=device).unsqueeze(1).expand_as(pos_grid)
                scatter_rows = row_ids[scatter]
                src_a = pos_grid[scatter]
                dst_a = torch.where(before, pos_grid, pos_grid - 1)[scatter]
                remaining_goals[scatter_rows, dst_a] = active_states[scatter_rows, src_a]

        preds = queries[:, 0]
        # Use the updated active_next_var that was adjusted for current state variables
        # (defined earlier when checking max_vars_in_states)

        # ---- 2) rules
        rule_states, rule_counts = _unify_with_rules(
            self, queries, remaining_goals, remaining_counts, preds, active_next_var,
            max_output_per_query=max_derived_per_state
        )
        
        # ii) Print rule unifications
        if verbose > 0:
            print_states("ii) RULE UNIFICATIONS", rule_states, rule_counts)

        # ---- 3) facts
        fact_states, fact_counts = _unify_with_facts(
            self, queries, remaining_goals, remaining_counts, preds,
            excluded_queries=excluded_queries[active_idx] if excluded_queries is not None else None,
            verbose=verbose,
            max_output_per_query=max_derived_per_state
        )
        
        # iii) Print fact unifications
        if verbose > 0:
            print_states("iii) FACT UNIFICATIONS", fact_states, fact_counts)

        # ---- 4) detect immediate proof (remaining goals vanish due to fact match)
        immediate = (remaining_counts == 0) & (fact_counts > 0)
        
        # Also detect early-exit proofs from _unify_with_facts
        # These are cases where fact unification detected that all remaining goals are facts
        # and returned True() directly (matching string engine behavior)
        if fact_counts.any() and self.true_pred_idx is not None:
            for i in range(A):
                if fact_counts[i] > 0:
                    # Check if first fact state is True()
                    first_fact = fact_states[i, 0, 0]  # [max_atoms, 3] -> [3]
                    if first_fact[0].item() == self.true_pred_idx:
                        immediate[i] = True
        
        immediate_idx = torch.arange(A, device=device)[immediate]

        # ---- 5) combine candidates
        Rmax, Fmax = (rule_states.shape[1], fact_states.shape[1])
        max_atoms_rules = rule_states.shape[2] if rule_states.numel() > 0 else max_atoms
        max_atoms_facts = fact_states.shape[2] if fact_states.numel() > 0 else max_atoms
        max_atoms_comb = max(max_atoms_rules, max_atoms_facts, max_atoms)
        total_slots = Rmax + Fmax

        packed = torch.full((A, total_slots, max_atoms_comb, 3), pad, dtype=torch.long, device=device)
        if Rmax > 0:
            packed[:, :Rmax, :max_atoms_rules] = rule_states
        if Fmax > 0:
            packed[:, Rmax:Rmax + Fmax, :max_atoms_facts] = fact_states

        owner_rule_mask = torch.arange(Rmax, device=device).unsqueeze(0).expand(A, -1) < rule_counts.unsqueeze(1)
        owner_fact_mask = torch.arange(Fmax, device=device).unsqueeze(0).expand(A, -1) < fact_counts.unsqueeze(1)
        owner_mask = torch.cat([owner_rule_mask, owner_fact_mask], dim=1) & (~immediate).unsqueeze(1)

        # Check which active states have NO unifications at all
        has_any_unification = (rule_counts > 0) | (fact_counts > 0)
        
        if not owner_mask.any():
            # Handle immediate proofs (True())
            if immediate_idx.numel() > 0:
                dst = active_idx[immediate_idx]
                all_derived[dst, 0, 0] = self.true_tensor.squeeze(0)
                derived_counts[dst] = 1
            
            # Handle states with NO unifications (False())
            # These are active states that had no rule matches and no fact matches
            no_match = ~has_any_unification
            no_match_idx = torch.arange(A, device=device)[no_match]
            if no_match_idx.numel() > 0:
                dst = active_idx[no_match_idx]
                if self.false_tensor is not None:
                    all_derived[dst, 0, 0] = self.false_tensor.squeeze(0)
                    derived_counts[dst] = 1
            
            return (all_derived, derived_counts, updated_next_var_indices)

        row_ids = torch.arange(A, device=device).unsqueeze(1).expand_as(owner_mask)
        col_ids = torch.arange(total_slots, device=device).unsqueeze(0).expand_as(owner_mask)
        owner_ids = row_ids[owner_mask]
        slot_ids = col_ids[owner_mask]
        candidates = packed[owner_ids, slot_ids]
        cand_counts = (candidates[:, :, 0] != pad).sum(dim=1)

        # ---- 4-bis) prune ground facts & mark proofs
        # Pass excluded_queries mapped to active owners
        excluded_for_owners = excluded_queries[active_idx[owner_ids]] if excluded_queries is not None else None
        pruned_states, pruned_counts, is_proof = self._prune_and_prove(candidates, cand_counts, excluded_for_owners)
        owners_for_cands = active_idx[owner_ids]
        
        # iv) Print after pruning ground facts
        if verbose > 0:
            print(f"\n{'='*60}")
            print("iv) PRUNE GROUND FACTS & DETECT IMMEDIATE PROOFS")
            print(f"{'='*60}")
            if self.debug_helper is not None:
                for i in range(pruned_states.shape[0]):
                    state = pruned_states[i]
                    valid = state[:, 0] != pad
                    if valid.any():
                        atoms = state[valid]
                        atoms_str = [self.debug_helper.atom_to_str(atom) for atom in atoms]
                        proof_str = " [PROOF]" if is_proof[i] else ""
                        print(f"  Candidate {i} (owner={owners_for_cands[i].item()}){proof_str}: [{', '.join(atoms_str)}]")
                    else:
                        print(f"  Candidate {i} (owner={owners_for_cands[i].item()}) [PROOF]: <empty>")
            if immediate_idx.numel() > 0:
                print(f"  Immediate proofs detected for states: {immediate_idx.tolist()}")


        if immediate_idx.numel() > 0:
            dst = active_idx[immediate_idx]
            all_derived[dst, 0, 0] = self.true_tensor.squeeze(0)
            derived_counts[dst] = 1
        
        # Collect all owners that have at least one proof
        if is_proof.any():
            proof_owners = torch.unique(owners_for_cands[is_proof])
            all_derived[proof_owners, 0, 0] = self.true_tensor.squeeze(0)
            derived_counts[proof_owners] = 1
            
            # When a proof is found for an owner, only return True() - discard non-proof states
            # This is correct: if we found one way to prove the query, we're done!
            has_proof_mask = torch.zeros(A, dtype=torch.bool, device=device)
            has_proof_mask[proof_owners] = True
            owners_have_proof = has_proof_mask[owners_for_cands]
            
            # Keep only non-proof states from owners that have NO proof
            keep_mask = ~is_proof & ~owners_have_proof
        else:
            # No proofs found, keep all non-proof states
            keep_mask = ~is_proof
        
        if not keep_mask.any():
            return (all_derived, derived_counts, updated_next_var_indices)

        owners_kept = owners_for_cands[keep_mask]
        states_kept = pruned_states[keep_mask]
        counts_kept = pruned_counts[keep_mask]

        # ---- 6) canonicalise + update next var indices
        canon_states, next_vars_per_state = self._canonicalize_variables(
            states_kept, counts_kept, self.constant_no,
            next_var_indices[owners_kept], pad
        )
        
        # v) Print after canonicalization (before dedup)
        if verbose > 0:
            print(f"\n{'='*60}")
            print("v) CANONICALIZE (before dedup)")
            print(f"{'='*60}")
            if self.debug_helper is not None:
                for i in range(canon_states.shape[0]):
                    state = canon_states[i]
                    valid = state[:, 0] != pad
                    if valid.any():
                        atoms = state[valid]
                        atoms_str = [self.debug_helper.atom_to_str(atom) for atom in atoms]
                        print(f"  Candidate {i} (owner={owners_kept[i].item()}): [{', '.join(atoms_str)}]")
            if debug:
                print(f"\n[DEBUG] Before packing:")
                print(f"  canon_states.shape: {canon_states.shape}")
                print(f"  counts_kept: {counts_kept.tolist()}")
                print(f"  owners_kept: {owners_kept.tolist()}")



        if canon_states.shape[0] > 0:
            # keep the maximum next_var index across expansions for each owner
            agg = torch.full((B,), 0, dtype=next_var_indices.dtype, device=device)
            agg.scatter_reduce_(0, owners_kept, next_vars_per_state, reduce='amax', include_self=False)
            updated_next_var_indices = torch.maximum(updated_next_var_indices, agg)

        # ---- pack back per original owner
        K = min(total_slots, max_derived_per_state)
        
        # Check if we'll hit the cap and print warning
        if owners_kept.numel() > 0:
            states_per_owner = torch.bincount(owners_kept, minlength=B)
            max_states = states_per_owner.max().item() if states_per_owner.numel() > 0 else 0
            if max_states > K:
                for i in range(B):
                    if states_per_owner[i] > K:
                        print(f"WARNING (get_derived_states): State {i} generated {states_per_owner[i].item()} derived states, capping to {K}")
        
        packed_states = torch.full((B, K, max_atoms_comb, 3), pad, dtype=torch.long, device=device)
        packed_counts = torch.zeros(B, dtype=torch.long, device=device)
        if owners_kept.numel() > 0:
            sort_vals, sort_idx = torch.sort(owners_kept)
            states_sorted = canon_states[sort_idx]
            counts_sorted = counts_kept[sort_idx]
            seg_start = torch.ones_like(sort_vals, dtype=torch.bool, device=device)
            seg_start[1:] = sort_vals[1:] != sort_vals[:-1]
            
            # Compute position within each owner's segment correctly
            # Assign a segment ID to each element
            seg_id = torch.cumsum(seg_start.long(), dim=0) - 1
            # For each segment, count positions 0, 1, 2, ...
            # Use scatter to find segment starts
            # OPTIMIZED: Use B as upper bound (max possible segments = batch size)
            num_segs_val = B
            seg_starts = torch.full((num_segs_val,), len(seg_id), dtype=torch.long, device=device)  # Initialize to max value
            if seg_id.numel() > 0:
                seg_starts.scatter_reduce_(0, seg_id, torch.arange(len(seg_id), device=device), reduce='amin', include_self=False)
            
            # Position within segment = current_index - segment_start
            pos = torch.arange(len(seg_id), device=device) - seg_starts[seg_id]
            pos = torch.minimum(pos, torch.full_like(pos, K - 1))
            dst_b = sort_vals
            dst_s = pos
            
            # Ensure states_sorted matches max_atoms_comb dimension
            if states_sorted.shape[1] < max_atoms_comb:
                # Pad atoms dimension
                padding = torch.full((states_sorted.shape[0], max_atoms_comb - states_sorted.shape[1], 3),
                                   pad, dtype=torch.long, device=device)
                states_sorted = torch.cat([states_sorted, padding], dim=1)
            elif states_sorted.shape[1] > max_atoms_comb:
                # Truncate atoms dimension
                states_sorted = states_sorted[:, :max_atoms_comb]
            
            packed_states[dst_b, dst_s] = states_sorted
            owner_maxpos = torch.full((B,), -1, dtype=torch.long, device=device)
            # Use scatter_reduce with 'amax' to find the maximum position per owner
            owner_maxpos.scatter_reduce_(0, dst_b, dst_s, reduce='amax', include_self=False)
            packed_counts = torch.clamp(owner_maxpos + 1, min=0, max=K)

        # ---- 6-bis) dedup (optional)
        if debug and verbose > 0:
            print(f"\n[DEBUG] Before dedup:")
            print(f"  packed_states.shape: {packed_states.shape}")
            print(f"  packed_counts: {packed_counts.tolist()}")
            if self.debug_helper is not None:
                for b in range(B):
                    if packed_counts[b] > 0:
                        print(f"  Batch {b}:")
                        for i in range(min(packed_counts[b].item(), 5)):
                            state = packed_states[b, i]
                            valid = state[:, 0] != pad
                            if valid.any():
                                atoms = state[valid]
                                atoms_str = [self.debug_helper.atom_to_str(atom) for atom in atoms]
                                print(f"    Packed state {i}: [{', '.join(atoms_str)}]")
                            else:
                                print(f"    Packed state {i}: <empty> [BUG!]")
        
        if deduplicate:
            unique_states, unique_counts = deduplicate_states_packed(
                packed_states, packed_counts, pad, self.hash_cache
            )
            dedup_suffix = " (after dedup)"
        else:
            unique_states, unique_counts = packed_states, packed_counts
            dedup_suffix = " (no dedup)"
        
        # vi) Print final next states
        if verbose > 0:
            print_states(f"vi) FINAL NEXT STATES{dedup_suffix}", unique_states, unique_counts)

        # ---- 7) pad back to original batch
        write_mask = unique_counts > 0
        write_idx = torch.arange(B, device=device)[write_mask]
        if write_idx.numel() > 0:
            # Write as many atoms as fit in the output tensor
            max_atoms_write = min(max_atoms_comb, max_atoms_output)
            all_derived[write_idx, :unique_states.shape[1], :max_atoms_write] = unique_states[write_idx, :, :max_atoms_write]
            derived_counts[write_idx] = unique_counts[write_idx]
        
        # ---- 8) Handle active states that ended up with NO derived states -> False()
        # These are active states that had unifications but after pruning/dedup have nothing
        # OPTIMIZED: Create mask directly on GPU without .item() calls
        is_active = torch.zeros(B, dtype=torch.bool, device=device)
        is_active[active_idx] = True
        no_derived = (derived_counts == 0) & is_active
        no_derived_idx = torch.arange(B, device=device)[no_derived]
        if no_derived_idx.numel() > 0 and self.false_tensor is not None:
            all_derived[no_derived_idx, 0, 0] = self.false_tensor.squeeze(0)
            derived_counts[no_derived_idx] = 1

        return (all_derived, derived_counts, updated_next_var_indices)

    # ---- helpers ----
    @torch.no_grad()
    def _prune_and_prove(self, candidates: Tensor, atom_counts: Tensor, 
                         excluded_queries: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Remove ground atoms known to be facts and detect empty (proof) states.
        Also removes atoms that match the original excluded query to prevent circular proofs.
        
        candidates:  [N, M, 3]
        atom_counts: [N]
        excluded_queries: [N, max_atoms, 3] - original query for each candidate to exclude
        """
        device, pad = candidates.device, self.padding_idx
        if candidates.numel() == 0:
            return (candidates, atom_counts, atom_counts == 0)

        N, M = candidates.shape[:2]
        valid = candidates[:, :, 0] != pad
        args = candidates[:, :, 1:3]
        is_const = (args <= self.constant_no).all(dim=2)
        ground = valid & is_const

        drop_mask = torch.zeros_like(valid)
        
        # Drop ground atoms that are facts
        if ground.any():
            atoms = candidates.reshape(-1, 3)[ground.flatten()]
            is_fact = self.fact_index.contains(atoms)
            dm = torch.zeros_like(ground.flatten(), dtype=torch.bool, device=device)
            dm[ground.flatten()] = is_fact
            drop_mask = dm.view(N, M)
        
        # DON'T drop ground atoms that match the original excluded query
        # We want to keep them so the state is NOT considered a proof (no circular reasoning)
        if excluded_queries is not None and ground.any():
            # For each candidate, check if any ground atom matches the first atom of excluded_query
            excluded_first_atoms = excluded_queries[:, 0, :]  # [N, 3]
            # Expand for broadcasting: [N, 1, 3] vs [N, M, 3]
            excluded_expanded = excluded_first_atoms.unsqueeze(1).expand(-1, M, -1)
            # Check if ground atoms match excluded query
            matches_excluded = ground & (candidates == excluded_expanded).all(dim=2)
            # INVERT the drop mask: keep (don't drop) atoms that match the excluded query
            # If an atom was marked to drop because it's a fact, but it matches excluded query, unmark it
            drop_mask = drop_mask & ~matches_excluded

        keep = valid & ~drop_mask
        pruned_counts = keep.sum(dim=1)
        is_proof = pruned_counts == 0

        pos = torch.cumsum(keep.long(), dim=1) - 1
        # OPTIMIZED: Use M as upper bound to avoid .item() call
        out_len_val = M
        out = torch.full((N, out_len_val, 3), pad, dtype=torch.long, device=device)
        if keep.any():
            src_idx = torch.arange(M, device=device).unsqueeze(0).expand_as(keep)
            row_ids = torch.arange(N, device=device).unsqueeze(1).expand_as(keep)
            rows = row_ids[keep]
            cols = src_idx[keep]
            out[rows, pos[keep]] = candidates[rows, cols]
        return (out, pruned_counts, is_proof)

    @torch.no_grad()
    def _canonicalize_variables(self,
                                states: Tensor,
                                counts: Tensor,
                                constant_no: int,
                                next_var_start: Tensor,
                                pad: int) -> Tuple[Tensor, Tensor]:
        """
        Alpha-rename variables per state to a compact, deterministic range:
          newVar_i := next_var_start[state] + rank(i)
        Deterministic by first-occurrence position (row-major over arg slots).
        Returns:
          canon_states:         [N, M, 3]
          updated_next_var_end: [N]
        """
        if states.numel() == 0:
            return states, next_var_start

        device = states.device
        N, M, _ = states.shape
        args = states[:, :, 1:3]
        is_var = (args > constant_no) & (args != pad)

        # Linear position per arg slot (0..M*2-1), used to order first occurrences
        lin_pos = (torch.arange(M, device=device).unsqueeze(1).repeat(1, 2)
                   .reshape(1, M, 2)
                   .expand(N, -1, -1))
        lin_pos = lin_pos.clone()
        lin_pos[:, :, 1] += M  # distinguish arg-0 and arg-1 slots

        # Build per-occurrence vectors filtered to variable positions
        occ_state = torch.arange(N, device=device).view(N, 1, 1).expand_as(args)
        SHIFT = self.runtime_var_end_index + 2  # > any index
        combined = args + occ_state * SHIFT  # unique per (state, var_id)
        occ_mask = is_var

        # Gather occurrences
        occ_keys = combined[occ_mask]               # [T]
        occ_lin  = lin_pos[occ_mask]                # [T]
        if occ_keys.numel() == 0:
            # no variables anywhere
            return states, next_var_start

        # Map each (state,var) group to its first-occurrence position
        # 1) group id per occurrence (global, dense): by unique()
        uniq, inv = torch.unique(occ_keys, return_inverse=True, sorted=True)  # uniq sorted ascending
        # sort occurrences by (group, first position)
        BIG = (M * 2 + 1)
        key2 = inv.long() * BIG + occ_lin.long()
        order = torch.argsort(key2)
        inv_sorted = inv[order]
        lin_sorted = occ_lin[order]
        seg_start = torch.ones_like(inv_sorted, dtype=torch.bool)
        seg_start[1:] = inv_sorted[1:] != inv_sorted[:-1]
        first_pos_per_group = lin_sorted[seg_start]   # [G]
        # 2) rank groups inside each state by first occurrence
        group_state = (uniq // SHIFT).long()          # [G] -> state id
        key3 = group_state * BIG + first_pos_per_group
        ord3 = torch.argsort(key3)
        sorted_state = group_state[ord3]
        # Detect state boundaries
        state_boundary = torch.ones_like(sorted_state, dtype=torch.bool)
        state_boundary[1:] = sorted_state[1:] != sorted_state[:-1]
        # Compute rank within state using cumsum with reset at boundaries
        # state_group_id assigns a unique ID to each contiguous run of the same state
        state_group_id = torch.cumsum(state_boundary.long(), dim=0) - 1
        # Count position within each state group
        # For each state_group_id, we want positions 0, 1, 2, ...
        # Use bincount to get counts per group, then cumsum within each group
        ones = torch.ones_like(state_group_id)
        # Cumulative count within group: current_position - group_start_position
        group_starts = torch.zeros(state_group_id.max() + 1, device=device, dtype=torch.long)
        group_starts.scatter_reduce_(0, state_group_id, torch.arange(len(state_group_id), device=device), reduce='amin', include_self=True)
        rank_sorted = torch.arange(len(state_group_id), device=device) - group_starts[state_group_id]
        # scatter back to original group order
        ranks = torch.empty_like(rank_sorted)
        ranks[ord3] = rank_sorted                     # [G]

        # base per group
        base_per_group = next_var_start[group_state]
        new_id_per_group = base_per_group + ranks
        max_runtime_idx = self.runtime_var_end_index
        if new_id_per_group.numel() > 0:
            overflow_groups = new_id_per_group > max_runtime_idx
            if overflow_groups.any():
                overflow_states = torch.unique(group_state[overflow_groups])
                raise RuntimeError(
                    f"Canonicalization exceeded runtime variable budget "
                    f"(limit={self.runtime_var_end_index}) for {overflow_states.numel()} states. "
                    "Increase max_total_vars or adjust rule complexity."
                )

        # assign new variable ids per occurrence, then back to states
        new_id_per_occ = new_id_per_group[inv]        # [T]
        new_args = args.clone()
        new_args[occ_mask] = new_id_per_occ.long()
        canon_states = torch.cat([states[:, :, 0:1], new_args], dim=2)

        updated_next = next_var_start
        # update next_var index: base + (#unique vars in each state)
        vars_per_state = torch.bincount(group_state, minlength=N)
        updated_next = updated_next + vars_per_state
        return canon_states, updated_next

    # ---- misc ----
    def is_true_state(self, state: Tensor) -> bool:
        """
        Check if state contains only the True predicate (with padding).
        
        Args:
            state: [max_atoms, 3] tensor
            
        Returns:
            True if state has exactly 1 non-padding atom and it equals True predicate
        """
        if self.true_tensor is None:
            return False
        # Count non-padding atoms
        non_padding = (state[:, 0] != self.padding_idx).sum()
        if non_padding != 1:
            return False
        # Compare first atom only
        return torch.equal(state[0], self.true_tensor.squeeze(0))

    def is_false_state(self, state: Tensor) -> bool:
        """
        Check if state contains only the False predicate (with padding).
        
        Args:
            state: [max_atoms, 3] tensor
            
        Returns:
            True if state has exactly 1 non-padding atom and it equals False predicate
        """
        if self.false_tensor is None:
            return False
        # Count non-padding atoms
        non_padding = (state[:, 0] != self.padding_idx).sum()
        if non_padding != 1:
            return False
        # Compare first atom only
        return torch.equal(state[0], self.false_tensor.squeeze(0))

    def get_false_state(self) -> Tensor:
        if self.false_tensor is None:
            raise ValueError('False predicate not defined')
        return self.false_tensor.clone()

    def get_true_state(self) -> Tensor:
        if self.true_tensor is None:
            raise ValueError('True predicate not defined')
        return self.true_tensor.clone()

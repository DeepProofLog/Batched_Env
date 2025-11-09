# unification_engine.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor




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
    
    # Find unique mask: consecutive elements with different hashes
    # [B, max_states]
    unique_mask = torch.ones((B, max_states), dtype=torch.bool, device=device)
    unique_mask[:, 1:] = sorted_hashes[:, 1:] != sorted_hashes[:, :-1]
    
    # Also mask out invalid entries (those past state_counts)
    valid_after_sort = sort_indices < state_counts.unsqueeze(1)
    unique_mask = unique_mask & valid_after_sort
    
    # Count uniques per batch
    unique_counts = unique_mask.sum(dim=1)  # [B]
    # OPTIMIZED: Cache max result
    max_unique_tensor = unique_counts.max() if unique_counts.numel() > 0 else torch.tensor(0, device=device)
    max_unique = int(max_unique_tensor.item())
    
    if max_unique == 0:
        return torch.full((B, 0, max_atoms, arity), padding_idx, dtype=states.dtype, device=device), unique_counts
    
    # Compact using cumsum to create packed indices (NO LOOP!)
    # Create position indices for each unique element
    # cumsum gives us the position within the unique subset
    positions = torch.cumsum(unique_mask.long(), dim=1) - 1  # [B, max_states] - position in output (0-indexed)
    
    # Mask to valid range
    valid_positions = positions < max_unique
    final_mask = unique_mask & valid_positions
    
    # Create scatter indices: [B, max_unique, max_atoms, 3]
    # We'll gather from sorted_states into unique_states
    # Use masked_scatter or advanced indexing
    
    # Simpler approach: use boolean indexing with reshape
    # For each batch, select the unique states
    # Since we need to pad to max_unique, we'll use a different approach:
    
    # Create output tensor
    unique_states = torch.full((B, max_unique, max_atoms, arity), padding_idx, dtype=states.dtype, device=device)
    
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
            h = _pack_triples_64(facts.int(), pack_base)
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
    args = goals_batch[:, :, 1:]

    # Build a dense mapping per batch row.
    # OPTIMIZED: Compute max once, cache the result
    goals_flat = goals_batch[:, :, 1:].flatten().long()
    subs_flat = substitutions_batch.flatten().long().clamp_min(0)
    max_idx_tensor = torch.max(torch.cat([goals_flat, subs_flat], dim=0))
    max_idx = int(max_idx_tensor.item()) + 1

    mapping = torch.arange(max_idx, device=goals_batch.device).unsqueeze(0).expand(B, -1).clone()
    valid = substitutions_batch[..., 0] != padding_idx
    if valid.any():
        br = torch.arange(B, device=goals_batch.device).unsqueeze(1).expand_as(substitutions_batch[..., 0])[valid]
        frm = substitutions_batch[valid][:, 0].long()
        to  = substitutions_batch[valid][:, 1].long()
        mapping[br, frm] = to

    out_args = mapping[torch.arange(B, device=goals_batch.device).view(B, 1, 1),
                       args.long()]
    return torch.cat([preds, out_args], dim=2)


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
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return (mask, subs)

    q = q_args[idx]
    t = t_args[idx]
    qv = q >= var_start
    tv = t >= var_start

    # Bind variable to constant or variable (both directions)
    # slot 0 corresponds to arg #0, slot 1 to arg #1
    # Case 1: qVar - tConst
    tc = ~tv & ~t.eq(0)  # t is a valid index (not padding)
    qv_tc = qv & tc
    if qv_tc.any():
        rows, cols = qv_tc.nonzero(as_tuple=True)
        subs[idx[rows], cols, 0] = q[rows, cols].long()
        subs[idx[rows], cols, 1] = t[rows, cols].long()

    # Case 2: tVar - qConst
    qc = ~qv & ~q.eq(0)
    tv_qc = tv & qc
    if tv_qc.any():
        rows, cols = tv_qc.nonzero(as_tuple=True)
        subs[idx[rows], cols, 0] = t[rows, cols].long()
        subs[idx[rows], cols, 1] = q[rows, cols].long()

    # Case 3: both variables: bind one variable to the other
    vv = qv & tv
    if vv.any():
        rows, cols = vv.nonzero(as_tuple=True)
        # consistently bind tVar -> qVar (arbitrary choice, but deterministic)
        subs[idx[rows], cols, 0] = t[rows, cols].long()
        subs[idx[rows], cols, 1] = q[rows, cols].long()

    # ---- Consistency check: the same variable cannot map to two different targets
    s = subs[idx]  # [K,2,2]
    froms = s[:, :, 0]
    tos   = s[:, :, 1]
    have_two = (froms != padding_idx).sum(1) == 2
    if have_two.any():
        f0 = froms[have_two, 0]
        f1 = froms[have_two, 1]
        t0 = tos[have_two, 0]
        t1 = tos[have_two, 1]
        conflict = (f0 == f1) & (t0 != t1)
        if conflict.any():
            bad_rows = idx[have_two.nonzero(as_tuple=True)[0][conflict]]
            mask[bad_rows] = False
            subs[bad_rows] = padding_idx

    return (mask, subs.long())


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
    # OPTIMIZED: Check using tensor operations, avoid .item()
    if lens.numel() == 0 or lens.sum() == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z
    keep = lens > 0
    if not keep.any():
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z
    lens = lens[keep]
    starts = starts[keep]
    kept_qidx = keep.nonzero(as_tuple=True)[0]             # [A_kept]
    # Flat cartesian product without loops
    row_ids = torch.repeat_interleave(torch.arange(lens.numel(), device=device), lens)
    prefix = torch.cumsum(lens, dim=0) - lens
    offs = torch.repeat_interleave(prefix, lens)
    # OPTIMIZED: Use lens.sum() directly as a tensor until needed for arange
    total_len = lens.sum()
    pos_in_row = torch.arange(total_len.item(), device=device) - offs
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
    max_output_per_query: int = 100,
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
        excluded_queries: [A] bool/0-1 mask; when True for row *i*, drop the trivial
                          self-match `queries[i] == fact` if the query is ground.
        labels:           (ignored) kept only for API compatibility.
        max_output_per_query: per-owner cap.

    Returns:
        states: [A, K, M, 3] packed successors.
        counts: [A]          valid successors per input row (≤ K).
    """
    device = queries.device
    pad = engine.padding_idx
    A, G = remaining_goals.shape[:2]

    # Fast exits --------------------------------------------------------------
    if A == 0 or engine.facts_idx.numel() == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    # 1) Ragged pairing (query → contiguous block of matching facts) ----------
    # OPTIMIZATION: predicate_range_map is already int32 on device, avoid redundant .to()
    prm = engine.predicate_range_map
    seg_starts = prm[:, 0].long()
    seg_lens   = (prm[:, 1] - prm[:, 0]).long()

    qi, fi = _pairs_via_predicate_ranges(pred_indices, seg_starts, seg_lens)  # [L], [L]
    if qi.numel() == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    # Optional: exclude trivial self-matches (labels not needed) --------------
    if excluded_queries is not None:
        # excluded_queries is [A, 3] - the original queries to exclude
        # For each (qi, fi) pair, check if queries[qi] == excluded_queries[qi] == facts[fi]
        # If so, this is a trivial self-match and should be dropped
        
        ex_queries = excluded_queries.index_select(0, qi)  # [L, 3] - get excluded query for each pair
        q_tmp = queries.index_select(0, qi)                # [L, 3] - get current query for each pair
        f_tmp = engine.facts_idx.index_select(0, fi)       # [L, 3] - get matched fact for each pair
        
        # Check if query matches excluded query (i.e., this is the original query)
        query_is_excluded = (q_tmp == ex_queries).all(dim=1)  # [L]
        
        # Check if query is ground (all arguments are constants)
        ground_q = (q_tmp[:, 1:] <= engine.constant_no).all(dim=1)  # [L]
        
        # Check if query matches fact
        same = (q_tmp == f_tmp).all(dim=1)  # [L]
        
        # Drop if: query is excluded AND ground AND matches fact (trivial self-match)
        drop = query_is_excluded & ground_q & same  # [L]
        
        if drop.any():
            keep = ~drop
            qi = qi[keep]
            fi = fi[keep]
            if qi.numel() == 0:
                empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
                return empty, torch.zeros(A, dtype=torch.long, device=device)

    # Gather pairs and unify pairwise (still loop-free) -----------------------
    q_pairs = queries.index_select(0, qi)                  # [L, 3]
    f_pairs = engine.facts_idx.index_select(0, fi)         # [L, 3]

    ok, subs = _unify_one_to_one(q_pairs, f_pairs, engine.constant_no, pad)
    if not ok.any():
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    # Keep successful matches only -------------------------------------------
    qi = qi[ok]
    subs = subs[ok].view(-1, 2, 2)                         # [M', 2, 2]

    # 2) Instantiate remaining goals with the substitutions ------------------
    remaining_sel = remaining_goals.index_select(0, qi)    # [M', G, 3]
    remain_inst = _apply_substitutions(
        remaining_sel, subs.view(subs.shape[0], -1, 2), pad
    )                                                      # [M', G, 3]

    # 3) Compact to active goal count per-row (drop padded tail) -------------
    pos_g = torch.arange(G, device=device).unsqueeze(0).expand_as(remain_inst[:, :, 0])
    take_g = pos_g < remaining_counts.index_select(0, qi).unsqueeze(1)   # [M', G]

    counts = take_g.sum(1)
    # OPTIMIZED: Cache max result
    max_cols_tensor = counts.max() if counts.numel() > 0 else torch.tensor(1, device=device)
    max_cols = int(max_cols_tensor.item())
    compact = torch.full((counts.shape[0], max_cols, 3), pad, dtype=torch.int32, device=device)
    if take_g.any():
        row = torch.arange(counts.shape[0], device=device).unsqueeze(1).expand_as(take_g)
        pos = torch.cumsum(take_g.long(), dim=1) - 1
        src = torch.arange(G, device=device).unsqueeze(0).expand_as(take_g)
        compact[row[take_g], pos[take_g]] = remain_inst[row[take_g], src[take_g]]

    # 4) Cap per-owner fanout (K per original query row) ---------------------
    K_per_owner = torch.bincount(qi, minlength=A)
    # OPTIMIZED: Cache max result
    K_max_tensor = K_per_owner.max() if K_per_owner.numel() > 0 else torch.tensor(0, device=device)
    K = int(min(max_output_per_query, K_max_tensor.item()))
    if K == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    out = torch.full((A, K, compact.shape[1], 3), pad, dtype=torch.int32, device=device)
    counts_per_q = torch.zeros(A, dtype=torch.long, device=device)

    # Stable segmented take by owner id (loop-free) --------------------------
    owner_sorted, perm = torch.sort(qi)
    comp_sorted = compact.index_select(0, perm)
    seg_start = torch.ones_like(owner_sorted, dtype=torch.bool, device=device)
    seg_start[1:] = owner_sorted[1:] != owner_sorted[:-1]
    pos_in_owner = torch.cumsum(seg_start, dim=0) - 1
    keep = pos_in_owner < K
    if keep.any():
        db = owner_sorted[keep]
        ds = pos_in_owner[keep]
        out[db, ds] = comp_sorted[keep]
        maxpos = torch.full((A,), -1, dtype=torch.long, device=device)
        maxpos.index_put_((db,), ds, accumulate=True)
        counts_per_q = torch.clamp(maxpos + 1, min=0, max=K)

    return out, counts_per_q

@torch.no_grad()
def _unify_with_rules(
    engine,
    queries: Tensor,
    remaining_goals: Tensor,
    remaining_counts: Tensor,
    pred_indices: Tensor,
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
        max_output_per_query: cap on branching factor per active state.

    Returns:
        states: [A, K, M, 3] packed successor states (rule body ⊕ remaining goals), padded.
        counts: [A]          number of valid successors per input row (≤ K).
    """
    device = queries.device
    pad = engine.padding_idx
    A, G = remaining_goals.shape[:2]

    # Fast exits --------------------------------------------------------------
    if A == 0 or engine.rules_heads_idx.numel() == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    # 1) Sort rule heads by predicate and build ranges -----------------------
    # heads_sorted/bodies_sorted/lens_sorted share the same permutation.
    order = torch.argsort(engine.rules_heads_idx[:, 0])
    heads_sorted = engine.rules_heads_idx.index_select(0, order)  # [R, 3]
    bodies_sorted = engine.rules_idx.index_select(0, order)       # [R, Bmax, 3]
    lens_sorted = engine.rule_lens.index_select(0, order)         # [R]

    pcol = heads_sorted[:, 0]
    uniq, counts = torch.unique_consecutive(pcol, return_counts=True)
    # OPTIMIZATION: counts is already long, avoid redundant .to()
    starts = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts[:-1]]),
        dim=0,
    )
    ends = torch.cumsum(counts, dim=0)

    # Compact range arrays sized to max predicate id encountered
    # OPTIMIZED: Cache max result
    max_pred_tensor = pcol.max()
    max_pred = int(max_pred_tensor.item())
    seg_starts = torch.zeros((max_pred + 1,), dtype=torch.long, device=device)
    seg_lens   = torch.zeros((max_pred + 1,), dtype=torch.long, device=device)
    seg_starts[uniq] = starts
    seg_lens[uniq] = (ends - starts)

    # 2) Ragged pairing (query, rule) via ranges -----------------------------
    qi, ri = _pairs_via_predicate_ranges(pred_indices, seg_starts, seg_lens)
    if qi.numel() == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    q_pairs = queries.index_select(0, qi)                 # [L, 3]
    h_pairs = heads_sorted.index_select(0, ri)            # [L, 3]

    ok, subs = _unify_one_to_one(q_pairs, h_pairs, engine.constant_no, pad)
    if not ok.any():
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    # Keep successful matches only -------------------------------------------
    qi = qi[ok]
    ri = ri[ok]
    subs = subs[ok].view(-1, 2, 2)

    # 3) Instantiate (rule body ⊕ remaining goals) ---------------------------
    bodies_all = bodies_sorted.index_select(0, ri)        # [M', Bmax, 3]
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
    # OPTIMIZED: Cache max result
    max_cols_tensor = counts.max() if counts.numel() > 0 else torch.tensor(1, device=device)
    max_cols = int(max_cols_tensor.item())

    # Concatenate body and remaining, then compact to `max_cols`
    cat_full = torch.full((counts.shape[0], Bmax + G, 3), pad, dtype=torch.int32, device=device)
    cat_full[:, :Bmax] = bodies_inst
    cat_full[:, Bmax:] = remain_inst
    keep_mask = torch.cat([take_b, take_g], dim=1)

    pos = torch.cumsum(keep_mask.long(), dim=1) - 1
    row = torch.arange(counts.shape[0], device=device).unsqueeze(1).expand_as(pos)
    if keep_mask.any():
        cat_compact = torch.full((counts.shape[0], max_cols, 3), pad, dtype=torch.int32, device=device)
        src = torch.arange(keep_mask.shape[1], device=device).unsqueeze(0).expand_as(keep_mask)
        cat_compact[row[keep_mask], pos[keep_mask]] = cat_full[row[keep_mask], src[keep_mask]]
    else:
        cat_compact = torch.full((counts.shape[0], 1, 3), pad, dtype=torch.int32, device=device)

    # 4) Cap per-owner fanout (K per original query row) ---------------------
    K_per_owner = torch.bincount(qi, minlength=A)
    # OPTIMIZED: Cache max result
    K_max_tensor = K_per_owner.max() if K_per_owner.numel() > 0 else torch.tensor(0, device=device)
    K = int(min(max_output_per_query, K_max_tensor.item()))
    if K == 0:
        empty = torch.full((A, 0, 1, 3), pad, dtype=torch.int32, device=device)
        return empty, torch.zeros(A, dtype=torch.long, device=device)

    out = torch.full((A, K, cat_compact.shape[1], 3), pad, dtype=torch.int32, device=device)
    counts_per_q = torch.zeros(A, dtype=torch.long, device=device)

    # Stable segmented take by owner id (loop-free) --------------------------
    owner_sorted, perm = torch.sort(qi)
    cat_sorted = cat_compact.index_select(0, perm)
    seg_start = torch.ones_like(owner_sorted, dtype=torch.bool, device=device)
    seg_start[1:] = owner_sorted[1:] != owner_sorted[:-1]
    pos_in_owner = torch.cumsum(seg_start, dim=0) - 1
    keep = pos_in_owner < K
    if keep.any():
        db = owner_sorted[keep]
        ds = pos_in_owner[keep]
        out[db, ds] = cat_sorted[keep]
        maxpos = torch.full((A,), -1, dtype=torch.long, device=device)
        maxpos.index_put_((db,), ds, accumulate=True)
        counts_per_q = torch.clamp(maxpos + 1, min=0, max=K)

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
                 pack_base: Optional[int] = None):
        self.device = device
        self.padding_idx = int(padding_idx)
        self.constant_no = int(constant_no)
        self.runtime_var_end_index = int(runtime_var_end_index)
        self.max_arity = int(max_arity)
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx

        # Handle None properly for tensors
        if facts_idx is None:
            self.facts_idx = torch.empty((0, 3), dtype=torch.int32, device=device)
        else:
            self.facts_idx = facts_idx.to(device=device, dtype=torch.int32)
            
        if rules_idx is None:
            self.rules_idx = torch.empty((0, 1, 3), dtype=torch.int32, device=device)
        else:
            self.rules_idx = rules_idx.to(device=device, dtype=torch.int32)
            
        if rule_lens is None:
            self.rule_lens = torch.empty((0,), dtype=torch.int32, device=device)
        else:
            self.rule_lens = rule_lens.to(device=device, dtype=torch.int32)
            
        if rules_heads_idx is None:
            self.rules_heads_idx = torch.empty((0, 3), dtype=torch.int32, device=device)
        else:
            self.rules_heads_idx = rules_heads_idx.to(device=device, dtype=torch.int32)

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
            self.predicate_range_map = predicate_range_map.to(device=device, dtype=torch.int32)

        # helper tensors for TRUE/FALSE atoms
        pad = self.padding_idx
        if self.true_pred_idx is not None:
            self.true_tensor = torch.tensor([[self.true_pred_idx, pad, pad]], dtype=torch.int32, device=device)
        else:
            self.true_tensor = None
        if self.false_pred_idx is not None:
            self.false_tensor = torch.tensor([[self.false_pred_idx, pad, pad]], dtype=torch.int32, device=device)
        else:
            self.false_tensor = None

        # indices for fast membership & dedup
        self.fact_index = GPUFactIndex(self.facts_idx, self.pack_base)
        self.hash_cache = GPUHashCache(device)

    # ---- factory ----
    @classmethod
    def from_index_manager(cls, im, take_ownership: bool = False) -> 'UnificationEngine':
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
            pack_base=getattr(im, 'total_vocab_size', None)
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
                           verbose: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        One-step expansion, strictly ordered:

        1) preprocessing
        2) rule unification
        3) fact unification
        4) prune ground facts & detect immediate proofs
        5) combine
        6) canonicalise + dedup
        7) pad back
        """
        device = current_states.device
        pad = self.padding_idx
        B, max_atoms, _ = current_states.shape

        # ---- 1) preprocessing
        valid_mask = current_states[:, :, 0] != pad
        has_any = valid_mask.any(dim=1)
        empty_states = ~has_any

        has_false = (current_states[:, :, 0] == self.false_pred_idx).any(dim=1) if self.false_pred_idx is not None \
                    else torch.zeros(B, dtype=torch.bool, device=device)
        only_true = has_any & ((current_states[:, :, 0] == self.true_pred_idx) | ~valid_mask).all(dim=1) \
                    if self.true_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=device)

        terminal = empty_states | has_false | only_true

        max_derived_per_state = 128
        all_derived = torch.full((B, max_derived_per_state, max_atoms, 3), pad, dtype=torch.int32, device=device)
        derived_counts = torch.zeros(B, dtype=torch.long, device=device)
        updated_next_var_indices = next_var_indices.clone()

        # emit trivials
        if only_true.any():
            all_derived[only_true, 0, 0] = self.true_tensor.squeeze(0)
            derived_counts[only_true] = 1
        if (has_false | empty_states).any():
            mf = (has_false | empty_states) & ~only_true
            if self.false_tensor is None:
                raise ValueError("False predicate not defined")
            all_derived[mf, 0, 0] = self.false_tensor.squeeze(0)
            derived_counts[mf] = 1

        active = ~terminal
        if not active.any():
            return (all_derived, derived_counts, updated_next_var_indices)

        active_idx = active.nonzero(as_tuple=True)[0]
        A = active_idx.shape[0]
        active_states = current_states[active_idx]
        active_valid  = valid_mask[active_idx]

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
                src_b = arangeA.unsqueeze(1).expand(-1, max_atoms)[scatter]
                src_a = pos_grid[scatter]
                dst_b = src_b
                dst_a = torch.where(before, pos_grid, pos_grid - 1)[scatter]
                remaining_goals[dst_b, dst_a] = active_states[src_b, src_a]

        preds = queries[:, 0]

        # ---- 2) rules
        rule_states, rule_counts = _unify_with_rules(
            self, queries, remaining_goals, remaining_counts, preds
        )

        # ---- 3) facts
        fact_states, fact_counts = _unify_with_facts(
            self, queries, remaining_goals, remaining_counts, preds,
            excluded_queries=excluded_queries[active_idx] if excluded_queries is not None else None,
        )

        # ---- 4) detect immediate proof (remaining goals vanish due to fact match)
        immediate = (remaining_counts == 0) & (fact_counts > 0)

        # ---- 5) combine candidates
        Rmax, Fmax = (rule_states.shape[1], fact_states.shape[1])
        max_atoms_rules = rule_states.shape[2] if rule_states.numel() > 0 else max_atoms
        max_atoms_facts = fact_states.shape[2] if fact_states.numel() > 0 else max_atoms
        max_atoms_comb = max(max_atoms_rules, max_atoms_facts, max_atoms)
        total_slots = Rmax + Fmax

        packed = torch.full((A, total_slots, max_atoms_comb, 3), pad, dtype=torch.int32, device=device)
        if Rmax > 0:
            packed[:, :Rmax, :max_atoms_rules] = rule_states
        if Fmax > 0:
            packed[:, Rmax:Rmax + Fmax, :max_atoms_facts] = fact_states

        owner_rule_mask = torch.arange(Rmax, device=device).unsqueeze(0).expand(A, -1) < rule_counts.unsqueeze(1)
        owner_fact_mask = torch.arange(Fmax, device=device).unsqueeze(0).expand(A, -1) < fact_counts.unsqueeze(1)
        owner_mask = torch.cat([owner_rule_mask, owner_fact_mask], dim=1) & (~immediate).unsqueeze(1)

        if not owner_mask.any():
            if immediate.any():
                dst = active_idx[immediate]
                all_derived[dst, 0, 0] = self.true_tensor.squeeze(0)
                derived_counts[dst] = 1
            return (all_derived, derived_counts, updated_next_var_indices)

        owner_ids, slot_ids = owner_mask.nonzero(as_tuple=True)
        candidates = packed[owner_ids, slot_ids]
        cand_counts = (candidates[:, :, 0] != pad).sum(dim=1)

        # ---- 4-bis) prune ground facts & mark proofs
        pruned_states, pruned_counts, is_proof = self._prune_and_prove(candidates, cand_counts)
        owners_for_cands = active_idx[owner_ids]

        if immediate.any():
            dst = active_idx[immediate]
            all_derived[dst, 0, 0] = self.true_tensor.squeeze(0)
            derived_counts[dst] = 1
        if is_proof.any():
            proof_owners = torch.unique(owners_for_cands[is_proof])
            all_derived[proof_owners, 0, 0] = self.true_tensor.squeeze(0)
            derived_counts[proof_owners] = 1

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

        if canon_states.shape[0] > 0:
            # keep the maximum next_var index across expansions for each owner
            agg = torch.full((B,), 0, dtype=next_var_indices.dtype, device=device)
            agg.index_put_((owners_kept,), next_vars_per_state, accumulate=True)
            updated_next_var_indices = torch.maximum(updated_next_var_indices, agg)

        # ---- pack back per original owner
        K = min(total_slots, max_derived_per_state)
        packed_states = torch.full((B, K, max_atoms_comb, 3), pad, dtype=torch.int32, device=device)
        packed_counts = torch.zeros(B, dtype=torch.long, device=device)
        if owners_kept.numel() > 0:
            sort_vals, sort_idx = torch.sort(owners_kept)
            states_sorted = canon_states[sort_idx]
            counts_sorted = counts_kept[sort_idx]
            seg_start = torch.ones_like(sort_vals, dtype=torch.bool, device=device)
            seg_start[1:] = sort_vals[1:] != sort_vals[:-1]
            pos = torch.cumsum(seg_start.long(), dim=0) - 1
            pos = torch.minimum(pos, torch.full_like(pos, K - 1))
            dst_b = sort_vals
            dst_s = pos
            
            # Ensure states_sorted matches max_atoms_comb dimension
            if states_sorted.shape[1] < max_atoms_comb:
                # Pad atoms dimension
                padding = torch.full((states_sorted.shape[0], max_atoms_comb - states_sorted.shape[1], 3),
                                   pad, dtype=torch.int32, device=device)
                states_sorted = torch.cat([states_sorted, padding], dim=1)
            elif states_sorted.shape[1] > max_atoms_comb:
                # Truncate atoms dimension
                states_sorted = states_sorted[:, :max_atoms_comb]
            
            packed_states[dst_b, dst_s] = states_sorted
            owner_maxpos = torch.full((B,), -1, dtype=torch.long, device=device)
            owner_maxpos.index_put_((dst_b,), dst_s, accumulate=True)
            packed_counts = torch.clamp(owner_maxpos + 1, min=0, max=K)

        # ---- 6-bis) dedup
        unique_states, unique_counts = deduplicate_states_packed(
            packed_states, packed_counts, pad, self.hash_cache
        )

        # ---- 7) pad back to original batch
        write_mask = unique_counts > 0
        write_idx = write_mask.nonzero(as_tuple=True)[0]
        if write_idx.numel() > 0:
            max_atoms_write = min(max_atoms_comb, max_atoms)
            all_derived[write_idx, :unique_states.shape[1], :max_atoms_write] = unique_states[write_idx, :, :max_atoms_write]
            derived_counts[write_idx] = unique_counts[write_idx]

        return (all_derived, derived_counts, updated_next_var_indices)

    # ---- helpers ----
    @torch.no_grad()
    def _prune_and_prove(self, candidates: Tensor, atom_counts: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Remove ground atoms known to be facts and detect empty (proof) states.
        candidates:  [N, M, 3]
        atom_counts: [N]
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
        if ground.any():
            atoms = candidates.reshape(-1, 3)[ground.flatten()]
            is_fact = self.fact_index.contains(atoms)
            dm = torch.zeros_like(ground.flatten(), dtype=torch.bool, device=device)
            dm[ground.flatten()] = is_fact
            drop_mask = dm.view(N, M)

        keep = valid & ~drop_mask
        pruned_counts = keep.sum(dim=1)
        is_proof = pruned_counts == 0

        pos = torch.cumsum(keep.long(), dim=1) - 1
        row = torch.arange(N, device=device).unsqueeze(1).expand_as(pos)
        # OPTIMIZED: Cache max result
        out_len_tensor = pruned_counts.max() if pruned_counts.numel() > 0 else torch.tensor(1, device=device)
        out_len = int(out_len_tensor.item())
        out = torch.full((N, out_len, 3), pad, dtype=torch.int32, device=device)
        if keep.any():
            src_idx = torch.arange(M, device=device).unsqueeze(0).expand_as(keep)
            out[row[keep], pos[keep]] = candidates[row[keep], src_idx[keep]]
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
        boundary = torch.ones_like(sorted_state, dtype=torch.bool)
        boundary[1:] = sorted_state[1:] != sorted_state[:-1]
        rank_sorted = torch.cumsum(boundary.long(), dim=0) - 1
        # scatter back to original group order
        ranks = torch.empty_like(rank_sorted)
        ranks[ord3] = rank_sorted                     # [G]

        # base per group
        base_per_group = next_var_start[group_state]
        new_id_per_group = base_per_group + ranks

        # assign new variable ids per occurrence, then back to states
        new_id_per_occ = new_id_per_group[inv]        # [T]
        new_args = args.clone()
        new_args[occ_mask] = new_id_per_occ.int()
        canon_states = torch.cat([states[:, :, 0:1], new_args], dim=2)

        # update next_var index: base + (#unique vars in each state)
        vars_per_state = torch.bincount(group_state, minlength=N)
        updated_next = next_var_start + vars_per_state
        return canon_states, updated_next

    # ---- misc ----
    def is_true_state(self, state: Tensor) -> bool:
        return self.true_tensor is not None and torch.equal(state.view(-1, 3), self.true_tensor)

    def is_false_state(self, state: Tensor) -> bool:
        return self.false_tensor is not None and torch.equal(state.view(-1, 3), self.false_tensor)

    def get_false_state(self) -> Tensor:
        if self.false_tensor is None:
            raise ValueError('False predicate not defined')
        return self.false_tensor.clone()

    def get_true_state(self) -> Tensor:
        if self.true_tensor is None:
            raise ValueError('True predicate not defined')
        return self.true_tensor.clone()

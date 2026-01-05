"""
Vectorized Unification Engine for torch.compile() compatibility.

This module provides a fully compilable version of the unification engine
with fixed-shape tensors throughout. All dynamic operations (nonzero, 
repeat_interleave, .any() branches) are replaced with masked operations.

Key Design Principles:
    - All tensors have fixed shapes [B, K_max, ...] where K_max = padding_states
    - Invalid entries are marked with padding_idx
    - No .item(), .any(), or .nonzero() calls
    - All branches replaced with torch.where

Usage:
    from unification import UnificationEngineVectorized
    
    engine = UnificationEngineVectorized.from_index_manager(index_manager)
    derived, counts, new_vars = engine.get_derived_states_compiled(
        states, next_vars, excluded
    )
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor

# Import base classes and utilities from main unification module
from typing import List, Dict, Any, Union
import torch._dynamo


# Cache for pack base tensors - avoids repeated torch.as_tensor allocations
_pack_base_cache: dict = {}

@torch.no_grad()
def _pack_triples_64(atoms: Tensor, base: int) -> Tensor:
    """
    Pack triples [pred, a, b] into single 64-bit integers for efficient set operations.
    Formula: ((pred * base) + arg0) * base + arg1
    """
    if atoms.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=atoms.device)
    
    # Extract components: [N] each
    p, a, b = atoms[:, 0].long(), atoms[:, 1].long(), atoms[:, 2].long()
    
    # Cache the base tensor per (base, device) to avoid repeated allocations
    cache_key = (base, atoms.device)
    if cache_key not in _pack_base_cache:
        _pack_base_cache[cache_key] = torch.tensor(base, dtype=torch.int64, device=atoms.device)
    base_t = _pack_base_cache[cache_key]
    
    return ((p * base_t) + a) * base_t + b


class GPUFactIndex:
    """Efficient GPU-based index for fast fact membership testing."""
    
    def __init__(self, facts: Tensor, pack_base: int):
        self.device = facts.device
        self.pack_base = int(pack_base)
        
        if facts.numel() == 0:
            self.fact_hashes = torch.empty(0, dtype=torch.int64, device=self.device)
        else:
            self.fact_hashes = _pack_triples_64(facts.long(), self.pack_base).sort()[0]

    @torch.no_grad()
    def contains(self, atoms: Tensor) -> Tensor:
        if atoms.numel() == 0 or self.fact_hashes.numel() == 0:
            return torch.zeros((atoms.shape[0],), dtype=torch.bool, device=atoms.device)
            
        keys = _pack_triples_64(atoms.long(), self.pack_base)
        idx  = torch.searchsorted(self.fact_hashes, keys)
        valid = idx < self.fact_hashes.shape[0]
        
        mask = torch.zeros_like(keys, dtype=torch.bool)
        mask[valid] = (self.fact_hashes[idx[valid]] == keys[valid])
        
        return mask


@torch.no_grad()
def apply_substitutions(goals: Tensor, subs_pairs: Tensor, padding_idx: int) -> Tensor:
    """Apply variable substitutions to goal atoms (optimized for S=2)."""
    if goals.numel() == 0:
        return goals
    
    N, M = goals.shape[:2]
    S = subs_pairs.shape[1]
    pad = padding_idx
    
    preds = goals[:, :, 0:1]
    args = goals[:, :, 1:]
    
    # OPTIMIZATION: Loop-unrolled for common S=2 case
    if S == 2:
        frm_0 = subs_pairs[:, 0, 0].view(N, 1, 1)
        to_0 = subs_pairs[:, 0, 1].view(N, 1, 1)
        frm_1 = subs_pairs[:, 1, 0].view(N, 1, 1)
        to_1 = subs_pairs[:, 1, 1].view(N, 1, 1)
        
        valid_0 = (frm_0 != pad)
        valid_1 = (frm_1 != pad)
        
        result_args = torch.where((args == frm_0) & valid_0, to_0.expand_as(args), args)
        result_args = torch.where((result_args == frm_1) & valid_1, to_1.expand_as(result_args), result_args)
        
        return torch.cat([preds, result_args], dim=2)
    
    # General case for S != 2
    valid = subs_pairs[..., 0] != pad
    frm = subs_pairs[:, :, 0].view(N, S, 1, 1)
    to_ = subs_pairs[:, :, 1].view(N, S, 1, 1)
    args_exp = args.view(N, 1, M, 2)
    valid_exp = valid.view(N, S, 1, 1)
    
    match = (args_exp == frm) & valid_exp
    any_match = match.any(dim=1)
    match_idx = match.long().argmax(dim=1)
    
    to_flat = subs_pairs[:, :, 1]
    match_idx_flat = match_idx.view(N, M * 2)
    to_gathered = to_flat.gather(1, match_idx_flat).view(N, M, 2)
    
    result_args = torch.where(any_match, to_gathered, args)
    return torch.cat([preds, result_args], dim=2)


@torch.no_grad()
def unify_one_to_one(
    queries: Tensor, 
    terms: Tensor, 
    constant_no: int, 
    padding_idx: int
) -> Tuple[Tensor, Tensor]:
    """Perform pairwise unification between queries and terms (optimized)."""
    device = queries.device
    L = queries.shape[0]
    
    if L == 0:
        return (torch.empty(0, dtype=torch.bool, device=device),
                torch.full((0, 2, 2), padding_idx, dtype=torch.long, device=device))

    var_start = constant_no + 1
    pad = padding_idx
    
    # Extract predicates and args
    pred_ok = (queries[:, 0] == terms[:, 0])
    q_args, t_args = queries[:, 1:], terms[:, 1:]  # [L, 2]
    
    # Compute masks once
    q_const = (q_args <= constant_no)
    t_const = (t_args <= constant_no)
    qv = (q_args >= var_start) & (q_args != pad)
    tv = (t_args >= var_start) & (t_args != pad)
    
    # Constant conflict check
    const_conflict = (q_const & t_const & (q_args != t_args)).any(dim=1)
    mask = pred_ok & ~const_conflict
    
    # OPTIMIZED: Compute substitutions in single vectorized pass
    case1 = qv & ~tv & (t_args != 0)
    case2 = ~qv & (q_args != 0) & tv
    case3 = qv & tv
    
    # Default: padding
    from_val = torch.full_like(q_args, pad)
    to_val = torch.full_like(q_args, pad)
    
    # Apply in reverse priority (case3 last gets overwritten by case2, then case1)
    from_val = torch.where(case3, t_args, from_val)
    to_val = torch.where(case3, q_args, to_val)
    
    from_val = torch.where(case2, t_args, from_val)
    to_val = torch.where(case2, q_args, to_val)
    
    from_val = torch.where(case1, q_args, from_val)
    to_val = torch.where(case1, t_args, to_val)
    
    # Stack into subs: [L, 2, 2]
    subs = torch.stack([from_val, to_val], dim=2)  # [L, 2, 2]
    
    # Consistency check: same var bound to different values
    same_var = (subs[:, 0, 0] == subs[:, 1, 0]) & (subs[:, 0, 0] != pad)
    diff_tgt = subs[:, 0, 1] != subs[:, 1, 1]
    conflict = same_var & diff_tgt
    mask = mask & ~conflict
    
    # Clear subs for failed unifications
    fail_mask = ~mask
    subs = torch.where(fail_mask.view(L, 1, 1), torch.full_like(subs, pad), subs)
    
    return mask, subs


# ============================================================================
# Vectorized Fact Lookup
# ============================================================================

def fact_contains(
    atoms: Tensor,              # [N, 3]
    fact_hashes: Tensor,        # [F] sorted fact hashes
    pack_base: int,
) -> Tensor:
    """
    Check if atoms exist in the fact set. Fully vectorized - no nonzero().
    
    This replaces GPUFactIndex.contains for compilation compatibility.
    Uses only searchsorted, gather, and element-wise ops.
    
    Args:
        atoms: [N, 3] query atoms
        fact_hashes: [F] sorted hashes of known facts
        pack_base: Packing base for hash computation
        
    Returns:
        mask: [N] boolean mask where True = atom is a known fact
    """
    device = atoms.device
    N = atoms.shape[0]
    
    if N == 0 or fact_hashes.numel() == 0:
        return torch.zeros(N, dtype=torch.bool, device=device)
    
    # Pack query atoms to hashes
    keys = _pack_triples_64(atoms.long(), pack_base)  # [N]
    
    # Binary search for potentially matching indices
    F = fact_hashes.shape[0]
    idx = torch.searchsorted(fact_hashes, keys)  # [N]
    
    # Clamp to valid range (instead of using boolean mask)
    idx_clamped = idx.clamp(max=F - 1)  # [N]
    
    # Gather potentially matching facts
    candidate_hashes = fact_hashes[idx_clamped]  # [N]
    
    # Match if:
    # 1. Index was in range (idx < F), AND
    # 2. Hash matches
    in_range = idx < F
    hash_match = candidate_hashes == keys
    
    return in_range & hash_match


def all_atoms_are_ground_facts(
    states: Tensor,                 # [N, M, 3] states with atoms
    fact_hashes: Tensor,            # [F] sorted fact hashes
    pack_base: int,
    constant_no: int,
    padding_idx: int,
    excluded_queries: Optional[Tensor] = None,  # [B, 1, 3] or None - atoms to NOT count as facts
) -> Tensor:
    """
    Check if ALL valid atoms in each state are known ground facts.
    
    Args:
        states: [N, M, 3] where N is batch, M is max atoms per state
        fact_hashes: sorted fact hashes for lookup
        pack_base: packing base for hash computation
        constant_no: maximum constant index (vars are > constant_no)
        padding_idx: padding value
        excluded_queries: [B, 1, 3] If provided, atoms matching these are NOT counted as facts.
                         This prevents excluded queries from triggering proof detection.
        
    Returns:
        is_proof: [N] boolean - True if all valid atoms are known facts
    """
    N, M, _ = states.shape
    device = states.device
    pad = padding_idx
    
    if N == 0:
        return torch.zeros(N, dtype=torch.bool, device=device)
    
    # Identify valid (non-padding) atoms
    preds = states[:, :, 0]  # [N, M]
    valid_atom = (preds != pad) & (preds != 0)  # [N, M]
    
    # Check if atoms are ground (all args are constants, not vars)
    args = states[:, :, 1:3]  # [N, M, 2]
    is_ground = (args <= constant_no).all(dim=-1)  # [N, M]
    
    # An atom can only be a fact if it's valid AND ground
    can_be_fact = valid_atom & is_ground  # [N, M]
    
    # Flatten states for fact lookup: [N*M, 3]
    flat_states = states.view(-1, 3)
    is_fact_flat = fact_contains(flat_states, fact_hashes, pack_base)
    is_fact = is_fact_flat.view(N, M)  # [N, M]
    
    # Atom is confirmed fact if: can_be_fact AND is_fact
    confirmed_fact = can_be_fact & is_fact  # [N, M]
    
    # Handle Exclusion: Atoms matching excluded_queries are NOT counted as facts
    # This prevents proofs that lead back to the excluded query
    if excluded_queries is not None:
        # excluded_queries: [B, 1, 3] - need to broadcast to [N, M]
        # Note: N may be B*K_f, so we need to determine the original B
        B = excluded_queries.shape[0]
        K_f = N // B  # Number of candidates per batch
        
        excl_first = excluded_queries[:, 0, :]  # [B, 3]
        # Expand to [N, 1, 3] by repeating each B element K_f times
        excl_exp = excl_first.unsqueeze(1).expand(-1, K_f, -1).reshape(N, 1, 3)  # [N, 1, 3]
        
        # Check if any atom matches the excluded query
        is_excluded_atom = (states == excl_exp).all(dim=-1)  # [N, M]
        
        # Excluded atoms are NOT counted as confirmed facts
        confirmed_fact = confirmed_fact & ~is_excluded_atom
    
    # A state is a proof if:
    # - All valid atoms are confirmed facts, OR
    # - There are no valid atoms (empty state = already proved)
    # 
    # is_proof = (valid_atom => confirmed_fact) for all atoms
    # Equivalently: is_proof = (NOT valid_atom OR confirmed_fact).all()
    # Also covers: if no valid atoms, all conditions trivially satisfied
    
    atoms_ok = (~valid_atom) | confirmed_fact  # [N, M]
    is_proof = atoms_ok.all(dim=-1)  # [N]
    
    # Also need at least one valid atom OR empty state for it to be a real proof
    # Empty state (all padding) should be a proof
    # State with some valid non-fact atoms should NOT be a proof
    has_valid_atoms = valid_atom.sum(dim=-1) > 0  # [N]
    
    # If has valid atoms, must all be facts. If no valid atoms, it's a proof (empty).
    is_proof = is_proof & (has_valid_atoms | ~has_valid_atoms)  # Always true, simplifies to is_proof
    
    return is_proof


def standardize_vars_parity(
    states: Tensor,          # [B, K, M, 3] derived states
    counts: Tensor,          # [B] valid count per batch
    next_var_indices: Tensor, # [B] starting variable index per batch
    constant_no: int,
    padding_idx: int,
    input_states: Optional[Tensor], # [B, M_in, 3] Input states for variable seeding
    max_vocab_size: int = 4096,
) -> Tuple[Tensor, Tensor]:
    """
    Vectorized implementation of parity mode variable standardization.
    Standardizes each candidate state INDEPENDENTLY but fully vectorized across batch and candidates.
    
    This replaces the loop-based logic to avoid graph breaks in torch.compile.
    
    Logic:
    1. Reshape to treat (Batch * Candidates) as a single large batch dimension (BK).
    2. Seed assignments using input_states (mapped to self).
    3. Iterate sequentially over atoms in derived states to assign new variables
       in appearance order (only if not already assigned).
    4. Compute new next_var based on max used variable per batch.
    """
    B, K, M, _ = states.shape
    device = states.device
    pad = padding_idx
    
    # 1. Reshape to [BK] batch dimension
    BK = B * K
    
    # Flatten states to [BK, M, 3]
    states_flat = states.reshape(BK, M, 3)
    
    # Prepare assignment table: [BK, MaxVocab] (Init -1)
    assign_table = torch.full((BK, max_vocab_size), -1, dtype=torch.long, device=device)
    
    # 2. Seed `input_states` assignments (map to self)
    if input_states is not None:
        # input_states: [B, Min, 3] -> expand to [B, K, Min, 3] -> reshape [BK, Min, 3]
        Min = input_states.shape[1]
        input_exp = input_states.unsqueeze(1).expand(-1, K, -1, -1).reshape(BK, Min, 3)
        
        # Extract args: [BK, Min*2]
        input_args = input_exp[:, :, 1:3].reshape(BK, -1)
        
        # Identify variables
        is_var_in = (input_args > constant_no) & (input_args != pad)
        
        # Scatter self-assignment (write to table index=value)
        # Handle masking by writing to index 0 for non-vars (0 is constant/pad, we ignore its entry)
        safe_idx = torch.where(is_var_in, input_args, torch.zeros_like(input_args))
        assign_table.scatter_(1, safe_idx, safe_idx)
        
    # 3. Initialize next variable counters [BK]
    next_gen = next_var_indices.unsqueeze(1).expand(-1, K).reshape(BK)
    
    # 4. Sequential Scan over Derived State Args
    # Flatten args: [BK, M*2]
    # CAUTION: We must clone because we update state_args conceptually, 
    # but practically we just read from original states and write to output
    state_args = states_flat[:, :, 1:3].reshape(BK, -1)
    L = state_args.shape[1]
    
    # To store results
    out_args = state_args.clone()
    
    # Iterate over the sequence length L (e.g., 40-60 iterations)
    for t in range(L):
        # Current column of args: [BK]
        vals = state_args[:, t]
        
        # Check if valid variable
        is_var = (vals > constant_no) & (vals != pad)
        
        # Determine current assignment from table
        curr_assign = assign_table.gather(1, vals.unsqueeze(1)).squeeze(1)
        
        # Unassigned if -1 AND is a variable
        is_unassigned = (curr_assign == -1) & is_var
        
        # Determine new value to assign
        new_assign = torch.where(is_unassigned, next_gen, curr_assign)
        
        # Update next_gen counters
        next_gen = torch.where(is_unassigned, next_gen + 1, next_gen)
        
        # Update assignment table (scatter new assignment)
        safe_indices = torch.where(is_var, vals, torch.zeros_like(vals))
        assign_table.scatter_(1, safe_indices.unsqueeze(1), new_assign.unsqueeze(1))
        
        # Store standardized value in output
        out_args[:, t] = torch.where(is_var, new_assign, vals)

    # 5. Reconstruct standardized states
    out_args_reshaped = out_args.reshape(BK, M, 2)
    states_flat = states_flat.clone() # Clone to avoid modifying original if needed
    states_flat[:, :, 1:3] = out_args_reshaped
    
    # Reshape back to [B, K, M, 3]
    standardized = states_flat.reshape(B, K, M, 3)
    
    # 6. Compute new_next_var [B] based on max used variable per batch
    # Mask out invalid candidates using counts
    next_gen_b = next_gen.reshape(B, K)
    
    # mask: [B, K]
    k_indices = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
    is_valid_cand = k_indices < counts.unsqueeze(1)
    
    # Filter next_gen: use baseline for invalid ones
    base_next = next_var_indices.unsqueeze(1).expand(-1, K)
    valid_next_gen = torch.where(is_valid_cand, next_gen_b, base_next)
    
    # Take max over K
    new_next_var, _ = valid_next_gen.max(dim=1)
    
    return standardized, new_next_var


def standardize_vars(
    states: Tensor,          # [B, K, M, 3]
    counts: Tensor,          # [B] valid count per batch
    next_var_indices: Tensor, # [B] starting variable index per batch
    constant_no: int,
    padding_idx: int,
    input_states: Optional[Tensor] = None, # [B, A, 3] Input states used to determine variable offset
    extra_new_vars: int = 15, # Safe upper bound for new variables per step
    **kwargs
) -> Tuple[Tensor, Tensor]:
    """
    Renumber runtime variables to canonical form per batch element.
    
    OPTIMIZED: Uses input_states to determine variable ranges instead of scanning 
    the full derived states [B, K, M]. This makes the operation O(1) relative 
    to the number of candidates K.
    """
    device = states.device
    B, K, M, _ = states.shape
    pad = padding_idx
    
    if B == 0 or states.numel() == 0:
        return states, next_var_indices
    
    LARGE = 1_000_000

    # 1. Determine Offset from INPUT stats (Small Tensor [B, A])
    # The output variables are a superset of input variables + new variables.
    
    # Default: No variables in input -> No offset needed (inputs were ground)
    min_var_in = torch.full((B,), LARGE, dtype=torch.long, device=device)
    max_var_in = torch.zeros(B, dtype=torch.long, device=device)
    has_input_vars = torch.zeros(B, dtype=torch.bool, device=device)
    
    if input_states is not None and input_states.numel() > 0:
        # Args: [B, A, 2]
        in_args = input_states[:, :, 1:3]
        is_var_in = (in_args > constant_no) & (in_args != pad)
        
        # Fast reduction on small tensor
        masked_min = torch.where(is_var_in, in_args, torch.full_like(in_args, LARGE))
        min_var_in = masked_min.min(dim=-1).values.min(dim=-1).values # [B]
        
        masked_max = torch.where(is_var_in, in_args, torch.zeros_like(in_args))
        max_var_in = masked_max.max(dim=-1).values.max(dim=-1).values # [B]
        
        has_input_vars = (min_var_in < LARGE)

    # Offset Calculation
    # If Input Has Vars: Offset = next_var_indices - min_var_in
    # If Input No Vars: Offset = 0 (We assume new vars start at next_var_indices)
    offset = torch.where(
        has_input_vars,
        next_var_indices - min_var_in,
        torch.zeros_like(next_var_indices)
    )
    
    # 2. Apply Offset to Output [B, K, M]
    args = states[:, :, :, 1:3]
    is_var_out = (args > constant_no) & (args != pad)
    
    offset_exp = offset.view(B, 1, 1, 1).expand(-1, K, M, 2)
    standardized_args = torch.where(
        is_var_out,
        args + offset_exp,
        args
    )
    
    # In-place update (states is fresh from pack_results)
    # Removing .clone() saves 100MB+ per step and significant bandwidth
    states[:, :, :, 1:3] = standardized_args
    standardized = states
    
    # 3. Compute New Next Var (Fast Estimation)
    # We need to bound the MAX variable used.
    # Max Used = Max(Input_Max + Offset, Generated_Max)
    # Generated vars end at next_var_indices + extra_new_vars
    
    # Only shift max_in if we actually had input vars
    max_in_shifted = torch.where(
        has_input_vars,
        max_var_in + offset,
        torch.zeros_like(max_var_in)
    )
    
    # Generated vars are always relative to next_var_indices
    max_gen_shifted = next_var_indices + extra_new_vars
    
    current_max_new = torch.maximum(max_in_shifted, max_gen_shifted)
    new_next_var = current_max_new + 1
    
    return standardized, new_next_var


def pairs_via_predicate_ranges(
    query_preds: Tensor,        # [B] predicate ID per query
    seg_starts: Tensor,         # [P] start index for each predicate
    seg_lens: Tensor,           # [P] count for each predicate  
    max_pairs: int,             # Max pairs per query (pre-computed)
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Generate (query_idx, item_idx) pairs with fixed output shape.
    
    Returns:
        item_idx: [B, max_pairs] - indices into facts/rules
        valid_mask: [B, max_pairs] - which pairs are valid
        query_idx: [B, max_pairs] - repeated query indices (for convenience)
    """
    B = query_preds.shape[0]
    
    if B == 0:
        return (
            torch.zeros((0, max_pairs), dtype=torch.long, device=device),
            torch.zeros((0, max_pairs), dtype=torch.bool, device=device),
            torch.zeros((0, max_pairs), dtype=torch.long, device=device),
        )
    
    # Lookup counts and starts per query
    lens = seg_lens[query_preds.long()]         # [B]
    starts = seg_starts[query_preds.long()]     # [B]
    
    # Create offset grid [1, max_pairs]
    offsets = torch.arange(max_pairs, device=device, dtype=torch.long).unsqueeze(0)
    
    # Item indices = start + offset
    item_idx = starts.unsqueeze(1) + offsets    # [B, max_pairs]
    
    # Valid where offset < len
    valid_mask = offsets < lens.unsqueeze(1)    # [B, max_pairs]
    
    # Query indices (just repeated B indices)
    query_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(-1, max_pairs)
    
    return item_idx, valid_mask, query_idx


def unify_with_facts(
    queries: Tensor,                # [B, 3] query atoms
    remaining: Tensor,              # [B, G, 3] remaining atoms
    remaining_counts: Tensor,       # [B] valid remaining count
    item_idx: Tensor,               # [B, max_pairs] fact indices
    valid_mask: Tensor,             # [B, max_pairs] valid pairs
    facts: Tensor,                  # [F, 3] all facts
    constant_no: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Vectorized fact unification with fixed output shape.
    
    Returns:
        derived_states: [B, max_pairs, G, 3] successor states
        success_mask: [B, max_pairs] which unifications succeeded
        subs: [B, max_pairs, 2, 2] substitutions applied
    """
    B, max_pairs = item_idx.shape
    G = remaining.shape[1]
    device = queries.device
    pad = padding_idx
    
    if B == 0 or facts.numel() == 0:
        return (
            torch.full((B, max_pairs, G, 3), pad, dtype=torch.long, device=device),
            torch.zeros((B, max_pairs), dtype=torch.bool, device=device),
            torch.full((B, max_pairs, 2, 2), pad, dtype=torch.long, device=device),
        )
    
    # Clamp indices to valid range (mask handles invalids)
    safe_idx = item_idx.clamp(0, facts.shape[0] - 1)
    
    # Gather facts for all pairs: [B, max_pairs, 3]
    fact_atoms = facts[safe_idx.view(-1)].view(B, max_pairs, 3)
    
    # Expand queries for comparison: [B, max_pairs, 3]
    q_expanded = queries.unsqueeze(1).expand(-1, max_pairs, -1)
    
    # Check predicate match
    pred_match = (q_expanded[:, :, 0] == fact_atoms[:, :, 0])
    
    # Check if query is ground (all args are constants)
    q_args = q_expanded[:, :, 1:3]  # [B, max_pairs, 2]
    is_ground = (q_args <= constant_no).all(dim=-1)  # [B, max_pairs]
    
    # For ground queries: direct equality check
    ground_match = (q_expanded == fact_atoms).all(dim=-1) & is_ground
    
    # For non-ground queries: unification
    # Flatten for unify_one_to_one
    flat_q = q_expanded.reshape(-1, 3)  # [B*max_pairs, 3]
    flat_f = fact_atoms.reshape(-1, 3)  # [B*max_pairs, 3]
    
    ok_flat, subs_flat = unify_one_to_one(flat_q, flat_f, constant_no, pad)
    
    ok = ok_flat.view(B, max_pairs)
    subs = subs_flat.view(B, max_pairs, 2, 2)
    
    # Success = (ground_match OR unification_ok) AND valid_mask AND pred_match
    success_mask = ((ground_match | ok) & valid_mask & pred_match)
    
    # Apply substitutions to remaining atoms
    # remaining: [B, G, 3] -> expand to [B, max_pairs, G, 3]
    remaining_exp = remaining.unsqueeze(1).expand(-1, max_pairs, -1, -1).clone()
    
    # For each successful pair, apply its substitution
    # Reshape for apply_substitutions: [B*max_pairs, G, 3]
    rem_flat = remaining_exp.reshape(B * max_pairs, G, 3)
    subs_for_apply = subs.reshape(B * max_pairs, 2, 2)
    
    # Apply substitutions
    rem_subst = apply_substitutions(rem_flat, subs_for_apply, pad)
    derived_states = rem_subst.view(B, max_pairs, G, 3)
    
    # Zero out invalid entries
    invalid = ~success_mask
    derived_states[invalid] = pad
    
    return derived_states, success_mask, subs


def unify_with_rules(
    queries: Tensor,                    # [B, 3]
    remaining: Tensor,                  # [B, G, 3]
    remaining_counts: Tensor,           # [B]
    item_idx: Tensor,                   # [B, max_pairs] rule indices
    valid_mask: Tensor,                 # [B, max_pairs]
    rules_heads: Tensor,                # [R, 3]
    rules_bodies: Tensor,               # [R, Bmax, 3]
    rule_lens: Tensor,                  # [R]
    next_var_indices: Tensor,           # [B]
    constant_no: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Vectorized rule unification with fixed output shape.
    
    Returns:
        derived_states: [B, max_pairs, M, 3] where M = Bmax + G
        success_mask: [B, max_pairs]
        subs: [B, max_pairs, 2, 2]
        body_lens: [B, max_pairs] length of rule body for each pair
    """
    B, max_pairs = item_idx.shape
    G = remaining.shape[1]
    device = queries.device
    pad = padding_idx
    
    if B == 0 or rules_heads.numel() == 0:
        M = G + 1
        return (
            torch.full((B, max_pairs, M, 3), pad, dtype=torch.long, device=device),
            torch.zeros((B, max_pairs), dtype=torch.bool, device=device),
            torch.full((B, max_pairs, 2, 2), pad, dtype=torch.long, device=device),
            torch.zeros((B, max_pairs), dtype=torch.long, device=device),
        )
    
    Bmax = rules_bodies.shape[1]
    M = Bmax + G
    
    # Clamp indices
    safe_idx = item_idx.clamp(0, rules_heads.shape[0] - 1)
    
    # Gather rule heads: [B, max_pairs, 3]
    rule_heads_sel = rules_heads[safe_idx.view(-1)].view(B, max_pairs, 3)
    
    # Gather rule bodies: [B, max_pairs, Bmax, 3]
    rule_bodies_sel = rules_bodies[safe_idx.view(-1)].view(B, max_pairs, Bmax, 3)
    
    # Gather rule lengths: [B, max_pairs]
    rule_lens_sel = rule_lens[safe_idx.view(-1)].view(B, max_pairs)
    
    # -------------------------------------------------------------------------
    # Standardization Apart: Rename template variables to runtime variables
    # -------------------------------------------------------------------------
    template_start = constant_no + 1
    next_per_pair = next_var_indices.unsqueeze(1).expand(-1, max_pairs)  # [B, max_pairs]
    
    # Rename head args
    h_args = rule_heads_sel[:, :, 1:3]  # [B, max_pairs, 2]
    is_template_h = (h_args >= template_start) & (h_args != pad)
    h_args_renamed = torch.where(
        is_template_h,
        next_per_pair.unsqueeze(-1) + (h_args - template_start),
        h_args
    )
    heads_renamed = torch.cat([rule_heads_sel[:, :, :1], h_args_renamed], dim=-1)
    
    # Rename body args
    b_args = rule_bodies_sel[:, :, :, 1:3]  # [B, max_pairs, Bmax, 2]
    is_template_b = (b_args >= template_start) & (b_args != pad)
    b_args_renamed = torch.where(
        is_template_b,
        next_per_pair.unsqueeze(-1).unsqueeze(-1) + (b_args - template_start),
        b_args
    )
    bodies_renamed = torch.cat([rule_bodies_sel[:, :, :, :1], b_args_renamed], dim=-1)
    
    # -------------------------------------------------------------------------
    # Unification
    # -------------------------------------------------------------------------
    q_expanded = queries.unsqueeze(1).expand(-1, max_pairs, -1)  # [B, max_pairs, 3]
    
    # Flatten for unify_one_to_one
    flat_q = q_expanded.reshape(-1, 3)
    flat_h = heads_renamed.reshape(-1, 3)
    
    ok_flat, subs_flat = unify_one_to_one(flat_q, flat_h, constant_no, pad)
    ok = ok_flat.view(B, max_pairs)
    subs = subs_flat.view(B, max_pairs, 2, 2)
    
    # -------------------------------------------------------------------------
    # Apply substitutions and combine body + remaining
    # -------------------------------------------------------------------------
    # Expand remaining to [B, max_pairs, G, 3]
    remaining_exp = remaining.unsqueeze(1).expand(-1, max_pairs, -1, -1)
    
    # Concatenate body + remaining: [B, max_pairs, M, 3]
    combined = torch.cat([bodies_renamed, remaining_exp], dim=2)
    
    # Apply substitutions to combined
    # Reshape: [B*max_pairs, M, 3]
    combined_flat = combined.reshape(B * max_pairs, M, 3)
    subs_flat_for_apply = subs.reshape(B * max_pairs, 2, 2)
    
    combined_subst = apply_substitutions(combined_flat, subs_flat_for_apply, pad)
    derived_states = combined_subst.view(B, max_pairs, M, 3)
    
    # Success mask - original behavior allows any number of atoms and truncates later
    success_mask = ok & valid_mask
    
    # Zero out invalid entries using torch.where for compile-friendliness
    derived_states = torch.where(
        success_mask.unsqueeze(-1).unsqueeze(-1),
        derived_states,
        torch.full_like(derived_states, pad)
    )
    
    return derived_states, success_mask, subs, rule_lens_sel


def prune_ground_facts(
    candidates: Tensor,         # [B, K, M, 3]
    valid_mask: Tensor,         # [B, K]
    fact_hashes: Tensor,        # [F] sorted fact hashes
    pack_base: int,
    constant_no: int,
    padding_idx: int,
    true_pred_idx: Optional[int] = None,  # True predicate index for proof detection
    excluded_queries: Optional[Tensor] = None,  # [B, 1, 3] atoms to NOT prune
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Remove known ground facts from candidates (fixed shape).
    Also treats True predicate atoms as "facts" (proof indicators).
    Fully vectorized - NO nonzero() calls.
    
    Args:
        excluded_queries: [B, 1, 3] If provided, atoms matching these are NOT pruned.
                         This matches original prune_and_collapse behavior for cycle prevention.
    
    Returns:
        pruned_states: [B, K, M, 3] with facts removed
        pruned_counts: [B, K] new atom counts per candidate
        is_proof: [B, K] whether candidate became empty (proof found)
    """
    B, K, M, _ = candidates.shape
    device = candidates.device
    pad = padding_idx
    
    # Check which atoms are ground facts
    preds = candidates[:, :, :, 0]               # [B, K, M]
    args = candidates[:, :, :, 1:3]              # [B, K, M, 2]
    
    valid_atom = (preds != pad)                  # [B, K, M]
    is_ground = (args <= constant_no).all(dim=-1)  # [B, K, M]
    ground_atoms = valid_atom & is_ground        # [B, K, M]
    
    # Check ALL atoms against fact hashes (fully vectorized)
    flat_atoms = candidates.view(-1, 3)          # [B*K*M, 3]
    is_fact_flat = fact_contains(flat_atoms, fact_hashes, pack_base)
    is_fact = is_fact_flat.view(B, K, M)
    
    # Only mark as fact if it was actually a ground atom
    is_fact = is_fact & ground_atoms
    
    # Handle Exclusion: Keep excluded atoms (don't prune them)
    # This matches original prune_and_collapse behavior at lines 47-52 of unification.py
    if excluded_queries is not None:
        excl_first = excluded_queries[:, 0, :]  # [B, 3]
        # Expand to match candidates shape: [B, K, M, 3] vs [B, 3]
        excl_exp = excl_first.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
        is_excluded_atom = (candidates == excl_exp).all(dim=-1) & ground_atoms  # [B, K, M]
        # Keep excluded atoms by not marking them as facts
        is_fact = is_fact & ~is_excluded_atom
    
    # Also treat True predicate atoms as "facts" (proof indicators)
    if true_pred_idx is not None:
        is_true_pred = (preds == true_pred_idx)  # [B, K, M]
        is_fact = is_fact | is_true_pred
    
    # Atoms to keep: valid AND NOT a known fact
    keep_atom = valid_atom & ~is_fact  # [B, K, M]
    
    # Compute new counts
    pruned_counts = keep_atom.sum(dim=-1)  # [B, K]
    
    # Detect proofs: candidate with zero remaining atoms
    is_proof = (pruned_counts == 0) & valid_mask
    
    # Mask out removed atoms (gaps remain, downstream handles this)
    pruned_states = torch.where(
        keep_atom.unsqueeze(-1),
        candidates,
        torch.full_like(candidates, pad)
    )
    
    return pruned_states, pruned_counts, is_proof


def compact_atoms(
    states: Tensor,             # [B, K, M, 3]
    padding_idx: int,
) -> Tensor:
    """
    Left-align atoms by removing gaps (padding) in the M dimension.
    
    After prune_ground_facts, atoms may have gaps where facts were removed.
    This function compacts atoms to the front (position 0, 1, 2, ...) and
    fills the rest with padding.
    
    This matches the tensor engine's compact operation for parity.
    Uses scatter_ operations that are compatible with torch.compile.
    
    Args:
        states: [B, K, M, 3] states with potential gaps in M dimension
        padding_idx: value used for padding
        
    Returns:
        compacted: [B, K, M, 3] states with atoms left-aligned
    """
    B, K, M, _ = states.shape
    device = states.device
    pad = padding_idx
    
    # Identify valid (non-padding) atoms
    valid_atom = (states[:, :, :, 0] != pad)  # [B, K, M]
    
    # Compute target positions via cumsum
    # cumsum gives 1-indexed positions, subtract 1 for 0-indexed
    pos = torch.cumsum(valid_atom.long(), dim=2) - 1  # [B, K, M]
    
    # For invalid atoms, set position to M (out of bounds that we won't use)
    # This ensures they don't interfere with valid scatter targets
    pos_safe = torch.where(valid_atom, pos, torch.tensor(M - 1, dtype=pos.dtype, device=device))
    
    # Create output tensor filled with padding
    compacted = torch.full_like(states, pad)
    
    # CRITICAL: We must scatter in order from position 0 to M-1 (low to high)
    # so that valid atoms at lower positions are not overwritten by padding from higher positions.
    # But scatter_ doesn't guarantee order when multiple sources target same destination.
    
    # Instead, use argsort on positions to get a reordering where we process in target order
    # This way, valid atoms with their correct target positions come before invalid atoms
    # that all have position M-1.
    
    # Actually, let's use a cleaner approach: for each target position j in [0, M),
    # find which source position maps to it (the one with valid_atom[i] and pos[i]==j)
    
    # Build inverse mapping: for each target position, which source gives it?
    # We can use scatter with the source position as value, then gather
    
    # Approach: Create sorted order within each (B, K) slice by (valid descending, pos ascending)
    # This puts valid atoms with low target positions first.
    
    # Simpler: use scatter with the maximum valid target as the source
    # If valid_atom[i] and pos[i] = j, then compacted[j] = states[i]
    
    # To avoid conflicts, we can iterate conceptually: 
    # For gather-based approach, we need to know which source index maps to each target index.
    # source_for_target[j] = i where valid_atom[i] and pos[i] == j, or M-1 (padding) if none
    
    # Build source_for_target via scatter of source indices
    source_indices = torch.arange(M, device=device).view(1, 1, M).expand(B, K, M)  # [B, K, M]
    
    # Scatter source indices to target positions
    # Initialize with M (invalid marker that will gather padding)
    source_for_target = torch.full((B, K, M), M - 1, dtype=torch.long, device=device)
    
    # Only scatter valid atoms' source indices to their target positions
    # Use scatter_ with the source index as the value
    # For pos_safe, valid atoms have their correct target, invalid have M-1
    # Scatter source_indices to positions pos_safe
    source_for_target.scatter_(2, pos_safe, source_indices)
    
    # Now for invalid atoms (that all went to M-1), we have multiple sources targeting M-1
    # Last one wins, but M-1 position should be padding anyway, so it's fine.
    
    # But wait - valid atoms might also have target position M-1 if there are M valid atoms.
    # In that case, we want the valid one, not the invalid one.
    # The scatter order depends on iteration order which is problematic.
    
    # Let's fix: For invalid atoms, don't scatter at all. Use where to filter.
    # Reset and scatter only valid
    source_for_target = torch.full((B, K, M), M - 1, dtype=torch.long, device=device)
    
    # Scatter only where valid - we need to mask invalid positions to not participate
    # Unfortunately scatter_ doesn't support masking directly
    
    # Alternative: scatter with values that indicate "no valid source" for invalids
    # Then in a second pass, overwrite with valid sources
    
    # Actually, the key insight: if we process scatter in increasing order of source position,
    # and valid atoms appear at various positions, we need the valid source to "win".
    
    # Simple solution: scatter twice - first scatter all as if valid, then 
    # the result will have the last source index at each target position.
    # If we order sources so valid ones come last, they win.
    
    # But easier: use the fact that for a given target position j, there's at most one 
    # valid source i with pos[i] = j (since cumsum is strictly increasing for valid positions).
    # Invalid sources all map to M-1 (by our pos_safe definition).
    
    # So the only conflict is at position M-1 where multiple invalids and possibly one valid collide.
    # For positions 0..M-2, there's no conflict - at most one valid source maps there.
    
    # Fix: change invalid target to M (out of bounds) and use a temporary buffer of size M+1,
    # then slice back to M.
    
    # Even simpler: just use the existing pos_safe with M-1 for invalids.
    # Positions 0 to count-1 will have exactly one valid source each.
    # Position M-1 might have both valid (if count==M) and invalid sources.
    # If count==M, valid source should win - but scatter order is undefined.
    
    # Safest fix: use scatter_reduce with 'max' or 'min' on source indices, then gather.
    # But scatter_reduce might not be compile-friendly.
    
    # Simplest fix that works: re-implement without scatter, using pure gather logic.
    # For each target position j, find if there exists a valid source with pos==j.
    
    # Use eq and any: is_target[j] = any(valid_atom & (pos == j))
    # But this is O(M^2) comparisons.
    
    # Let's just try the simple scatter and hope the order works in practice.
    # If valid atoms appear before invalids in memory order, and valid sources for 
    # positions 0..count-1 are scattered first, they should be fine.
    # Only M-1 position is problematic if count==M.
    
    # Actually, let's try: scatter valid sources first, then don't scatter invalids at all.
    # We can do this by setting invalid sources' target to their own source index 
    # (a no-op that just reads what's already there)... but that doesn't work either.
    
    # Let's go back to the working nonzero approach but guard it with torch._dynamo.disable
    # for the compact operation, so it doesn't break compilation of the outer function.
    
    # For now, use a cleaner approach: argsort-based reorder
    # Sort each (B, K) row by (valid desc, pos asc) and take first M elements
    # This is equivalent to stable argsort on pos where invalid atoms have pos=M.
    
    # Set invalid atoms' positions to M (so they sort to the end)
    sort_key = torch.where(valid_atom, pos, torch.tensor(M, dtype=pos.dtype, device=device))
    
    # Argsort gives indices that would sort the tensor
    sorted_indices = torch.argsort(sort_key, dim=2, stable=True)  # [B, K, M]
    
    # Gather states in sorted order
    sorted_indices_exp = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, 3)  # [B, K, M, 3]
    compacted = torch.gather(states, 2, sorted_indices_exp)
    
    return compacted


def pack_results_parity(
    fact_states: Tensor,        # [B, K_f, G, 3]
    fact_mask: Tensor,          # [B, K_f]
    rule_states: Tensor,        # [B, K_r, M, 3]
    rule_mask: Tensor,          # [B, K_r]
    K_max: int,
    M_max: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor]:
    """
    Combine fact and rule results using stable sort for parity.
    Slower but deterministic and exact matching original order.
    """
    B = fact_states.shape[0]
    device = fact_states.device
    pad = padding_idx
    K_f = fact_states.shape[1]
    K_r = rule_states.shape[1]
    G = fact_states.shape[2]
    M_r = rule_states.shape[2]
    
    # Normalize dimensions
    if G < M_max:
        fact_pad = torch.full((B, K_f, M_max - G, 3), pad, dtype=fact_states.dtype, device=device)
        fact_states = torch.cat([fact_states, fact_pad], dim=2)
    elif G > M_max:
        fact_states = fact_states[:, :, :M_max, :]
    
    if M_r < M_max:
        rule_pad = torch.full((B, K_r, M_max - M_r, 3), pad, dtype=rule_states.dtype, device=device)
        rule_states = torch.cat([rule_states, rule_pad], dim=2)
    elif M_r > M_max:
        rule_states = rule_states[:, :, :M_max, :]
    
    # Concatenate rules before facts
    all_states = torch.cat([rule_states, fact_states], dim=1)  # [B, K_r+K_f, M_max, 3]
    all_masks = torch.cat([rule_mask, fact_mask], dim=1)  # [B, K_r+K_f]
    
    counts = all_masks.sum(dim=1).clamp(max=K_max)
    
    # PARITY MODE: Use stable argsort
    sorted_idx = torch.argsort(all_masks.int(), dim=1, descending=True, stable=True)
    top_idx = sorted_idx[:, :K_max]
    
    top_idx_exp = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M_max, 3)
    derived = torch.gather(all_states, 1, top_idx_exp)
    
    return derived, counts


def pack_results(
    fact_states: Tensor,        # [B, K_f, G, 3]
    fact_mask: Tensor,          # [B, K_f]
    rule_states: Tensor,        # [B, K_r, M, 3]
    rule_mask: Tensor,          # [B, K_r]
    K_max: int,
    M_max: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor]:
    """
    Combine fact and rule results into single fixed-shape output.
    Valid entries are COMPACTED to the front (indices 0 to count-1).
    
    Returns:
        derived: [B, K_max, M_max, 3] - valid entries at front, rest is padding
        counts: [B] number of valid derived states per batch element
    """
    B = fact_states.shape[0]
    device = fact_states.device
    pad = padding_idx
    K_f = fact_states.shape[1]
    K_r = rule_states.shape[1]
    G = fact_states.shape[2]
    M_r = rule_states.shape[2]
    
    # First, normalize both tensors to M_max atoms dimension
    # This handles both padding (if smaller) and truncation (if larger)
    if G < M_max:
        fact_pad = torch.full((B, K_f, M_max - G, 3), pad, dtype=fact_states.dtype, device=device)
        fact_states = torch.cat([fact_states, fact_pad], dim=2)
    elif G > M_max:
        fact_states = fact_states[:, :, :M_max, :]
    
    if M_r < M_max:
        rule_pad = torch.full((B, K_r, M_max - M_r, 3), pad, dtype=rule_states.dtype, device=device)
        rule_states = torch.cat([rule_states, rule_pad], dim=2)
    elif M_r > M_max:
        rule_states = rule_states[:, :, :M_max, :]
    
    # Now both have shape [B, K_*, M_max, 3] - safe to concatenate
    # Concatenate in REVERSED order (facts first, then rules) to avoid flip later
    # This gives us the reversed tensor directly
    all_states_rev = torch.cat([fact_states, rule_states], dim=1)  # [B, K_f+K_r, M_max, 3] pre-reversed
    all_masks_rev = torch.cat([fact_mask, rule_mask], dim=1)  # [B, K_f+K_r] pre-reversed

    K_total = K_f + K_r

    # Count valid (capped at K_max)
    counts = all_masks_rev.sum(dim=1).clamp(max=K_max)

    # FAST MODE: Scatter-based compaction for torch.compile compatibility
    # Strategy: Reverse the source order so invalid entries are scattered first,
    # then valid entries overwrite them (scatter processes in source index order)
    # NOTE: We already concatenated in reversed order above to avoid .flip() call

    # Compute target indices for compaction using cumsum on REVERSED masks
    cumsum_rev = all_masks_rev.long().cumsum(dim=1)  # [B, K_total]

    # Target index for valid = cumsum - 1 (0-indexed position)
    # For invalid: use K_max-1 (safe position, will be overwritten by later valid entries)
    # Optimize: avoid torch.full_like by using direct arithmetic
    target_idx_rev = torch.where(
        all_masks_rev,
        cumsum_rev - 1,  # Valid: sequential positions 0, 1, 2, ...
        K_max - 1  # Invalid: last position (broadcast scalar)
    ).clamp(max=K_max - 1)

    # Allocate output
    derived = torch.full((B, K_max, M_max, 3), pad, dtype=fact_states.dtype, device=device)

    # Expand target_idx for scatter
    target_idx_exp = target_idx_rev.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M_max, 3)

    # Scatter: invalid entries write pad to K_max-1 first, valid entries overwrite to 0,1,2...
    derived.scatter_(1, target_idx_exp, all_states_rev)
    
    return derived, counts


# ============================================================================
# Main Vectorized Engine Class
# ============================================================================

class UnificationEngineVectorized:
    """
    Fully compilable unification engine with fixed tensor shapes.
    
    This wraps an existing UnificationEngine and provides a get_derived_states
    method that can be compiled with torch.compile(). All operations use
    fixed-shape tensors with masking for invalid entries.
    """
    
    def __init__(
        self,
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
        predicate_no: Optional[int] = None,
        max_derived_per_state: Optional[int] = None,
        # Vectorized specific options
        max_fact_pairs: int = None,
        max_rule_pairs: int = None,
        padding_atoms: int = None,
        parity_mode: bool = False,
        end_pred_idx: Optional[int] = None,
        end_proof_action: bool = False,
    ):
        """
        Initialize the Vectorized Unification Engine.
        """
        self.device = device
        self.padding_idx = int(padding_idx)
        self.constant_no = int(constant_no)
        self.runtime_var_end_index = int(runtime_var_end_index)
        self.max_arity = int(max_arity)
        self.predicate_no = int(predicate_no) if predicate_no is not None else None
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.end_pred_idx = end_pred_idx
        self.end_proof_action = bool(end_proof_action)
        self.max_derived_per_state = int(max_derived_per_state) if max_derived_per_state is not None else 120
        self.K_max = self.max_derived_per_state
        
        self.parity_mode = parity_mode
        self.padding_atoms_limit = padding_atoms

        # Tensors
        self.facts_idx = facts_idx.to(device=device, dtype=torch.long)
        self.rules_idx = rules_idx.to(device=device, dtype=torch.long)
        self.rule_lens = rule_lens.to(device=device, dtype=torch.long)
        self.rules_heads_idx = rules_heads_idx.to(device=device, dtype=torch.long)
        
        # Determine max rule body size
        if rule_lens.numel() > 0:
            self.max_rule_body_size = int(rule_lens.max().item())
        else:
            self.max_rule_body_size = 1

        # M_max computation
        max_rule_body = self.max_rule_body_size
        max_depth_estimate = 10
        if padding_atoms is not None:
            self.M_max = padding_atoms + max_rule_body * max_depth_estimate
        else:
            self.M_max = max_rule_body * max_depth_estimate + 10
            
        # Pack base for hashing
        if pack_base is None:
             self.pack_base = int(runtime_var_end_index + 2000)
        else:
             self.pack_base = int(pack_base)

        # Build Fact Index and Hash Cache
        self.fact_index = GPUFactIndex(self.facts_idx, self.pack_base)
        self.fact_hashes = self.fact_index.fact_hashes
        
        # Sort rules by predicate
        if self.rules_heads_idx.numel() > 0:
            order = torch.argsort(self.rules_heads_idx[:, 0], stable=True)
            self.rules_heads_sorted = self.rules_heads_idx.index_select(0, order)
            self.rules_idx_sorted = self.rules_idx.index_select(0, order)
            self.rule_lens_sorted = self.rule_lens.index_select(0, order)

            preds = self.rules_heads_sorted[:, 0]
            uniq, counts = torch.unique_consecutive(preds, return_counts=True)
            starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts[:-1]]), dim=0)
            
            if self.predicate_no is not None:
                num_pred = self.predicate_no + 1
            else:
                num_pred = int(preds.max().item()) + 2
                
            self.rule_seg_starts = torch.zeros((num_pred,), dtype=torch.long, device=device)
            self.rule_seg_lens = torch.zeros((num_pred,), dtype=torch.long, device=device)
            mask = uniq < num_pred
            self.rule_seg_starts[uniq[mask]] = starts[mask]
            self.rule_seg_lens[uniq[mask]] = counts[mask]
        else:
            self.rules_heads_sorted = self.rules_heads_idx
            self.rules_idx_sorted = self.rules_idx
            self.rule_lens_sorted = self.rule_lens
            self.rule_seg_starts = torch.zeros((1,), dtype=torch.long, device=device)
            self.rule_seg_lens = torch.zeros((1,), dtype=torch.long, device=device)
            
        # Fact predicate ranges
        if predicate_range_map is not None and predicate_range_map.numel() > 0:
            self.fact_seg_starts = predicate_range_map[:, 0].long()
            self.fact_seg_lens = (predicate_range_map[:, 1] - predicate_range_map[:, 0]).long()
        else:
            self._build_fact_ranges()

        # Terminal atoms
        pad = self.padding_idx
        self.true_atom = torch.tensor([self.true_pred_idx, pad, pad], dtype=torch.long, device=device) if self.true_pred_idx is not None else None
        self.false_atom = torch.tensor([self.false_pred_idx, pad, pad], dtype=torch.long, device=device) if self.false_pred_idx is not None else None
        
        # Compute/Set max pairs
        # TODO: Optimize to use K_max cap after resolving correctness issues
        if max_fact_pairs is None:
            if predicate_range_map is not None:
                fact_lens = (
                    predicate_range_map[:, 1] - 
                    predicate_range_map[:, 0]
                )
                max_fact_pairs = int(fact_lens.max().item()) if fact_lens.numel() > 0 else 50
                max_fact_pairs = max(max_fact_pairs, 50)
            else:
                max_fact_pairs = 50
            
        if max_rule_pairs is None:
            if self.rule_seg_lens.numel() > 0:
                max_rule_pairs = int(self.rule_seg_lens.max().item())
                max_rule_pairs = max(max_rule_pairs, 100)
            else:
                max_rule_pairs = 100
        
        self.max_fact_pairs = max_fact_pairs
        self.max_rule_pairs = max_rule_pairs

    def _create_block_sparse_index(self) -> None:
        """
        Restructures flat facts [F, 3] into block-sparse [P, MAX_K, 3].
        
        This enables O(1) fact lookup via block_index[predicate_id],
        eliminating the need for predicate range scanning.
        """
        facts = self.facts_idx  # [F, 3]
        device = self.device
        pad = self.padding_idx
        
        if facts.numel() == 0:
            # Empty facts - create minimal block index
            num_preds = self.predicate_no + 1 if self.predicate_no else 1
            self.block_index = torch.full((num_preds, 1, 3), pad, dtype=torch.long, device=device)
            self.max_facts_per_pred = 1
            return
        
        # 1. Count facts per predicate
        preds = facts[:, 0]  # [F]
        num_preds = self.predicate_no + 1 if self.predicate_no else int(preds.max().item()) + 2
        counts = torch.bincount(preds.long(), minlength=num_preds)  # [P]
        
        # 2. Determine MAX_K (cap to prevent OOM for bursty predicates)
        MAX_K_CAP = 4096  # Increased to handle datasets with many facts per predicate
        max_k = int(counts.max().item())
        max_k = min(max_k, MAX_K_CAP)
        max_k = max(max_k, 1)  # Ensure at least 1
        self.max_facts_per_pred = max_k
        
        # 3. Allocate block tensor [P, MAX_K, 3]
        self.block_index = torch.full(
            (num_preds, max_k, 3), pad, dtype=torch.long, device=device
        )
        
        # 4. Fill blocks (vectorized scatter)
        # Sort facts by predicate for contiguous insertion
        order = torch.argsort(preds, stable=True)
        facts_sorted = facts[order]  # [F, 3]
        preds_sorted = preds[order]  # [F]
        
        # Compute position within each predicate group
        group_starts = torch.zeros(num_preds + 1, dtype=torch.long, device=device)
        group_starts[1:] = counts.cumsum(0)
        
        # For each fact, compute its local index within predicate group
        global_idx = torch.arange(facts.shape[0], device=device)
        local_idx = global_idx - group_starts[preds_sorted.long()]  # Position 0,1,2... within group
        
        # Clamp to MAX_K to handle predicates exceeding cap
        valid = local_idx < max_k
        local_idx_clamped = local_idx.clamp(max=max_k - 1)
        
        # Scatter facts into block_index
        # block_index[pred, local_idx] = fact
        self.block_index[preds_sorted[valid].long(), local_idx_clamped[valid]] = facts_sorted[valid]
        
        # Update max_fact_pairs to match block index dimension
        # This ensures compatibility with existing code that uses max_fact_pairs

    def _sparse_unify_with_facts(
        self,
        queries: Tensor,                 # [B, 3]
        remaining: Tensor,               # [B, G, 3]
        remaining_counts: Tensor,        # [B]
        active_mask: Tensor,             # [B]
        excluded_queries: Optional[Tensor] = None,  # [B, 1, 3]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        O(1) fact unification using block-sparse index.
        
        Instead of scanning predicate ranges, directly indexes into the
        pre-built block_index tensor for instant fact retrieval.
        
        Args:
            queries: [B, 3] Query atoms (predicate, arg1, arg2)
            remaining: [B, G, 3] Remaining goal atoms after query
            remaining_counts: [B] Valid count of remaining atoms
            active_mask: [B] Which batch elements to process
            excluded_queries: [B, 1, 3] Optional queries to exclude (cycle prevention)
            
        Returns:
            fact_states: [B, MAX_K, G, 3] Derived states from fact unification
            fact_success: [B, MAX_K] Success mask for each candidate
            subs: [B, MAX_K, 2, 2] Substitutions applied
        """
        B = queries.shape[0]
        G = remaining.shape[1]
        device = queries.device
        pad = self.padding_idx
        MAX_K = self.max_facts_per_pred
        
        if B == 0 or self.block_index.numel() == 0:
            return (
                torch.full((B, MAX_K, G, 3), pad, dtype=torch.long, device=device),
                torch.zeros((B, MAX_K), dtype=torch.bool, device=device),
                torch.full((B, MAX_K, 2, 2), pad, dtype=torch.long, device=device),
            )
        
        # 1. O(1) Lookup: Get all candidate facts for each query predicate
        query_preds = queries[:, 0]  # [B]
        
        # Clamp to valid predicate range (padding pred -> empty block)
        safe_preds = query_preds.clamp(0, self.block_index.shape[0] - 1)
        
        # Direct indexing: [B, MAX_K, 3]
        candidate_facts = self.block_index[safe_preds.long()]
        
        # 2. Vectorized Unification
        # Expand queries for broadcast: [B, MAX_K, 3]
        q_exp = queries.unsqueeze(1).expand(-1, MAX_K, -1)
        
        # Check predicate match (should be guaranteed by lookup, but handle padding)
        pred_match = (q_exp[:, :, 0] == candidate_facts[:, :, 0])
        
        # Check for non-padding facts
        not_padding = (candidate_facts[:, :, 0] != pad)
        
        # Flatten for unification: [B*MAX_K, 3]
        flat_q = q_exp.reshape(-1, 3)
        flat_f = candidate_facts.reshape(-1, 3)
        
        ok_flat, subs_flat = unify_one_to_one(flat_q, flat_f, self.constant_no, pad)
        ok = ok_flat.view(B, MAX_K)
        subs = subs_flat.view(B, MAX_K, 2, 2)
        
        # Success mask: unification OK AND predicate matches AND not padding AND active
        fact_success = ok & pred_match & not_padding & active_mask.unsqueeze(1)
        
        # 3. Handle excluded queries (cycle prevention)
        if excluded_queries is not None:
            excl_first = excluded_queries[:, 0, :]  # [B, 3]
            # Compare candidate facts to excluded query
            is_excluded = (candidate_facts == excl_first.unsqueeze(1)).all(dim=-1)  # [B, MAX_K]
            fact_success = fact_success & ~is_excluded
        
        # 4. Apply substitutions to remaining atoms
        remaining_exp = remaining.unsqueeze(1).expand(-1, MAX_K, -1, -1)  # [B, MAX_K, G, 3]
        rem_flat = remaining_exp.reshape(B * MAX_K, G, 3)
        subs_for_apply = subs.reshape(B * MAX_K, 2, 2)
        
        rem_subst = apply_substitutions(rem_flat, subs_for_apply, pad)
        fact_states = rem_subst.view(B, MAX_K, G, 3)
        
        # Zero out invalid entries using where for compile-friendliness
        fact_states = torch.where(
            fact_success.unsqueeze(-1).unsqueeze(-1),
            fact_states,
            torch.full_like(fact_states, pad)
        )
        
        return fact_states, fact_success, subs


    @classmethod
    def from_index_manager(
        cls,
        im,
        max_fact_pairs: int = None,
        max_rule_pairs: int = None,
        padding_atoms: int = None,
        parity_mode: bool = False,
        max_derived_per_state: Optional[int] = None,
        end_proof_action: bool = False,
        # kept for compatibility with train caller signature
        **kwargs 
    ):
        """
        Create UnificationEngineVectorized from an IndexManager.
        """
        return cls(
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
            predicate_no=getattr(im, 'predicate_no', None),
            max_derived_per_state=max_derived_per_state,
            max_fact_pairs=max_fact_pairs,
            max_rule_pairs=max_rule_pairs,
            padding_atoms=padding_atoms,
            parity_mode=parity_mode,
            end_pred_idx=im.end_pred_idx,
            end_proof_action=end_proof_action,
        )
    

    @torch.no_grad()
    def get_derived_states_compiled(
        self,
        current_states: Tensor,     # [B, A, 3]
        next_var_indices: Tensor,   # [B]
        excluded_queries: Optional[Tensor] = None,  # [B, 1, 3]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate derived states with fixed output shapes.
        
        This method is designed to be compilable with torch.compile().
        All operations use fixed shapes and masked operations.
        
        Args:
            current_states: [B, A, 3] Current state atoms
            next_var_indices: [B] Next available variable index per batch
            excluded_queries: [B, 1, 3] Optional queries to exclude (cycle prevention)
        
        Returns:
            derived_states: [B, K_max, M_max, 3] Successor states
            counts: [B] Number of valid successors per batch element
            new_vars: [B] Updated next variable indices
        """
        B, A, _ = current_states.shape
        device = current_states.device
        pad = self.padding_idx
        
        # Pre-allocate output
        derived = torch.full(
            (B, self.K_max, self.M_max, 3), 
            pad, dtype=torch.long, device=device
        )
        counts = torch.zeros(B, dtype=torch.long, device=device)
        new_vars = next_var_indices.clone()
        
        # ---------------------------------------------------------------------
        # 1. Extract queries (first non-padding atom)
        # ---------------------------------------------------------------------
        # For left-aligned input, query is always first atom
        queries = current_states[:, 0, :]  # [B, 3]
        remaining = current_states[:, 1:, :]  # [B, A-1, 3]
        G = remaining.shape[1]
        
        # Valid remaining counts
        remaining_valid = (remaining[:, :, 0] != pad)  # [B, A-1]
        remaining_counts = remaining_valid.sum(dim=1)  # [B]
        
        # Query predicates
        query_preds = queries[:, 0]  # [B]
        
        # Check for terminal states
        is_empty = (query_preds == pad)
        is_true = (query_preds == self.true_pred_idx) if self.true_pred_idx is not None else torch.zeros_like(is_empty)
        is_false = (query_preds == self.false_pred_idx) if self.false_pred_idx is not None else torch.zeros_like(is_empty)
        is_terminal = is_empty | is_true | is_false
        
        # Handle terminals with torch.where
        if self.true_atom is not None:
            derived[:, 0, 0, :] = torch.where(
                is_true.unsqueeze(-1),
                self.true_atom.unsqueeze(0).expand(B, -1),
                derived[:, 0, 0, :]
            )
            counts = torch.where(is_true, torch.ones_like(counts), counts)
        
        if self.false_atom is not None:
            derived[:, 0, 0, :] = torch.where(
                (is_false | is_empty).unsqueeze(-1),
                self.false_atom.unsqueeze(0).expand(B, -1),
                derived[:, 0, 0, :]
            )
            counts = torch.where(is_false | is_empty, torch.ones_like(counts), counts)
        
        # Mask for active (non-terminal) states
        active_mask = ~is_terminal
        
        # If all terminal, return early
        # Note: We can't use .all() for compilation, so we process all anyway
        
        # ---------------------------------------------------------------------
        # 2. Fact Unification
        # ---------------------------------------------------------------------
        fact_item_idx, fact_valid, _ = pairs_via_predicate_ranges(
            query_preds, self.fact_seg_starts, self.fact_seg_lens,
            self.max_fact_pairs, device
        )
        
        fact_states, fact_success, _ = unify_with_facts(
            queries, remaining, remaining_counts,
            fact_item_idx, fact_valid,
            self.facts_idx, self.constant_no, pad
        )
        
        # Mask out terminals
        fact_success = fact_success & active_mask.unsqueeze(1)
        
        # ---------------------------------------------------------------------
        # 2a. Excluded Queries Filtering (Cycle Prevention)
        # Filter out fact candidates that match the excluded query.
        # This prevents trivial loops where the query resolves to itself as a fact.
        # ---------------------------------------------------------------------
        if excluded_queries is not None and self.facts_idx.numel() > 0:
            # excluded_queries: [B, 1, 3] - first atom of each excluded query
            excl_first = excluded_queries[:, 0, :]  # [B, 3]
            
            # Get the matched facts for each pair
            K_f = fact_states.shape[1]
            safe_idx = fact_item_idx.clamp(0, max(self.facts_idx.shape[0] - 1, 0))
            matched_facts = self.facts_idx[safe_idx.view(-1)].view(B, K_f, 3)
            
            # Compare each matched fact to the excluded query
            is_excluded = (matched_facts == excl_first.unsqueeze(1)).all(dim=-1)
            
            # Filter out excluded facts from success mask
            fact_success = fact_success & ~is_excluded
        
        # ---------------------------------------------------------------------
        # 2b. Check for proofs from fact unification
        # After substitution, if ALL remaining atoms are now known ground facts,
        # the state represents a proof.
        # ---------------------------------------------------------------------
        # fact_states: [B, K_f, G, 3] - remaining atoms after substitution
        K_f = fact_states.shape[1]
        G_f = fact_states.shape[2]
        
        # Flatten to [B*K_f, G, 3] for all_atoms_are_ground_facts
        fact_states_flat = fact_states.view(B * K_f, G_f, 3)
        fact_is_proof_flat = all_atoms_are_ground_facts(
            fact_states_flat, self.fact_hashes, self.pack_base, 
            self.constant_no, pad,
            excluded_queries=excluded_queries
        )
        fact_is_proof = fact_is_proof_flat.view(B, K_f)  # [B, K_f]
        
        # Only consider successful unifications that lead to proofs
        fact_is_proof = fact_is_proof & fact_success
        
        # For states that are proofs, replace with True atom
        if self.true_atom is not None and G_f > 0:
            # Create True atom state: [G, 3] with True at first position
            true_state = torch.full((G_f, 3), pad, dtype=torch.long, device=device)
            true_state[0, :] = self.true_atom

            # Replace proof states with True atom
            fact_states = torch.where(
                fact_is_proof.unsqueeze(-1).unsqueeze(-1),
                true_state.unsqueeze(0).unsqueeze(0).expand(B, K_f, -1, -1),
                fact_states
            )
        
        # CRITICAL: Detect batch-level proofs from fact unification
        # If ANY fact candidate is a proof for a batch, that batch has found a proof
        # This must happen BEFORE combining with rules to prevent proof truncation at K_max
        fact_proof_batch = fact_is_proof.sum(dim=1) > 0  # [B] - True if batch has any proof
        
        # ---------------------------------------------------------------------
        # 3. Rule Unification
        # ---------------------------------------------------------------------
        rule_item_idx, rule_valid, _ = pairs_via_predicate_ranges(
            query_preds, self.rule_seg_starts, self.rule_seg_lens,
            self.max_rule_pairs, device
        )
        
        rule_states, rule_success, _, rule_lens = unify_with_rules(
            queries, remaining, remaining_counts,
            rule_item_idx, rule_valid,
            self.rules_heads_sorted, self.rules_idx_sorted, self.rule_lens_sorted,
            next_var_indices, self.constant_no, pad
        )
        
        # Mask out terminals
        rule_success = rule_success & active_mask.unsqueeze(1)
        
        # Update variable indices for successful rule unifications
        # (simplified: just increment by max rule body size for all)
        max_body = self.rules_idx_sorted.shape[1] if self.rules_idx_sorted.numel() else 0
        # Replace .any() with sum() > 0 for compilation
        has_rule_success = rule_success.sum(dim=1) > 0
        new_vars = torch.where(
            has_rule_success,
            next_var_indices + max_body,
            new_vars
        )
        
        # ---------------------------------------------------------------------
        # 4. Combine Results
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        # 4. Combine Results
        # ---------------------------------------------------------------------
        if self.parity_mode:
            combined, combined_counts = pack_results_parity(
                fact_states, fact_success,
                rule_states, rule_success,
                self.K_max, self.M_max, pad,
            )
        else:
            combined, combined_counts = pack_results(
                fact_states, fact_success,
                rule_states, rule_success,
                self.K_max, self.M_max, pad,
            )
        
        # ---------------------------------------------------------------------
        # 5. Prune Ground Facts
        # ---------------------------------------------------------------------
        combined_valid = (combined_counts > 0)
        combined_valid_exp = torch.arange(self.K_max, device=device).unsqueeze(0) < combined_counts.unsqueeze(1)
        
        pruned, pruned_atom_counts, is_proof = prune_ground_facts(
            combined, combined_valid_exp,
            self.fact_hashes, self.pack_base, self.constant_no, pad,
            true_pred_idx=self.true_pred_idx,
            excluded_queries=excluded_queries
        )
        
        # Compact atoms in parity mode to match tensor engine behavior
        # This left-aligns atoms after pruning (removes gaps)
        if self.parity_mode:
            pruned = compact_atoms(pruned, pad)
        
        # Handle proofs - combine fact-based proofs with prune-detected proofs
        prune_proof_batch = is_proof.sum(dim=1) > 0  # [B]
        proof_batch = fact_proof_batch | prune_proof_batch  # Either source counts
        
        # For proof batches, replace the ENTIRE first state with True() + padding
        # This ensures no artifacts from rule states remain
        if self.true_atom is not None:
            # Create a full True() state: True atom at position 0, rest is padding
            M = pruned.shape[2]
            true_full_state = torch.full((M, 3), pad, dtype=torch.long, device=device)
            true_full_state[0, :] = self.true_atom  # [M, 3]
            
            # Replace entire first candidate state for proof batches
            pruned[:, 0, :, :] = torch.where(
                proof_batch.view(B, 1, 1),
                true_full_state.unsqueeze(0).expand(B, -1, -1),
                pruned[:, 0, :, :]
            )
        
        # Use combined_counts as the starting point for new_counts
        new_counts = combined_counts.clone()
        
        # Handle proofs - set count to 1 for proof batches
        new_counts = torch.where(
            proof_batch,
            torch.ones_like(new_counts),
            new_counts
        )
        
        # Handle case where active (non-terminal) batch elements have no derivations
        # They should be marked as FALSE (dead end in proof search)
        # This matches original engine behavior at line 1977-1980
        no_derivations = (new_counts == 0) & ~is_terminal
        if self.false_atom is not None:
            new_counts = torch.where(no_derivations, torch.ones_like(new_counts), new_counts)
        
        # Merge with terminal results
        final_derived = torch.where(
            is_terminal.view(B, 1, 1, 1),
            derived,
            pruned
        )
        
        # Also inject FALSE for no-derivation cases
        if self.false_atom is not None:
            final_derived[:, 0, 0, :] = torch.where(
                no_derivations.unsqueeze(-1),
                self.false_atom.unsqueeze(0).expand(B, -1),
                final_derived[:, 0, 0, :]
            )
        
        final_counts = torch.where(
            is_terminal,
            counts,
            new_counts
        )
        
        # ---------------------------------------------------------------------
        # 6. Standardize Variables
        # Renumber runtime variables to start from next_var_indices
        # This matches the original engine's standardize_derived_states step
        # ---------------------------------------------------------------------
        if self.parity_mode:
            std_derived, std_new_vars = standardize_vars_parity(
                final_derived, final_counts, next_var_indices,
                self.constant_no, pad,
                input_states=current_states
            )
        else:
            std_derived, std_new_vars = standardize_vars(
                final_derived, final_counts, next_var_indices,
                self.constant_no, pad,
                input_states=current_states,
                extra_new_vars=self.max_rule_body_size + 2  # Safe margin for new vars
            )
        
        return std_derived, final_counts, std_new_vars


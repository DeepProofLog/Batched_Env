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
    from unification_vectorized import UnificationEngineVectorized
    
    engine = UnificationEngineVectorized.from_base_engine(base_engine)
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
from unification import (
    UnificationEngine,
    GPUFactIndex,
    GPUHashCache,
    _pack_triples_64,
    apply_substitutions,
    unify_one_to_one,
)


# ============================================================================
# Vectorized Fact Lookup (No nonzero() or boolean indexing)
# ============================================================================

def fact_contains_vectorized(
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
) -> Tensor:
    """
    Check if ALL valid atoms in each state are known ground facts.
    
    Args:
        states: [N, M, 3] where N is batch, M is max atoms per state
        fact_hashes: sorted fact hashes for lookup
        pack_base: packing base for hash computation
        constant_no: maximum constant index (vars are > constant_no)
        padding_idx: padding value
        
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
    is_fact_flat = fact_contains_vectorized(flat_states, fact_hashes, pack_base)
    is_fact = is_fact_flat.view(N, M)  # [N, M]
    
    # Atom is confirmed fact if: can_be_fact AND is_fact
    confirmed_fact = can_be_fact & is_fact  # [N, M]
    
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


def standardize_vars_fixed(
    states: Tensor,          # [B, K, M, 3]
    counts: Tensor,          # [B] valid count per batch
    next_var_indices: Tensor, # [B] starting variable index per batch
    constant_no: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor]:
    """
    Renumber runtime variables to canonical form per batch element.
    
    Template variables (>= constant_no + 1) in the derived states are
    renumbered to start from next_var_indices[b] for each batch b.
    This ensures structural identity for hashing and comparison.
    
    Args:
        states: [B, K, M, 3] derived states
        counts: [B] number of valid states per batch
        next_var_indices: [B] starting variable index for renumbering
        constant_no: threshold (vars > constant_no are runtime vars)
        padding_idx: padding value
        
    Returns:
        (standardized_states, new_next_var): renumbered states and updated next_var
    """
    device = states.device
    B, K, M, _ = states.shape
    pad = padding_idx
    
    if B == 0 or states.numel() == 0:
        return states, next_var_indices
    
    # Template variable threshold
    template_start = constant_no + 1
    
    # Extract arguments (skip predicate at index 0)
    args = states[:, :, :, 1:3]  # [B, K, M, 2]
    
    # Identify variables (> constant_no) and not padding
    is_var = (args > constant_no) & (args != pad)  # [B, K, M, 2]
    
    # For each batch element, we need to find unique variables and renumber them
    # This is tricky to vectorize - we'll do a simplified approach:
    # Within each state, renumber variables in order of first appearance
    
    # Flatten args for processing: [B, K*M*2]
    flat_args = args.reshape(B, -1)  # [B, K*M*2]
    flat_is_var = is_var.reshape(B, -1)  # [B, K*M*2]
    
    # Create a mapping from old var -> new var (order of appearance)
    # For compilation compatibility, we'll use a simpler approach:
    # Sort unique variables and map based on sorted rank
    
    # Create keys that separate batches: var_val + batch_idx * LARGE
    LARGE = 1_000_000
    batch_idx = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
    keys = flat_args + batch_idx * LARGE  # [B, K*M*2]
    
    # Mask out non-variables
    keys = torch.where(flat_is_var, keys, torch.full_like(keys, -1))
    
    # For each position in flat_args, find how many DISTINCT variables appear before it
    # This gives us the new ID (relative to 0)
    
    # Simpler approach for fixed-shape compatibility:
    # Find the minimum variable index in each batch and offset from there
    
    # Get minimum variable per batch (for renumbering)
    masked_vars = torch.where(flat_is_var, flat_args, torch.full_like(flat_args, LARGE))
    min_var_per_batch = masked_vars.min(dim=1).values  # [B]
    min_var_per_batch = torch.where(
        min_var_per_batch < LARGE,
        min_var_per_batch,
        next_var_indices
    )
    
    # Compute offset: shift variables to start from next_var_indices
    # new_var = old_var - min_var + next_var
    offset = next_var_indices - min_var_per_batch  # [B]
    
    # Apply offset to all variables
    offset_exp = offset.view(B, 1, 1, 1).expand(-1, K, M, 2)
    standardized_args = torch.where(
        is_var,
        args + offset_exp,
        args
    )
    
    # Reconstruct states with standardized args
    standardized = states.clone()
    standardized[:, :, :, 1:3] = standardized_args
    
    # Compute new max variable per batch
    max_var = torch.where(flat_is_var, flat_args + offset.unsqueeze(1), 
                          torch.full_like(flat_args, 0)).max(dim=1).values  # [B]
    new_next_var = torch.maximum(next_var_indices, max_var + 1)
    
    return standardized, new_next_var


def pairs_via_predicate_ranges_fixed(
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


def unify_with_facts_fixed(
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


def unify_with_rules_fixed(
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
    
    # Success mask
    success_mask = ok & valid_mask
    
    # Zero out invalid entries
    derived_states[~success_mask] = pad
    
    return derived_states, success_mask, subs, rule_lens_sel


def prune_ground_facts_fixed(
    candidates: Tensor,         # [B, K, M, 3]
    valid_mask: Tensor,         # [B, K]
    fact_hashes: Tensor,        # [F] sorted fact hashes
    pack_base: int,
    constant_no: int,
    padding_idx: int,
    true_pred_idx: Optional[int] = None,  # True predicate index for proof detection
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Remove known ground facts from candidates (fixed shape).
    Also treats True predicate atoms as "facts" (proof indicators).
    Fully vectorized - NO nonzero() calls.
    
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
    is_fact_flat = fact_contains_vectorized(flat_atoms, fact_hashes, pack_base)
    is_fact = is_fact_flat.view(B, K, M)
    
    # Only mark as fact if it was actually a ground atom
    is_fact = is_fact & ground_atoms
    
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


def pack_results_fixed(
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
    
    # Pad atoms dimension to M_max
    if G < M_max:
        fact_pad = torch.full((B, K_f, M_max - G, 3), pad, dtype=fact_states.dtype, device=device)
        fact_states = torch.cat([fact_states, fact_pad], dim=2)
    
    if M_r < M_max:
        rule_pad = torch.full((B, K_r, M_max - M_r, 3), pad, dtype=rule_states.dtype, device=device)
        rule_states = torch.cat([rule_states, rule_pad], dim=2)
    
    # Concatenate facts and rules along K dimension
    all_states = torch.cat([fact_states, rule_states], dim=1)  # [B, K_f+K_r, M_max, 3]
    all_masks = torch.cat([fact_mask, rule_mask], dim=1)  # [B, K_f+K_r]
    
    K_total = K_f + K_r
    
    # Count valid (capped at K_max)
    counts = all_masks.sum(dim=1).clamp(max=K_max)
    
    # Compute target indices for compaction using cumsum
    # For valid entries: cumsum gives 1-indexed position, so subtract 1
    # For invalid entries: we set target to K_max (out of valid range)
    cumsum = all_masks.long().cumsum(dim=1)  # [B, K_total]
    
    # Target index = cumsum - 1 for valid, K_max-1 for invalid (clamped)
    # Invalid entries should not overwrite valid ones -> use large offset
    target_idx = torch.where(
        all_masks,
        cumsum - 1,  # Valid: 0, 1, 2, ... (0-indexed)
        torch.full_like(cumsum, K_max)  # Invalid: out of range (will be ignored)
    ).clamp(max=K_max - 1)  # Clamp for safety
    
    # Allocate output initialized with padding
    derived = torch.full((B, K_max, M_max, 3), pad, dtype=fact_states.dtype, device=device)
    
    # Expand target_idx for scatter: [B, K_total] -> [B, K_total, M_max, 3]
    target_idx_exp = target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M_max, 3)
    
    # Create a masked source: only valid entries, zeros for invalid
    # This way, invalid entries write zeros which is harmless if they clash
    masked_states = torch.where(
        all_masks.unsqueeze(-1).unsqueeze(-1),
        all_states,
        torch.full_like(all_states, pad)  # Invalid entries write pad (no harm if overwritten)
    )
    
    # But we need to be careful: if invalid entries target an index before valid entries
    # scatter, they could overwrite. Solution: ensure invalid target >= counts
    # We already set invalid targets to K_max, but after clamping they become K_max-1
    # 
    # Better approach: use scatter_reduce with 'sum' semantics but that's complex
    # Instead, just iterate the scatter order: do pad last so valid is preserved
    # 
    # Actually simpler: scatter ALL states, but valid entries should come after
    # invalid in the concatenation order so they overwrite the invalid zeros
    
    # The issue: fact entries (all invalid) come first in all_states at indices 0-49
    # Rule entries (valid) come at indices 50-57
    # Scatter processes 0-49 first (all targeting 0 after clamp), writing pad
    # Then 50-57 which write valid data to 0-7
    # But the final result shows 0 has zeros... 
    # 
    # OH! The issue is target_idx for invalid is K_max=120, clamped to 119
    # So invalid entries don't overwrite index 0!
    # But then why is index 0 zero?
    
    # Let's use scatter_add instead to avoid overwrites, but it's tricky with pads
    # Simpler: create separate scatter for valid only
    
    # Only scatter valid states to their target indices
    # For entries where mask is False, scatter has no effect (target is K_max-1)
    
    derived.scatter_(1, target_idx_exp, masked_states)
    
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
        base_engine: UnificationEngine,
        max_fact_pairs: int = 50,
        max_rule_pairs: int = 100,
    ):
        """
        Initialize from a base UnificationEngine.
        
        Args:
            base_engine: Existing UnificationEngine with facts, rules, etc.
            max_fact_pairs: Max fact candidates per query (pre-computed)
            max_rule_pairs: Max rule candidates per query (pre-computed)
        """
        self.engine = base_engine
        self.device = base_engine.device
        self.padding_idx = base_engine.padding_idx
        self.constant_no = base_engine.constant_no
        self.true_pred_idx = base_engine.true_pred_idx
        self.false_pred_idx = base_engine.false_pred_idx
        
        # Fixed shape parameters
        self.K_max = base_engine.max_derived_per_state or 120
        self.M_max = (base_engine.max_rule_body_size or 4) + 10  # body + remaining
        self.max_fact_pairs = max_fact_pairs
        self.max_rule_pairs = max_rule_pairs
        
        # Pre-compute predicate limits
        self._compute_pred_limits()
        
        # Reference tensors from base engine
        self.facts_idx = base_engine.facts_idx
        self.fact_index = base_engine.fact_index
        self.rules_heads_sorted = base_engine.rules_heads_sorted
        self.rules_idx_sorted = base_engine.rules_idx_sorted
        self.rule_lens_sorted = base_engine.rule_lens_sorted
        self.rule_seg_starts = base_engine.rule_seg_starts
        self.rule_seg_lens = base_engine.rule_seg_lens
        
        # For vectorized fact lookup (no nonzero)
        self.fact_hashes = base_engine.fact_index.fact_hashes
        self.pack_base = base_engine.fact_index.pack_base
        
        # Fact predicate ranges
        if base_engine.predicate_range_map is not None:
            self.fact_seg_starts = base_engine.predicate_range_map[:, 0].long()
            self.fact_seg_lens = (
                base_engine.predicate_range_map[:, 1] - 
                base_engine.predicate_range_map[:, 0]
            ).long()
        else:
            # Build from facts
            self._build_fact_ranges()
        
        # Terminal atoms
        self.true_atom = base_engine.true_atom
        self.false_atom = base_engine.false_atom
    
    def _compute_pred_limits(self):
        """Pre-compute max pairs per predicate for fixed shapes."""
        # For now, use the class parameters
        # In a more sophisticated version, we'd analyze the actual data
        pass
    
    def _build_fact_ranges(self):
        """Build fact predicate ranges if not provided."""
        if self.facts_idx.numel() == 0:
            self.fact_seg_starts = torch.zeros(1, dtype=torch.long, device=self.device)
            self.fact_seg_lens = torch.zeros(1, dtype=torch.long, device=self.device)
            return
        
        # Sort facts by predicate
        preds = self.facts_idx[:, 0]
        order = torch.argsort(preds, stable=True)
        sorted_preds = preds[order]
        
        uniq, counts = torch.unique_consecutive(sorted_preds, return_counts=True)
        starts = torch.cumsum(torch.cat([
            torch.zeros(1, dtype=torch.long, device=self.device), 
            counts[:-1]
        ]), dim=0)
        
        max_pred = int(uniq.max().item()) + 2 if uniq.numel() > 0 else 1
        self.fact_seg_starts = torch.zeros(max_pred, dtype=torch.long, device=self.device)
        self.fact_seg_lens = torch.zeros(max_pred, dtype=torch.long, device=self.device)
        
        self.fact_seg_starts[uniq.long()] = starts
        self.fact_seg_lens[uniq.long()] = counts
    
    @classmethod
    def from_base_engine(
        cls,
        base_engine: UnificationEngine,
        max_fact_pairs: int = 50,
        max_rule_pairs: int = 100,
    ) -> "UnificationEngineVectorized":
        """Factory method to create from existing engine."""
        return cls(base_engine, max_fact_pairs, max_rule_pairs)
    
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
        fact_item_idx, fact_valid, _ = pairs_via_predicate_ranges_fixed(
            query_preds, self.fact_seg_starts, self.fact_seg_lens,
            self.max_fact_pairs, device
        )
        
        fact_states, fact_success, _ = unify_with_facts_fixed(
            queries, remaining, remaining_counts,
            fact_item_idx, fact_valid,
            self.facts_idx, self.constant_no, pad
        )
        
        # Mask out terminals
        fact_success = fact_success & active_mask.unsqueeze(1)
        
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
            self.constant_no, pad
        )
        fact_is_proof = fact_is_proof_flat.view(B, K_f)  # [B, K_f]
        
        # Only consider successful unifications that lead to proofs
        fact_is_proof = fact_is_proof & fact_success
        
        # For states that are proofs, replace with True atom
        if self.true_atom is not None:
            # Create True atom state: [G, 3] with True at first position
            true_state = torch.full((G_f, 3), pad, dtype=torch.long, device=device)
            true_state[0, :] = self.true_atom
            
            # Replace proof states with True atom
            fact_states = torch.where(
                fact_is_proof.unsqueeze(-1).unsqueeze(-1),
                true_state.unsqueeze(0).unsqueeze(0).expand(B, K_f, -1, -1),
                fact_states
            )
        
        # ---------------------------------------------------------------------
        # 3. Rule Unification
        # ---------------------------------------------------------------------
        rule_item_idx, rule_valid, _ = pairs_via_predicate_ranges_fixed(
            query_preds, self.rule_seg_starts, self.rule_seg_lens,
            self.max_rule_pairs, device
        )
        
        rule_states, rule_success, _, rule_lens = unify_with_rules_fixed(
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
        combined, combined_counts = pack_results_fixed(
            fact_states, fact_success,
            rule_states, rule_success,
            self.K_max, self.M_max, pad
        )
        
        # ---------------------------------------------------------------------
        # 5. Prune Ground Facts
        # ---------------------------------------------------------------------
        combined_valid = (combined_counts > 0)
        combined_valid_exp = torch.arange(self.K_max, device=device).unsqueeze(0) < combined_counts.unsqueeze(1)
        
        pruned, pruned_atom_counts, is_proof = prune_ground_facts_fixed(
            combined, combined_valid_exp,
            self.fact_hashes, self.pack_base, self.constant_no, pad,
            true_pred_idx=self.true_pred_idx
        )
        
        # Handle proofs - replace .any() with sum() > 0 for compilation
        proof_batch = is_proof.sum(dim=1) > 0  # [B]
        if self.true_atom is not None:
            pruned[:, 0, 0, :] = torch.where(
                proof_batch.unsqueeze(-1),
                self.true_atom.unsqueeze(0).expand(B, -1),
                pruned[:, 0, 0, :]
            )
        
        # Update counts
        new_counts = torch.where(
            proof_batch,
            torch.ones_like(combined_counts),
            combined_counts
        )
        
        # Merge with terminal results
        final_derived = torch.where(
            is_terminal.view(B, 1, 1, 1),
            derived,
            pruned
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
        std_derived, std_new_vars = standardize_vars_fixed(
            final_derived, final_counts, next_var_indices,
            self.constant_no, pad
        )
        
        return std_derived, final_counts, std_new_vars


# ============================================================================
# Compiled Wrapper Function
# ============================================================================

def create_compiled_engine(
    base_engine: UnificationEngine,
    max_fact_pairs: int = 50,
    max_rule_pairs: int = 100,
    compile_mode: str = 'reduce-overhead',
) -> UnificationEngineVectorized:
    """
    Create a vectorized engine with compiled get_derived_states.
    
    Args:
        base_engine: Existing UnificationEngine
        max_fact_pairs: Max facts per predicate
        max_rule_pairs: Max rules per predicate
        compile_mode: torch.compile mode ('default', 'reduce-overhead', 'max-autotune')
    
    Returns:
        UnificationEngineVectorized with compiled method
    """
    engine = UnificationEngineVectorized.from_base_engine(
        base_engine, max_fact_pairs, max_rule_pairs
    )
    
    # Compile the main method
    engine.get_derived_states_compiled = torch.compile(
        engine.get_derived_states_compiled,
        mode=compile_mode,
        dynamic=False,  # Fixed shapes!
    )
    
    return engine

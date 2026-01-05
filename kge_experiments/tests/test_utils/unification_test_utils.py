"""
Test utilities for comparing unification engine outputs.

Provides normalization functions for structural comparison between
different implementations that may produce semantically equivalent
but structurally different results (e.g., different variable numbering,
different ordering of derived states).
"""

import torch
from torch import Tensor
from typing import Tuple, List, Set, FrozenSet, Dict, Any


def canonicalize_atom(atom: Tensor, constant_no: int, padding_idx: int) -> Tuple[int, ...]:
    """
    Canonicalize a single atom by renumbering variables to appearance order.

    Args:
        atom: [3] tensor (predicate, arg0, arg1)
        constant_no: Number of constants (variables have index > constant_no)
        padding_idx: Padding index to ignore

    Returns:
        Tuple of (predicate, canon_arg0, canon_arg1)
    """
    var_map = {}
    next_var = constant_no + 1

    result = []
    for i, val in enumerate(atom.tolist()):
        if i == 0:
            # Predicate - keep as is
            result.append(val)
        elif val > constant_no and val != padding_idx:
            # Variable - map to canonical form
            if val not in var_map:
                var_map[val] = next_var
                next_var += 1
            result.append(var_map[val])
        else:
            # Constant or padding - keep as is
            result.append(val)

    return tuple(result)


def canonicalize_state(
    state: Tensor,
    constant_no: int,
    padding_idx: int,
) -> FrozenSet[Tuple[int, ...]]:
    """
    Canonicalize a state (set of atoms) for comparison.

    Steps:
    1. Remove padding atoms
    2. Sort atoms by predicate then args (canonical order)
    3. Renumber variables to appearance order

    Args:
        state: [M, 3] tensor of atoms
        constant_no: Number of constants
        padding_idx: Padding index

    Returns:
        Frozenset of sorted, canonicalized atom tuples
    """
    # Filter out padding atoms
    valid_mask = state[:, 0] != padding_idx
    valid_atoms = state[valid_mask]

    if valid_atoms.numel() == 0:
        return frozenset()

    # Sort atoms by (predicate, arg0, arg1) for canonical order
    # This ensures deterministic ordering regardless of internal arrangement
    keys = valid_atoms[:, 0] * 1_000_000 + valid_atoms[:, 1] * 1_000 + valid_atoms[:, 2]
    sorted_indices = keys.argsort()
    sorted_atoms = valid_atoms[sorted_indices]

    # Renumber variables to appearance order across all atoms
    var_map = {}
    next_var = constant_no + 1
    result = []

    for atom in sorted_atoms:
        canonical_atom = []
        for i, val in enumerate(atom.tolist()):
            if i == 0:
                # Predicate
                canonical_atom.append(val)
            elif val > constant_no and val != padding_idx:
                # Variable
                if val not in var_map:
                    var_map[val] = next_var
                    next_var += 1
                canonical_atom.append(var_map[val])
            else:
                # Constant or padding
                canonical_atom.append(val)
        result.append(tuple(canonical_atom))

    return frozenset(result)


def compare_derived_states(
    prod_states: Tensor,      # [B, K, M, 3]
    prod_counts: Tensor,      # [B]
    ref_states: Tensor,       # [B, K, M, 3]
    ref_counts: Tensor,       # [B]
    constant_no: int,
    padding_idx: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Compare derived states using canonicalization.

    Two implementations are considered equivalent if for each batch element,
    the set of canonical states matches (ignoring order and variable naming).

    Args:
        prod_states: Production derived states [B, K, M, 3]
        prod_counts: Production counts [B]
        ref_states: Reference derived states [B, K, M, 3]
        ref_counts: Reference counts [B]
        constant_no: Number of constants
        padding_idx: Padding index

    Returns:
        Tuple of (num_matches, list_of_mismatches)
        Each mismatch dict contains: batch_idx, prod_set, ref_set, only_in_prod, only_in_ref
    """
    B = prod_states.shape[0]
    matches = 0
    mismatches = []

    for b in range(B):
        prod_count = int(prod_counts[b].item())
        ref_count = int(ref_counts[b].item())

        # Canonicalize each valid derived state
        prod_set = {
            canonicalize_state(prod_states[b, k], constant_no, padding_idx)
            for k in range(prod_count)
        }
        ref_set = {
            canonicalize_state(ref_states[b, k], constant_no, padding_idx)
            for k in range(ref_count)
        }

        if prod_set == ref_set:
            matches += 1
        else:
            only_in_prod = prod_set - ref_set
            only_in_ref = ref_set - prod_set
            mismatches.append({
                'batch_idx': b,
                'prod_count': prod_count,
                'ref_count': ref_count,
                'only_in_prod': only_in_prod,
                'only_in_ref': only_in_ref,
            })

    return matches, mismatches


def compare_counts(
    prod_counts: Tensor,
    ref_counts: Tensor,
    tolerance: int = 0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare derivation counts between two implementations.

    Args:
        prod_counts: [B] Production counts
        ref_counts: [B] Reference counts
        tolerance: Allow difference of up to tolerance

    Returns:
        Tuple of (all_match, stats_dict)
    """
    diff = (prod_counts - ref_counts).abs()
    max_diff = diff.max().item()
    mean_diff = diff.float().mean().item()

    all_match = max_diff <= tolerance

    return all_match, {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'prod_total': prod_counts.sum().item(),
        'ref_total': ref_counts.sum().item(),
        'prod_mean': prod_counts.float().mean().item(),
        'ref_mean': ref_counts.float().mean().item(),
    }


def normalize_next_vars(
    prod_next_vars: Tensor,
    ref_next_vars: Tensor,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare next_var indices between implementations.

    The absolute values may differ but the relative changes should be similar.

    Args:
        prod_next_vars: [B] Production next variable indices
        ref_next_vars: [B] Reference next variable indices

    Returns:
        Tuple of (direction_match, stats_dict)
        direction_match is True if both increased/decreased by similar amounts
    """
    prod_delta = prod_next_vars[1:] - prod_next_vars[:-1] if prod_next_vars.numel() > 1 else torch.zeros(0)
    ref_delta = ref_next_vars[1:] - ref_next_vars[:-1] if ref_next_vars.numel() > 1 else torch.zeros(0)

    # Check if deltas have same sign (both increase or both stay same)
    if prod_delta.numel() > 0:
        prod_sign = (prod_delta > 0).float() - (prod_delta < 0).float()
        ref_sign = (ref_delta > 0).float() - (ref_delta < 0).float()
        direction_match = (prod_sign == ref_sign).all().item()
    else:
        direction_match = True

    return direction_match, {
        'prod_mean_delta': prod_delta.float().mean().item() if prod_delta.numel() > 0 else 0,
        'ref_mean_delta': ref_delta.float().mean().item() if ref_delta.numel() > 0 else 0,
        'prod_range': (prod_next_vars.min().item(), prod_next_vars.max().item()),
        'ref_range': (ref_next_vars.min().item(), ref_next_vars.max().item()),
    }

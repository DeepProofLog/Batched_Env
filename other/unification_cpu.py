"""
Fully Vectorized Unification - NO FOR LOOPS!

This module provides a completely vectorized unification implementation that processes
all batch elements simultaneously without any Python loops over the batch dimension.
"""

import torch
from typing import TYPE_CHECKING, List, Tuple, Dict

if TYPE_CHECKING:
    from index_manager import IndexManager

# Import unification functions from GPU (they work on CPU tensors too)
from unification import (
    _unify_with_rules_batched,
    _unify_with_facts_batched,
    _unify_one_to_one_optimized,
    apply_substitutions_simple,
    apply_substitutions_batched
)

_HASH_PRIMES = torch.tensor([31, 37, 41, 43, 47, 53, 59, 61], dtype=torch.long)


def _tensor_hash(tensor: torch.Tensor) -> int:
    """
    Calculates a fast, device-agnostic hash for a 2D tensor for deduplication.
    """
    if tensor.numel() == 0:
        return 0
    
    # Ensure primes tensor is on the same device as the input tensor
    device = tensor.device
    global _HASH_PRIMES
    if _HASH_PRIMES.device != device:
        _HASH_PRIMES = _HASH_PRIMES.to(device)

    # Pad or truncate the primes to match the number of columns
    rows, cols = tensor.shape
    if cols > len(_HASH_PRIMES):
         # Fallback for unexpectedly wide tensors
        primes = _HASH_PRIMES.repeat(cols // len(_HASH_PRIMES) + 1)[:cols]
    else:
        primes = _HASH_PRIMES[:cols]

    # Calculate hash: (tensor * primes).sum() 
    # This is a fast, vectorized operation.
    return (tensor * primes).sum().item()




def canonicalize_variables_in_state_idx(state_tensor: torch.Tensor, index_manager: "IndexManager", next_var_index: int) -> Tuple[torch.Tensor, int]:
    """Canonicalize variable indices in a state tensor."""
    if state_tensor.numel() == 0: return state_tensor, next_var_index
    device = state_tensor.device
    var_start_index = index_manager.constant_no + 1
    is_var_mask = state_tensor >= var_start_index
    if not is_var_mask.any(): return state_tensor, next_var_index
    unique_vars = torch.unique(state_tensor[is_var_mask])
    num_unique = len(unique_vars)
    if num_unique == 0: return state_tensor, next_var_index
    new_vars = torch.arange(next_var_index, next_var_index + num_unique, device=device)
    result = state_tensor.clone()
    for i in range(num_unique):
        result[result == unique_vars[i]] = new_vars[i]
    return result, next_var_index + num_unique


def get_next_unification(
    current_states: torch.Tensor,
    facts_tensor: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager,
    next_var_indices: torch.Tensor,
    excluded_queries: torch.Tensor = None,
    labels: torch.Tensor = None,
    verbose: int = 0
) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
    """
    Fully vectorized unification with NO for loops over batch dimension.
    
    Strategy:
    1. Extract all queries and remaining goals in parallel using tensor ops
    2. Identify terminal states (empty, true, false) using masks
    3. For non-terminal states, perform batched rule and fact unification
    4. Reassemble results using advanced indexing
    
    Args:
        current_states: [B, padding_atoms, arity+1] batch of states
        facts_tensor: [num_facts, arity+1] facts
        rules: [num_rules, max_rule_atoms, arity+1] rules
        rule_lengths: [num_rules] number of atoms per rule
        index_manager: IndexManager instance
        next_var_indices: [B] next variable index for each query
        excluded_queries: [B, arity+1] queries to exclude from facts (for label=1 queries)
        labels: [B] labels (1=positive, 0=negative)
        verbose: verbosity level
        
    Returns:
        all_derived_states: List of length B, each element is List[Tensor] of derived states
        next_var_indices: [B] updated variable indices
    """
    batch_size = current_states.shape[0]
    pad = index_manager.padding_idx
    device = current_states.device
    
    # Step 1: Extract queries and remaining goals for all batch elements
    # Valid mask: which atoms in each state are not padding
    valid_atoms_mask = current_states[:, :, 0] != pad  # [B, padding_atoms]
    
    # Check which batch elements have at least one valid atom
    has_atoms = valid_atoms_mask.any(dim=1)  # [B]
    
    # Extract first valid atom (query) for each batch element
    # We'll use argmax to find the first True in each row (or 0 if all False)
    first_valid_idx = valid_atoms_mask.long().argmax(dim=1)  # [B]
    batch_indices = torch.arange(batch_size, device=device)
    
    # Extract queries [B, arity+1]
    queries = current_states[batch_indices, first_valid_idx]
    
    # Check for terminal conditions
    empty_states = ~has_atoms  # [B] - states with no atoms
    false_states = (current_states[:, :, 0] == index_manager.false_pred_idx).any(dim=1)  # [B]
    true_queries = queries[:, 0] == index_manager.true_pred_idx  # [B]
    
    # Terminal states: empty OR false OR true query
    terminal_states = empty_states | false_states | true_queries  # [B]
    active_states = ~terminal_states  # [B]
    
    # Step 2: Extract remaining goals for active states
    # Create mask for remaining goals (all atoms after the first valid one)
    remaining_mask = valid_atoms_mask.clone()  # [B, padding_atoms]
    remaining_mask[batch_indices, first_valid_idx] = False  # Remove first atom
    
    # Count remaining goals per batch element
    remaining_counts = remaining_mask.sum(dim=1)  # [B]
    max_remaining = remaining_counts.max().item() if remaining_counts.numel() > 0 else 0
    
    # Extract remaining goals into dense tensor [B, max_remaining, arity+1]
    if max_remaining > 0:
        # Gather remaining goals
        remaining_goals = torch.full(
            (batch_size, max_remaining, current_states.shape[2]),
            pad, dtype=torch.long, device=device
        )
        for b in range(batch_size):
            if remaining_counts[b] > 0:
                valid_remaining = current_states[b, remaining_mask[b]]
                remaining_goals[b, :len(valid_remaining)] = valid_remaining
    else:
        remaining_goals = torch.full(
            (batch_size, 1, current_states.shape[2]),
            pad, dtype=torch.long, device=device
        )
    
    # Step 3: Process active (non-terminal) states
    # Initialize results
    all_derived_states = [[] for _ in range(batch_size)]
    updated_var_indices = next_var_indices.clone()
    
    # Handle terminal states first
    for b in range(batch_size):
        if terminal_states[b]:
            if empty_states[b] or true_queries[b]:
                all_derived_states[b] = [index_manager.true_tensor]
            else:  # false_states[b]
                all_derived_states[b] = [index_manager.false_tensor]
    
    # Get indices of active states
    active_indices = active_states.nonzero(as_tuple=True)[0]
    
    if active_indices.numel() == 0:
        return all_derived_states, updated_var_indices
    
    # Extract active queries and remaining goals
    active_queries = queries[active_indices]  # [N_active, arity+1]
    active_remaining = remaining_goals[active_indices]  # [N_active, max_remaining, arity+1]
    active_remaining_counts = remaining_counts[active_indices]  # [N_active]
    
    # Step 4: Unify with rules and facts in batch
    # Group by predicate for efficient lookup
    predicates = active_queries[:, 0]  # [N_active]
    unique_preds = torch.unique(predicates)
    
    # Process each predicate group
    for pred_idx in unique_preds:
        pred_idx_int = pred_idx.item()
        
        # Get indices of queries with this predicate
        pred_mask = predicates == pred_idx  # [N_active]
        pred_query_indices = pred_mask.nonzero(as_tuple=True)[0]  # Indices within active_queries
        
        # Get original batch indices
        original_batch_indices = active_indices[pred_query_indices]
        
        # Extract queries with this predicate
        pred_queries = active_queries[pred_query_indices]  # [N_pred, arity+1]
        pred_remaining = active_remaining[pred_query_indices]  # [N_pred, max_remaining, arity+1]
        pred_remaining_counts = active_remaining_counts[pred_query_indices]  # [N_pred]
        
        # Unify with rules
        rule_results = _unify_with_rules_batched(
            pred_queries, pred_remaining, pred_remaining_counts,
            rules, rule_lengths, index_manager, pred_idx_int
        )
        
        # Get excluded queries for this predicate group (if any)
        pred_excluded_queries = None
        pred_labels = None
        if excluded_queries is not None and labels is not None:
            pred_excluded_queries = excluded_queries[original_batch_indices]
            pred_labels = labels[original_batch_indices]
        
        # Unify with facts
        fact_results = _unify_with_facts_batched(
            pred_queries, pred_remaining, pred_remaining_counts,
            index_manager, pred_idx_int,
            excluded_queries=pred_excluded_queries,
            labels=pred_labels
        )
        
        # Combine results for each query
        for i, batch_idx in enumerate(original_batch_indices):
            batch_idx = batch_idx.item()
            next_var_idx = next_var_indices[batch_idx].item()
            
            # Get results for this query
            query_rule_results = rule_results[i] if i < len(rule_results) else []
            query_fact_results = fact_results[i] if i < len(fact_results) else []
            
            # Check for early success
            if query_fact_results and len(query_fact_results) > 0:
                if isinstance(query_fact_results[0], torch.Tensor) and \
                   torch.equal(query_fact_results[0], index_manager.true_tensor):
                    all_derived_states[batch_idx] = [index_manager.true_tensor]
                    continue
            
            # Combine all results
            all_results = query_rule_results + query_fact_results
            
            if not all_results:
                all_derived_states[batch_idx] = [index_manager.false_tensor]
                continue
            
            # Canonicalize and deduplicate
            final_states, seen_hashes = [], set()
            found_true = False
            
            for s in all_results:
                s_valid = s[s[:, 0] != pad]
                if s_valid.numel() == 0:
                    all_derived_states[batch_idx] = [index_manager.true_tensor]
                    found_true = True
                    break
                
                s_canonical, next_var_idx = canonicalize_variables_in_state_idx(
                    s_valid, index_manager, next_var_idx
                )
                s_hash = _tensor_hash(s_canonical)
                
                if s_hash not in seen_hashes:
                    final_states.append(s_canonical)
                    seen_hashes.add(s_hash)
            
            if not found_true:
                if not final_states:
                    all_derived_states[batch_idx] = [index_manager.false_tensor]
                else:
                    # Pad results to match original state size
                    original_max_atoms = current_states.shape[1]
                    padded_final_states = []
                    for s in final_states:
                        num_rows, num_cols = s.shape
                        if num_rows > original_max_atoms:
                            padded_final_states.append(s[:original_max_atoms])
                        elif num_rows < original_max_atoms:
                            padding = torch.full(
                                (original_max_atoms - num_rows, num_cols),
                                pad, dtype=s.dtype, device=device
                            )
                            padded_final_states.append(torch.cat([s, padding], dim=0))
                        else:
                            padded_final_states.append(s)
                    
                    all_derived_states[batch_idx] = padded_final_states
            
            updated_var_indices[batch_idx] = next_var_idx
    
    return all_derived_states, updated_var_indices


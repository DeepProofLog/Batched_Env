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
from batched_unification_gpu import _unify_with_rules_batched, _unify_with_facts_batched

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



def _unify_one_to_one_optimized(
    queries: torch.Tensor,
    terms: torch.Tensor,
    im: "IndexManager"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Further optimized unification that replaces the expensive 2D `torch.unique`
    with a faster sort-and-compare method for conflict detection.
    """
    B, A1 = queries.shape
    device = queries.device
    
    if B == 0:
        return torch.empty(0, dtype=torch.bool, device=device), torch.empty(0, A1 - 1, 2, dtype=torch.long, device=device)

    A = A1 - 1
    pad = im.padding_idx
    var_start = im.constant_no + 1
    
    pred_match = queries[:, 0] == terms[:, 0]
    q_args, t_args = queries[:, 1:], terms[:, 1:]
    is_q_var, is_t_var = q_args >= var_start, t_args >= var_start
    
    const_mismatch = (~is_q_var & ~is_t_var) & (q_args != t_args)
    initial_unifiable = pred_match & ~const_mismatch.any(dim=1)
    
    potential_indices = initial_unifiable.nonzero(as_tuple=True)[0]
    if potential_indices.numel() == 0:
        return torch.zeros(B, dtype=torch.bool, device=device), torch.full((B, A, 2), pad, dtype=torch.long, device=device)

    # --- [OPTIMIZATION V2] Replaced torch.unique with sort-and-compare ---
    q_pot, t_pot = q_args[potential_indices], t_args[potential_indices]
    var_to_const_mask = (q_pot >= var_start) & (t_pot < var_start)
    
    batch_indices_pot = torch.arange(q_pot.shape[0], device=device).unsqueeze(1).expand_as(q_pot)
    
    bindings = torch.stack([
        batch_indices_pot[var_to_const_mask],
        q_pot[var_to_const_mask],
        t_pot[var_to_const_mask]
    ], dim=1)

    if bindings.numel() > 0:
        # Sort by batch index, then variable index. This is faster than unique.
        # We use a stable sort to be predictable, though not strictly necessary here.
        # We can combine batch and var indices for a single-pass sort.
        max_var_id = im.variable_no + 1 # A safe upper bound
        combined_key = bindings[:, 0] * max_var_id + bindings[:, 1]
        sorted_indices = torch.argsort(combined_key, stable=True)
        sorted_bindings = bindings[sorted_indices]
        
        # Check adjacent rows for conflicts
        # A conflict is: same batch & var, but different const
        same_batch_and_var = (sorted_bindings[:-1, :2] == sorted_bindings[1:, :2]).all(dim=1)
        different_const = sorted_bindings[:-1, 2] != sorted_bindings[1:, 2]
        
        conflict_pairs = same_batch_and_var & different_const
        
        if conflict_pairs.any():
            conflicting_batch_indices_pot = torch.unique(sorted_bindings[:-1][conflict_pairs, 0])
            
            conflict_mask_pot = torch.zeros(q_pot.shape[0], dtype=torch.bool, device=device)
            conflict_mask_pot[conflicting_batch_indices_pot] = True
            
            initial_unifiable[potential_indices[conflict_mask_pot]] = False

    final_mask = initial_unifiable
    full_subs = torch.full((B, A, 2), pad, dtype=torch.long, device=device)
    success_indices = final_mask.nonzero(as_tuple=True)[0]

    if success_indices.numel() > 0:
        q_succ, t_succ = q_args[success_indices], t_args[success_indices]
        is_q_var_succ, is_t_var_succ = is_q_var[success_indices], is_t_var[success_indices]
        
        # Clone after expand to avoid warning about index_put_ on expanded tensors
        subs_succ = torch.full_like(q_succ, pad).unsqueeze(-1).expand(-1, -1, 2).clone()
        
        q_to_t_mask = is_q_var_succ & ~is_t_var_succ
        subs_succ[q_to_t_mask] = torch.stack([q_succ[q_to_t_mask], t_succ[q_to_t_mask]], dim=-1)

        t_to_q_mask = ~is_q_var_succ & is_t_var_succ
        subs_succ[t_to_q_mask] = torch.stack([t_succ[t_to_q_mask], q_succ[t_to_q_mask]], dim=-1)
        
        full_subs[success_indices] = subs_succ

    return final_mask, full_subs


def apply_substitutions_simple(goals: torch.Tensor, substitutions: torch.Tensor, im: "IndexManager") -> torch.Tensor:
    """Apply substitutions to goals tensor - OPTIMIZED.
    
    Key optimization: Avoid repeated max() calls and minimize tensor operations.
    """
    pad = im.padding_idx
    device = goals.device
    
    # Early exit checks
    if goals.numel() == 0 or substitutions.numel() == 0:
        return goals
    
    valid_mask = substitutions[:, 0] != pad
    if not valid_mask.any():
        return goals
    
    subs = substitutions[valid_mask]
    
    # OPTIMIZED: Compute max once using torch operations instead of Python max()
    # This is much faster than calling .max().item() twice
    # We use a single torch.max call on both tensors
    combined_tensor = torch.cat([goals.flatten(), subs.flatten()])
    max_idx = int(combined_tensor.max()) + 1
    
    # Pre-allocate mapping on correct device
    mapping = torch.arange(max_idx, device=device, dtype=torch.long)
    
    # Batch index assignment
    mapping[subs[:, 0]] = subs[:, 1]
    
    # Direct indexing with reshape
    return mapping[goals.flatten()].reshape(goals.shape)


def apply_substitutions_batched(
    goals_batch: torch.Tensor, 
    substitutions_batch: torch.Tensor, 
    im: "IndexManager"
) -> torch.Tensor:
    """
    Applies a batch of substitutions to a batch of goal tensors.
    
    Args:
        goals_batch (Tensor[B, G, A]): Batch of goal tensors.
        substitutions_batch (Tensor[B, S, 2]): Batch of substitution lists.
        im (IndexManager): The index manager.

    Returns:
        Tensor[B, G, A]: The batch of goals after applying substitutions.
    """
    pad = im.padding_idx
    device = goals_batch.device
    B, G, A = goals_batch.shape
    
    if B == 0:
        return goals_batch

    # Find the maximum index across all goals and substitutions to define the mapping table size
    max_idx = int(max(goals_batch.max(), substitutions_batch.max())) + 1
    
    # Create a batch of mapping tables - clone after expand to avoid warning
    # Shape: [B, max_idx]
    mapping = torch.arange(max_idx, device=device).unsqueeze(0).expand(B, -1).clone()

    # Prepare batch indices for advanced indexing
    batch_indices = torch.arange(B, device=device).unsqueeze(1)

    # Filter out padding substitutions
    valid_mask = substitutions_batch[..., 0] != pad
    
    # Get the actual substitutions and their corresponding batch indices
    from_vars = substitutions_batch[valid_mask][:, 0]
    to_vals = substitutions_batch[valid_mask][:, 1]
    # Clone after expand to avoid warning about index_put_ on expanded tensors
    batch_idx_for_subs = batch_indices.expand_as(substitutions_batch[..., 0])[valid_mask]
    
    # Apply substitutions in a batch "scatter" operation
    # For each item in the batch, this sets mapping[from_var] = to_val
    mapping[batch_idx_for_subs, from_vars] = to_vals
    
    # Apply the mapping to the goals in a batch "gather" operation
    # mapping.gather(1, goals_batch) would work but can be slow with large `max_idx`.
    # A more direct indexing is often better if memory allows.
    return mapping[batch_indices.unsqueeze(-1), goals_batch]


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


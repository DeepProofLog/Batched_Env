"""
Batched Unification with GPU Optimizations

This module implements GPU-optimized unification using GPU-native batch unique operations.

Key Features:
1. GPU-native hashing for deduplication
2. Batched unique operations without CPU transfers
3. Vectorized operations where possible
"""

import torch
from typing import List, Tuple, Set
from gpu_optimizations import gpu_parallel_hash, gpu_batch_unique
from index_manager import IndexManager

def deduplicate_states_gpu(
    states: List[torch.Tensor],
    var_threshold: int,
    padding_idx: int
) -> List[torch.Tensor]:
    """
    GPU-optimized deduplication using batch unique operations.
    
    This is faster than CPU-based deduplication because:
    1. All hashing happens on GPU without transfers
    2. Uses GPU-native unique operations
    3. Fully vectorized batch processing
    
    Args:
        states: List of state tensors to deduplicate
        var_threshold: minimum value for variables
        padding_idx: padding index
        
    Returns:
        List of unique states
    """
    if not states:
        return []
    
    # Quick path: if very few states, deduplication overhead not worth it
    if len(states) <= 2:
        return states
    
    # Stack all states for batch processing
    # First, ensure all have the same shape
    device = states[0].device
    max_atoms = max(s.shape[0] for s in states)
    arity_p1 = states[0].shape[1]
    
    # Pad all states to same size
    padded_states = []
    for s in states:
        if s.shape[0] < max_atoms:
            padding = torch.full(
                (max_atoms - s.shape[0], arity_p1),
                padding_idx, dtype=s.dtype, device=device
            )
            padded = torch.cat([s, padding], dim=0)
        else:
            padded = s[:max_atoms]
        padded_states.append(padded)
    
    # Stack into batch tensor [B, max_atoms, arity+1]
    states_batch = torch.stack(padded_states, dim=0)
    
    # Compute hashes in parallel on GPU
    hashes = gpu_parallel_hash(states_batch, var_threshold, padding_idx)
    
    # Get unique states using GPU batch unique
    unique_states_batch, _ = gpu_batch_unique(states_batch, hashes, return_inverse=False)
    
    # Convert back to list of tensors, removing padding
    unique_list = []
    for i in range(unique_states_batch.shape[0]):
        state = unique_states_batch[i]
        valid_mask = state[:, 0] != padding_idx
        if valid_mask.any():
            unique_list.append(state[valid_mask])
    
    return unique_list



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


def _unify_with_rules_batched(
    queries: torch.Tensor,
    remaining_goals: torch.Tensor,
    remaining_counts: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager,
    pred_idx: int
) -> List[List[torch.Tensor]]:
    """
    Batched rule unification for multiple queries with the same predicate.
    
    Args:
        queries: [N, arity+1] batch of queries with same predicate
        remaining_goals: [N, max_remaining, arity+1] remaining goals for each query
        remaining_counts: [N] number of remaining goals for each query
        rules: [num_rules, max_rule_atoms, arity+1] all rules
        rule_lengths: [num_rules] atoms per rule
        index_manager: IndexManager instance
        pred_idx: predicate index being processed
        
    Returns:
        List of length N, each element is List[Tensor] of derived states from rules
    """
    device = queries.device
    pad = index_manager.padding_idx
    num_queries = queries.shape[0]
    
    # Get relevant rules for this predicate
    relevant_rule_indices = index_manager.rule_index.get(pred_idx, [])
    if not relevant_rule_indices:
        return [[] for _ in range(num_queries)]
    
    num_rules = len(relevant_rule_indices)
    rule_heads = rules[relevant_rule_indices, 0, :]  # [num_rules, arity+1]
    
    # Expand queries to match all rules: [N * num_rules, arity+1]
    queries_expanded = queries.unsqueeze(1).expand(-1, num_rules, -1).reshape(-1, queries.shape[1])
    
    # Expand rule heads to match all queries: [N * num_rules, arity+1]
    rule_heads_expanded = rule_heads.unsqueeze(0).expand(num_queries, -1, -1).reshape(-1, rule_heads.shape[1])
    
    # Batch unify all query-rule pairs
    mask, subs = _unify_one_to_one_optimized(queries_expanded, rule_heads_expanded, index_manager)
    
    # Reshape mask to [N, num_rules]
    mask = mask.reshape(num_queries, num_rules)
    subs = subs.reshape(num_queries, num_rules, subs.shape[1], subs.shape[2])
    
    # OPTIMIZED: Batch convert rule lengths to CPU once
    rule_lengths_cpu = rule_lengths[relevant_rule_indices].cpu().numpy()
    
    # Process results for each query
    results = []
    for q_idx in range(num_queries):
        query_results = []
        
        # Get successful rule matches for this query
        success_mask = mask[q_idx]
        if not success_mask.any():
            results.append(query_results)
            continue
        
        success_indices = success_mask.nonzero(as_tuple=True)[0]
        
        # OPTIMIZED: Convert to CPU once for loop
        success_indices_cpu = success_indices.cpu().numpy()
        
        for rule_idx_local in success_indices_cpu:
            rule_idx_global = relevant_rule_indices[int(rule_idx_local)]
            
            # Get rule body - use pre-converted lengths
            rule_len = int(rule_lengths_cpu[rule_idx_local])
            body = rules[rule_idx_global, 1:rule_len]  # [body_len, arity+1]
            
            # Apply substitutions to body
            current_subs = subs[q_idx, rule_idx_local]
            instantiated_body = apply_substitutions_simple(body, current_subs, index_manager)
            
            # Add remaining goals if any
            if remaining_counts[q_idx] > 0:
                remaining = remaining_goals[q_idx, :remaining_counts[q_idx]]
                instantiated_remaining = apply_substitutions_simple(remaining, current_subs, index_manager)
                next_state = torch.cat([instantiated_body, instantiated_remaining], dim=0)
            else:
                next_state = instantiated_body
            
            query_results.append(next_state)
        
        results.append(query_results)
    
    return results


def _unify_with_facts_batched(
    queries: torch.Tensor,
    remaining_goals: torch.Tensor,
    remaining_counts: torch.Tensor,
    index_manager,
    pred_idx: int,
    excluded_queries: torch.Tensor = None,
    labels: torch.Tensor = None
) -> List[List[torch.Tensor]]:
    """
    VECTORIZED fact unification for multiple queries with the same predicate.
    
    For positive queries (label=1), excludes the original query from facts to prevent
    trivial self-unification.
    
    Args:
        queries: [N, arity+1] batch of queries with same predicate
        remaining_goals: [N, max_remaining, arity+1] remaining goals for each query
        remaining_counts: [N] number of remaining goals for each query
        index_manager: IndexManager instance
        pred_idx: predicate index being processed
        excluded_queries: [N, arity+1] queries to exclude from facts (for label=1)
        labels: [N] labels (1=positive, 0=negative)
        
    Returns:
        List of length N, each element is List[Tensor] of derived states from facts
    """
    device = queries.device
    pad = index_manager.padding_idx
    num_queries = queries.shape[0]
    
    # OPTIMIZED: Batch the .item() calls by converting query args to CPU once
    q_args_cpu = queries[:, 1:].cpu().numpy()  # [N, arity]
    
    # VECTORIZED: Lookup all candidates at once
    # Collect all candidate facts for all queries
    all_candidates = []
    query_to_candidate_ranges = []  # (start_idx, end_idx) for each query
    
    start_idx = 0
    for q_idx in range(num_queries):
        # Use numpy array instead of .item() calls
        q_args = q_args_cpu[q_idx]
        
        # Construct lookup key - now using numpy ints directly
        constant_args_with_pos = []
        for arg_pos in range(index_manager.max_arity):
            arg_val = int(q_args[arg_pos])
            if not (index_manager.constant_no < arg_val <= index_manager.runtime_var_end_index) and arg_val != pad:
                constant_args_with_pos.append((arg_pos, arg_val))
        
        lookup_key = (pred_idx,) + tuple(constant_args_with_pos)
        
        # Lookup candidates
        candidate_indices = index_manager.fact_index.get(lookup_key)
        
        if candidate_indices is not None and candidate_indices.numel() > 0:
            relevant_facts = index_manager.facts_tensor[candidate_indices]
            
            # EXCLUDE the original query for positive labels (label=1)
            if excluded_queries is not None and labels is not None:
                if labels[q_idx].item() == 1:
                    # Exclude facts that match the original excluded query
                    excluded_query = excluded_queries[q_idx]
                    # Check which facts match the excluded query
                    match_mask = (relevant_facts == excluded_query.unsqueeze(0)).all(dim=1)
                    # Keep only non-matching facts
                    if match_mask.any():
                        relevant_facts = relevant_facts[~match_mask]
            
            # Only add if we still have candidates after exclusion
            if relevant_facts.shape[0] > 0:
                all_candidates.append(relevant_facts)
                end_idx = start_idx + relevant_facts.shape[0]
                query_to_candidate_ranges.append((start_idx, end_idx))
                start_idx = end_idx
            else:
                query_to_candidate_ranges.append((start_idx, start_idx))  # Empty range
        else:
            query_to_candidate_ranges.append((start_idx, start_idx))  # Empty range
    
    # If no candidates found for any query, return empty results
    if len(all_candidates) == 0:
        return [[] for _ in range(num_queries)]
    
    # VECTORIZED: Stack all candidates and all queries
    all_candidates_tensor = torch.cat(all_candidates, dim=0)  # [total_candidates, arity+1]
    
    # Create expanded queries to match candidates
    expanded_queries = []
    for q_idx, (start, end) in enumerate(query_to_candidate_ranges):
        num_candidates = end - start
        if num_candidates > 0:
            expanded_queries.append(queries[q_idx:q_idx+1].expand(num_candidates, -1))
    
    if len(expanded_queries) == 0:
        return [[] for _ in range(num_queries)]
    
    expanded_queries_tensor = torch.cat(expanded_queries, dim=0)  # [total_candidates, arity+1]
    
    # VECTORIZED: Unify all at once
    mask, subs = _unify_one_to_one_optimized(
        expanded_queries_tensor,
        all_candidates_tensor,
        index_manager
    )
    
    # VECTORIZED: Process results for each query
    results = []
    for q_idx, (start, end) in enumerate(query_to_candidate_ranges):
        num_candidates = end - start
        
        if num_candidates == 0:
            results.append([])
            continue
        
        # Get results for this query
        query_mask = mask[start:end]
        query_subs = subs[start:end]
        
        success_indices = query_mask.nonzero(as_tuple=True)[0]
        if success_indices.numel() == 0:
            results.append([])
            continue
        
        # If no remaining goals, return true
        if remaining_counts[q_idx] == 0:
            results.append([index_manager.true_tensor])
            continue
        
        # Apply substitutions to remaining goals
        succ_subs = query_subs[success_indices]
        remaining = remaining_goals[q_idx:q_idx+1, :remaining_counts[q_idx]]  # [1, rem_count, arity+1]
        
        # Expand remaining goals for all successful unifications
        num_succ = succ_subs.shape[0]
        expanded_goals = remaining.expand(num_succ, -1, -1).clone()  # [num_succ, rem_count, arity+1]
        
        # Apply substitutions in batch
        instantiated_goals_batch = apply_substitutions_batched(expanded_goals, succ_subs, index_manager)
        
        # Convert to list of tensors
        query_results = list(torch.unbind(instantiated_goals_batch, dim=0))
        results.append(query_results)
    
    return results

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
    Fully vectorized unification WITHOUT canonicalization.
    
    Key difference from standard approach:
    - Uses structural hashing for deduplication
    - No variable renaming (canonicalization) needed
    - Faster because it skips the expensive canonicalization step
    
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
        index_manager: IndexManager instance
        next_var_indices: [B] next variable index for each query
        verbose: verbosity level
        
    Returns:
        all_derived_states: List of length B, each element is List[Tensor] of derived states
        next_var_indices: [B] updated variable indices (not modified in this version)
    """
    batch_size = current_states.shape[0]
    pad = index_manager.padding_idx
    var_threshold = index_manager.constant_no + 1
    device = current_states.device
    
    # Step 1: Extract queries and remaining goals for all batch elements
    valid_atoms_mask = current_states[:, :, 0] != pad
    has_atoms = valid_atoms_mask.any(dim=1)
    
    first_valid_idx = valid_atoms_mask.long().argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=device)
    
    queries = current_states[batch_indices, first_valid_idx]
    
    # Check for terminal conditions
    empty_states = ~has_atoms
    false_states = (current_states[:, :, 0] == index_manager.false_pred_idx).any(dim=1)
    true_queries = queries[:, 0] == index_manager.true_pred_idx
    
    terminal_states = empty_states | false_states | true_queries
    active_states = ~terminal_states
    
    # Step 2: Extract remaining goals
    remaining_mask = valid_atoms_mask.clone()
    remaining_mask[batch_indices, first_valid_idx] = False
    
    remaining_counts = remaining_mask.sum(dim=1)
    max_remaining = remaining_counts.max().item() if remaining_counts.numel() > 0 else 0
    
    if max_remaining > 0:
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
    
    # Step 3: Initialize results
    all_derived_states = [[] for _ in range(batch_size)]
    updated_var_indices = next_var_indices.clone()
    
    # Handle terminal states
    for b in range(batch_size):
        if terminal_states[b]:
            if empty_states[b] or true_queries[b]:
                all_derived_states[b] = [index_manager.true_tensor]
            else:
                all_derived_states[b] = [index_manager.false_tensor]
    
    active_indices = active_states.nonzero(as_tuple=True)[0]
    
    if active_indices.numel() == 0:
        return all_derived_states, updated_var_indices
    
    active_queries = queries[active_indices]
    active_remaining = remaining_goals[active_indices]
    active_remaining_counts = remaining_counts[active_indices]
    
    # Step 4: Process by predicate
    predicates = active_queries[:, 0]
    unique_preds = torch.unique(predicates)
    
    for pred_idx in unique_preds:
        pred_idx_int = pred_idx.item()
        
        pred_mask = predicates == pred_idx
        pred_query_indices = pred_mask.nonzero(as_tuple=True)[0]
        original_batch_indices = active_indices[pred_query_indices]
        
        pred_queries = active_queries[pred_query_indices]
        pred_remaining = active_remaining[pred_query_indices]
        pred_remaining_counts = active_remaining_counts[pred_query_indices]
        
        # Ensure all tensors are on GPU
        pred_queries = pred_queries.to(device)
        pred_remaining = pred_remaining.to(device)
        rules = rules.to(device)
        
        rule_results = _unify_with_rules_batched(
            pred_queries, pred_remaining, pred_remaining_counts,
            rules, rule_lengths, index_manager, pred_idx_int
        )
        
        # Get excluded queries for this predicate group (if any)
        pred_excluded_queries = None
        pred_labels = None
        if excluded_queries is not None and labels is not None:
            pred_excluded_queries = excluded_queries[original_batch_indices].to(device)
            pred_labels = labels[original_batch_indices].to(device)
        
        fact_results = _unify_with_facts_batched(
            pred_queries, pred_remaining, pred_remaining_counts,
            index_manager, pred_idx_int,
            excluded_queries=pred_excluded_queries,
            labels=pred_labels
        )
        
        # Process results WITHOUT canonicalization
        for i, batch_idx in enumerate(original_batch_indices):
            batch_idx = batch_idx.item()
            next_var_idx = next_var_indices[batch_idx].item()
            
            query_rule_results = rule_results[i] if i < len(rule_results) else []
            query_fact_results = fact_results[i] if i < len(fact_results) else []
            
            # Check for early success
            if query_fact_results and len(query_fact_results) > 0:
                if isinstance(query_fact_results[0], torch.Tensor) and \
                   torch.equal(query_fact_results[0], index_manager.true_tensor):
                    all_derived_states[batch_idx] = [index_manager.true_tensor]
                    continue
            
            all_results = query_rule_results + query_fact_results
            
            if not all_results:
                all_derived_states[batch_idx] = [index_manager.false_tensor]
                continue
            
            # CRITICAL CHANGE: Use structural deduplication instead of canonicalization
            final_states = []
            found_true = False
            
            for s in all_results:
                s_valid = s[s[:, 0] != pad]
                if s_valid.numel() == 0:
                    all_derived_states[batch_idx] = [index_manager.true_tensor]
                    found_true = True
                    break
                
                final_states.append(s_valid)
            
            if not found_true:
                if not final_states:
                    all_derived_states[batch_idx] = [index_manager.false_tensor]
                else:
                    # OPTIMIZATION: Truncate early if too many states (before deduplication)
                    # This avoids expensive deduplication on large state sets
                    max_states_before_dedup = 100  # Reasonable limit
                    if len(final_states) > max_states_before_dedup:
                        final_states = final_states[:max_states_before_dedup]
                    
                    # Deduplicate using GPU batch unique operations
                    unique_states = deduplicate_states_gpu(
                        final_states, var_threshold, pad
                    )
                    
                    # Pad results
                    original_max_atoms = current_states.shape[1]
                    padded_final_states = []
                    for s in unique_states:
                        num_rows, num_cols = s.shape
                        if num_rows > original_max_atoms:
                            padded_final_states.append(s[:original_max_atoms])
                        elif num_rows < original_max_atoms:
                            padding_tensor = torch.full(
                                (original_max_atoms - num_rows, num_cols),
                                pad, dtype=s.dtype, device=device
                            )
                            padded_final_states.append(torch.cat([s, padding_tensor], dim=0))
                        else:
                            padded_final_states.append(s)
                    
                    all_derived_states[batch_idx] = padded_final_states
            
            # Note: We don't update next_var_indices since we're not canonicalizing
            # Variables keep their original IDs
    
    return all_derived_states, updated_var_indices

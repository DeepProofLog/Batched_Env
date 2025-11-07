"""
Batched Unification with GPU Optimizations

This module implements GPU-optimized unification using GPU-native batch unique operations.

Key Features:
1. GPU-native hashing for deduplication
2. Batched unique operations without CPU transfers
3. Vectorized operations where possible
"""

import torch
from typing import List, Tuple, Optional
from index_manager import IndexManager



def gpu_parallel_hash(states: torch.Tensor, var_threshold: int, padding_idx: int) -> torch.Tensor:
    """
    GPU-parallelized hashing for multiple states - OPTIMIZED VERSION.
    
    Uses vectorized operations and avoids .item() calls for massive speedup.
    
    Args:
        states: [B, max_atoms, arity+1] batch of states
        var_threshold: minimum value for variables
        padding_idx: padding index
        
    Returns:
        hashes: [B] hash values for each state
    """
    B, max_atoms, arity_p1 = states.shape
    device = states.device
    
    # Simple but fast hash: just flatten and use torch hashing
    # This avoids expensive canonicalization and .item() calls
    
    # Valid atoms mask
    valid_mask = states[:, :, 0] != padding_idx  # [B, max_atoms]
    
    # Create a simple hash based on the flattened representation
    # We'll use a polynomial rolling hash computed fully on GPU
    prime = 31
    mod_val = 2**31 - 1  # Large prime for modulo
    
    # Flatten states: [B, max_atoms * arity_p1]
    flat_states = states.reshape(B, -1).long()
    
    # Compute powers of prime: [max_atoms * arity_p1]
    max_len = max_atoms * arity_p1
    powers = torch.arange(max_len, device=device, dtype=torch.long)
    prime_powers = torch.pow(prime, powers) % mod_val
    
    # Compute hash as sum of (value * prime^position) for each position
    # [B, max_atoms * arity_p1] * [max_atoms * arity_p1] -> [B]
    hashes = (flat_states * prime_powers.unsqueeze(0)).sum(dim=1) % mod_val
    
    # Zero out hashes for empty states
    has_atoms = valid_mask.any(dim=1)
    hashes = hashes * has_atoms.long()
    
    return hashes


def gpu_batch_unique(
    states: torch.Tensor,
    hashes: torch.Tensor,
    return_inverse: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    GPU-based deduplication using precomputed hashes.
    
    Args:
        states: [B, max_atoms, arity+1] states to deduplicate
        hashes: [B] hash values for each state
        return_inverse: whether to return inverse mapping
        
    Returns:
        unique_states: [U, max_atoms, arity+1] unique states
        inverse_indices: [B] mapping from original to unique (if return_inverse=True)
    """
    device = states.device
    B = states.shape[0]
    
    # Sort by hash
    sorted_hashes, sort_indices = torch.sort(hashes)
    
    # Find unique hashes
    if B == 0:
        return states[:0], torch.tensor([], dtype=torch.long, device=device) if return_inverse else None
    
    # Identify unique positions
    unique_mask = torch.ones(B, dtype=torch.bool, device=device)
    if B > 1:
        unique_mask[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
    
    # Get unique indices
    unique_indices = sort_indices[unique_mask]
    
    # Extract unique states
    unique_states = states[unique_indices]
    
    if return_inverse:
        # Create inverse mapping
        inverse = torch.zeros(B, dtype=torch.long, device=device)
        unique_pos = torch.arange(unique_mask.sum(), device=device)
        
        # Map each original index to its unique position
        cumsum = unique_mask.cumsum(0) - 1
        inverse[sort_indices] = cumsum
        
        return unique_states, inverse
    
    return unique_states, None



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
    
    # Ensure substitutions are on same device as goals
    substitutions_batch = substitutions_batch.to(device)
    
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
    pred_idx: int,
    verbose_engine: int = 0
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
    
    if verbose_engine >= 1:
        print(f"\n  [RULE UNIFICATION] Predicate: {index_manager.predicate_idx2str.get(pred_idx, f'pred_{pred_idx}')}")
        print(f"    Queries ({num_queries}):")
        for i in range(num_queries):
            print(f"      Q{i}: {index_manager.debug_print_state_from_indices(queries[i:i+1], oneline=True)}")
    
    # Get relevant rules for this predicate
    relevant_rule_indices = index_manager.rule_index.get(pred_idx, [])
    if not relevant_rule_indices:
        if verbose_engine >= 1:
            print(f"    No rules found for this predicate.")
        return [[] for _ in range(num_queries)]
    
    num_rules = len(relevant_rule_indices)
    rule_heads = rules[relevant_rule_indices, 0, :]  # [num_rules, arity+1]
    
    if verbose_engine >= 2:
        print(f"    Relevant rules ({num_rules}):")
        for idx, rule_idx in enumerate(relevant_rule_indices):
            rule_len = int(rule_lengths[rule_idx].item())
            rule_atoms = rules[rule_idx, :rule_len]
            print(f"      R{idx}: {index_manager.debug_print_state_from_indices(rule_atoms, oneline=True)}")
    
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
            if verbose_engine >= 1:
                print(f"    Q{q_idx}: No successful rule unifications")
            results.append(query_results)
            continue
        
        success_indices = success_mask.nonzero(as_tuple=True)[0]
        
        if verbose_engine >= 1:
            print(f"    Q{q_idx}: {success_indices.numel()} successful rule unifications")
        
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
            
            if verbose_engine >= 2:
                print(f"      R{rule_idx_local}: {index_manager.debug_print_state_from_indices(next_state, oneline=True)}")
            
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
    labels: torch.Tensor = None,
    verbose_engine: int = 0
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
    
    if verbose_engine >= 1:
        print(f"\n  [FACT UNIFICATION] Predicate: {index_manager.predicate_idx2str.get(pred_idx, f'pred_{pred_idx}')}")
        print(f"    Queries ({num_queries}):")
        for i in range(num_queries):
            print(f"      Q{i}: {index_manager.debug_print_state_from_indices(queries[i:i+1], oneline=True)}")
    
    # OPTIMIZED: Batch the .item() calls by converting query args to CPU once
    q_args_cpu = queries[:, 1:].cpu().numpy()  # [N, arity]
    
    # VECTORIZEDLookup all candidates at once
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
                    excluded_query = excluded_queries[q_idx].to(relevant_facts.device)
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
        if verbose_engine >= 1:
            print(f"    No candidate facts found for any query.")
        return [[] for _ in range(num_queries)]
    
    if verbose_engine >= 2:
        print(f"    Total candidate facts: {sum(end - start for start, end in query_to_candidate_ranges)}")
        for i, (start, end) in enumerate(query_to_candidate_ranges):
            if end > start:
                print(f"      Q{i}: {end - start} candidates")
    
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
    
    # Ensure queries are on same device as candidates
    expanded_queries_tensor = expanded_queries_tensor.to(all_candidates_tensor.device)
    
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
            if verbose_engine >= 1:
                print(f"    Q{q_idx}: No candidate facts")
            results.append([])
            continue
        
        # Get results for this query
        query_mask = mask[start:end]
        query_subs = subs[start:end]
        
        success_indices = query_mask.nonzero(as_tuple=True)[0]
        if success_indices.numel() == 0:
            if verbose_engine >= 1:
                print(f"    Q{q_idx}: No successful fact unifications")
            results.append([])
            continue
        
        if verbose_engine >= 1:
            print(f"    Q{q_idx}: {success_indices.numel()} successful fact unifications")
            for idx in success_indices[:20]:
                fact_idx = start + idx.item()
                fact = all_candidates_tensor[fact_idx]
                print(f"      Fact: {index_manager.debug_print_state_from_indices(fact.unsqueeze(0), oneline=True)}")
                print(f"            Substitution: {index_manager.debug_print_state_from_indices(fact.unsqueeze(0), oneline=True)}")
        
        # If no remaining goals, return true
        if remaining_counts[q_idx] == 0:
            if verbose_engine >= 2:
                print(f"      -> TRUE (no remaining goals)")
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
        
        if verbose_engine >= 2:
            for idx, result in enumerate(query_results[:5]):  # Show first 5
                print(f"      Result {idx}: {index_manager.debug_print_state_from_indices(result, oneline=True)}")
            if len(query_results) > 5:
                print(f"      ... and {len(query_results) - 5} more")
        
        results.append(query_results)
    
    return results



def get_queries_and_remaining_goals(current_states: torch.Tensor,
                                    index_manager,
                                    next_var_indices: torch.Tensor,
                                    batch_size,
                                    device,
                                    pad) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract queries and remaining goals from current states.
    Also checks for terminal conditions (true/false states).
    """
    valid_atoms_mask = current_states[:, :, 0] != pad
    has_atoms = valid_atoms_mask.any(dim=1)
    
    first_valid_idx = valid_atoms_mask.long().argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=device)
    
    queries = current_states[batch_indices, first_valid_idx]
    
    # Check for terminal conditions
    empty_states = ~has_atoms
    # if there is any empty state, raise an error
    if empty_states.any():
        raise ValueError("Empty states detected in current_states; should not happen.")
    false_states = (current_states[:, :, 0] == index_manager.false_pred_idx).any(dim=1)
    true_queries = queries[:, 0] == index_manager.true_pred_idx
    
    terminal_states = false_states | true_queries
    active_states = ~terminal_states
    
    # Extract remaining goals
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

    # Initialize results
    all_derived_states = [[] for _ in range(batch_size)]
    updated_var_indices = next_var_indices.clone()
    
    # Handle terminal states
    for b in range(batch_size):
        if terminal_states[b]:
            if true_queries[b]:
                all_derived_states[b] = [index_manager.true_tensor]
            elif false_states[b]:
                all_derived_states[b] = [index_manager.false_tensor]
            else:
                raise ValueError("Unexpected terminal state encountered.")
    return queries, remaining_goals, remaining_counts, active_states, all_derived_states, updated_var_indices


def get_next_unification(
    current_states: torch.Tensor,
    facts_tensor: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager,
    next_var_indices: torch.Tensor,
    excluded_queries: torch.Tensor = None,
    labels: torch.Tensor = None,
    verbose: int = 0,
    verbose_engine: int = 0
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
    
    if verbose_engine >= 1:
        print(f"\n{'='*80}")
        print(f"GET_NEXT_UNIFICATION - Batch size: {batch_size}")
        print(f"{'='*80}")
    
    # Step 1: Extract queries and remaining goals and initial checks

    queries, remaining_goals, remaining_counts, active_states, all_derived_states, updated_var_indices = get_queries_and_remaining_goals(
        current_states, index_manager, next_var_indices, batch_size, device, pad)

    if verbose_engine >= 1:
        print(f"\n[INITIAL QUERIES]")
        for i in range(batch_size):
            query_str = index_manager.debug_print_state_from_indices(queries[i:i+1], oneline=True)
            remaining_str = ""
            if remaining_counts[i] > 0:
                remaining = remaining_goals[i, :remaining_counts[i]]
                remaining_str = f" | Remaining: {index_manager.debug_print_state_from_indices(remaining, oneline=True)}"
            print(f"  Env {i}: {query_str}{remaining_str}")
            if not active_states[i]:
                print(f"    -> TERMINAL (skipping unification)")
    
    active_indices = active_states.nonzero(as_tuple=True)[0]

    if active_indices.numel() == 0: # all states are terminal
        if verbose_engine >= 1:
            print(f"\nAll states are terminal. Returning.")
        return all_derived_states, updated_var_indices
    
    active_queries = queries[active_indices]
    active_remaining = remaining_goals[active_indices]
    active_remaining_counts = remaining_counts[active_indices]
    
    # Process by predicate
    predicates = active_queries[:, 0]
    unique_preds = torch.unique(predicates)
    
    if verbose_engine >= 1:
        print(f"\n[PROCESSING BY PREDICATE]")
        print(f"  Unique predicates: {[index_manager.predicate_idx2str.get(p.item(), f'pred_{p.item()}') for p in unique_preds]}")
    
    for pred_idx in unique_preds:
        if verbose_engine >= 1:
            print(f"\n{'-'*60}\nProcessing predicate index: {pred_idx.item()} ({index_manager.predicate_idx2str.get(pred_idx.item(), f'pred_{pred_idx.item()}')})")
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
        
        # Step 2: Rule unification
        # List of length N as next states means that there are N queries, 
        # and each query has a list with derived states, 
        # and each derived state is a tensor of (pad_atoms,embedding_size)
        rule_results = _unify_with_rules_batched(
            pred_queries, 
            pred_remaining, 
            pred_remaining_counts,
            rules, 
            rule_lengths, 
            index_manager, 
            pred_idx_int,
            verbose_engine=verbose_engine
        )
        
        # Get excluded queries for this predicate group (if any)
        pred_excluded_queries = None
        pred_labels = None
        if excluded_queries is not None and labels is not None:
            pred_excluded_queries = excluded_queries[original_batch_indices].to(device)
            pred_labels = labels[original_batch_indices].to(device)
        
        # Step 3: Fact unification
        fact_results = _unify_with_facts_batched(
            pred_queries, 
            pred_remaining, 
            pred_remaining_counts,
            index_manager, 
            pred_idx_int,
            excluded_queries=pred_excluded_queries,
            labels=pred_labels,
            verbose_engine=verbose_engine
        )

        if verbose_engine >= 1:
            print(f"\n  [SUMMARY] Predicate {index_manager.predicate_idx2str.get(pred_idx_int, f'pred_{pred_idx_int}')}: "
                  f"len(rule_results)={len(rule_results)}, len(fact_results)={len(fact_results)}")

        # Step 4: Combine and deduplicate results
        if verbose_engine >= 1:
            print(f"\n  [COMBINE & DEDUPLICATE]")
            
        for i, batch_idx in enumerate(original_batch_indices):
            batch_idx = batch_idx.item()
            next_var_idx = next_var_indices[batch_idx].item()
            
            # Combine results
            query_rule_results = rule_results[i] if i < len(rule_results) else []
            query_fact_results = fact_results[i] if i < len(fact_results) else []
            
            if verbose_engine >= 1:
                print(f"    Env {batch_idx}: {len(query_rule_results)} rule results + {len(query_fact_results)} fact results")
            
            # Check for early success
            if query_fact_results and len(query_fact_results) > 0:
                if isinstance(query_fact_results[0], torch.Tensor) and \
                    torch.equal(query_fact_results[0].cpu(), index_manager.true_tensor.cpu()):
                    all_derived_states[batch_idx] = [index_manager.true_tensor]
                    if verbose_engine >= 1:
                        print(f"      -> Early success: TRUE")
                    continue
            
            all_results = query_rule_results + query_fact_results

            if not all_results:
                all_derived_states[batch_idx] = [index_manager.false_tensor]
                if verbose_engine >= 1:
                    print(f"      -> No results: FALSE")
                continue
            
            # Deduplicate states. Use structural deduplication instead of canonicalization
            final_states = []
            found_true = False
            
            # get the combined states by filtering padding and true states
            for s in all_results:
                s_valid = s[s[:, 0] != pad]

                if s_valid.numel() == 0:
                    raise ValueError("Derived state should not be empty after filtering padding.")                
                final_states.append(s_valid)
            
            if not found_true:
                if not final_states:
                    # Set to false if no valid states
                    all_derived_states[batch_idx] = [index_manager.false_tensor]
                    if verbose_engine >= 1:
                        print(f"      -> No valid states: FALSE")
                else:
                    if len(final_states) > 100:
                        print(f"Warning: query index (batch idx) {batch_idx} has a large number of derived states ({len(final_states)}).")
                        # raise ValueError(f"query index (batch idx) {batch_idx}: Large number of derived states ({len(final_states)})")
                        final_states = final_states[:100]
                    
                    # Deduplicate using GPU batch unique operations
                    unique_states = deduplicate_states_gpu(
                        final_states, var_threshold, pad
                    )
                    
                    if verbose_engine >= 2:
                        print(f"      -> After deduplication: {len(unique_states)} unique states (from {len(final_states)})")
                    
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

    if verbose_engine >= 1:
        print(f"\n[FINAL NEXT STATES]")
        for i in range(batch_size):
            num_states = len(all_derived_states[i])
            print(f"  Env {i}: {num_states} derived states")
            if num_states > 0:
                for j, state in enumerate(all_derived_states[i][:3]):  # Show first 3
                    print(f"    State {j}: {index_manager.debug_print_state_from_indices(state, oneline=True)}")
                if num_states > 3:
                    print(f"    ... and {num_states - 3} more")
        print(f"{'='*80}\n")

    return all_derived_states, updated_var_indices

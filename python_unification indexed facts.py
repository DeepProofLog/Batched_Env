import torch
from typing import List, Dict, Tuple, Optional
from index_manager import IndexManager

# [OPTIMIZATION V2] Pre-calculate a tensor of primes for fast hashing.
# The size should be larger than the max number of columns in a state tensor.
# Using different primes reduces the chance of hash collisions.
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
        
        subs_succ = torch.full_like(q_succ, pad).unsqueeze(-1).expand(-1, -1, 2)
        
        q_to_t_mask = is_q_var_succ & ~is_t_var_succ
        subs_succ[q_to_t_mask] = torch.stack([q_succ[q_to_t_mask], t_succ[q_to_t_mask]], dim=-1)

        t_to_q_mask = ~is_q_var_succ & is_t_var_succ
        subs_succ[t_to_q_mask] = torch.stack([t_succ[t_to_q_mask], q_succ[t_to_q_mask]], dim=-1)
        
        full_subs[success_indices] = subs_succ

    return final_mask, full_subs

# The other helper functions (apply_substitutions_simple, canonicalize_variables_in_state_idx)
# remain the same as the previous version. They are not the primary bottleneck right now.
def apply_substitutions_simple(goals: torch.Tensor, substitutions: torch.Tensor, im: IndexManager) -> torch.Tensor:
    pad = im.padding_idx
    device = goals.device
    if goals.numel() == 0 or substitutions.numel() == 0: return goals
    valid_mask = substitutions[:, 0] != pad
    if not valid_mask.any(): return goals
    subs = substitutions[valid_mask]
    max_idx = max(int(goals.max()), int(subs.max())) + 1
    mapping = torch.arange(max_idx, device=device)
    mapping[subs[:, 0]] = subs[:, 1]
    original_shape = goals.shape
    return mapping[goals.flatten()].reshape(original_shape)

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
    
    # Create a batch of mapping tables
    # Shape: [B, max_idx]
    mapping = torch.arange(max_idx, device=device).unsqueeze(0).expand(B, -1)

    # Prepare batch indices for advanced indexing
    batch_indices = torch.arange(B, device=device).unsqueeze(1)

    # Filter out padding substitutions
    valid_mask = substitutions_batch[..., 0] != pad
    
    # Get the actual substitutions and their corresponding batch indices
    from_vars = substitutions_batch[valid_mask][:, 0]
    to_vals = substitutions_batch[valid_mask][:, 1]
    batch_idx_for_subs = batch_indices.expand_as(substitutions_batch[..., 0])[valid_mask]
    
    # Apply substitutions in a batch "scatter" operation
    # For each item in the batch, this sets mapping[from_var] = to_val
    mapping[batch_idx_for_subs, from_vars] = to_vals
    
    # Apply the mapping to the goals in a batch "gather" operation
    # mapping.gather(1, goals_batch) would work but can be slow with large `max_idx`.
    # A more direct indexing is often better if memory allows.
    return mapping[batch_indices.unsqueeze(-1), goals_batch]

def canonicalize_variables_in_state_idx(state_tensor: torch.Tensor, index_manager: "IndexManager", next_var_index: int) -> Tuple[torch.Tensor, int]:
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

def _unify_with_rules(
    query: torch.Tensor,
    remaining_goals: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager: "IndexManager"
) -> List[torch.Tensor]:
    """
    Unifies a query with rule heads and generates new goal states.
    Returns a list of new, unpadded, unsubstituted goal states.
    """
    device = query.device
    query_pred = query[0, 0].item()
    next_states_list = []
    
    # Use the rule_index to find relevant rules (this part is already efficient)
    relevant_rule_indices = index_manager.rule_index.get(query_pred, [])
    if not relevant_rule_indices:
        return []

    rule_heads = rules[relevant_rule_indices, 0, :]
    mask, subs = _unify_one_to_one_optimized(query.expand(len(relevant_rule_indices), -1), rule_heads, index_manager)
    
    success_indices = mask.nonzero(as_tuple=True)[0]
    if success_indices.numel() == 0:
        return []

    succ_rel_indices = torch.tensor(relevant_rule_indices, device=device)[success_indices]
    succ_subs = subs[success_indices]

    # This Python loop is over a small number of successful unifications, which is generally fine.
    for i, rule_idx in enumerate(succ_rel_indices):
        body = rules[rule_idx, 1:rule_lengths[rule_idx].item()]
        current_subs = succ_subs[i]
        
        instantiated_body = apply_substitutions_simple(body, current_subs, index_manager)
        
        if remaining_goals.numel() > 0:
            instantiated_rest = apply_substitutions_simple(remaining_goals, current_subs, index_manager)
            next_states_list.append(torch.cat([instantiated_body, instantiated_rest], dim=0))
        else:
            next_states_list.append(instantiated_body)
            
    return next_states_list


def _unify_with_facts_fully_indexed(
    query: torch.Tensor,
    remaining_goals: torch.Tensor,
    index_manager: "IndexManager"
) -> List[torch.Tensor]:
    """
    [OPTIMIZED V5] This function uses the faithfully replicated composite index.
    It performs a single dictionary lookup to get the exact candidate set.
    """
    device = query.device
    pad = index_manager.padding_idx
    query_pred = query[0, 0].item()
    q_args = query[0, 1:]

    # --- Step 1: Construct the canonical lookup key from the query ---
    constant_args_with_pos = []
    for arg_pos in range(index_manager.max_arity):
        arg_val = q_args[arg_pos].item()
        if not index_manager.is_var_idx(arg_val) and arg_val != pad:
            constant_args_with_pos.append((arg_pos, arg_val))
            
    # The key must match the canonical format stored in the index.
    # Note: The combination automatically creates sorted-by-position tuples.
    lookup_key = (query_pred,) + tuple(constant_args_with_pos)

    # --- Step 2: Perform a single, O(1) lookup to get candidate indices ---
    candidate_indices = index_manager.fact_index.get(lookup_key)

    # If the key doesn't exist, no facts could possibly match.
    if candidate_indices is None or candidate_indices.numel() == 0:
        return []

    # Retrieve the small, precise set of candidate facts.
    relevant_facts = index_manager.facts_tensor[candidate_indices]

    # --- Step 3: Unify with the tiny, pre-filtered set of facts ---
    mask, subs = _unify_one_to_one_optimized(query.expand(relevant_facts.shape[0], -1), relevant_facts, index_manager)
    
    success_indices = mask.nonzero(as_tuple=True)[0]
    if success_indices.numel() == 0:
        return []

    if remaining_goals.numel() == 0:
        return [index_manager.true_tensor]

    succ_subs = subs[success_indices]
    num_succ = succ_subs.shape[0]
    
    expanded_goals = remaining_goals.unsqueeze(0).expand(num_succ, -1, -1)
    instantiated_goals_batch = apply_substitutions_batched(expanded_goals, succ_subs, index_manager)
    
    return list(torch.unbind(instantiated_goals_batch, dim=0))

def get_next_unification_pt(
    current_state: torch.Tensor,
    facts_tensor: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager: "IndexManager",
    next_var_index: int,
    verbose: int = 0
) -> Tuple[List[torch.Tensor], int]:
    """
    [Refactored for Profiling]
    Orchestrates unification by calling separate functions for rules and facts.
    """
    pad = index_manager.padding_idx
    device = current_state.device
    
    valid_mask = current_state[:, 0] != pad
    if not valid_mask.any(): return [index_manager.true_tensor], next_var_index
    
    state = current_state[valid_mask]
    
    if (state[:, 0] == index_manager.false_pred_idx).any(): return [index_manager.false_tensor], next_var_index
    if state.shape[0] == 0: return [index_manager.true_tensor], next_var_index
    
    query, remaining_goals = state[0:1], state[1:]
    
    # --- 1. Unify with Rules ---
    rule_next_states = _unify_with_rules(query, remaining_goals, rules, rule_lengths, index_manager)

    # --- 2. Unify with Facts ---
    fact_next_states = _unify_with_facts_fully_indexed(query, remaining_goals, index_manager)
    
    # --- Check for early successful proof from fact unification ---
    # The unification functions now return a list with a special tensor for this case.
    if fact_next_states and torch.equal(fact_next_states[0], index_manager.true_tensor):
        return [index_manager.true_tensor], next_var_index
        
    next_states_list = rule_next_states + fact_next_states
    
    # --- 3. Process, Canonicalize, and Deduplicate Results ---
    if not next_states_list: return [index_manager.false_tensor], next_var_index
        
    final_states, seen_hashes = [], set()
    
    for s in next_states_list:
        s_valid = s[s[:, 0] != pad]
        if s_valid.numel() == 0:
            # An empty goal set is a successful proof
            return [index_manager.true_tensor], next_var_index
        
        # We still need to canonicalize variables from both rule and fact applications
        s_canonical, next_var_index = canonicalize_variables_in_state_idx(s_valid, index_manager, next_var_index)
        s_hash = _tensor_hash(s_canonical)
        
        if s_hash not in seen_hashes:
            final_states.append(s_canonical)
            seen_hashes.add(s_hash)

    if not final_states: return [index_manager.false_tensor], next_var_index

    # --- 4. Pad results to a uniform shape ---
    original_max_atoms = current_state.shape[0]
    padded_final_states = []
    for s in final_states:
        num_rows, num_cols = s.shape
        if num_rows > original_max_atoms:
            padded_final_states.append(s[:original_max_atoms])
        elif num_rows < original_max_atoms:
            padding = torch.full((original_max_atoms - num_rows, num_cols), pad, dtype=s.dtype, device=device)
            padded_final_states.append(torch.cat([s, padding], dim=0))
        else:
            padded_final_states.append(s)
            
    return padded_final_states, next_var_index



def get_next_unification_pt_with_correct_logic(
    current_state: torch.Tensor,
    facts_tensor: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager: "IndexManager",
    next_var_index: int,
    verbose: int = 0
) -> Tuple[List[torch.Tensor], int]:
    """
    Orchestrates unification by first unifying with rules, and then, for each
    resulting state, unifying its first goal with facts. It then processes the
    remaining atoms by filtering out any known facts.

    This function is modified to match the logic of the original Python version.
    """
    pad = index_manager.padding_idx
    device = current_state.device

    # --- Initial Checks and Setup ---
    valid_mask = current_state[:, 0] != pad
    if not valid_mask.any():
        return [index_manager.true_tensor], next_var_index

    state = current_state[valid_mask]

    if (state[:, 0] == index_manager.false_pred_idx).any():
        return [index_manager.false_tensor], next_var_index
    if state.shape[0] == 0:
        return [index_manager.true_tensor], next_var_index

    query, remaining_goals = state[0:1], state[1:]

    # --- Step 1: Unify with Rules to get intermediate states ---
    intermediate_states = _unify_with_rules(query, remaining_goals, rules, rule_lengths, index_manager)

    if not intermediate_states:
        return [index_manager.false_tensor], next_var_index

    states_after_facts = []

    # --- Step 2: For each intermediate state, unify its new first goal with Facts ---
    for inter_state in intermediate_states:
        inter_state_valid = inter_state[inter_state[:, 0] != pad]

        if inter_state_valid.numel() == 0:
            return [index_manager.true_tensor], next_var_index

        first_goal, rest_of_goals = inter_state_valid[0:1], inter_state_valid[1:]

        # Unify this new goal with the facts
        fact_derived_states = _unify_with_facts_fully_indexed(first_goal, rest_of_goals, index_manager)

        # --- LOGIC CORRECTION: Check for immediate proof if there were no remaining goals ---
        # This now perfectly mirrors the `if not rest_of_goals:` check.
        # If the list of fact-derived states is not empty, it means unification succeeded.
        # If unification succeeded AND there were no goals left, it's a proof.
        if rest_of_goals.numel() == 0 and fact_derived_states:
            # We must ensure the returned state isn't a 'False' state. The helper function
            # returns an empty list for failure, so `if fact_derived_states:` is a valid check.
            if not torch.equal(fact_derived_states[0], index_manager.false_tensor):
                 return [index_manager.true_tensor], next_var_index

        # --- Step 3: Process results of fact unification ---
        for new_state in fact_derived_states:
            # If the new state is the true tensor, we can append the rest of the goals as a proof.
            if torch.equal(new_state, index_manager.true_tensor):
                # return the rest of the goals as a proof.
                new_state = rest_of_goals
            # otherwise, apply the substitutions to the rest of the goals.
            else:
                new_state = apply_substitutions_simple(rest_of_goals, new_state, index_manager)

            new_state_valid = new_state[new_state[:, 0] != pad]

            # Create a mask to identify which atoms in the new state are known facts.
            is_a_fact_mask = (new_state_valid.unsqueeze(1) == facts_tensor.unsqueeze(0)).all(dim=2).any(dim=1)

            # Filter out the atoms that are facts
            filtered_state = new_state_valid[~is_a_fact_mask]
            if filtered_state.numel() == 0:
                # If the filtered state is empty, it means all atoms were facts.
                # This is equivalent to a proof, so we return the true tensor.
                return [index_manager.true_tensor], next_var_index
            states_after_facts.append(filtered_state)

    # --- Step 4: Process, Canonicalize, Deduplicate, and Pad Final Results ---
    if not states_after_facts:
        return [index_manager.false_tensor], next_var_index

    final_states, seen_hashes = [], set()

    for s in states_after_facts:
        if s.numel() == 0:
            return [index_manager.true_tensor], next_var_index

        s_canonical, next_var_index = canonicalize_variables_in_state_idx(s, index_manager, next_var_index)
        s_hash = _tensor_hash(s_canonical)

        if s_hash not in seen_hashes:
            final_states.append(s_canonical)
            seen_hashes.add(s_hash)

    if not final_states:
        return [index_manager.false_tensor], next_var_index

    original_max_atoms = current_state.shape[0]
    padded_final_states = []
    for s in final_states:
        num_rows, num_cols = s.shape
        if num_rows > original_max_atoms:
            padded_final_states.append(s[:original_max_atoms])
        elif num_rows < original_max_atoms:
            padding = torch.full((original_max_atoms - num_rows, num_cols), pad, dtype=s.dtype, device=device)
            padded_final_states.append(torch.cat([s, padding], dim=0))
        else:
            padded_final_states.append(s)

    return padded_final_states, next_var_index
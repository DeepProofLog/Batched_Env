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

def get_next_unification_pt(
    current_state: torch.Tensor,
    facts_tensor: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager: "IndexManager",
    next_var_index: int,
    excluded_fact: Optional[torch.Tensor] = None,
    verbose: int = 0
) -> Tuple[List[torch.Tensor], int]:
    """
    Main unification function, now using a much faster tensor-based hash for deduplication.
    """
    pad = index_manager.padding_idx
    device = current_state.device
    
    valid_mask = current_state[:, 0] != pad
    if not valid_mask.any(): return [index_manager.true_tensor], next_var_index
    
    state = current_state[valid_mask]
    
    if (state[:, 0] == index_manager.false_pred_idx).any(): return [index_manager.false_tensor], next_var_index
    if state.shape[0] == 0: return [index_manager.true_tensor], next_var_index
    
    query, remaining_goals = state[0:1], state[1:]
    query_pred = query[0, 0].item()
    
    next_states_list = []

    # Rule Unification
    relevant_rule_indices = index_manager.rule_index.get(query_pred, [])
    if relevant_rule_indices:
        rule_heads = rules[relevant_rule_indices, 0, :]
        mask, subs = _unify_one_to_one_optimized(query.expand(len(relevant_rule_indices), -1), rule_heads, index_manager)
        success_indices = mask.nonzero(as_tuple=True)[0]
        
        for i in success_indices:
            original_rule_idx = relevant_rule_indices[i.item()]
            body = rules[original_rule_idx, 1:rule_lengths[original_rule_idx].item()]
            instantiated_body = apply_substitutions_simple(body, subs[i], index_manager)
            if remaining_goals.numel() > 0:
                instantiated_rest = apply_substitutions_simple(remaining_goals, subs[i], index_manager)
                next_states_list.append(torch.cat([instantiated_body, instantiated_rest], dim=0))
            else:
                next_states_list.append(instantiated_body)

    # Fact Unification
    fact_mask = facts_tensor[:, 0] == query_pred
    if fact_mask.any():
        relevant_facts = facts_tensor[fact_mask]
        mask, subs = _unify_one_to_one_optimized(query.expand(relevant_facts.shape[0], -1), relevant_facts, index_manager)
        success_indices = mask.nonzero(as_tuple=True)[0]
        
        for i in success_indices:
            if remaining_goals.numel() > 0:
                next_states_list.append(apply_substitutions_simple(remaining_goals, subs[i], index_manager))
            else:
                return [index_manager.true_tensor], next_var_index

    # Process, Canonicalize, and Deduplicate Results
    if not next_states_list: return [index_manager.false_tensor], next_var_index
        
    final_states, seen_hashes = [], set()
    
    for s in next_states_list:
        s_valid = s[s[:, 0] != pad]
        if s_valid.numel() == 0: return [index_manager.true_tensor], next_var_index
        
        s_canonical, next_var_index = canonicalize_variables_in_state_idx(s_valid, index_manager, next_var_index)
        
        # [OPTIMIZATION V2] Replaced slow .tobytes() hash with fast tensor hash
        s_hash = _tensor_hash(s_canonical)
        
        if s_hash not in seen_hashes:
            final_states.append(s_canonical)
            seen_hashes.add(s_hash)

    if not final_states: return [index_manager.false_tensor], next_var_index

    # Pad results
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
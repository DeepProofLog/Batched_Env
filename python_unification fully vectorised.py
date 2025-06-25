# THIS IS A FULLY VECTORISED VERSION, THE PROBLEM IS THAT IN THE FACTS WE DO UNIFICATION WITH WAY TOO MANY FACTS
# IN THE INDEXED FACTS VERSION THIS IS SOLVED

import torch
from typing import List, Dict, Set, Tuple, FrozenSet, Optional, NamedTuple, Any
from utils import Term, Rule
from index_manager import IndexManager

# These NamedTuple classes are great, let's keep them.
class MGUResult(NamedTuple):
    """Holds the results of a vectorized Most General Unifier (MGU) operation."""
    unification_mask: torch.Tensor
    substitutions: torch.Tensor

class RuleUnificationResult(NamedTuple):
    """Holds the results of a vectorized rule unification operation."""
    instantiated_bodies: torch.Tensor
    resolved_substitutions: torch.Tensor
    success_indices: torch.Tensor

# This low-level function is well-optimized for small batches. We will keep it.
def _unify_one_to_one(
    queries: torch.Tensor,
    terms: torch.Tensor,
    im: "IndexManager"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ultra-optimized unification for small, one-to-one batches.
    This function is efficient when the number of queries and terms is small.
    """
    B, A1 = queries.shape
    if B == 0:
        return torch.empty(0, dtype=torch.bool, device=queries.device), torch.empty(0, A1 - 1, 2, dtype=torch.long, device=queries.device)
        
    A = A1 - 1
    pad = im.padding_idx
    var_start = im.constant_no + 1
    device = queries.device

    # 1. Fast predicate check
    pred_match_mask = queries[:, 0] == terms[:, 0]
    if not pred_match_mask.any():
        return torch.zeros(B, dtype=torch.bool, device=device), torch.full((B, A, 2), pad, dtype=torch.long, device=device)

    # 2. Filter to only work on pairs with matching predicates
    matching_indices = pred_match_mask.nonzero(as_tuple=True)[0]
    if matching_indices.numel() == 0:
         return torch.zeros(B, dtype=torch.bool, device=device), torch.full((B, A, 2), pad, dtype=torch.long, device=device)

    q_args = queries[matching_indices, 1:]
    t_args = terms[matching_indices, 1:]
    sub_B = q_args.shape[0]

    # 3. Core unification logic
    q_var_mask = q_args >= var_start
    t_var_mask = t_args >= var_start
    
    both_const = ~(q_var_mask | t_var_mask)
    const_equal = q_args == t_args
    const_match = both_const & const_equal
    var_positions = q_var_mask | t_var_mask
    unifiable_positions = const_match | var_positions
    term_unifiable = unifiable_positions.all(dim=1)

    # This conflict check is a good optimization
    if A == 2:
        q_same = (q_args[:, 0] == q_args[:, 1]) & (q_args[:, 0] >= var_start)
        t_diff = (t_args[:, 0] != t_args[:, 1]) & (t_args[:, 0] < var_start) & (t_args[:, 1] < var_start)
        conflict1 = q_same & t_diff
        
        t_same = (t_args[:, 0] == t_args[:, 1]) & (t_args[:, 0] >= var_start)
        q_diff = (q_args[:, 0] != q_args[:, 1]) & (q_args[:, 0] < var_start) & (q_args[:, 1] < var_start)
        conflict2 = t_same & q_diff
        
        has_conflict = conflict1 | conflict2
    else:
        # General conflict check for any arity (more expensive, so only use if needed)
        # Find variables that are mapped to two different constants
        sub_vars = torch.where(q_var_mask, q_args, t_args)
        sub_vals = torch.where(q_var_mask, t_args, q_args)
        is_var_to_const = (sub_vars >= var_start) & (sub_vals < var_start)

        # A simple iterative check can be faster on CPU for small batches than a full vectorized solution
        # This part can be further optimized if it becomes a bottleneck again
        has_conflict = torch.zeros(sub_B, dtype=torch.bool, device=device)
        for i in range(sub_B):
            if not term_unifiable[i]: continue
            
            sample_v_to_c = is_var_to_const[i]
            if not sample_v_to_c.any(): continue
            
            vars_ = sub_vars[i, sample_v_to_c]
            vals_ = sub_vals[i, sample_v_to_c]
            
            unique_vars, inverse_indices = torch.unique(vars_, return_inverse=True)
            # Check if any unique variable maps to more than one unique constant
            for j in range(unique_vars.shape[0]):
                mapped_vals = vals_[inverse_indices == j]
                if torch.unique(mapped_vals).shape[0] > 1:
                    has_conflict[i] = True
                    break
            if has_conflict[i]:
                break

    # 6. Final success determination & substitution generation
    success_mask = term_unifiable & ~has_conflict
    
    final_subs = torch.full((sub_B, A, 2), pad, dtype=torch.long, device=device)
    if success_mask.any():
        sub_vars = torch.where(q_var_mask, q_args, torch.where(t_var_mask, t_args, pad))
        sub_vals = torch.where(q_var_mask, t_args, torch.where(t_var_mask, q_args, pad))
        substitutions = torch.stack((sub_vars, sub_vals), dim=-1)
        final_subs = torch.where(success_mask.unsqueeze(1).unsqueeze(2), substitutions, pad)

    # 8. Map results back to the original input size
    full_mask = torch.zeros(B, dtype=torch.bool, device=device)
    full_subs = torch.full((B, A, 2), pad, dtype=torch.long, device=device)
    full_mask[matching_indices] = success_mask
    full_subs[matching_indices] = final_subs
    return full_mask, full_subs

# This function is now used on SMALL, PRE-FILTERED batches.
def unify_with_facts(
    queries: torch.Tensor, # Shape (Q, A1) where Q is small
    facts: torch.Tensor,   # Shape (F, A1) where F is small
    im: IndexManager
) -> MGUResult:
    """
    Vectorized fact unification, intended for use with a small, pre-filtered set of facts.
    """
    Q, A1 = queries.shape
    F = facts.shape[0]
    if Q == 0 or F == 0:
        return MGUResult(torch.empty(Q, F, dtype=torch.bool), torch.empty(Q, F, A1-1, 2, dtype=torch.long))
    
    device = queries.device
    pad = im.padding_idx
    
    # Expand queries and facts to create all pairs for the small batch
    expanded_queries = queries.unsqueeze(1).expand(Q, F, A1)
    expanded_facts = facts.unsqueeze(0).expand(Q, F, A1)
    
    # Flatten for one-to-one unification
    flat_queries = expanded_queries.reshape(Q * F, A1)
    flat_facts = expanded_facts.reshape(Q * F, A1)
    
    unify_mask, unify_subs = _unify_one_to_one(flat_queries, flat_facts, im)
    
    # Reshape back to (Q, F)
    result_mask = unify_mask.view(Q, F)
    result_subs = unify_subs.view(Q, F, A1-1, 2)
    
    return MGUResult(result_mask, result_subs)

# This function is now used on SMALL, PRE-FILTERED batches.
def unify_with_rules(
    queries: torch.Tensor,      # Shape (Q, A1) where Q is small
    rules: torch.Tensor,        # Shape (R, B+1, A1) where R is small
    rule_lengths: torch.Tensor, # Shape (R,)
    im: IndexManager
) -> RuleUnificationResult:
    """ 
    Vectorized rule unification, intended for use with a small, pre-filtered set of rules.
    """
    Q, A1 = queries.shape
    A = A1 - 1
    R, B_plus_1, _ = rules.shape
    B_max = B_plus_1 - 1
    device = queries.device
    pad = im.padding_idx

    if Q == 0 or R == 0:
        return RuleUnificationResult(torch.empty(0,0,A1,dtype=torch.long), torch.empty(0,A,2,dtype=torch.long), torch.empty(0,2,dtype=torch.long))
    
    rule_heads = rules[:, 0, :]
    
    # Unify queries with all rule heads in the small batch
    head_mask, head_subs = unify_with_facts(queries, rule_heads, im) # Re-use fact unification logic
    
    q_indices, r_indices = head_mask.nonzero(as_tuple=True)
    if q_indices.numel() == 0:
        return RuleUnificationResult(torch.empty(0,0,A1,dtype=torch.long), torch.empty(0,A,2,dtype=torch.long), torch.empty(0,2,dtype=torch.long))

    success_indices = torch.stack((q_indices, r_indices), dim=1)
    S = success_indices.shape[0]
    
    # Gather data for successful unifications
    sel_subs = head_subs[q_indices, r_indices]
    sel_bodies = rules[r_indices, 1:, :]
    sel_body_lens = rule_lengths[r_indices] - 1
    
    # Apply substitutions. This is still a bottleneck, but much less so on small tensors.
    # On CPU, a Python loop might even be faster here, but we'll keep the torch version for now.
    max_idx = im.runtime_var_end_index + 1
    # FIX: .clone() is added to satisfy the deprecation warning.
    maps = torch.arange(max_idx, device=device).unsqueeze(0).expand(S, -1).clone()
    
    valid_subs_mask = sel_subs[:, :, 0] != pad
    if valid_subs_mask.any():
        var_indices = sel_subs[valid_subs_mask]
        batch_idx = torch.arange(S, device=device).unsqueeze(1).expand(-1, A)[valid_subs_mask]
        maps[batch_idx, var_indices[:, 0]] = var_indices[:, 1]
        
        for _ in range(3): # Fixed-point iteration
             maps = torch.gather(maps, 1, maps)
    
    expanded_maps = maps.unsqueeze(1).expand(S, B_max, max_idx)
    safe_indices = torch.clamp(sel_bodies, 0, max_idx - 1)
    inst_bodies = torch.gather(expanded_maps, 2, safe_indices)
    
    body_mask = torch.arange(B_max, device=device).unsqueeze(0) < sel_body_lens.unsqueeze(1)
    inst_bodies_masked = torch.where(body_mask.unsqueeze(-1), inst_bodies, pad)

    final_subs = torch.where(valid_subs_mask.unsqueeze(-1), sel_subs, pad)

    return RuleUnificationResult(inst_bodies_masked, final_subs, success_indices)

# This substitution function is much simpler and more CPU-friendly.
def apply_substitutions_to_goals(
    goals: torch.Tensor, # Shape (G, A1)
    substitutions: torch.Tensor, # Shape (Sub, 2)
    im: IndexManager
) -> torch.Tensor:
    """
    Applies a single set of substitutions to a list of goals.
    This is a more CPU-friendly version for the hybrid model.
    """
    if goals.numel() == 0 or substitutions.numel() == 0:
        return goals

    pad = im.padding_idx
    device = goals.device
    
    # Filter out padding from substitutions
    valid_subs_mask = substitutions[:, 0] != pad
    subs = substitutions[valid_subs_mask]
    if subs.numel() == 0:
        return goals

    # Create a map for this single substitution set
    max_idx = im.runtime_var_end_index + 1
    sub_map = torch.arange(max_idx, device=device)
    sub_map[subs[:, 0]] = subs[:, 1]
    
    # Resolve chains (X->Y, Y->c becomes X->c)
    for _ in range(3): # A few iterations are enough for typical depths
        sub_map = sub_map[sub_map]

    # Apply the map to the goals
    # We only apply substitutions to variables
    var_start = im.constant_no + 1
    is_var = goals >= var_start
    
    # Use the map to substitute variables
    substituted_goals = sub_map[goals]
    
    # Combine original constants with substituted variables
    result = torch.where(is_var, substituted_goals, goals)
    return result

def canonicalize_variables_in_state_idx(
    state_tensor: torch.Tensor,
    index_manager: "IndexManager",
    next_var_index: int
) -> Tuple[torch.Tensor, int]:
    """Optimized variable canonicalization with early exits."""
    if state_tensor.numel() == 0: 
        return state_tensor, next_var_index
    
    var_start_index = index_manager.constant_no + 1
    is_var_mask = state_tensor >= var_start_index
    
    if not is_var_mask.any():
        return state_tensor, next_var_index
    
    unique_vars, inverse_indices = torch.unique(state_tensor[is_var_mask], return_inverse=True)
    num_unique_vars = unique_vars.shape[0]

    if num_unique_vars == 0: 
        return state_tensor, next_var_index

    if next_var_index + num_unique_vars > index_manager.runtime_var_end_index:
        raise ValueError("No more available variable indices.")

    new_indices = torch.arange(next_var_index, next_var_index + num_unique_vars, device=state_tensor.device)
    new_state = state_tensor.clone()
    new_state[is_var_mask] = new_indices[inverse_indices]
    
    next_var_index += num_unique_vars
    return new_state, next_var_index

# **** NEW HYBRID ORCHESTRATION FUNCTION ****
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
    Orchestrates the unification process using a hybrid CPU-first approach.
    It uses Python control flow to prune the search space and calls vectorized
    functions on small, pre-filtered batches of data.
    """
    pad = index_manager.padding_idx
    device = current_state.device
    true_idx = index_manager.true_pred_idx
    false_idx = index_manager.false_pred_idx

    # 1. State preprocessing (same as before)
    valid_mask = current_state[:, 0] != pad
    # FIX: Return correct format List[Tensor] instead of List[List[Tensor]]
    if not valid_mask.any(): return [index_manager.true_tensor], next_var_index
    state = current_state[valid_mask]
    if (state[:, 0] == false_idx).any(): return [index_manager.false_tensor], next_var_index
    state = state[state[:, 0] != true_idx]
    if state.shape[0] == 0: return [index_manager.true_tensor], next_var_index

    # 2. Goal selection (Python handles the high-level loop)
    query = state[0:1] # Keep as 2D tensor (1, A1)
    remaining_goals = state[1:]

    intermediate_states = []

    # 3. Rule Unification (Hybrid Approach)
    query_pred = query[0, 0].item()
    
    # Use Python dict index to get relevant rule indices
    relevant_rule_indices = index_manager.rule_index.get(query_pred, [])
    
    if relevant_rule_indices:
        # Create a SMALL tensor of only relevant rules
        relevant_rules_tensor = rules[relevant_rule_indices]
        relevant_lengths_tensor = rule_lengths[relevant_rule_indices]
        
        # Call vectorized function on the small batch
        rule_result = unify_with_rules(query, relevant_rules_tensor, relevant_lengths_tensor, index_manager)

        if rule_result.success_indices.numel() > 0:
            # This loop is in Python, over a small number of successes
            for i in range(rule_result.success_indices.shape[0]):
                subs = rule_result.resolved_substitutions[i]
                body = rule_result.instantiated_bodies[i]
                
                # Filter padding from body
                body = body[body[:, 0] != pad]

                # Apply subs to the original remaining goals
                substituted_remaining = apply_substitutions_to_goals(remaining_goals, subs, index_manager)
                
                # Combine to form a new intermediate state
                if body.numel() > 0:
                    new_state = torch.cat([body, substituted_remaining], dim=0)
                else:
                    new_state = substituted_remaining
                intermediate_states.append(new_state)

    # 4. Fact Unification (applied to each intermediate state)
    final_states = []
    
    # If no rules applied, the query itself is the first goal to unify with facts
    if not intermediate_states:
        intermediate_states.append(state)

    for inter_state in intermediate_states:
        if inter_state.shape[0] == 0:
            final_states.append(index_manager.true_tensor)
            continue

        first_goal = inter_state[0:1]
        rest_of_goals = inter_state[1:]
        
        # Use Python dict index to get relevant fact indices
        first_goal_pred = first_goal[0, 0].item()
        # This assumes a fact index exists. If not, it needs to be built.
        # For now, we fall back to filtering the full tensor, which is slow but works.
        if hasattr(index_manager, 'fact_index') and first_goal_pred in index_manager.fact_index:
             cand_indices = index_manager.fact_index[first_goal_pred]
             candidate_facts = facts_tensor[cand_indices]
        else: # Fallback: filter all facts by predicate
             candidate_facts = facts_tensor[facts_tensor[:, 0] == first_goal_pred]

        if candidate_facts.shape[0] > 0:
            fact_result = unify_with_facts(first_goal, candidate_facts, index_manager)
            success_mask = fact_result.unification_mask.squeeze(0)

            if success_mask.any():
                successful_fact_indices = success_mask.nonzero().squeeze(-1)
                for idx in successful_fact_indices:
                    subs = fact_result.substitutions[0, idx]
                    new_final_state = apply_substitutions_to_goals(rest_of_goals, subs, index_manager)
                    final_states.append(new_final_state)

    # FIX: Return correct format List[Tensor] instead of List[List[Tensor]]
    if not final_states:
        return [index_manager.false_tensor], next_var_index
    
    # 5. Deduplication and Canonicalization
    # A robust way to deduplicate tensors is to convert them to a hashable format
    seen_states = set()
    deduplicated_states = []
    for s in final_states:
        # Remove padding before creating the tuple for hashing
        s_valid = s[s[:, 0] != pad]
        # FIX: Return correct format List[Tensor] instead of List[List[Tensor]]
        if s_valid.numel() == 0: # An empty state means success
            return [index_manager.true_tensor], next_var_index
        
        state_tuple = tuple(s_valid.flatten().tolist())
        if state_tuple not in seen_states:
            deduplicated_states.append(s_valid)
            seen_states.add(state_tuple)

    if not deduplicated_states:
         return [index_manager.false_tensor], next_var_index

    # Canonicalize variables for each new state
    canonical_states = []
    for s in deduplicated_states:
        canonical_s, next_var_index = canonicalize_variables_in_state_idx(s, index_manager, next_var_index)
        canonical_states.append(canonical_s)

    return canonical_states, next_var_index

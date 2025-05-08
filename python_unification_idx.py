import torch
from typing import List, Dict, Set, Tuple, FrozenSet, Optional, NamedTuple, Any
from utils import Term, Rule
from index_manager_idx import IndexManager, facts_to_tensor_im, rules_to_tensor_im, state_to_tensor_im, \
                        debug_print_atom, debug_print_state_from_indices, debug_print_states_from_indices


def format_substitutions_dict(subs: Dict[int, int], index_manager: IndexManager) -> Dict[str, str]:
    if not subs : return {}
    if subs.get('ground_match') is True : 
        return {"ground_match": "True"}
    readable_subs = {}
    for var_idx, val_idx in subs.items():
        var_str = index_manager.get_str_for_term_idx(var_idx) 
        val_str = index_manager.get_str_for_term_idx(val_idx) 
        readable_subs[var_str] = val_str
    return readable_subs

def is_variable_idx(idx: int, index_manager: IndexManager) -> bool: 
    return index_manager.is_var_idx(idx)

def resolve_idx(idx: int, substitutions: Dict[int, int], index_manager: IndexManager) -> int: 
    seen = {idx}
    while is_variable_idx(idx, index_manager) and idx in substitutions: 
        idx_before = idx
        idx = substitutions[idx]
        if idx == idx_before: 
            break
        if idx in seen: 
            return idx 
    return idx

def unify_terms_idx(term1_idx: torch.Tensor, 
                    term2_idx: torch.Tensor, 
                    index_manager: IndexManager, 
                    initial_substitutions: Optional[Dict[int, int]] = None
                   ) -> Optional[Dict[int, int]]:
    if term1_idx[0].item() != term2_idx[0].item(): 
        return None
    subs = initial_substitutions.copy() if initial_substitutions else {}
    for i in range(1, index_manager.max_arity + 1): 
        arg1 = term1_idx[i].item()
        arg2 = term2_idx[i].item()
        r_arg1 = resolve_idx(arg1, subs, index_manager)
        r_arg2 = resolve_idx(arg2, subs, index_manager) 
        if r_arg1 == r_arg2: 
            if r_arg1 == index_manager.padding_idx and arg1 != arg2 : 
                 pass 
            elif r_arg1 == index_manager.padding_idx and arg1 == index_manager.padding_idx and arg2 == index_manager.padding_idx:
                 pass 
            else: 
                continue
        is_r_arg1_var = is_variable_idx(r_arg1, index_manager)
        is_r_arg2_var = is_variable_idx(r_arg2, index_manager)
        if is_r_arg1_var and is_r_arg2_var: 
            print(f"WARNING: Both args are variables: {r_arg1}, {r_arg2}")
            if index_manager.is_fixed_var_idx(r_arg1) and index_manager.is_dynamic_var_idx(r_arg2):
                 subs[r_arg2] = r_arg1
            elif index_manager.is_fixed_var_idx(r_arg2) and index_manager.is_dynamic_var_idx(r_arg1):
                 subs[r_arg1] = r_arg2
            else: 
                 subs[max(r_arg1, r_arg2)] = min(r_arg1, r_arg2) 
        elif is_r_arg1_var: 
            subs[r_arg1] = r_arg2
        elif is_r_arg2_var: 
            subs[r_arg2] = r_arg1
        elif r_arg1 != r_arg2: 
            return None       
    return subs

def apply_substitutions_to_term_idx(term_idx: torch.Tensor, substitutions: Dict[int, int], index_manager: IndexManager) -> torch.Tensor:
    new_term_list = [term_idx[0].item()] 
    for i in range(1, index_manager.max_arity + 1): 
        arg = term_idx[i].item()
        new_term_list.append(resolve_idx(arg, substitutions, index_manager)) 
    return torch.tensor(new_term_list, dtype=torch.long, device=term_idx.device)

def apply_substitutions_to_state_idx(state_idx: torch.Tensor, substitutions: Dict[int, int], index_manager: IndexManager) -> torch.Tensor:
    if state_idx.numel() == 0: 
        return state_idx
    new_atoms = [apply_substitutions_to_term_idx(state_idx[i], substitutions, index_manager) for i in range(state_idx.shape[0])] 
    if not new_atoms: 
        return torch.empty((0, index_manager.max_arity + 1), dtype=torch.long, device=state_idx.device)
    return torch.stack(new_atoms)

def standardize_apart_rule_idx(
    rule_tensor_single: torch.Tensor, 
    rule_length: int,                 
    index_manager: IndexManager 
) -> torch.Tensor:
    if rule_length == 0: 
        return rule_tensor_single 
    local_original_var_to_fresh_dynamic_map: Dict[int, int] = {}
    new_rule_atoms_list = []
    for i in range(rule_length): 
        atom = rule_tensor_single[i]
        new_atom_elements = [atom[0].item()] 
        for k in range(1, index_manager.max_arity + 1): 
            original_arg_idx = atom[k].item()
            if original_arg_idx == index_manager.padding_idx:
                new_atom_elements.append(index_manager.padding_idx)
                continue
            if is_variable_idx(original_arg_idx, index_manager): 
                if original_arg_idx not in local_original_var_to_fresh_dynamic_map:
                    local_original_var_to_fresh_dynamic_map[original_arg_idx] = index_manager.get_next_var() 
                new_atom_elements.append(local_original_var_to_fresh_dynamic_map[original_arg_idx])
            else: 
                new_atom_elements.append(original_arg_idx)
        new_rule_atoms_list.append(torch.tensor(new_atom_elements, dtype=torch.long, device=index_manager.device))
    
    if not new_rule_atoms_list: 
        standardized_part = torch.empty((0, index_manager.max_arity+1), dtype=torch.long, device=index_manager.device)
    else:
        standardized_part = torch.stack(new_rule_atoms_list) 
    if rule_length < rule_tensor_single.shape[0]:
        padding = rule_tensor_single[rule_length:] 
        if padding.device != index_manager.device: 
            padding = padding.to(index_manager.device)
        return torch.cat([standardized_part, padding], dim=0)
    return standardized_part

def unify_with_facts_idx(
    query_idx: torch.Tensor, 
    facts_tensor: torch.Tensor, 
    index_manager: IndexManager, 
    facts_as_set: Optional[FrozenSet[Tuple[int, int, int]]] = None, 
    excluded_fact_idx: Optional[torch.Tensor] = None, 
    verbose: int = 0 
) -> List[Dict[int, int]]:
    substitutions_found = []
    is_ground_query = True
    for i in range(1, index_manager.max_arity + 1):
        arg_val = query_idx[i].item()
        if arg_val != index_manager.padding_idx and is_variable_idx(arg_val, index_manager):
            is_ground_query = False
            break
    if is_ground_query:
        if facts_as_set is not None: 
            query_tuple = tuple(query_idx.tolist()) 
            if query_tuple in facts_as_set:
                if excluded_fact_idx is None or not torch.equal(query_idx, excluded_fact_idx):
                    if verbose > 1: print(f"    DEBUG (unify_with_facts): Ground query {debug_print_atom(query_idx, index_manager)} matched in facts_as_set.")
                    substitutions_found.append({'ground_match': True}) 
            return substitutions_found
    for i in range(facts_tensor.shape[0]):
        fact_idx = facts_tensor[i]
        if excluded_fact_idx is not None and torch.equal(fact_idx, excluded_fact_idx):
            continue
        subs = unify_terms_idx(fact_idx, query_idx, index_manager) 
        if subs is not None:
            if verbose > 1: print(f"    DEBUG (unify_with_facts): Query {debug_print_atom(query_idx, index_manager)} unified with fact {debug_print_atom(fact_idx, index_manager)} -> subs {format_substitutions_dict(subs, index_manager)}")
            substitutions_found.append(subs)
    return substitutions_found

def unify_with_rules_idx(
    query_idx: torch.Tensor, 
    rules_tensor: torch.Tensor, 
    rule_lengths_tensor: torch.Tensor, 
    index_manager: IndexManager, 
    original_rules_list: List[Rule], 
    verbose: int = 0
) -> List[Tuple[torch.Tensor, Dict[int, int]]]: 
    results = []
    for i in range(rules_tensor.shape[0]):
        rule_full_tensor = rules_tensor[i] 
        rule_len = rule_lengths_tensor[i].item()
        if rule_len == 0: continue 
        standardized_rule_tensor = standardize_apart_rule_idx(rule_full_tensor, rule_len, index_manager)
        rule_head_idx = standardized_rule_tensor[0] 
        subs = unify_terms_idx(query_idx, rule_head_idx, index_manager) 
        if subs is not None:
            body_template_idx = standardized_rule_tensor[1:rule_len] 
            if body_template_idx.numel() > 0:
                instantiated_body = apply_substitutions_to_state_idx(body_template_idx, subs, index_manager) 
            else: 
                instantiated_body = torch.empty((0, index_manager.max_arity + 1), dtype=torch.long, device=index_manager.device)
            if verbose >= 1:
                original_rule_obj = original_rules_list[i]
                original_rule_head_str = f"{original_rule_obj.head.predicate}{original_rule_obj.head.args}"
                original_rule_body_str_list = [f"{t.predicate}{t.args}" for t in original_rule_obj.body]
                original_rule_body_str = ", ".join(original_rule_body_str_list) if original_rule_body_str_list else "{}"
                print(f"    Rule {i}: {original_rule_head_str} :- {original_rule_body_str}")
                print(f"      Query: {debug_print_atom(query_idx, index_manager)}")
                print(f"      Standardized Rule Head: {debug_print_atom(rule_head_idx, index_manager)}")
                print(f"      Subs (vars in query/std_rule_head): {format_substitutions_dict(subs, index_manager)}")
                print(f"      Instantiated Body (after subs): {debug_print_state_from_indices(instantiated_body, index_manager, oneline=True)}")
            results.append((instantiated_body, subs))
        else:
            if verbose >= 1: 
                print(f"    Rule head {str(original_rules_list[i])} did not unify with query {debug_print_atom(query_idx, index_manager)}. Rule idx:{rule_head_idx}, Query idx: {query_idx}")
    return results

def get_next_unification_pt(
    current_state_idx: torch.Tensor, 
    facts_tensor: torch.Tensor,      
    facts_as_set: FrozenSet[Tuple[int, int, int]], 
    rules_tensor: torch.Tensor,      
    rule_lengths_tensor: torch.Tensor, 
    index_manager: IndexManager, 
    original_rules_list: List[Rule], 
    excluded_fact_idx: Optional[torch.Tensor] = None,
    include_intermediate_rule_states: bool = False, # New parameter
    verbose: int = 0
) -> Tuple[List[torch.Tensor], Any]: 
    
    if current_state_idx.device != index_manager.device:
        current_state_idx = current_state_idx.to(index_manager.device)

    if verbose >= 1: print(f"\n++++++++++++++ Input State: {debug_print_state_from_indices(current_state_idx, index_manager, oneline=True)} ++++++++++++++")

    if current_state_idx.numel() > 0:
        for atom_idx_val in range(current_state_idx.shape[0]):
            if current_state_idx[atom_idx_val, 0].item() == index_manager.false_pred_idx:
                if verbose >= 1: print(f"State contains FALSE, returning [[FALSE]]")
                return [index_manager.false_tensor.unsqueeze(0)], None 
    if current_state_idx.numel() > 0:
        true_rows = current_state_idx[:, 0] == index_manager.true_pred_idx
        current_state_idx_filtered = current_state_idx[~true_rows]
    else:
        current_state_idx_filtered = current_state_idx
    if current_state_idx_filtered.shape[0] == 0: 
        if verbose >= 1: print("Current state empty after filtering TRUEs, returning [[TRUE]]")
        return [index_manager.true_tensor.unsqueeze(0)], None 

    query_atom_idx = current_state_idx_filtered[0]      
    remaining_goals_idx = current_state_idx_filtered[1:] 

    if verbose >= 1:
        print(f"Processing Query Atom: {debug_print_atom(query_atom_idx, index_manager)}")
        print(f"Rest of Current Goal List: {debug_print_state_from_indices(remaining_goals_idx, index_manager, oneline=True) if remaining_goals_idx.numel() > 0 else '[]'}")

    all_resolvents: List[torch.Tensor] = []
    states_derived_from_rules: List[torch.Tensor] = [] # To store states after rule unification

    # --- Step 1: Unify query_atom_idx with RULE HEADS ---
    if verbose >= 1: print(f"\n  --- Step 1: Rule Unification for Query Atom ---")
    rule_unification_results = unify_with_rules_idx(
        query_atom_idx, rules_tensor, rule_lengths_tensor, 
        index_manager, original_rules_list, verbose=verbose
    )
    if verbose >=1 and not rule_unification_results: print("    No rule unifications found for the query atom.")
    
    for instantiated_body, rule_subs in rule_unification_results:
        if verbose >= 1: print(f"    Rule Subs on Query Atom: {format_substitutions_dict(rule_subs, index_manager)}")
        substituted_remaining_goals = apply_substitutions_to_state_idx(remaining_goals_idx, rule_subs, index_manager)
        
        # Form the new state from rule application
        state_from_rule: torch.Tensor
        if instantiated_body.numel() > 0 and substituted_remaining_goals.numel() > 0:
            state_from_rule = torch.cat([instantiated_body, substituted_remaining_goals], dim=0)
        elif instantiated_body.numel() > 0:
            state_from_rule = instantiated_body
        elif substituted_remaining_goals.numel() > 0:
            state_from_rule = substituted_remaining_goals
        else: # Both empty, means rule led to empty body and original remaining goals were also empty or resolved
            state_from_rule = torch.empty((0, index_manager.max_arity + 1), dtype=torch.long, device=index_manager.device)

        # Clean TRUEs from this state_from_rule
        if state_from_rule.numel() > 0:
             true_rows_sfr = state_from_rule[:, 0] == index_manager.true_pred_idx
             state_from_rule_cleaned = state_from_rule[~true_rows_sfr]
        else:
             state_from_rule_cleaned = state_from_rule
        
        if state_from_rule_cleaned.shape[0] == 0: # Rule application directly led to TRUE
            if verbose >= 1: print(f"      Rule path directly led to TRUE.")
            # Add TRUE to all_resolvents to ensure it's a possible final state
            # And skip adding to states_derived_from_rules as it's already solved
            all_resolvents.append(index_manager.true_tensor.unsqueeze(0))
        else:
            if verbose >= 1: print(f"      State from rule (cleaned): {debug_print_state_from_indices(state_from_rule_cleaned, index_manager, oneline=True)}")
            states_derived_from_rules.append(state_from_rule_cleaned)


    # --- Step 2: Optionally include these intermediate states_derived_from_rules ---
    if include_intermediate_rule_states:
        if verbose >= 1 and states_derived_from_rules: print(f"\n  --- Including Intermediate Rule States ---")
        for sfr_cleaned in states_derived_from_rules:
            # No need to re-check for empty here as it was handled when sfr_cleaned was created
            all_resolvents.append(sfr_cleaned)
            if verbose >=1: print(f"    Added intermediate: {debug_print_state_from_indices(sfr_cleaned, index_manager, oneline=True)}")


    # --- Step 3: For each state in states_derived_from_rules, unify its first goal with FACTS ---
    if verbose >= 1 and states_derived_from_rules: print(f"\n  --- Step 3: Fact Unification on First Goal of Rule-Derived States ---")
    
    for state_from_rule_item in states_derived_from_rules: # These are already cleaned of TRUEs
        if verbose >= 1: print(f"    Processing rule-derived state for fact unification: {debug_print_state_from_indices(state_from_rule_item, index_manager, oneline=True)}")
        
        # If state_from_rule_item is already empty, it means it resolved to TRUE.
        # This case should have been caught when states_derived_from_rules was populated.
        # However, as a safeguard:
        if state_from_rule_item.shape[0] == 0:
            # This should ideally not be reached if logic above is correct,
            # as empty cleaned states are added as TRUE to all_resolvents directly.
            # If it does, means it's a TRUE state.
            # all_resolvents.append(index_manager.true_tensor.unsqueeze(0))
            continue # Already handled

        first_goal_of_sfr = state_from_rule_item[0]
        rest_of_sfr_goals = state_from_rule_item[1:]

        fact_unif_substitutions_sfr = unify_with_facts_idx(
            first_goal_of_sfr, facts_tensor, index_manager, facts_as_set, excluded_fact_idx, verbose=verbose
        )
        if verbose >=1 and not fact_unif_substitutions_sfr: print(f"      No fact unifications found for: {debug_print_atom(first_goal_of_sfr, index_manager)}")

        for fact_subs_sfr in fact_unif_substitutions_sfr:
            if verbose >= 1: print(f"      Fact Subs on '{debug_print_atom(first_goal_of_sfr, index_manager)}': {format_substitutions_dict(fact_subs_sfr, index_manager)}")
            
            final_next_state: torch.Tensor
            if fact_subs_sfr.get('ground_match'): 
                final_next_state = rest_of_sfr_goals
            else:
                final_next_state = apply_substitutions_to_state_idx(rest_of_sfr_goals, fact_subs_sfr, index_manager)
            
            if final_next_state.numel() > 0:
                true_rows_final = final_next_state[:, 0] == index_manager.true_pred_idx
                final_next_state = final_next_state[~true_rows_final]
            
            if final_next_state.shape[0] == 0: 
                if verbose >= 1: print(f"        Resulting state resolved to TRUE.")
                all_resolvents.append(index_manager.true_tensor.unsqueeze(0))
            else:
                if verbose >= 1: print(f"        New state after fact application: {debug_print_state_from_indices(final_next_state, index_manager, oneline=True)}")
                all_resolvents.append(final_next_state)
    
    # --- Finalize and Deduplicate ---
    if not all_resolvents:
        # This happens if rule unification yielded no states, or if subsequent fact unification on those also yielded nothing.
        if verbose >= 1: print("\n  No further states derived after all steps. Returning [[FALSE]].")
        return [index_manager.false_tensor.unsqueeze(0)], None 

    unique_states_list = []
    seen_state_tuples = set()
    for s_tensor in all_resolvents:
        s_tuple = tuple(tuple(atom.tolist()) for atom in s_tensor)
        if s_tuple not in seen_state_tuples:
            unique_states_list.append(s_tensor)
            seen_state_tuples.add(s_tuple)
    
    if verbose >= 1:
        print(f"\nFinal Next Potential States (Unique: {len(unique_states_list)}):")
        # for s_idx, s_tensor in enumerate(unique_states_list):
        #     print(f"  - {debug_print_state_from_indices(s_tensor, index_manager, oneline=True)}")
        print(debug_print_states_from_indices(unique_states_list, index_manager)) # Use the new helper
        print(f"++++++++++++++ End Tensor Unification Step ++++++++++++++\n")

    return unique_states_list, None 




if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_constants_main = {"john", "mary", "peter", "chocolate", "wine", "a", "b", "c", "d"}
    all_predicates_main = {"parent", "loves", "grandparent", "ancestor", "p", "q"} 
    max_fixed_vars_main = 2 
    rules_str_main = [
        Rule(Term("grandparent", ("X", "Z")), 
             [Term("parent", ("X", "Y")), Term("parent", ("Y", "Z"))]),
        Rule(Term("ancestor", ("A", "B")), 
             [Term("parent", ("A", "B"))]),
        Rule(Term("ancestor", ("C", "E")),
             [Term("parent", ("C", "D")), Term("ancestor", ("D", "E"))]),
        Rule(Term("q", ("U", "W")), 
             [Term("p", ("U", "V")), Term("parent", ("V", "W"))])
    ]
    im_main = IndexManager(constants=all_constants_main,
                           predicates=all_predicates_main,
                           max_total_vars=max_fixed_vars_main,
                           rules=rules_str_main, 
                           max_arity=2, 
                           device=DEVICE)
    facts_str_main = [
        Term("parent", ("john", "mary")),
        Term("parent", ("mary", "peter")),
        Term("loves", ("john", "chocolate")),
        Term("loves", ("mary", "wine")),
        Term("p", ("a", "b")), 
        Term("parent",("b","c")) 
    ]
    facts_tensor_main = facts_to_tensor_im(facts_str_main, im_main)
    facts_as_set_main = frozenset(tuple(f.tolist()) for f in facts_tensor_main) 
    max_atoms_main = 0
    if rules_str_main:
        max_atoms_main = max(1 + len(r.body) for r in rules_str_main) if rules_str_main else 1
    rules_tensor_main, rule_lengths_tensor_main = rules_to_tensor_im(rules_str_main, max_atoms_main, im_main)
    print("--- IndexManager Initialized ---")
    print(f"True Idx: {im_main.true_pred_idx}, False Idx: {im_main.false_pred_idx}, Pad Idx: {im_main.padding_idx}")
    print("\n--- Facts (Tensor Debug) ---")
    print(debug_print_state_from_indices(facts_tensor_main, im_main))
    print("\n--- Rules (Tensor Debug) ---")
    for i in range(rules_tensor_main.shape[0]):
        print(f"Rule {i} (len {rule_lengths_tensor_main[i].item()}): {debug_print_state_from_indices(rules_tensor_main[i,:rule_lengths_tensor_main[i].item()], im_main, oneline=True)}")

    print("\n\n--- Test 1: Query grandparent(john, Who) ---")
    initial_query_vars_gp = {} 
    im_main.next_dynamic_var_idx_counter = im_main.dynamic_variable_start_index 
    initial_state_str_gp = [Term("grandparent", ("john", "Who"))]
    initial_state_tensor_gp = state_to_tensor_im(initial_state_str_gp, im_main, initial_query_vars_gp)
    
    current_proof_state = initial_state_tensor_gp
    for step_num in range(1, 4): # Max 3 steps for this test
        print(f"\n--- GP Step {step_num} ---")
        if current_proof_state is None or current_proof_state.numel() == 0 or \
           (current_proof_state.shape[0] == 1 and current_proof_state[0,0].item() == im_main.true_pred_idx) or \
           (current_proof_state.shape[0] == 1 and current_proof_state[0,0].item() == im_main.false_pred_idx):
            print(f"Proof terminated or reached single TRUE/FALSE before step {step_num}.")
            break

        next_possible_states, _ = get_next_unification_pt(
            current_proof_state, 
            facts_tensor_main,    
            facts_as_set_main,    
            rules_tensor_main,    
            rule_lengths_tensor_main, 
            im_main,
            original_rules_list=rules_str_main, 
            verbose=1,
            include_intermediate_rule_states=False # Test with default
        )
        if not next_possible_states:
            print("No next states derived.")
            current_proof_state = None # End proof
            break
        
        # In a real scenario, an agent would pick one. For testing, we take the first.
        current_proof_state = next_possible_states[0] 
        print(f"  Selected next state for GP Step {step_num+1} (if any): {debug_print_state_from_indices(current_proof_state, im_main, oneline=True)}")

    if current_proof_state is not None and current_proof_state.shape[0] == 1 and current_proof_state[0,0].item() == im_main.true_pred_idx:
        print("\nGRANDPARENT TEST SUCCEEDED TO PROVE TRUE!")
    else:
        print(f"\nGRANDPARENT TEST FINISHED. Final state: {debug_print_state_from_indices(current_proof_state, im_main, oneline=True) if current_proof_state is not None else 'None'}")

    print("\n\n--- Test 2: Query q(a, What) ---")
    initial_query_vars_q = {}
    im_main.next_dynamic_var_idx_counter = im_main.dynamic_variable_start_index
    initial_state_str_q = [Term("q", ("a", "What"))]
    initial_state_tensor_q = state_to_tensor_im(initial_state_str_q, im_main, initial_query_vars_q)
    current_proof_state_q = initial_state_tensor_q
    for step_num_q in range(1, 4):
        print(f"\n--- Q Step {step_num_q} ---")
        if current_proof_state_q is None or current_proof_state_q.numel() == 0 or \
           (current_proof_state_q.shape[0] == 1 and current_proof_state_q[0,0].item() == im_main.true_pred_idx) or \
           (current_proof_state_q.shape[0] == 1 and current_proof_state_q[0,0].item() == im_main.false_pred_idx):
            print(f"Proof terminated or reached single TRUE/FALSE before step {step_num_q}.")
            break
        next_possible_states_q, _ = get_next_unification_pt(
            current_proof_state_q,
            facts_tensor_main, facts_as_set_main,
            rules_tensor_main, rule_lengths_tensor_main, im_main,
            original_rules_list=rules_str_main, verbose=1
        )
        if not next_possible_states_q:
            print("No next states derived for Q query.")
            current_proof_state_q = None
            break
        current_proof_state_q = next_possible_states_q[0]
        print(f"  Selected next state for Q Step {step_num_q+1} (if any): {debug_print_state_from_indices(current_proof_state_q, im_main, oneline=True)}")
    
    if current_proof_state_q is not None and current_proof_state_q.shape[0] == 1 and current_proof_state_q[0,0].item() == im_main.true_pred_idx:
        print("\nQUERY Q TEST SUCCEEDED TO PROVE TRUE!")
    else:
        print(f"\nQUERY Q TEST FINISHED. Final state: {debug_print_state_from_indices(current_proof_state_q, im_main, oneline=True) if current_proof_state_q is not None else 'None'}")


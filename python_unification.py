import torch
from typing import List, Dict, Set, Tuple, FrozenSet, Optional, NamedTuple, Any
from utils import Term, Rule
from index_manager import IndexManager, facts_to_tensor_im, rules_to_tensor_im, state_to_tensor_im, \
                        debug_print_atom, debug_print_state_from_indices, debug_print_states_from_indices


def format_substitutions_dict(subs: Dict[int, int], index_manager: IndexManager) -> Dict[str, str]:
    """
    Converts a substitutions dictionary to a human-readable format.
    Keys are variable names, values are the corresponding term values.
    If 'ground_match' is True, it indicates a ground match.
    """
    if not subs: return {}
    if subs.get('ground_match') is True:
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
    """
    Resolves an index through substitutions, handling cycles.
    Returns the resolved index or the original index if no resolution is possible.
    """
    seen = {idx}
    while is_variable_idx(idx, index_manager) and idx in substitutions:
        idx_before = idx
        idx = substitutions[idx]
        if idx == idx_before: 
            break
        if idx in seen: 
            return idx
    return idx

def unify_terms_idx(
    term1_idx: torch.Tensor,
    term2_idx: torch.Tensor,
    index_manager: IndexManager
) -> Optional[Dict[int, int]]:
    """
    Attempts to unify two terms represented by tensors. `term1` is the target, `term2` is the source.
    Variables in `term2` are substituted by values from `term1`.
    Returns a substitution dictionary if successful, otherwise None.
    """
    if term1_idx[0].item() != term2_idx[0].item():
        return None  # Predicates must match

    subs = {}
    # Arguments start at index 1
    for i in range(1, index_manager.max_arity + 1):
        arg1 = term1_idx[i].item()
        arg2 = term2_idx[i].item()

        # Resolve both arguments with the current substitution set
        r_arg1 = resolve_idx(arg1, subs, index_manager)
        r_arg2 = resolve_idx(arg2, subs, index_manager)

        # Skip if they already resolve to the same value or are both padding
        if r_arg1 == r_arg2:
            continue

        is_r_arg1_var = is_variable_idx(r_arg1, index_manager)
        is_r_arg2_var = is_variable_idx(r_arg2, index_manager)

        if is_r_arg2_var:
            subs[r_arg2] = r_arg1  # Unify: substitute variable from term2 with value from term1
        elif is_r_arg1_var:
            subs[r_arg1] = r_arg2  # Unify: substitute variable from term1 with value from term2
        elif r_arg1 != r_arg2:
            return None  # Mismatch between two different constants

    return subs

def apply_substitutions_to_term_idx(term_idx: torch.Tensor, substitutions: Dict[int, int], index_manager: IndexManager) -> torch.Tensor:
    new_term_list = [term_idx[0].item()] 
    for i in range(1, index_manager.max_arity + 1): 
        arg = term_idx[i].item()
        new_term_list.append(resolve_idx(arg, substitutions, index_manager)) 
    return torch.tensor(new_term_list, dtype=torch.long, device=term_idx.device)

def apply_substitutions_to_state_idx(state: torch.Tensor, substitutions: Dict[int, int], index_manager: IndexManager) -> torch.Tensor:
    if state.numel() == 0 or not substitutions:
        return state
    new_atoms = [apply_substitutions_to_term_idx(state[i], substitutions, index_manager) for i in range(state.shape[0])] 
    if not new_atoms:
        return torch.empty((0, index_manager.max_arity + 1), dtype=torch.long, device=state.device)
    return torch.stack(new_atoms)

def standardize_apart_rule_idx(
    rule_tensor_single: torch.Tensor,
    rule_length: int,
    index_manager: IndexManager,
    next_var_index: int
) -> Tuple[torch.Tensor, int]:
    """Renames variables in a rule to fresh, unique dynamic variables to prevent clashes."""
    if rule_length == 0:
        return rule_tensor_single, next_var_index

    local_map: Dict[int, int] = {}
    new_atoms = []
    for i in range(rule_length):
        atom = rule_tensor_single[i]
        new_atom_elements = [atom[0].item()]
        for k in range(1, index_manager.max_arity + 1):
            original_idx = atom[k].item()
            if is_variable_idx(original_idx, index_manager):
                if original_idx not in local_map:
                    if next_var_index > index_manager.variable_end_index:
                        raise ValueError(f"No more available variable indices: {next_var_index} exceeds max {index_manager.variable_end_index}.")
                    local_map[original_idx] = next_var_index
                    next_var_index += 1
                new_atom_elements.append(local_map[original_idx])
            else:
                new_atom_elements.append(original_idx)
        new_atoms.append(torch.tensor(new_atom_elements, dtype=torch.long, device=index_manager.device))

    standardized_part = torch.stack(new_atoms)

    # Re-attach padding if it existed
    if rule_length < rule_tensor_single.shape[0]:
        padding = rule_tensor_single[rule_length:]
        return torch.cat([standardized_part, padding], dim=0), next_var_index

    return standardized_part, next_var_index


def unify_with_facts(
    query: torch.Tensor,
    fact_indexed: Dict[Tuple, Set[Tuple[int, ...]]],  # Takes the built index
    facts_set: FrozenSet[Tuple[int, int, int]],
    excluded_fact: Optional[torch.Tensor],
    index_manager: IndexManager,
    verbose: int = 0
) -> List[Dict[int, int]]:
    """
    Attempts to unify a query with facts using an index for efficiency.
    """
    substitutions_found = []

    # Check if the query is ground (contains no variables)
    is_ground = not any(is_variable_idx(query[i].item(), index_manager) for i in range(1, index_manager.max_arity + 1))

    # --- Case 1: The query is a ground atom ---
    if is_ground:
        query_tuple = tuple(query.tolist())
        excluded_fact_tuple = tuple(excluded_fact.tolist()) if excluded_fact is not None else None
        if query_tuple in facts_set and query_tuple != excluded_fact_tuple:
            if verbose >= 1:
                print(f"Ground query {debug_print_atom(query, index_manager)} matched in fact set.")
            substitutions_found.append({'ground_match': True})
        return substitutions_found

    # --- Case 2: The query is not ground (contains variables) ---
    # Construct a lookup key from the query's predicate and constant arguments
    query_constant_args_with_pos = []
    for i in range(1, index_manager.max_arity + 1):
        arg_idx = query[i].item()
        if not is_variable_idx(arg_idx, index_manager) and arg_idx != index_manager.padding_idx:
            # Store argument position (0 or 1) and its constant index
            query_constant_args_with_pos.append((i - 1, arg_idx))

    # The key is the predicate index plus the sorted constant arguments for canonical representation
    lookup_key = (query[0].item(),) + tuple(sorted(query_constant_args_with_pos, key=lambda x: x[0]))

    # Retrieve only the candidate facts that could possibly unify using the index
    candidate_fact_tuples = fact_indexed.get(lookup_key, set())

    if verbose >= 1:
        # print(f"DEBUG: Lookup key {lookup_key} found {len(candidate_fact_tuples)} candidate facts.")
        print(f"First 10 candidate facts: {[debug_print_atom(torch.tensor(fact_tuple, dtype=torch.long, device=query.device), index_manager)\
                             for fact_tuple in list(candidate_fact_tuples)[:10]]}") if candidate_fact_tuples else None
    excluded_fact_tuple = tuple(excluded_fact.tolist()) if excluded_fact is not None else None

    # Iterate over the much smaller set of candidate facts
    for fact_tuple in candidate_fact_tuples:
        if fact_tuple == excluded_fact_tuple:
            continue

        fact_tensor = torch.tensor(fact_tuple, dtype=torch.long, device=query.device)

        # Attempt to unify the fact with the query (fact provides values, query has variables)
        subs = unify_terms_idx(fact_tensor, query, index_manager)

        if subs is not None:
            if verbose >= 1:
                print(f"    Query {debug_print_atom(query, index_manager)} unified with fact {debug_print_atom(fact_tensor, index_manager)} -> subs {format_substitutions_dict(subs, index_manager)}")
            substitutions_found.append(subs)

    return substitutions_found


def unify_with_rules(
    query: torch.Tensor,
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager: IndexManager,
    rules_term: List[Rule],
    # next_var_index: int,
    verbose: int = 0
) -> List[Tuple[torch.Tensor, Dict[int, int]]]:
# ) -> Tuple[List[Tuple[torch.Tensor, Dict[int, int]]], int]:
    """Unifies a query with all rule heads, returning instantiated bodies and substitutions."""
    results = []
    # current_next_var_index = next_var_index
    for i in range(rules.shape[0]):
        rule_len = rule_lengths[i].item()

        # Standardize apart to get fresh variables for this rule application
        rule = rules[i]
        # rule, current_next_var_index = standardize_apart_rule_idx(
        #     rule, rule_len, index_manager, current_next_var_index
        # )
        rule_head_idx = rule[0]

        # Unify query with the standardized rule head
        subs = unify_terms_idx(query, rule_head_idx, index_manager)

        if subs is not None:
            body_template = rule[1:rule_len]
            instantiated_body = apply_substitutions_to_state_idx(body_template, subs, index_manager)
            
            if verbose >= 1:
                # print(f"  Rule {i}: {str(rules_term[i])}")
                print(f"  Rule {i}: {debug_print_atom(rule_head_idx, index_manager)} --> {debug_print_state_from_indices(body_template, index_manager, True)}")
                # print(f"    Query: {debug_print_atom(query, index_manager)}. Subs: {format_substitutions_dict(subs, index_manager)}")
                print(f"    Subs: {format_substitutions_dict(subs, index_manager)}. New Body: {debug_print_state_from_indices(instantiated_body, index_manager, True)}")

            results.append((instantiated_body, subs))
    return results #, current_next_var_index


# To be added to dd/python_unification.py

def canonicalize_variables_in_state_idx(
    state_tensor: torch.Tensor,
    index_manager: "IndexManager",
    next_var_index: int
) -> Tuple[torch.Tensor, int]:
    """
    Renames all variables within a single state tensor to a compact,
    sequential set of new variable indices starting from `next_var_index`.
    """
    if state_tensor.numel() == 0:
        return state_tensor, next_var_index

    local_map: Dict[int, int] = {}
    new_state_atoms = []

    # Work only on the non-padding part of the state
    non_padding_mask = state_tensor[:, 0] != index_manager.padding_idx
    unpadded_state = state_tensor[non_padding_mask]

    if unpadded_state.numel() == 0:
        return state_tensor, next_var_index

    # 1. Collect all unique variable indices in the current state
    vars_in_state = set()
    for i in range(unpadded_state.shape[0]):
        for k in range(1, index_manager.max_arity + 1):
            arg_idx = unpadded_state[i, k].item()
            if index_manager.is_var_idx(arg_idx):
                vars_in_state.add(arg_idx)

    if not vars_in_state:
        return state_tensor, next_var_index

    # 2. Create a deterministic mapping from old indices to new, compact ones
    for old_var_idx in sorted(list(vars_in_state)):
        if next_var_index > index_manager.variable_end_index:
            raise ValueError(f"No more available variable indices: {next_var_index} exceeds max {index_manager.variable_end_index}.")
        local_map[old_var_idx] = next_var_index
        next_var_index += 1

    # 3. Apply the mapping to create the new, canonicalized state
    for i in range(unpadded_state.shape[0]):
        atom = unpadded_state[i]
        new_atom_elements = [atom[0].item()]
        for k in range(1, index_manager.max_arity + 1):
            original_idx = atom[k].item()
            # Replace with the new canonical index if it's a mapped variable
            new_idx = local_map.get(original_idx, original_idx)
            new_atom_elements.append(new_idx)

        new_state_atoms.append(torch.tensor(new_atom_elements, dtype=torch.long, device=index_manager.device))

    new_unpadded_state = torch.stack(new_state_atoms)

    # 4. Re-apply padding to restore the original tensor shape
    padding_rows = state_tensor.shape[0] - new_unpadded_state.shape[0]
    if padding_rows > 0:
        padding = torch.full((padding_rows, index_manager.max_arity + 1), index_manager.padding_idx, dtype=torch.long, device=index_manager.device)
        return torch.cat([new_unpadded_state, padding], dim=0), next_var_index

    return new_unpadded_state, next_var_index

def get_next_unification_pt(
    current_state: torch.Tensor,
    fact_indexed: Dict[Tuple, Set[Tuple[int, ...]]],
    facts_set: FrozenSet[Tuple[int, ...]],
    rules: torch.Tensor,
    rule_lengths: torch.Tensor,
    index_manager: "IndexManager",
    rules_term: List["Rule"],
    next_var_index: int,
    excluded_fact: Optional[torch.Tensor] = None,
    unification_strategy: str = 'rules_then_facts', # 'rules_and_facts', 'rules_then_facts'
    verbose: int = 0
) -> Tuple[List[torch.Tensor], int]:
    """
    Processes a state by first unifying its primary goal with rules,
    and then unifying the first goal of each resulting state with facts.
    This function is designed to match the logic of the original string-based implementation.
    """

    if verbose: 
        print(f"\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # --- 1. Initial State Checks (handles padded input) ---
    non_padding_mask = current_state[:, 0] != index_manager.padding_idx
    state = current_state[non_padding_mask]
 
    if state.numel() == 0:
        # This means the original state was empty or all padding. This is a success condition (empty goal list).
        print("True state resolved to padding or empty.") if verbose >= 1 else None
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++") if verbose >= 1 else None
        return [index_manager.false_tensor.unsqueeze(0)], next_var_index

    if torch.any(state[:, 0] == index_manager.false_pred_idx):
        print("Current state contains a FALSE predicate. Terminating path.") if verbose >= 1 else None
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++") if verbose >= 1 else None
        return [index_manager.false_tensor.unsqueeze(0)], next_var_index
    
    # Filter out TRUE predicates as they are considered proven
    state = state[state[:, 0] != index_manager.true_pred_idx]
    if state.shape[0] == 0:
        print("All goals resolved to TRUE.") if verbose >= 1 else None
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++") if verbose >= 1 else None
        return [index_manager.true_tensor.unsqueeze(0)], next_var_index

    # --- 2. Goal Selection ---
    query_atom = state[0]
    remaining_goals = state[1:]
    if verbose >= 1:
        print(f"Query: {debug_print_atom(query_atom, index_manager)}")
        print(f"Remaining Goals: {debug_print_state_from_indices(remaining_goals, index_manager, True)} ")

    # --- 3. Step 1: Unify Query with Rule Heads ---
    if verbose >= 1: print("\n--- Rule unification ---")
    states_from_rules: List[torch.Tensor] = []
    rule_unification_results = unify_with_rules(
        query_atom, rules, rule_lengths, index_manager, rules_term, verbose
    )

    for body, subs in rule_unification_results:
        # Apply substitutions from the rule unification to the rest of the goals
        substituted_remaining_goals = apply_substitutions_to_state_idx(remaining_goals, subs, index_manager)
        # The new state is the rule body followed by the updated remaining goals
        new_state = torch.cat([body, substituted_remaining_goals], dim=0)
        states_from_rules.append(new_state)

    if not states_from_rules:
        if verbose >= 1: 
            print("No rule unifications found. Path terminates.")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return [index_manager.false_tensor.unsqueeze(0)], next_var_index

    # --- 4. Step 2: Unify First Goal of Intermediate States with Facts ---
    final_resolvents: List[torch.Tensor] = []
    if verbose >= 1: print("\n--- Fact Unification ---")

    for state_from_rules in states_from_rules:
        assert state_from_rules.shape[0] > 0, "Intermediate state should not be empty at this point."

        first_goal, rest_of_goals = state_from_rules[0], state_from_rules[1:]

        fact_substitutions = unify_with_facts(
            first_goal, fact_indexed, facts_set, excluded_fact, index_manager, verbose
        )

        for subs in fact_substitutions:
            # If there are no other goals to prove, we are done with this path.
            if rest_of_goals.shape[0] == 0:
                if verbose >= 1: 
                    print("Fact unification succeeded and no goals remain. Proof found!")
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                return [index_manager.true_tensor.unsqueeze(0)], next_var_index

            # Apply substitutions from the fact unification to the rest of the goals
            new_goals = rest_of_goals if subs.get('ground_match') else apply_substitutions_to_state_idx(rest_of_goals, subs, index_manager)


            # Intermediate Fact Checking: Check if any of the new goals are now facts
            simplified_goals = []
            for atom in new_goals:
                # A goal is a fact if it's ground and exists in the fact set
                is_ground = not any(is_variable_idx(arg.item(), index_manager) for arg in atom[1:])
                if is_ground and tuple(atom.tolist()) in facts_set:
                    if verbose >= 1: print(f"    Goal {debug_print_atom(atom, index_manager)} resolved to a fact. Skipping.")
                    # Don't add it to the list, effectively replacing it with TRUE
                    continue
                simplified_goals.append(atom)

            if not simplified_goals:
                # All remaining goals were resolved to facts
                if verbose >= 1: 
                    print("All remaining goals resolved to facts. Proof found!")
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                return [index_manager.true_tensor.unsqueeze(0)], next_var_index
            final_resolvents.append(torch.stack(simplified_goals))

    # --- 5. Step 3: Combine States Based on Strategy ---
    if unification_strategy == 'rules_and_facts' and len(states_from_rules) > 1:
        # states_from_rules = rename_vars_local(states_from_rules, next_var_index, verbose=verbose)
        final_resolvents.extend(states_from_rules)

    if not final_resolvents:
        if verbose >= 1: 
            print("No states could be resolved via facts. Path terminates.")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return [index_manager.false_tensor.unsqueeze(0)], next_var_index

    # --- 6. Finalize and Deduplicate ---
    unique_states, seen_states = [], set()

    for state_tensor in final_resolvents:
        # Convert tensor to a hashable tuple to use with the set
        state_tuple = tuple(tuple(atom.tolist()) for atom in state_tensor)
        if state_tuple not in seen_states:
            unique_states.append(state_tensor)
            seen_states.add(state_tuple)

    # If we are not using the 'rules_and_facts' strategy, canonicalize variables in the unique states
    if unification_strategy != 'rules_and_facts':
        final_states = []
        for state in unique_states:
            f_state, next_var_index = canonicalize_variables_in_state_idx(state, index_manager, next_var_index)
            final_states.append(f_state)
    else:
        final_states = unique_states
        
    if verbose >= 1:
        print()
        print(f"\nFinal Next States: {debug_print_states_from_indices(final_states, index_manager)}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    return final_states, next_var_index






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
    facts_main = facts_to_tensor_im(facts_str_main, im_main)
    facts_set_main = frozenset(tuple(f.tolist()) for f in facts_main) 
    max_atoms_main = 0
    if rules_str_main:
        max_atoms_main = max(1 + len(r.body) for r in rules_str_main) if rules_str_main else 1
    rules_main, rule_lengths_tensor_main = rules_to_tensor_im(rules_str_main, max_atoms_main, im_main)
    print("--- IndexManager Initialized ---")
    print(f"True Idx: {im_main.true_pred_idx}, False Idx: {im_main.false_pred_idx}, Pad Idx: {im_main.padding_idx}")
    print("\n--- Facts (Tensor Debug) ---")
    print(debug_print_state_from_indices(facts_main, im_main))
    print("\n--- Rules (Tensor Debug) ---")
    for i in range(rules_main.shape[0]):
        print(f"Rule {i} (len {rule_lengths_tensor_main[i].item()}): {debug_print_state_from_indices(rules_main[i,:rule_lengths_tensor_main[i].item()], im_main, oneline=True)}")

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
            facts_main,    
            facts_set_main,    
            rules_main,    
            rule_lengths_tensor_main, 
            im_main,
            rules_term=rules_str_main, 
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
            facts_main, facts_set_main,
            rules_main, rule_lengths_tensor_main, im_main,
            rules_term=rules_str_main, verbose=1
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
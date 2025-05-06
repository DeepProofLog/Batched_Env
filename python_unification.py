from typing import List, Dict, Set, Tuple, FrozenSet, Optional
from utils import Term, Rule


def is_variable(arg: str) -> bool:
    """Check if an argument is a variable."""
    return arg[0].isupper() or arg[0] == '_'

def unify_terms(term1: Term, term2: Term, verbose: int=0) -> Dict[str, str]:
    """
    Attempts to unify two terms and returns a substitution dictionary if successful.
    Returns None if unification is not possible.
    We unify the first term with the second (term1 with term2)
    
    Args:
        term1: First Term object to unify
        term2: Second Term object to unify
    
    Returns:
        Dictionary mapping variables to their substitutions, or None if unification fails
    """
    assert len(term1.args) == len(term2.args) == 2, 'only support binary predicates'
    print('\nterm1:', term1, 'term2:', term2) if verbose else None
    substitutions = {}
    for arg1, arg2 in zip(term1.args, term2.args):
        print('arg1:', arg1, 'arg2:', arg2) if verbose else None

        # If both are constants, they must be equal
        if not (arg1[0].isupper() or arg1[0] == '_') and not (arg2[0].isupper() or arg2[0] == '_'):
            print('both are constants') if verbose else None
            if arg1 != arg2:
                print('constants are different') if verbose else None
                return None
        elif arg2 in substitutions:
            print('arg2 in substitutions') if verbose else None
            if substitutions[arg2] != arg1:
                print('different substitution') if verbose else None
                return None
        else:
            substitutions[arg2] = arg1
            print('substitutions:', substitutions) if verbose else None
    return substitutions

def unify_with_facts(query: Term, 
                    facts_indexed: Dict[Tuple, Set[Term]], 
                    facts_set: FrozenSet[Term], 
                    excluded_fact: Optional[Term] = None, # Added parameter
                    verbose: int=0) -> List[Dict[str, str]]:
    """
    Attempts to unify a query with a list of facts.
    
    Args:
        query: Term object representing the query
        facts: List of Term objects representing facts
    
    Returns:
        List of successful substitution dictionaries
    """

    substitutions = []
    query_is_ground = not any(arg[0].isupper() or arg[0] == '_' for arg in query.args)

    if query_is_ground:
        if query in facts_set and query != excluded_fact:
            substitutions.append({'True': 'True'})
        return substitutions

    # For non-ground queries, use the index
    query_constant_args_with_pos = [(i, arg) for i, arg in enumerate(query.args) if not (arg[0].isupper() or arg[0] == '_')]

    # # Sort constant args by position to create a canonical lookup key
    # sorted_query_constant_args = tuple(sorted(query_constant_args_with_pos, key=lambda x: x[0]))
    # lookup_key = (query.predicate,) + sorted_query_constant_args
    lookup_key = (query.predicate,) + tuple(query_constant_args_with_pos)
    # Retrieve candidate facts using the lookup key (later we remove the excluded_fact)
    candidate_facts = facts_indexed.get(lookup_key, set())

    # Iterate only over the candidate facts
    for fact in candidate_facts:
        if fact == excluded_fact:
            continue
        # Verify predicate and arity match before attempting unification
        if fact.predicate == query.predicate:
            subs = unify_terms(fact, query, verbose=0)
            if subs is not None:
                substitutions.append(subs)

    return substitutions

def unify_with_rules(query: Term, rules: List[Rule], verbose: int=0) -> List[Tuple[List[Term], Dict[str, str]]]:
    """
    Attempts to unify a query with the heads of rules and returns their bodies with substitutions.
    
    Args:
        query: Term object representing the query
        rules: List of Rule objects to try unifying with
    
    Returns:
        List of tuples containing (rule body terms, substitution dictionary)
    """
    results = []
    for rule in rules:
        if rule.head.predicate == query.predicate:
            subs = unify_terms(query, rule.head)
            if subs is not None:
                new_body = [
                    (term if (substituted_args := tuple(subs.get(arg, arg) for arg in term.args)) == term.args
                     else Term(term.predicate, substituted_args))
                    for term in rule.body
                ]
                print('Rule:', rule, '      Subs:', subs, '     New body:', new_body) if verbose else None
                results.append((new_body, subs))
    
    return results

def rename_vars_local(next_states: List[List[Term]],
                      global_next_var_index: int,
                      verbose: int = 0,
                      ) -> Tuple[List[List[Term]], int]:
    """
    Renames variables within each state locally, avoiding collisions with
    pre-existing 'Var_...' variables in that state. The global index counter
    tracks the highest index used across all states.

    Args:
        next_states: A list of states, where each state is a list of Terms.
        global_next_var_index: The starting index suggestion from the global scope.
        verbose: Verbosity level.

    Returns:
        A tuple containing:
        - The list of states with variables renamed locally and safely.
        - The updated global_next_var_index after processing all states.
    """
    if global_next_var_index is None:
        raise ValueError('global_next_var_index cannot be None')

    renamed_states_outer = []

    for idx, state in enumerate(next_states):
        local_var_mapping: Dict[str, str] = {} # Mapping is local to this state
        renamed_state_inner = [None] * len(state)

        # --- Collision Avoidance Step ---
        max_existing_k = -1
        existing_vars_in_state = set()
        for term in state:
            for arg in term.args:
                 if isinstance(arg, str) and arg.startswith('Var_'):
                      existing_vars_in_state.add(arg)
                      try:
                           k = int(arg[4:])
                           max_existing_k = max(max_existing_k, k)
                      except ValueError:
                           pass # Ignore malformed Var_ names

        local_start_index = max(global_next_var_index, max_existing_k + 1)
        current_state_var_index = local_start_index # Counter for new vars in this state

        # --- Renaming Loop ---
        for i, term in enumerate(state):
            original_args = term.args
            new_args_list = None
            term_changed = False

            for j, arg in enumerate(original_args):
                renamed_arg = arg

                # Only rename non-'Var_' variables
                if arg and (arg[0].isupper() or arg[0] == '_') and not arg.startswith('Var_'):
                    mapped_arg = local_var_mapping.get(arg)
                    if mapped_arg is None:
                        new_var_name = f"Var_{current_state_var_index}"
                        local_var_mapping[arg] = new_var_name
                        current_state_var_index += 1 # Increment index for next new var *in this state*
                        renamed_arg = new_var_name
                        term_changed = True
                    else:
                        renamed_arg = mapped_arg
                        if renamed_arg != arg: # Should always be true if mapped
                            term_changed = True
                # Else: Keep constants and existing 'Var_' variables as they are

                # --- Optimization: Build new args list only if necessary ---
                if term_changed and new_args_list is None:
                    new_args_list = list(original_args[:j])
                if new_args_list is not None:
                    new_args_list.append(renamed_arg)

            if term_changed:
                renamed_state_inner[i] = Term(term.predicate, tuple(new_args_list))
            else:
                renamed_state_inner[i] = term

        renamed_states_outer.append(renamed_state_inner)

    if renamed_states_outer != next_states:
        print('\n\nRenamed states:', renamed_states_outer) if verbose else None
        print('Original states:', next_states) if verbose else None
    return renamed_states_outer


def rename_vars(next_states: List[Term], 
                next_var_index: int,
                verbose: int = 0,
                ) -> Tuple[List[Term], int]:
    assert next_var_index is not None, 'next_var_index should not be None'

    renamed_states = []
    var_mapping: Dict[str, str] = {}
    current_next_var_index = next_var_index

    for state in next_states:
        new_state_list = [None] * len(state)

        for i, term in enumerate(state):
            term_pred = term.predicate
            original_args = term.args
            new_args_list = None  # Lazily create the list only if an arg changes
            term_changed = False # Flag to track if this specific term needs rebuilding

            for j, arg in enumerate(original_args):

                if arg and (arg[0].isupper() or arg[0] == '_') and not arg.startswith('Var_'):
                    mapped_arg = var_mapping.get(arg)
                    if mapped_arg is None:
                        new_var_name = f"Var_{current_next_var_index}"
                        var_mapping[arg] = new_var_name
                        current_next_var_index += 1
                        renamed_arg = new_var_name
                        term_changed = True
                    else:
                        renamed_arg = mapped_arg
                        if renamed_arg != arg:
                            term_changed = True
                        # If renamed_arg == arg, term_changed remains false *unless*
                        # set true by a previous arg in this term.

                    # --- Optimization: Streamlined lazy list handling ---
                    # If the term needs changing (either now or earlier)
                    if term_changed:
                        if new_args_list is None:
                            # Allocate and copy previous args only when needed
                            new_args_list = list(original_args[:j])
                        new_args_list.append(renamed_arg)

                elif new_args_list is not None:
                    new_args_list.append(arg)
                # Else (arg is constant/renamed AND new_args_list is None):
                    # Do nothing - we are still implicitly using original args

            if term_changed:
                new_state_list[i] = Term(term_pred, tuple(new_args_list))
            else:
                new_state_list[i] = term

        renamed_states.append(new_state_list)

    # Update the index that might be passed back or used later
    next_var_index = current_next_var_index

    if renamed_states != next_states:
        print('\n\nRenamed states:', renamed_states) if verbose else None
        print('Original states:', next_states) if verbose else None

    return renamed_states, next_var_index



def get_next_unification_python(state: List[Term], 
                                facts_set: FrozenSet, 
                                facts_indexed: Dict[Tuple, Set[Term]], 
                                rules: List[Rule], 
                                excluded_fact: Optional[Term] = None,
                                next_var_index: Optional[int] = None,
                                unification_strategy: str = 'only_rules', # 'only_rules' 'rules_and_facts'
                                verbose: int = 0) -> Tuple[List[List[Term]], Optional[int]]:
    """
    Processes a state: Rule Unification -> Intermediate States -> Fact Unification on First Goal
    -> Potential States -> Remove True Terms -> Return Next States
    If any state simplifies to empty (proof found), returns [[True]] immediately.
    """

    # --- Initial Checks and Setup ---
    if not state: return [[Term('True', ())]], next_var_index
    if any(term.predicate == 'False' for term in state): return [[Term('False', ())]], next_var_index
    state = [term for term in state if term.predicate != 'True']
    if not state: return [[Term('True', ())]], next_var_index

    # --- Goal Selection ---
    query = state[0]
    remaining_state = state[1:]
    rule_states = []

    print('\n\n++++++++++++++') if verbose else None
    print(f'Processing Query: {query}') if verbose else None
    print(f'Remaining State: {remaining_state}\n') if verbose else None

    next_states = []

    # --- Step 1: Unification ONLY with Rules ---
    rule_results = unify_with_rules(query, rules, verbose=verbose)
    for i, (body, rule_subs) in enumerate(rule_results):
        new_remaining = [
            (term if (substituted_args := tuple(rule_subs.get(arg, arg) for arg in term.args)) == term.args
             else Term(term.predicate, substituted_args))
            for term in remaining_state
        ]
        rule_derived_state = body + new_remaining
        rule_states.append(rule_derived_state)

    if not rule_states and unification_strategy == 'only_rules':
        print('No unification with rules') if verbose else None
        return [[Term('False', ())]], next_var_index

    print() if verbose else None
    
    # --- Step 2: Apply Fact Unification to First Goal of Intermediate States ---
    for state_from_rule in rule_states:

        first_goal = state_from_rule[0]
        rest_of_goals = state_from_rule[1:]

        fact_substitutions = unify_with_facts(first_goal, facts_indexed, facts_set, excluded_fact, verbose=verbose)
        print(F"State from rule: {state_from_rule}, subs: {fact_substitutions}") if verbose else None

        for subs in fact_substitutions:

            if not rest_of_goals: # if there are no remaining goals, we can return True because we found a fact
                print(f"    True next state and no other goals") if verbose else None
                return [[Term('True', ())]], next_var_index   

            else:
                # print('Remaining goals:', rest_of_goals) if verbose else None
                if subs.get('True') == 'True': # it is a fact
                    print(f"    Fact next state for sub {subs} in {state_from_rule}") if verbose else None
                    next_states.append(rest_of_goals)

                else: # Apply substitutions to remaining state
                    new_state = [
                        (term if (substituted_args := tuple(subs.get(arg, arg) for arg in term.args)) == term.args
                         else Term(term.predicate, substituted_args))
                        for term in rest_of_goals
                    ]

                    # substitute the facts by True
                    for j in range(len(new_state)):
                        atom = new_state[j]
                        if atom in facts_set and atom != excluded_fact:
                            new_state[j] = Term('True', ())

                    # if all atoms in the new state are True, we can return True
                    if all(term.predicate == 'True' for term in new_state):
                        print(f"    True next state for sub {subs} in {state_from_rule}") if verbose else None
                        return [[Term('True', ())]], next_var_index
                    
                    # if not all atoms in the new state are True, filter the True atoms
                    new_state = [term for term in new_state if term.predicate != 'True']
                    print(f"    New state after filtering facts: {new_state}") if verbose else None
                    next_states.append(new_state)

    if rule_states and unification_strategy == 'rules_and_facts':
        next_states.extend(rule_states)

    if not next_states:
        print('No unification with facts') if verbose else None
        return [[Term('False', ())]], next_var_index

    # --- Var renaming ---
    if unification_strategy == 'rules_and_facts':
        # next_states, next_var_index = rename_vars(next_states, next_var_index)
        next_states = rename_vars_local(next_states, next_var_index, verbose=verbose)

    print('\nNext states:', next_states) if verbose else None
    print('++++++++++++++\n') if verbose else None
    return next_states, next_var_index
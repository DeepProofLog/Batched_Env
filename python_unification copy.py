from typing import List, Dict, Set, Tuple
from collections import deque
from utils import Term, Rule, apply_substitution

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
    if term1.predicate != term2.predicate or len(term1.args) != len(term2.args):
        return None
    assert len(term1.args) == len(term2.args) == 2, 'only support binary predicates'
    print('\nterm1:', term1, 'term2:', term2) if verbose else None
    substitutions = {}
    for arg1, arg2 in zip(term1.args, term2.args):
        print('arg1:', arg1, 'arg2:', arg2) if verbose else None

        # If both are constants, they must be equal
        if not is_variable(arg1) and not is_variable(arg2):
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
    # print(asdas) if verbose else None
    return substitutions


def apply_substitution(term: Term, substitutions: Dict[str, str]) -> Term:
    """
    Applies variable substitutions to a term.
    
    Args:
        term: Term object to apply substitutions to
        substitutions: Dictionary of variable substitutions
    
    Returns:
        New Term object with substitutions applied
    """
    new_args = []
    for arg in term.args:
        if arg in substitutions:
            new_args.append(substitutions[arg])
        else:
            new_args.append(arg)
    return Term(term.predicate, new_args)

def unify_with_facts(query: Term, facts: List[Term], verbose: int=0) -> List[Dict[str, str]]:
    """
    Attempts to unify a query with a list of facts.
    
    Args:
        query: Term object representing the query
        facts: List of Term objects representing facts
    
    Returns:
        List of successful substitution dictionaries
    """
    substitutions = []
    
    # If query has no variables, just check if it exists in facts
    if not any(is_variable(arg) for arg in query.args):
        print('no variables in query') if verbose else None
        if query in facts:
            substitutions.append({'True': 'True'})
            print('Query is a fact') if verbose else None
        return substitutions
    
    # Try to unify with each matching fact
    for fact in facts:
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
            print('Rule:', rule, 'Subs:', subs) if verbose else None
            if subs is not None:
                # Apply substitutions to the body
                new_body = [apply_substitution(term, subs) for term in rule.body]
                print('New body:', new_body) if verbose else None
                results.append((new_body, subs))
    
    return results







def get_next_unification_python(state: List[Term], facts: List[Term], rules: List[Rule], verbose: int = 0) -> List[List[Term]]:
    """
    Processes a state: Rule Unification -> Intermediate States -> Fact Unification on First Goal
    -> Potential States -> Final Simplification & Proof Check.
    If any state simplifies to empty (proof found), returns [[True]] immediately.
    """

    # --- Initial Checks and Setup ---
    if not state: return [[Term('True', [])]]
    if any(term.predicate == 'False' for term in state): return [[Term('False', [])]]
    state = [term for term in state if term.predicate != 'True']
    if not state: return [[Term('True', [])]]

    facts_set = frozenset(facts) # Efficient fact lookup

    # --- Variable Renaming ---
    exclude = set()
    for term in state:
        for arg in term.args:
            if isinstance(arg, str) and arg.startswith('_'): exclude.add(arg)
    subs = {}
    var_counter = 0
    vars_in_state = set()
    for term in state:
        for arg in term.args:
            if is_variable(arg): vars_in_state.add(arg)
    for var in vars_in_state:
         while f'_{var_counter}' in exclude: var_counter += 1
         subs[var] = f'_{var_counter}'; var_counter += 1
    new_state_vars_renamed = []
    for term in state:
        new_args = [subs.get(arg, arg) for arg in term.args]
        new_state_vars_renamed.append(Term(term.predicate, new_args))
    state = new_state_vars_renamed
    # print(f"\nRenamed state: {state}") if verbose > 1 else None # Optional logging

    # --- Goal Selection ---
    query, *remaining_state = state
    intermediate_states = [] # Results from rule unification

    print('\n\n**********') if verbose else None
    print(f'Processing Query: {query}') if verbose else None
    print(f'Remaining State: {remaining_state}') if verbose else None
    print('**********\n') if verbose else None

    # --- Step 1: Unification ONLY with Rules ---
    print('Attempting unification with rules...') if verbose else None
    rule_results = unify_with_rules(query, rules, verbose=verbose)
    for i, (body, rule_subs) in enumerate(rule_results):
        rule_used = rules[i]
        # print(f"  Rule {i} matched: {rule_used}") if verbose else None # Optional logging
        new_remaining = [apply_substitution(term, rule_subs) for term in remaining_state]
        rule_derived_state = body + new_remaining
        # print(f"  Intermediate State (from Rule {i}): {rule_derived_state}") if verbose else None # Optional logging
        intermediate_states.append(rule_derived_state)

    # --- Step 2: Apply Fact Unification to First Goal of Intermediate States ---
    potential_next_states = [] # Store states before final simplification/proof check

    print('\nAttempting fact unification on first goal of intermediate states...') if verbose else None
    for state_from_rule in intermediate_states:
        # print(f"  Processing intermediate state: {state_from_rule}") if verbose > 1 else None # Optional logging

        if not state_from_rule: # Rule led directly to success
             potential_next_states.append([Term('True', [])])
             continue

        first_goal = state_from_rule[0]
        rest_of_goals = state_from_rule[1:]

        # print(f"    Checking first goal against facts: {first_goal}") if verbose else None # Optional logging
        fact_substitutions = unify_with_facts(first_goal, facts, verbose=verbose)

        fact_match_found = False
        if fact_substitutions:
            fact_match_found = True
            # print(f"    Fact matches found for {first_goal}") if verbose else None # Optional logging
            for fact_subs in fact_substitutions:
                # print(f"      Applying fact substitution: {fact_subs}") if verbose > 1 else None # Optional logging
                current_resulting_state_list = []
                if fact_subs.get('True') == 'True': # Ground fact match for first_goal
                    current_resulting_state_list = rest_of_goals.copy()
                else: # Fact match with variable binding for first_goal
                    current_resulting_state_list = [apply_substitution(term, fact_subs) for term in rest_of_goals]

                # Check immediate outcome - includes check if single remaining goal is a fact
                if not current_resulting_state_list: # Became empty -> success
                     # print("      Fact resolution led to success [True]") if verbose > 1 else None # Optional logging
                     potential_next_states.append([Term('True', [])])
                elif len(current_resulting_state_list) == 1:
                    single_goal = current_resulting_state_list[0]
                    is_ground = not any(is_variable(arg) for arg in single_goal.args)
                    if is_ground and single_goal in facts_set: # Use facts_set
                         # print(f"      Remaining single goal {single_goal} is a known fact. Path leads to success [True].") if verbose > 1 else None # Optional logging
                         potential_next_states.append([Term('True', [])])
                    else: # Single goal but not a known ground fact
                         potential_next_states.append(current_resulting_state_list)
                         # print(f"      Added fact-simplified state (single goal): {current_resulting_state_list}") if verbose > 1 else None # Optional logging
                else: # Multiple remaining goals
                    potential_next_states.append(current_resulting_state_list)
                    # print(f"      Added fact-simplified state (multiple goals): {current_resulting_state_list}") if verbose > 1 else None # Optional logging

        if not fact_match_found: # Keep the state derived from the rule if its first goal didn't match a fact
            # print(f"    No fact match for {first_goal}. Keeping state as is: {state_from_rule}") if verbose else None # Optional logging
            potential_next_states.append(state_from_rule)

    # --- Fallback: Check if original query is a fact (if no rules matched) ---
    if not intermediate_states: # No rules matched the original query
        print(f"No states derived from rules. Checking if original query '{query}' is a fact.") if verbose else None
        direct_fact_subs = unify_with_facts(query, facts, verbose=verbose)
        if direct_fact_subs:
             # print(f"Original query '{query}' matched facts. Applying to remaining state: {remaining_state}") if verbose else None # Optional logging
             for fact_subs in direct_fact_subs:
                  current_resulting_state_list = []
                  if fact_subs.get('True') == 'True':
                      current_resulting_state_list = remaining_state.copy()
                  else:
                      current_resulting_state_list = [apply_substitution(term, fact_subs) for term in remaining_state]

                  # Check outcome after direct fact match
                  if not current_resulting_state_list:
                       potential_next_states.append([Term('True', [])])
                  elif len(current_resulting_state_list) == 1: # Check single remaining goal
                       single_goal = current_resulting_state_list[0]
                       is_ground = not any(is_variable(arg) for arg in single_goal.args)
                       if is_ground and single_goal in facts_set: # Use facts_set
                            potential_next_states.append([Term('True', [])])
                       else: potential_next_states.append(current_resulting_state_list)
                  else: # Multiple goals
                      potential_next_states.append(current_resulting_state_list)

    # --- Final Post-processing: Simplify States & Check for Proof ---
    # This section iterates through all generated potential states.
    # It simplifies each by removing known ground facts.
    # If ANY state simplifies to empty, it returns [[True]] immediately.
    final_next_states = []
    processed_state_tuples = set() # Use set for efficient duplicate checking of *remaining* states

    print("\n--- Post-processing: Simplifying states and checking for proof ---") if verbose >= 1 else None

    for state_list in potential_next_states:
        original_state_repr = repr(state_list) # For logging

        # Check if a proof was found in the previous steps
        if state_list == [Term('True', [])]:
            print(f"  Found pre-resolved [True] state from earlier step. Proof found!") if verbose >= 1 else None
            return [[Term('True', [])]] # Immediate return

        # Simplify the current state list by removing known ground facts
        simplified_state = []
        simplification_occurred = False
        for term in state_list:
            is_ground = not any(is_variable(arg) for arg in term.args)
            # Check if it's a ground term and exists in the facts set
            if is_ground and term in facts_set:
                 print(f"  Simplifying: Removing ground fact {term} from state {original_state_repr}") if verbose >= 1 else None
                 simplification_occurred = True
                 continue # Skip adding this term (filter fact)
            else:
                 simplified_state.append(term) # Keep non-fact or non-ground terms

        # Check if simplification resulted in an empty list (proof found NOW)
        if not simplified_state:
            print(f"  State {original_state_repr} simplified to empty. Proof found!") if verbose >= 1 else None
            return [[Term('True', [])]] # Immediate return

        # If goals remain after simplification, add as a potential next state (if unique)
        else:
            if simplification_occurred:
                 print(f"  State {original_state_repr} simplified to {simplified_state}") if verbose >= 1 else None

            # Use tuple of sorted terms for canonical representation for duplicate check
            try:
                 state_tuple = tuple(sorted(simplified_state, key=repr))
                 if state_tuple not in processed_state_tuples:
                     final_next_states.append(simplified_state)
                     processed_state_tuples.add(state_tuple)
                     # print(f"  Adding unique simplified state: {simplified_state}") if verbose > 1 else None # Optional logging
                 else:
                      # print(f"  Skipping duplicate simplified state: {simplified_state}") if verbose >= 1 else None # Optional logging
                      pass # Silently skip duplicates unless verbose
            except TypeError as e:
                 # Fallback if state cannot be reliably hashed/sorted
                 print(f"  Post-processing: Error creating tuple for state {simplified_state} - {e}. Adding without duplicate check.")
                 final_next_states.append(simplified_state)

    # If the loop completes without finding a proof (no state simplified to empty)
    print('\n--- Final Next States (if no proof found yet) ---') if verbose else None
    if not final_next_states:
         print("  No viable next states remain after simplification, returning [False]") if verbose else None
         return [[Term('False', [])]]
    else:
         # Return all unique, simplified states found if no single proof was completed
         for i, s in enumerate(final_next_states): print(f"  State {i}: {s}") if verbose else None
         print('*************************\n') if verbose else None
         return final_next_states














def get_next_unification_python_old(state: List[Term], facts: List[Term], rules: List[Rule], verbose: int=0) -> List[List[Term]]:
    """
    Processes a state and returns all possible next states based on unification with facts and rules.
    
    Args:
        state: List of Term objects representing the current state
        facts: List of Term objects representing known facts
        rules: List of Rule objects representing inference rules
    
    Returns:
        List of possible next states, where each state is a list of Terms
    """

    # Handle terminal states
    if any(term.predicate == 'False' for term in state):
        print('\n\nState:', state) if verbose else None
        return [[Term('False', [])]]
    if any(term.predicate == 'True' for term in state):
        state = [term for term in state if term.predicate != 'True']
    if not state:
        return [[Term('True', [])]]
    
    # THIS CAN BE AVOIDED, by getting a set of vars ini initial state, and when getting rules, if some vars are in common, change the name of the vars in the rule
    # Substitue the vars that are capital letters in the state by numbers starting from '_0'
    # all the same variables will have the same number
    exclude = set()
    for term in state:
        for arg in term.args:
            if isinstance(arg, str) and arg.startswith('_'):
                exclude.add(arg)
    subs = {}
    var_counter = 0
    new_state = []
    for term in state:
        new_args = []
        for arg in term.args:
            if isinstance(arg, str) and arg.isupper():
                if arg not in subs:
                    while f'_{var_counter}' in exclude:
                        var_counter += 1
                    subs[arg] = f'_{var_counter}'
                new_args.append(subs[arg])
            else:
                new_args.append(arg)
        new_state.append(Term(term.predicate, new_args))
    state = new_state
    
    # if there are atoms with only variables, put them at the end of the state
    state.sort(key=lambda term: 1 if all(is_variable(arg) for arg in term.args) else 0)
    
    # Get the first query and remaining state
    query, *remaining_state = state
    next_states = []

    # Try unifying with facts
    print('\n\n**********\nQuery:', query, 'Remaining state:', remaining_state,'\n') if verbose else None
    print('unification with facts') if verbose else None
    fact_substitutions = unify_with_facts(query, facts, verbose=0)
    for subs in fact_substitutions:
        # print('Substitution:', subs) if verbose else None
        if subs.get('True') == 'True':
            # If it was a fact with no substitutions, continue with remaining state
            new_state = remaining_state.copy()
            if new_state:
                next_states.append(new_state)
            else:
                return [[Term('True', [])]]
        else:
            # Apply substitutions to remaining state
            new_state = [apply_substitution(term, subs) for term in remaining_state]
            next_states.append(new_state)

    # For the next states, the ones that are (true) facts, we can substite them by True term
    for i in range(len(next_states)):
        next_state = next_states[i]
        # substitute the facts by True
        for j in range(len(next_state)):
            atom = next_state[j]
            if not any(is_variable(arg) for arg in atom.args) and atom in facts:
                next_states[i][j] = Term('True', [])
        # if the next state i is True, return True (we found a True next state)
        if all(term.predicate == 'True' for term in next_state):
            return [[Term('True', [])]]

    print('Next states:', next_states) if verbose else None
    # Try unifying with rules
    print('\nunification with rules') if verbose else None
    rule_results = unify_with_rules(query, rules, verbose=0)
    for i, (body, subs) in enumerate(rule_results):
        # Apply substitutions to remaining state
        new_remaining = [apply_substitution(term, subs) for term in remaining_state]
        # Combine rule body with remaining state
        new_state = body + new_remaining 
        print('New state:', rules[i], new_state) if verbose else None
        next_states.append(new_state)
    
    print('\nNext states:', next_states,'\n**********\n') if verbose else None
    # If no unification was possible, return False
    if not next_states:
        return [[Term('False', [])]]

    return next_states
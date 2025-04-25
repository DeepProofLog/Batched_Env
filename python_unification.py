from typing import List, Dict, Set, Tuple, FrozenSet, Optional
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
    query_constant_args_with_pos = [(i, arg) for i, arg in enumerate(query.args) if not is_variable(arg)]
    # Sort constant args by position to create a canonical lookup key
    sorted_query_constant_args = tuple(sorted(query_constant_args_with_pos, key=lambda x: x[0]))
    lookup_key = (query.predicate,) + sorted_query_constant_args

    # Retrieve candidate facts using the lookup key (later we remove the excluded_fact)
    candidate_facts = facts_indexed.get(lookup_key, set())

    # Iterate only over the candidate facts
    for fact in candidate_facts:
        if fact == excluded_fact:
            continue
        # Verify predicate and arity match before attempting unification
        if fact.predicate == query.predicate and len(fact.args) == len(query.args):
             # Unify the fact with the query
            subs = unify_terms(fact, query, verbose=0) # Pass fact as term1, query as term2
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
                new_body = [apply_substitution(term, subs) for term in rule.body]
                print('Rule:', rule, '      Subs:', subs, '     New body:', new_body) if verbose else None
                results.append((new_body, subs))
    
    return results



def get_next_unification_python(state: List[Term], 
                                facts_set: FrozenSet, 
                                facts_indexed: Dict[Tuple, Set[Term]], 
                                rules: List[Rule], 
                                excluded_fact: Optional[Term] = None,
                                verbose: int = 0) -> List[List[Term]]:
    """
    Processes a state: Rule Unification -> Intermediate States -> Fact Unification on First Goal
    -> Potential States -> Remove True Terms -> Return Next States
    If any state simplifies to empty (proof found), returns [[True]] immediately.
    """

    # --- Initial Checks and Setup ---
    if not state: return [[Term('True', ())]]
    if any(term.predicate == 'False' for term in state): return [[Term('False', ())]]
    state = [term for term in state if term.predicate != 'True']
    if not state: return [[Term('True', ())]]


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
        # new_args = (subs.get(arg, arg) for arg in term.args)
        new_args = tuple(subs.get(arg, arg) for arg in term.args)
        new_state_vars_renamed.append(Term(term.predicate, new_args))
    if new_state_vars_renamed != state:
        print(f"Renamed state variables: {state} -> {new_state_vars_renamed}")
    state = new_state_vars_renamed 

    # --- Goal Selection ---
    # query, *remaining_state = state
    query = state[0]
    remaining_state = state[1:]
    intermediate_states = []

    print('\n\n**********') if verbose else None
    print(f'Processing Query: {query}') if verbose else None
    print(f'Remaining State: {remaining_state}') if verbose else None
    print('**********\n') if verbose else None

    # --- Step 1: Unification ONLY with Rules ---
    rule_results = unify_with_rules(query, rules, verbose=verbose)
    for i, (body, rule_subs) in enumerate(rule_results):
        new_remaining = [apply_substitution(term, rule_subs) for term in remaining_state]
        rule_derived_state = body + new_remaining
        intermediate_states.append(rule_derived_state)

    if not intermediate_states:
        print('No unification with rules') if verbose else None
        return [[Term('False', ())]]

    # --- Step 2: Apply Fact Unification to First Goal of Intermediate States ---
    next_states = []
    for state_from_rule in intermediate_states:

        first_goal = state_from_rule[0]
        rest_of_goals = state_from_rule[1:]

        fact_substitutions = unify_with_facts(first_goal, facts_indexed, facts_set, excluded_fact, verbose=verbose)

        for subs in fact_substitutions:
            if not rest_of_goals: # if there are no remaining goals, we can return True because we found a fact
                next_states.append([Term('True', ())])

            if subs.get('True') == 'True': # it is a fact
                next_states.append(rest_of_goals)

            else: # Apply substitutions to remaining state
                new_state = [apply_substitution(term, subs) for term in rest_of_goals]

                # substitute the facts by True
                for j in range(len(new_state)):
                    atom = new_state[j]
                    if atom in facts_set and atom != excluded_fact:
                        new_state[j] = Term('True', ())

                # if all atoms in the new state are True, we can return True
                if all(term.predicate == 'True' for term in new_state):
                    return [[Term('True', ())]]
                
                # if not all atoms in the new state are True, filter the True atoms
                new_state = [term for term in new_state if term.predicate != 'True']
                next_states.append(new_state)


    print('Next states:', next_states) if verbose else None
    return next_states
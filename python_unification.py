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

def get_next_unification_python(state: List[Term], facts: List[Term], rules: List[Rule], verbose: int=0) -> List[List[Term]]:
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
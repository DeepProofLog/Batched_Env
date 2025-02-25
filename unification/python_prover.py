from typing import List, Dict, Tuple
from utils import Term, Rule, apply_substitution, is_variable  

def unify_atom(rule: Rule, atom: Term, verbose=False) -> Tuple[List[Term], bool]:
    """
    Unify two terms and return the resulting child proofs (atoms) that need to be proven,
    along with the updated substitution as a list of atom paths.
    """
    print('---------------------------------') if verbose else None
    rule_head = rule.head
    rule_body = rule.body
    print (f'Unifying atom {atom} with rule head {rule_head} and rule body {rule_body}') if verbose else None
    # Apply substitution to the current atom and rule head
    term1 = atom
    term2 = rule_head
    if term1.predicate != term2.predicate or len(term1.args) != len(term2.args):
        print('Unification failed. Predicates or number of arguments do not match') if verbose else None
        print('---------------------------------') if verbose else None
        return None, False
    
    # Create an empty list to store the substitutions
    assignment = {}
    # Perform unification
    is_fact = not rule_body
    if is_fact:
        for arg_atom, arg_rulehead in zip(term1.args, term2.args):
            if is_variable(arg_atom): # if arg_atom is a variable, assign it to arg_rulehead
                assignment[arg_atom] = arg_rulehead
            else: # if arg_atom is a constant, check if it is the same as arg_rulehead
                if arg_atom != arg_rulehead:
                    print('Unification failed. Constants do not match') if verbose else None
                    print('---------------------------------') if verbose else None
                    return None, False
    else:
        for arg_atom, arg_rulehead in zip(term1.args, term2.args):
            if not is_variable(arg_atom): # if arg_atom is a constant, assign it to arg_rulehead
                assignment[arg_rulehead] = arg_atom
            else: # if arg_atom is a variable, assign it to arg_rulehead
                assignment[arg_rulehead] = arg_atom
                
    print(f'Unification successful') if verbose else None
    print(f'Substitution: {assignment}') if verbose else None
    # Apply the substitution to the body of the rule
    unified_body = [apply_substitution(term, assignment) for term in rule_body]

    print(f"Unification: {atom} -> {', '.join(str(body) for body in unified_body) if unified_body else '[]'}") if verbose else None
    print(f'---------------------') if verbose else None

    return unified_body, True


def unify_state( state: List[Term], rules: List[Rule], action: List[int], verbose=int) -> Tuple[List[Term], bool]:
    """
    Given a state and a atom of that state, unify that atom with the head of the rule.
    The main point is that if the atom unifies with a fact, then the rest of the atoms of the state need to update their variables value 
    
    Please take into account for the new state that the atoms need to have the same order as they had in the original state
    """
    atom, rule = state[action[0]], rules[action[1]]
    print('---------------------------------') if verbose>=1 else None
    print(f'State: {[str(atom) for atom in state]}') if verbose>=1 else None
    print(f'Atom: {atom}') if verbose>=1 else None
    print(f'Action: rule {action[1]}') if verbose>=1 else None
    print(f'Rule: {rule}') if verbose>=1 else None

    # Do unification with the atom and the head of the rule
    child_proof, unified = unify_atom(rule, atom, verbose=False)
    print(f'Unified: {unified}, unification: {[str(atom) for atom in child_proof] if child_proof else []}') if verbose>=1 else None
    if unified: # the atom unifies with the head of the rule
        new_state = []
        if not child_proof:   # If the rule has no body, we've found a fact (leaf node). Dont include the fact in the new state
            print('Found a fact') if verbose==2 else None
            # check which vars in the original atom have been assigned a value. This is useful for the other atoms in the state
            vars_assigned = {arg: value for arg, value in zip(atom.args, rule.head.args) if is_variable(arg)} # If there's a var in atom, check its assigned value
            print(f'Vars assigned: {vars_assigned}') if verbose==2 else None
            for i, state_atom in enumerate(state):
                print(f'Doing unification for atom {state_atom}') if verbose==2 else None
                if state_atom != atom:
                    # apply the substitution to the rest of the atoms in the state
                    new_state.append(Term(state_atom.predicate, [vars_assigned.get(arg, arg) for arg in state_atom.args]))
                    print(f'    Applying substitution to atom {state_atom}') if verbose==2 else None
                    print(f'    New state: {[str(state_atom) for state_atom in new_state]}') if verbose==2 else None

        else: # if the rule has a body, then the atom is not a fact. It is a rule, so we need to include the body of the rule in the new state
            # the rest of the atoms of the state dont need to update their variables value because the rule is not a fact, has a body
            for i, state_atom in enumerate(state):
                if state_atom != atom:
                    new_state.append(state_atom)
                else:
                    new_state.extend(child_proof)
        print(f'New state: {[str(state_atom) for state_atom in new_state]}') if verbose>=1 else None
        print('---------------------------------') if verbose>=1 else None
        return new_state, True
    else:
        print('---------------------------------') if verbose>=1 else None
        return None, False
    
def get_next_state_python(state: List[Term], rules_choice: List[int], rules: List[Rule]) -> Tuple[List[Term], bool, bool]:
    '''
    Given a state and an action (the rule to unify), return the next state of the environment, and whether it is a finished state
    '''
    # We are given as action only the rule choice, but unify_state needs the atom choice as well.
    # We can set by default the atom choice to 0 (the leftmost) as in prolog
    atom_choice = 0
    action = [[atom_choice, rule_choice] for rule_choice in rules_choice]
    for a in action: # if there is more than one action, for every action, calculate the next state
        state, unified = unify_state(state, rules, a, verbose=1)
        if not unified:
            break
        if unified and not state:
            break

    if unified and not state:
        done = successful_end = True
        state = ["True"]
    elif unified and state:
        done = successful_end = False
    else:
        done = True
        successful_end = False
        state = ["False"]
    return state, done, successful_end

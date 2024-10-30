import re
from collections import deque
from typing import List, Tuple
from utils import extract_var, Term
import janus_swi as janus


def get_actions_prolog(state: str, verbose: int=0) -> List[str]:
    
    print('State str:', state) if verbose>1 else None
    state = deque(re.findall(r'\w+\(.*?\)', state))
    print('State deque:', state) if verbose>1 else None
    # get the first element of the state
    s = state.popleft()
    print('State popleft:', s) if verbose>1 else None
    actions = []
    # We get 'truth' and 'B' from the query, which is whether the query is true or not, 
    # and the value of the query
    print('Query:', f"clause({s}, _B), term_string(_B, B)") if verbose>1 else None
    res = janus.query_once(f"clause({s}, _B), term_string(_B, B)")

    print('Res:', res) if verbose>1 else None
    # If res["truth"] is false, it means the clause is not true
    if not res["truth"]:
        print(f'There is no subsitution') if verbose>1 else None
        print(f'Actions: ["False"]') if verbose else None
        return ["False()"]
    # res["truth"] is true, i.e., there's a substitution
    # If res["B"] is true, it means it is a fact
    elif res["B"] == "true":
        print(f'There is a substitution for the clause, and it is a fact') if verbose>1 else None
        # if there are not vars in the clause, and there are no more atoms in the state, return True
        if not extract_var(s):
            print(f'    No vars in the clause {s}') if verbose>1 else None
            if not state:
                print(f'        No more atoms in the state, {state}') if verbose>1 else None
                print(f'Actions: ["True"]') if verbose else None
                return ["True()"]
        # if there are vars in the clause, substitute the var with 'REPLACE'
        else:
            print(f'    Vars in the clause') if verbose>1 else None
            var = extract_var(s)
            s = s.replace(var, "REPLACE")
            # get the value of the query with the var substituted
            print(f'        Substituted vars:', s) if verbose>1 else None
            res = janus.query_once(f"clause({s}, _B), term_string(_B, B)")
            print(f'        Res:', res) if verbose>1 else None
            # if the query is not true, return False
            if not res["truth"]:
                print(f'        The query is false') if verbose>1 else None
                print(f'Actions: ["False"]') if verbose else None
                return ["False()"]
            else:
                # if the query is true, substitute the var with the value of the query
                print(f'        subsitution found; Lets go through the rest of atoms of the state') if verbose>1 else None
                for d in janus.query(f"clause({s}, _B), term_string(_B, B)"):
                    unification = d['REPLACE']
                    print(f'            unification {unification},  {d}') if verbose>1 else None
                    actions.append(";".join(state).replace(var, unification))
                    print(f'            appended action {";".join(state).replace(var, unification)}') if verbose>1 else None
    # res["truth"] is true, but B is not true, i.e., it is not a fact, it is a rule (so there's a body)
    else:
        # substitute the var with the value of the query
        print(f'There is a substitution, and it is not a fact. Apply the unification to the rest of atoms in the clause') if verbose>1 else None
        for d in janus.query(f"clause({s}, _B), term_string(_B, B)"):
            print(f'    atom {d}') if verbose>1 else None
            actions.append(d['B']+";".join(state))
            print(f'    appended action {actions[-1]}') if verbose>1 else None
    print(f'Actions: {actions}') if verbose else None
    return actions
    
def from_str_to_term(next_state_str: str) -> List[Term]:
    ''' Convert a string to a list of Term objects '''
    n_atoms = next_state_str.count(')') 
    atoms = []
    for i in range(n_atoms): # to split atoms, take until ')'
        atom_str = next_state_str.split(')')[i] + ')'
        if atom_str[0] == ',':
            atom_str = atom_str[1:]
        if atom_str == 'False()' or atom_str == 'True()':
            atoms.append(Term(atom_str[:-2], []))
            continue
        else:
            predicate, args = atom_str.split('(')
            args = args[:-1].split(',')
            atoms.append(Term(predicate, args))
    return atoms

def get_next_state_prolog(state: List[Term], verbose=0) -> Tuple[List[Term], bool, bool]:
    print('     state str:', ", ".join([str(atom) for atom in state])) if verbose>0 else None
    # Handle terminal states: If any of the atoms in the state are False, return False. If all the atoms in the state are True, return True
    any_atom_false = any([atom.predicate == 'False' for atom in state]) 
    all_atoms_true = all([atom.predicate == 'True' for atom in state])
    # print('Any atom False:', any_atom_false, 'All atoms True:', all_atoms_true)
    if any_atom_false:
        next_states = [[Term('False', [])]]
    elif all_atoms_true:
        next_states = [[Term('True', [])]]
    else:
        # Remove the True atoms from the next states and convert them to a list of strings
        state = [str(atom) for atom in state if atom.predicate != 'True']
        state = ", ".join([str(atom) for atom in state])
        next_states_str = get_actions_prolog(state, verbose=0)
        # Convert the list of strings to a list of Term object
        next_states = []
        for next_state_str in next_states_str:
            atoms = from_str_to_term(next_state_str)
            next_states.append(atoms)
    print('         Next states:', [[str(atom) for atom in state] for state in next_states]) if verbose>0 else None
    return next_states

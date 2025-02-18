import re
from collections import deque
from typing import List
from utils import Term
import janus_swi as janus


def get_actions_prolog(state: str, verbose: int=0) -> List[str]:

    print('State str:', state) if verbose else None
    state = deque(re.findall(r'\w+\(.*?\)', state))
    print('State deque:', state) if verbose else None
    # get the first element of the state
    s = state.popleft()
    print('State popleft:', s) if verbose else None
    actions = []
    # for digit query only
    # if s.startswith('digit(') and s:

    # We get 'truth' and 'B' from the query, which is whether the query is true or not, 
    # and the value of the query
    print('Query:', f"clause({s}, _B), term_string(_B, B)") if verbose else None
    res = janus.query(f"clause({s}, _B), term_string(_B, B).")

    res_dict = {}
    for d in res:
        if "truth" in res_dict:
            res_dict["truth"].append(d['truth'])
        else:
            res_dict["truth"] = [d['truth']]
        if "B" in d:
            body = d['B']
        else:
            body = None
        if "B" in res_dict:
            res_dict["B"].append(body)
        else:
            res_dict["B"] = [body]
    print('Res_dict:', res_dict) if verbose else None

    # If res["truth"] is false, it means the clause is not true
    if res_dict["truth"] == [False]:
        print(f'There is no substitution') if verbose else None
        print(f'Actions: ["False"]') if verbose else None
        return ["False()"]
    # the clause is a fact
    elif any(t and (b is None or b == "true") for t, b in zip(res_dict["truth"], res_dict["B"])) and not state:
        print(f'it is a fact') if verbose else None
        print(f'Actions: ["True"]') if verbose else None
        return ["True()"]
    # substitution step
    elif all(b == "true" for b in res_dict["B"]) and state:
        state_list = "[" + s + ", " + ", ".join(state) + "]"
        res = janus.query(f"proof_first({str(state_list)}, _T), term_string(_T, T)")
        for d in res:
            actions.append(d["T"][1:-1])
        print(f' possible actions are {"; ".join(actions)}') if verbose else None
    # rule-matching step
    else:
        if len(res_dict["B"]) == 1 and not re.findall(r'\w+\(.*?\)', res_dict["B"][0]):
            res = janus.query_once(f"{res_dict["B"][0]}.")["truth"]
            # TODO: might not work for reordering
            if res:
                print(f'Action: True') if verbose else None
                return ["True()"]
            else:
                print(f'Action: False') if verbose else None
                return ["False()"]
        else:
            actions = res_dict["B"]
            print(f' possible actions are {"; ".join(actions)}') if verbose else None
    return actions


def from_str_to_term(next_state_str: str) -> List[Term]:
    ''' Convert a string to a list of Term objects '''
    atom_strs = re.findall(r'\w+\(.*?\)', next_state_str)
    atoms = []
    for atom_str in atom_strs:
        if atom_str == 'False()' or atom_str == 'True()':
            atoms.append(Term(atom_str[:-2], []))
        else:
            predicate, args = atom_str.split('(')
            args = args[:-1].split(',')
            atoms.append(Term(predicate, args))
    return atoms


def get_next_state_prolog(state: List[Term], verbose=0) -> List[List[Term]]:
    print('     state str:', ", ".join([str(atom) for atom in state])) if verbose else None
    # Handle terminal states: If any of the atoms in the state are False, return False. If all the atoms in the state are True, return True
    any_atom_false = any([atom.predicate == 'False' for atom in state]) 
    all_atoms_true = all([atom.predicate == 'True' for atom in state])
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
            for atom in atoms:
                if atom.predicate == 'is_addition' and atom.args[2][0] == '_':
                    atom.args[2] = '_'
            next_states.append(atoms)
    print('         Next states:', [[str(atom) for atom in state] for state in next_states]) if verbose else None
    return next_states
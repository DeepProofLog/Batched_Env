import re
from collections import deque
from typing import List
from utils import Term
import janus_swi as janus
from collections import defaultdict


# When we have a query that takes too much inference or stack, we get the following warning:
    # _swipl.close_query(self.state)
# janus_swi.janus.PrologError: swipl.next_solution(): not inner query
# This is because in the query we are trying to unify a variable with a value that is not instantiated.
# There is no problem, but if we want to avoid the warning, we can modify the del method of the janus.query object

_orig_del = janus.query.__del__

def patched_del(self):
    try:
        _orig_del(self)
    except janus.janus.PrologError as e:
        if "not inner query" in str(e):
            # Suppress the warning
            pass
        else:
            raise e

janus.query.__del__ = patched_del


def filter_actions_prolog(res, verbose: int=0) -> List[str]:
    """
    Filter the actions from the prolog query result
    When it takes too many inference steps or stack, it raises a warning, so we avoid it
    Also, sometime it enters infinte loop of possible next actions, so we break it
    """

    actions = []
    try:
        while True:
            d = res.next()
            if d:
                # assert that there is only one state in each next action
                assert d["T"].count('[') == 1 and d["T"].count(']') == 1
                print(f'        atom {d}') if verbose else None
                if d["T"][1:-1] in actions:
                    print(f'        action {d} already in actions') if verbose else None
                    break
                actions.append(d["T"][1:-1])
            else:
                break
    except janus.janus.PrologError as e:
        if "Arguments are not sufficiently instantiated" in str(e):
            print(f'        end of valid solutions: {e}') if verbose else None
            pass
        else:
            print(f'        re-raise error: {e}') if verbose else None
            pass
    finally:
        res.close()
    return actions


def get_actions_prolog(state: str, verbose: int=0) -> List[str]:
    # janus.query_once("set_prolog_flag(stack_limit, 2147483648)")
    print('\n\n--------------\nState str:', state) if verbose else None
    state = deque(re.findall(r'\w+\(.*?\)', state))
    # get the first element of the state
    s = state.popleft()
    print('State popleft:', s,' State:', state) if verbose else None

    if not state:
        res = janus.query(
            f"clause({s}, _B), term_string(_B, B)"
            # f"catch((clause({s}, _B), term_string(_B, B)), error(resource_error(stack), _), false)"
            # f"call_with_inference_limit((clause({s}, _B), term_string(_B, B)), 1000000, R)"
            )
        
        res_dict = defaultdict(list)
        for d in res:
            body = d.get('B')
            truth = d.get('truth')
            # If there's only one action and it's a fact, return True action
            if truth == True and body == "true":
                print('It is a fact') if verbose else None
                print('Actions: ["True"]') if verbose else None
                actions = ["True()"]
            # elif truth == False:
            #     pass
            elif truth == True and body != "true":
                res_dict["truth"].append(truth)
                res_dict["B"].append(body)
        
        print('Res dict:', res_dict,'\n') if verbose else None

        if not res_dict:
            print('There is no action/substitution') if verbose else None
            print('Actions: ["False"]') if verbose else None
            actions = ["False()"]
        else:
            print('It is a rule match') if verbose else None
            print('Actions:', res_dict["B"]) if verbose else None
            actions = res_dict["B"]
        
    else:
        state_list = "[" + s + ", " + ", ".join(state) + "]"
        print('State list:', state_list) if verbose else None
        res = janus.query(
            f"proof_first({state_list}, _T), term_string(_T, T)"
            # f"catch((proof_first({state_list}, _T), term_string(_T, T)), error(resource_error(stack), _), false)"
            # f"call_with_inference_limit((proof_first({state_list}, _T), term_string(_T, T)), 100000000, R)"
            )
        actions = filter_actions_prolog(res, verbose=verbose)
        if not actions:
            print('Actions: ["False"]') if verbose else None
            actions = ["False()"]
        print(f' Actions: {actions}') if verbose else None
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


def get_next_unification(state: List[Term], verbose=0) -> List[List[Term]]:
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
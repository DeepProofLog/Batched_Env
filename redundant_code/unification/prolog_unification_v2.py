import re
from collections import deque
from typing import List, Tuple
from utils_v1 import extract_var, Term
import janus_swi as janus

def get_res_dict(res: List[dict]) -> dict:
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
    return res_dict

def get_actions_prolog(state: str, verbose: int=0) -> List[str]:

    state = deque(re.findall(r'\w+\(.*?\)', state))
    # get the first element of the state
    s = state.popleft()
    print('     State deque:', state) if verbose else None
    print('     State popleft:', s) if verbose else None
    actions = []
    

    if not state:
        # We get 'truth' and 'B' from the query, which is whether the query is true or not, 
        # and the value of the query
        print('     Query:', f"clause({s}, _B), term_string(_B, B)") if verbose else None
        res = janus.query(f"clause({s}, _B), term_string(_B, B).")
        res_dict = get_res_dict(res)
        print('     Res_dict:', res_dict) if verbose else None
        # if res is empty, it means there is no substitution
        if not res_dict:
            print(f'     There is no substitution') if verbose else None
            print(f'     Actions: ["False"]') if verbose else None
            return ["False()"]

        # If it's false, there'll be only one next state. res["truth"] is false, it means the clause is not true
        if res_dict["truth"] == [False]:
            print(f'     There is no substitution') if verbose else None
            print(f'     Actions: ["False"]') if verbose else None
            return ["False()"]
        

        # if any next state is a fact or True, return True
        if any(t==True and (b is None or b == "true") for t, b in zip(res_dict["truth"], res_dict["B"])) and not state:
            print(f'     it is a fact') if verbose else None
            print(f'     Actions: ["True"]') if verbose else None
            return ["True()"]
        
        # otherwise, return the possible actions
        elif all(b == True for b in res_dict["truth"]):
            actions = res_dict["B"]
            print(f'     possible actions are {"; ".join(actions)}') if verbose else None
        else:
            raise ValueError(f'     Error: This case should not happen')

    else:
        # prove the first atom and do a unification with the remaining atoms of the state
        state_list = "[" + s + ", " + ", ".join(state) + "]"
        print('     State list:', state_list) if verbose else None
        res = janus.query(f"proof_first({str(state_list)}, _T), term_string(_T, T)")
        for d in res:
            if d["T"] == "true": # if any next state is true, return True
                actions = ["True()"]
                break
            elif d["T"] == "false": # if any next state is false, continue, and if later it's empty, return False
                continue
            actions.append(d["T"][1:-1])
        if not actions:
            actions = ["False()"]
        print(f'     possible actions_ are {"; ".join(actions)}') if verbose else None
    
    
    # # rule-matching step
    # else:
    #     # print(' all(b == "true" for b in res_dict["B"])',res_dict["B"],[b == "true" for b in res_dict["B"]],all(b == "true" for b in res_dict["B"]),'state',state)
    #     if len(res_dict["B"]) == 1 and not re.findall(r'\w+\(.*?\)', res_dict["B"][0]):
    #         res = janus.query_once(f"{res_dict["B"][0]}.")["truth"]
    #         # TODO: might not work for reordering
    #         if res:
    #             print(f'     Action: True') if verbose else None
    #             return ["True()"]
    #         else:
    #             print(f'     Action: False') if verbose else None
    #             return ["False()"]
    #     else:
    #         actions = res_dict["B"]
    #         print(f'     possible actions are {"; ".join(actions)}') if verbose else None

    return actions

# def get_actions_prolog(state: str, verbose: int=0) -> List[str]:

#     print('State str:', state) if verbose else None
#     state = deque(re.findall(r'\w+\(.*?\)', state))
#     print('State deque:', state) if verbose else None
#     # get the first element of the state
#     s = state.popleft()
#     print('State popleft:', s) if verbose else None
#     actions = []
#     # We get 'truth' and 'B' from the query, which is whether the query is true or not, 
#     # and the value of the query
#     print('Query:', f"clause({s}, _B), term_string(_B, B)") if verbose else None
#     res = janus.query_once(f"clause({s}, _B), term_string(_B, B)")

#     print('Res:', res) if verbose else None
#     # If res["truth"] is false, it means the clause is not true
#     if not res["truth"]:
#         print(f'There is no substitution') if verbose else None
#         print(f'Actions: ["False"]') if verbose else None
#         return ["False()"]
#     # res["truth"] is true, i.e., there's a substitution
#     # If res["B"] is true, it means it is a fact
#     elif res["B"] == "true":
#         print(f'There is a substitution for the clause, and it is a fact') if verbose else None
#         # if there are no more atoms in the state, return True
#         if not state:
#             print(f'    No more atoms in the state, {state}') if verbose else None
#             print(f'Actions: ["True"]') if verbose else None
#             return ["True()"]
#         # still atoms to proof
#         else:
#             print(f'    Prolog for one-step proof and unification') if verbose else None
#             state_list = "[" + s + ", " + ", ".join(state) + "]"
#             res = janus.query(f"proof_first({str(state_list)}, _T), term_string(_T, T)")
#             for d in res:
#                 print(f'        atom {d}') if verbose else None
#                 actions.append(d["T"][1:-1])
#                 print(f'        appended action {actions[-1]}') if verbose else None
#     # res["truth"] is true, but B is not true, i.e., it is not a fact, it is a rule (so there's a body)
#     else:
#         # substitute the var with the value of the query
#         print(f'There is a substitution, and it is not a fact. Apply the unification to the rest of atoms in the clause') if verbose else None
#         for d in janus.query(f"clause({s}, _B), term_string(_B, B)"):
#             print(f'    atom {d}') if verbose else None
#             actions.append(d['B']+",".join(state))
#             print(f'    appended action {actions[-1]}') if verbose else None
#     print(f'Actions: {actions}') if verbose else None
#     return actions

def from_str_to_term(next_state_str: str) -> List[Term]:
    ''' Convert a string to a list of Term objects '''
    if next_state_str == 'true' or next_state_str == 'True' or next_state_str == 'True()':
        return [Term('True', [])]
    elif next_state_str == 'false' or next_state_str == 'False' or next_state_str == 'False()':
        return [Term('False', [])]
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


def get_next_unification_prolog(state: List[Term], verbose=0) -> List[List[Term]]:
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
        next_states_str = get_actions_prolog(state, verbose=1)
        # Convert the list of strings to a list of Term object
        next_states = []
        for next_state_str in next_states_str:
            atoms = from_str_to_term(next_state_str)
            if atoms[0].predicate == 'True':
                next_states = [[Term('True', [])]]
                break
            next_states.append(atoms)
    print('     Next states:', [[str(atom) for atom in state] for state in next_states],'\n') if verbose else None
    return next_states

def get_next_state_prolog_mnist(state: List[Term], verbose=0) -> List[List[Term]]:
    # Handle terminal states: If any of the atoms in the state are False, return False. If all the atoms in the state are True, return True
    any_atom_false = any([atom.predicate == 'False' for atom in state]) 
    all_atoms_true = all([atom.predicate == 'True' for atom in state])
    if any_atom_false:
        next_states = [[Term('False', [])]]
    elif all_atoms_true:
        next_states = [[Term('True', [])]]
    else:
        # merge the arguments 0 and 1 into a single argument str([arg0, arg1]) and the arguments 2 and 3 into a single argument str([arg2, arg3])
        # we had them splitted because it is easier for embedding treatment, but for prolog we need to merge them
        # Now we check that the length of the arguments is 5 for the predicate addition and 3 for the predicate digit, but with N-digits it will be 1+2N and 1+N
        for atom in state:
            if atom.predicate == 'addition':
                assert len(atom.args) == 5, f'Error: The length of the arguments of the atom {atom} is not 5 in predicate addition'
                atom.args = ['['+str(atom.args[0])+'-'+str(atom.args[1])+']', '['+str(atom.args[2])+'-'+str(atom.args[3])+']', atom.args[4]]
            elif atom.predicate == 'digit':
                assert len(atom.args) == 3, f'Error: The length of the arguments of the atom {atom} is not 3 in predicate digit'
                atom.args = ['['+str(atom.args[0])+'-'+str(atom.args[1])+']', atom.args[2]]
        print('\nState:', state) if verbose else None
        # Remove the True atoms from the next states and convert them to a list of strings
        state = [str(atom) for atom in state if atom.predicate != 'True']
        state = ", ".join([str(atom) for atom in state])
        print('State str:',state) if verbose else None
        next_states_str = get_actions_prolog(state, verbose=0)
        # Convert the list of strings to a list of Term object
        next_states = []
        for next_state_str in next_states_str:
            print('Next state str:', next_state_str) if verbose else None
            atoms = from_str_to_term(next_state_str)
            for atom in atoms: 
                new_args = []
                for i, arg in enumerate(atom.args):
                    if (arg.startswith('[') and arg.endswith(']')) and arg.count('-') == 1 and 'im_' in arg:
                        split = arg[1:-1].split('-')
                        new_args.append(split[0])
                        new_args.append(split[1])
                    else:
                        new_args.append(arg)
                atom.args = new_args
            next_states.append(atoms)
    print('Next states', next_states) if verbose else None
    return next_states

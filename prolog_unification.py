import re
import janus_swi as janus
from typing import List, Optional, Tuple
from utils import Term, get_atom_from_string
from python_unification import rename_vars, rename_vars_local

# counter = 0
# while counter<20:
#     print("\n\n"+"*"*50)
#     print(f'current state is {state}')
#     res = janus.query_once(f"one_step_list({state}, _NewGoalList), term_string(_NewGoalList, NewGoalList)")
#     print(res)
#     actions = re.findall(r'\[[^\[\]]*\]', res['NewGoalList'])
#     if any(a=="[]" for a in actions):
#         print("proof succeeded")
#         break
#     if any(a=="[false]" for a in actions):
#         print("proof failed")
#         break
#     else:
#         agent = random.choice(actions)
#         state = agent
#         counter += 1




def get_next_unification_prolog(
                    state: List[Term],
                    next_var_index: Optional[int] = None,
                    verbose: int = 0,
                    ) -> Tuple[List[List[Term]], Optional[int]]:

    # --- Initial Checks and Setup ---
    if not state: return [[Term('True', ())]], next_var_index
    if any(term.predicate == 'False' for term in state): return [[Term('False', ())]], next_var_index
    state = [term for term in state if term.predicate != 'True']
    if not state: return [[Term('True', ())]], next_var_index

    state_str = str(state)
    print('\n\n++++++++++++++') if verbose else None
    print(f'Processing Query: {state}') if verbose else None


    prolog_query = f"one_step_list({state_str}, _NewGoalList), term_string(_NewGoalList, NewGoalList)"
    res = janus.query_once(prolog_query)


    if not res:
        print(f"Warning: No result found for query: {prolog_query}") if verbose else None
        return [[Term('False', ())]], next_var_index
    
    if action_list_str.strip() == '[false]':
        print("Prolog indicated failure ([false])") if verbose else None
        return [[Term('False', ())]], next_var_index

    actions = re.findall(r'\[[^\[\]]*\]', res['NewGoalList'])
    print('Actions:', actions) if verbose else None

    # --- Convert actions to Term objects ---
    next_states: List[List[Term]] = []
    term_regex = re.compile(r'([a-zA-Z0-9_]+(?:\([^)]*\))?)') # More specific: non-greedy match inside parens

    for action_list_str in actions:
        # Remove leading '[' and trailing ']' and strip whitespace
        content_str = action_list_str.strip()[1:-1].strip()

        if not content_str: # Handle empty lists like '[]'
            raise ValueError(f"Empty action list found: {action_list_str}")
            next_states.append([])
            continue

        # Find all individual term strings within the content string
        term_strings = term_regex.findall(content_str)

        current_action_terms: List[Term] = []
        for term_str in term_strings:
             term_str_cleaned = term_str.strip()
             if term_str_cleaned and term_str_cleaned != ',': # Ensure it's not just a comma if regex slips
                try:
                    # Use the provided function to parse each term string
                    term = get_atom_from_string(term_str_cleaned)
                    current_action_terms.append(term)
                except ValueError as e:
                    print(f"Warning: Skipping term due to parsing error: {e} in string '{action_list_str}'")
                    # Optionally, decide how to handle errors: skip term, skip list, raise exception?

        next_states.append(current_action_terms)

    print('\nParsed Actions (List[List[Term]]):', next_states) if verbose else None



    # --- Rename variables in the next states ---

    next_states, next_var_index = rename_vars(next_states, next_var_index)
    # next_states = rename_vars_local(next_states, next_var_index, verbose=verbose)

    print('\nNext states:', next_states) if verbose else None
    print('++++++++++++++\n') if verbose else None



    # --- Handle terminal states ---

    # If any of the atoms in the state are False, return False
    any_atom_false = any(atom.predicate == 'False' for state in next_states for atom in state)
    if any_atom_false:
        return [[Term('False', ())]], next_var_index
    
    # If all the atoms in the state are True, return True
    all_atoms_true = all(atom.predicate == 'True' for state in next_states for atom in state)
    if all_atoms_true:
        return [[Term('True', ())]], next_var_index
    
    # Filter out True atoms from the next states
    next_states = [
        [term for term in state if term.predicate != 'True']
        for state in next_states
    ]
    return next_states, next_var_index
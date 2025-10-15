import re
import janus_swi as janus
from typing import List, Optional, Tuple
from utils import Term, get_atom_from_string
from python_unification import rename_vars_local


def get_next_unification_prolog(
                    state: List[Term],
                    next_var_index: Optional[int] = None,
                    verbose: int = 0,
                    ) -> Tuple[List[List[Term]], Optional[int]]:

    print('\n\n++++++++++++++') if verbose else None
    print(f'State: {state}') if verbose else None

    # --- Initial Checks and Setup ---
    if not state: return [[Term('True', ())]], next_var_index
    if any(term.predicate == 'False' for term in state): return [[Term('False', ())]], next_var_index
    state = [term for term in state if term.predicate != 'True']
    if not state: return [[Term('True', ())]], next_var_index

    state_str = '[' + ','.join(term.prolog_str() for term in state) + ']'

    prolog_query = f"one_step_list({state_str}, _NewGoalList), term_string(_NewGoalList, NewGoalList)"
    res = janus.query_once(prolog_query)
    result = res.get('NewGoalList', None)
    
    if not res:
        raise ValueError(f"Prolog query returned no results for query: {prolog_query}")
    elif result.strip() == '[false]':
        return [[Term('False', ())]], next_var_index

    actions = re.findall(r'\[[^\[\]]*\]', result)
    # print('Actions:', actions) if verbose else None
    
    # --- Convert actions to Term objects ---
    next_states: List[List[Term]] = []
    term_regex = re.compile(r'([a-zA-Z0-9_]+(?:\([^)]*\))?)') # More specific: non-greedy match inside parens
    # remove any " present in the string
    actions = [action.replace('"', '') for action in actions]

    for action_list_str in actions:
        # Remove leading '[' and trailing ']' and strip whitespace
        content_str = action_list_str.strip()[1:-1].strip()

        if not content_str: # For empty lists, a proof is found
            print("Empty action list found, indicating a proof.") if verbose else None
            return [[Term('True', ())]], next_var_index

        # Find all individual term strings within the content string
        term_strings = term_regex.findall(content_str)

        current_action_terms: List[Term] = []
        for term_str in term_strings:
             term_str_cleaned = term_str.strip()
             if term_str_cleaned and term_str_cleaned != ',': # Ensure it's not just a comma if regex slips
                try:
                    term = get_atom_from_string(term_str_cleaned)
                    current_action_terms.append(term)
                except ValueError as e:
                    print(f"Warning: Skipping term due to parsing error: {e} in string '{action_list_str}'")

        next_states.append(current_action_terms)

    # print('\nParsed Actions (List[List[Term]]):', next_states) if verbose else None


    # --- Rename variables in the next states ---

    # next_states, next_var_index = rename_vars(next_states, next_var_index)
    next_states = rename_vars_local(next_states, next_var_index, verbose=0)

    if verbose:
        ordered_next_states = [sorted(state, key=lambda term: (term.predicate, term.args)) for state in next_states]
        ordered_next_states = sorted(ordered_next_states, 
                                     key=lambda state: [(term.predicate, term.args) for term in state])
        print('\nNext states:', ordered_next_states)
        print('++++++++++++++\n')

    # --- Handle terminal states ---
    any_atom_false = any(atom.predicate == 'False' for state in next_states for atom in state)
    if any_atom_false:
        return [[Term('False', ())]], next_var_index
    
    all_atoms_true = all(atom.predicate == 'True' for state in next_states for atom in state)
    if all_atoms_true:
        return [[Term('True', ())]], next_var_index
    
    next_states = [
        [term for term in state if term.predicate != 'True']
        for state in next_states
    ]
    return next_states, next_var_index






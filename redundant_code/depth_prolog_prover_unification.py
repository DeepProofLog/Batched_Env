# Required imports
import time
import re
import os
from typing import List, Dict, Tuple, Optional, Any
from dataset import get_queries
# *** Import janus-swi ***
try:
    import janus_swi as janus
except ImportError:
    print("Error: janus-swi library not found.")
    print("Please install it: pip install janus-swi")
    exit(1)

# Assuming utils.py contains Term class and is_variable function
# Or define them here as in the previous example:
class Term:
    def __init__(self, predicate, args):
        self.predicate = predicate
        self.args = list(args) # Ensure args are mutable lists
    def __repr__(self):
        # Handle zero args case correctly
        if not self.args:
            return f"{self.predicate}"
        # Ensure args are strings for join
        args_str = ','.join(map(str, self.args))
        return f"{self.predicate}({args_str})"
    def __eq__(self, other):
        return isinstance(other, Term) and self.predicate == other.predicate and self.args == other.args
    def __hash__(self):
        # Args must be hashable, convert list to tuple
        return hash((self.predicate, tuple(self.args)))
    # Add a comparison method for sorting if needed (based on representation)
    def __lt__(self, other):
        if not isinstance(other, Term):
            return NotImplemented
        return repr(self) < repr(other)


def is_variable(arg: Any) -> bool:
  # Allow for non-string types that might come from Janus
  return isinstance(arg, str) and (arg[0].isupper() or arg[0] == '_')

# --- Helper Functions for Prolog Interaction (Modified for Janus) ---

def term_to_prolog_str(term: Term) -> str:
    """Converts a Python Term object to a Prolog string representation. (Still needed for query construction)"""
    # Ensure args are properly formatted (e.g., handle strings vs atoms if necessary)
    # For simplicity, assume args are directly usable as atoms/variables
    args_str = ",".join(map(str, term.args))
    return f"{term.predicate}({args_str})"

def state_to_prolog_list_str(state: List[Term]) -> str:
    """Converts a list of Python Term objects (a state) to a Prolog list string. (Still needed for query construction)"""
    if not state:
        return "[]"
    terms_str = [term_to_prolog_str(t) for t in state]
    return f"[{','.join(terms_str)}]"

def parse_prolog_term_string(term_str: str) -> Optional[Term]:
    """Parses a predicate(arg1,..,argN) or atom string into a Term object."""
    term_str = term_str.strip()
    if not term_str: return None

    # Handle specific markers if you used them in Prolog (like '$$failed$$')
    if term_str == '$$failed$$':
        return Term('False', []) # Represent Prolog internal failure as False

    # Regex for predicate(args) - adjust regex if needed for your exact atoms/predicates
    match = re.match(r'^([a-zA-Z0-9_]+)\s*\((.*)\)$', term_str)
    if match:
        predicate = match.group(1)
        args_str = match.group(2).strip()
        if not args_str:
             args = []
        else:
             # Simple split - treats args like _9152 as strings.
             # This might be okay, or you might want specific variable handling.
             args = [arg.strip() for arg in args_str.split(',')]
        return Term(predicate, args)

    # Match atoms (like 'true', 'false', or other predicates/constants)
    elif re.match(r'^[a-zA-Z0-9_]+$', term_str):
        # Handle boolean atoms explicitly
        if term_str.lower() == 'true': return Term('True', [])
        if term_str.lower() == 'false': return Term('False', [])
        # Treat other atoms (constants or predicates without args)
        return Term(term_str, [])
    else:
        # This case might catch malformed strings or things you don't expect
        print(f"Warning: Could not parse term string from Prolog: '{term_str}'")
        return None

# --- Modify parse_janus_term ---
def parse_janus_term(janus_term_data: Any) -> Optional[Term]:
    """
    Parses data returned by Janus into a Python Term object.
    EXPECTS STRINGS for complex/atomic terms due to Prolog grounding.
    """
    # The primary expected type is now string
    if isinstance(janus_term_data, str):
        # Use the dedicated string parser
        term = parse_prolog_term_string(janus_term_data)
        if term is None:
             # The warning was already printed inside parse_prolog_term_string
             pass
        return term
    # Handle other potential types Janus might return (less likely now)
    elif isinstance(janus_term_data, (int, float)):
         return Term(str(janus_term_data), []) # Represent number as atom?
    # Handle list case ONLY if Prolog might return the failure marker list, e.g., ['$$failed$$']
    elif isinstance(janus_term_data, list) and janus_term_data == ['$$failed$$']:
         return Term('False', [])
    # If other list structures are returned unexpectedly:
    elif isinstance(janus_term_data, list):
         print(f"Warning: Unexpected list structure received in parse_janus_term (expected strings): {janus_term_data}")
         return None
    else:
        # Catchall for other unexpected types
        print(f"Warning: Unhandled Janus data type in parse_janus_term: {type(janus_term_data)}, value: {janus_term_data}")
        return None


# --- Modify parse_janus_result_to_states ---
def parse_janus_result_to_states(janus_result_list: Optional[List[Any]]) -> Optional[List[List[Term]]]:
    """Parses the list result from janus.query_once (expects list of lists of strings)."""
    if janus_result_list is None:
        print("Warning: Received None result from Janus query.")
        return None # Indicate failure or absence of result

    if not isinstance(janus_result_list, list):
        print(f"Warning: Expected list result from Janus findall, got {type(janus_result_list)}")
        return None

    all_next_states = []
    # [[str, str, ...], [str, ...], ...]
    for janus_state_string_list in janus_result_list:
        # Expect inner element to be a list of strings
        if not isinstance(janus_state_string_list, list):
            # Handle the case where Prolog might return something else unexpected
            # Or maybe the special failure marker if it wasn't a list?
             if janus_state_string_list == '$$failed$$': # If marker isn't a list
                 all_next_states.append([Term('False', [])])
                 continue
             else:
                 print(f"Warning: Expected list for state representation, got {type(janus_state_string_list)}: {janus_state_string_list}")
                 continue # Skip malformed state representation

        # Handle special empty list case or failure marker list
        if not janus_state_string_list: # Empty list might mean 'true'? Check Prolog logic.
             # If Prolog's next_state can result in [], what does it mean? Success?
             # Let's assume empty list means success for now if not failure marker.
             # all_next_states.append([Term('True', [])])
             continue # Or maybe just skip empty states?
        elif janus_state_string_list == ['$$failed$$']: # Check for failure marker list
             all_next_states.append([Term('False', [])])
             continue
        elif janus_state_string_list == ['true']: # Handle explicit success state
             all_next_states.append([Term('True', [])])
             continue

        current_py_state = []
        valid_state = True
        for term_string in janus_state_string_list:
            term = parse_janus_term(term_string) # Call the revised parser
            if term:
                current_py_state.append(term)
            else:
                print(f"Error: Failed to parse term string '{term_string}' within state list.")
                valid_state = False
                break # Stop processing this state if one term fails

        if valid_state and current_py_state:
            # Deduplication happens later, sorting Terms might be needed if not already comparable
            all_next_states.append(current_py_state)
        # If loop finished but state is empty (e.g., was ['true'] originally?)
        # elif valid_state and not current_py_state:
             # This case might occur if the state was just ['true'] which is handled above


    # What if parsing leads to no valid states?
    # if not all_next_states and janus_result_list: # If we started with data but ended with none
    #     print("Warning: Parsing resulted in no valid next states.")
        # Return failure? Or empty list?
        # return [[Term('False', [])]]

    return all_next_states # Return list of valid states found


def query_prolog_for_next_states(state: List[Term], verbose: int = 0) -> Optional[List[List[Term]]]:
    """
    Queries the already loaded SWI-Prolog engine via Janus to get next states.

    Args:
        state: The current state as a list of Term objects.
        verbose: Verbosity level.

    Returns:
        A list of next possible states, or None if an error occurred during query/parsing.
    """
    state_str = state_to_prolog_list_str(state)
    # Query uses the 'find_next_states/2' predicate defined in the loaded .pl file
    # We expect 'AllNextStates' to be bound to the result of findall/3
    query_string = f"find_next_states({state_str}, AllNextStates)."

    if verbose > 1:
        print(f"Executing Janus query: {query_string}")

    try:
        # janus.query_once returns the variable bindings of the *first* solution found.
        # Since find_next_states/2 uses findall/3, it always succeeds exactly once,
        # binding AllNextStates to the list of all possible next states.
        result_bindings = janus.query_once(query_string)

        if verbose > 1:
            print(f"Janus raw result bindings: {result_bindings}")

        if result_bindings is None or result_bindings is janus.false: # Query failed or returned false.
            print(f"Janus query failed or returned false for state: {state_str}")
            # Failure likely means the state has no successors according to Prolog's next_state
            return [] # Return empty list, signifying no next states (failure branch)

        if 'AllNextStates' not in result_bindings:
            print(f"Error: 'AllNextStates' variable not found in Janus result: {result_bindings}")
            return None # Indicate an unexpected error

        # Extract the list of states bound to the variable
        janus_states_list = result_bindings['AllNextStates']

        # Parse this Python structure (list of lists representing states/terms)
        parsed_states = parse_janus_result_to_states(janus_states_list)
        return parsed_states

    except Exception as e:
        # Catch potential exceptions during the query or parsing
        print(f"An error occurred during Janus query or parsing for state {state_str}: {e}")
        import traceback
        traceback.print_exc()
        return None # Indicate error


# Renamed from get_next_unification_prolog
def get_next_unification_janus(state: List[Term], facts_for_simplification: List[Term], verbose: int = 0) -> List[List[Term]]:
    """
    Calculates the next possible states using the embedded SWI-Prolog engine via Janus.
    Applies similar pre- and post-processing as the previous versions.

    Args:
        state: The current state (list of Term objects).
        facts_for_simplification: List of known facts (Term objects) used ONLY for post-simplification.
        verbose: Verbosity level.

    Returns:
        A list of next possible states (list of lists of Term objects).
    """
    # --- Initial Checks and Setup (same as before) ---
    if not state: return [[Term('True', [])]]
    if any(term.predicate == 'False' for term in state): return [[Term('False', [])]]
    original_state_repr_for_log = repr(state) # Log before removing True
    state = [term for term in state if term.predicate != 'True']
    if not state: return [[Term('True', [])]]

    facts_set = frozenset(facts_for_simplification)

    # --- Variable Renaming (same as before) ---
    # (Reusing the renaming logic - important for complex queries)
    exclude = set()
    for term in state:
        for arg in term.args:
            if isinstance(arg, str) and arg.startswith('_'): exclude.add(arg)
    subs = {}
    var_counter = 0
    vars_in_state = set()
    for term in state:
        for arg in term.args:
            # is_variable should handle potential non-string types from previous Janus steps if applicable
            if is_variable(arg): vars_in_state.add(str(arg)) # Convert to string for set consistency
    for var in vars_in_state:
        while f'_{var_counter}' in exclude or f'_{var_counter}' in vars_in_state: var_counter += 1
        subs[var] = f'_{var_counter}'; var_counter += 1

    renamed_state = []
    for term in state:
        # Apply substitution carefully, handling non-string args if they occur
        new_args = [subs.get(str(arg), arg) if is_variable(arg) else arg for arg in term.args]
        renamed_state.append(Term(term.predicate, new_args))

    if verbose > 1: print(f"Renamed state for Janus: {renamed_state}")


    # --- Call Prolog via Janus ---
    if verbose: print(f'Querying Prolog via Janus for state: {renamed_state}')
    potential_next_states = query_prolog_for_next_states(renamed_state, verbose)
    if verbose: print(f'Janus returned states (parsed): {potential_next_states}')


    # --- Handle Query Results ---
    if potential_next_states is None: # Error occurred during Janus call/parsing
        print(f"Janus query/parse failed for {renamed_state}, returning [False] state.") if verbose else None
        return [[Term('False', [])]] # Indicate failure

    # Note: query_prolog_for_next_states now returns [] if Prolog finds no solutions,
    # so we check for empty list here instead of a specific string.
    if not potential_next_states: # Prolog found no solutions
        print(f"Prolog found no next states for {renamed_state}, returning [False] state.") if verbose else None
        return [[Term('False', [])]]


    # --- Final Post-processing (Simplification & Duplicate Removal - same as before) ---
    final_next_states = []
    processed_state_tuples = set()

    if verbose >= 1: print("\n--- Post-processing Janus results ---")

    for state_list in potential_next_states:
        # Log the state as returned by Janus/parsing
        original_parsed_state_repr = repr(state_list)

        # Check if Prolog returned success directly (parsed as [Term('True', [])])
        if state_list == [Term('True', [])]:
            print(f"  Janus result indicates [True]. Proof found!") if verbose >= 1 else None
            return [[Term('True', [])]] # Immediate return

        # Check for False (should be handled by parser, but double check)
        if state_list == [Term('False', [])]:
             print(f"  Janus result indicates [False]. Branch failed.") if verbose >= 1 else None
             continue # Skip this branch

        # Simplify the current state list by removing known ground facts
        simplified_state = []
        simplification_occurred = False
        for term in state_list:
            # Use the safer is_variable check
            is_ground = not any(is_variable(arg) for arg in term.args)
            # Check against the facts_set
            if is_ground and term in facts_set:
                if verbose >= 1: print(f"  Simplifying: Removing ground fact {term} from state {original_parsed_state_repr}")
                simplification_occurred = True
                continue # Skip adding this term
            else:
                simplified_state.append(term)

        # Check if simplification resulted in an empty list (proof found NOW)
        if not simplified_state:
            print(f"  State {original_parsed_state_repr} simplified to empty. Proof found!") if verbose >= 1 else None
            return [[Term('True', [])]] # Immediate return

        # Add unique, simplified state
        else:
            if simplification_occurred and verbose >= 1:
                 print(f"  State {original_parsed_state_repr} simplified to {simplified_state}")

            # Use tuple of sorted terms for canonical representation
            try:
                # Ensure terms are sortable (add __lt__ to Term class if needed)
                state_tuple = tuple(sorted(simplified_state))
                if state_tuple not in processed_state_tuples:
                    final_next_states.append(simplified_state)
                    processed_state_tuples.add(state_tuple)
            except TypeError as e:
                 print(f"  Post-processing: Error creating tuple for state {simplified_state} - {e}. Adding without duplicate check.")
                 final_next_states.append(simplified_state)


    if verbose: print('\n--- Final Next States (from Janus + Post-processing) ---')
    if not final_next_states:
        if verbose: print("  No viable next states remain after post-processing, returning [False]")
        return [[Term('False', [])]]
    else:
        if verbose:
             for i, s in enumerate(final_next_states): print(f"  State {i}: {s}")
             print('*************************\n')
        return final_next_states


# --- Changes in the main execution flow ---

# No longer needs prolog_file argument
def check_provability_at_depth(state: List[Term], n: int, facts_for_simplification: List[Term], verbose: bool = False) -> str:
    """
    Checks provability using the Janus-based unification function.
    Assumes Prolog code is already consulted via janus.consult().
    """
    print(f"\nChecking query {state} at depth {n} using Janus (embedded Prolog)...") if verbose else None
    if not state: return 'error_empty_query'

    current_states = [state]
    for depth in range(n):
        if verbose: print(f"Depth {depth}, Num Current States: {len(current_states)}")
        if verbose > 1: print(f"Current states: {current_states}")

        next_generation_states = []
        any_branch_proved = False

        # Limit number of states explored per depth? Optional.
        # MAX_STATES_PER_DEPTH = 1000
        # if len(current_states) > MAX_STATES_PER_DEPTH:
        #     print(f"Warning: Exceeding state limit ({MAX_STATES_PER_DEPTH}) at depth {depth}. Pruning.")
        #     current_states = current_states[:MAX_STATES_PER_DEPTH]

        for i, current_state in enumerate(current_states):
            if any(term.predicate == 'True' for term in current_state):
                any_branch_proved = True; break # Already proven
            if any(term.predicate == 'False' for term in current_state):
                 continue # Branch failed

            # *** CALL THE JANUS VERSION ***
            # No longer needs prolog_file path
            branch_next_states = get_next_unification_janus(
                current_state,
                facts_for_simplification, # Still needed for post-simplification
                verbose=0 # Keep verbose low for internal calls
            )
            if verbose > 1: print(f"  Janus results for state {i}: {branch_next_states}")

            # Check results (handle [True] for success)
            if any(s == [Term('True', [])] for s in branch_next_states):
                 if verbose: print('  Provable branch found!')
                 any_branch_proved = True
                 break # Stop processing states at this depth if proof found

            # Filter out failed branches ([False]) and limit complexity
            valid_next_states = [
                branch_state for branch_state in branch_next_states
                if branch_state != [Term('False', [])] and len(branch_state) <= 20
            ]
            next_generation_states.extend(valid_next_states)
            # End of loop for current_states

        if any_branch_proved:
            return 'provable'

        # Deduplicate next_generation_states (same logic as before)
        unique_next_gen_tuples = set()
        unique_next_generation_states = []
        for s in next_generation_states:
            try:
                s_tuple = tuple(sorted(s)) # Assumes Term is sortable
                if s_tuple not in unique_next_gen_tuples:
                    unique_next_generation_states.append(s)
                    unique_next_gen_tuples.add(s_tuple)
            except TypeError: unique_next_generation_states.append(s)

        if not unique_next_generation_states:
            print('not_provable (no valid next states)') if verbose else None
            return 'not_provable'

        if depth == n - 1:
            print('depth_limit_exceeded') if verbose else None
            return 'depth_limit_exceeded'

        current_states = unique_next_generation_states

    print('error_should_not_arrive_here')
    return 'error_should_not_arrive_here'


# No longer needs prolog_file argument
def calculate_provability_ratios_by_depth(
    queries: list[Term],
    max_depth_check: int,
    facts: list[Term], # Keep facts for simplification
    is_train_data: bool = False, # Still has the same limitation
) -> tuple[dict[int, float], dict[str, int]] | None:
    """Calculates ratios using the Janus engine via check_provability_at_depth."""
    start_time = time.time()

    num_valid_queries = len(queries)
    if num_valid_queries == 0: return {}, {}

    print(f"Proving {num_valid_queries} valid queries up to depth {max_depth_check} using Janus...")

    min_proven_depth = [float('inf')] * num_valid_queries
    query_errors = [False] * num_valid_queries
    ratios_by_depth = {}
    proved_queries = {}

    if is_train_data:
        print("Warning: 'is_train_data' is set, but fact filtering is NOT performed when using the Prolog engine.")

    for depth in range(1, max_depth_check + 1):
        proven_count_cumulative = sum(1 for m_depth in min_proven_depth if m_depth != float('inf'))
        error_count_cumulative = sum(1 for err in query_errors if err)
        queries_checked_this_round = 0
        queries_newly_proven = 0
        queries_newly_errored = 0

        print(f"\n--- Processing Depth {depth}/{max_depth_check} ---")

        for i, goal in enumerate(queries):
            print(f"\r  Query {i+1}/{num_valid_queries} (Proven: {proven_count_cumulative}, Errors: {error_count_cumulative})", end='')

            if min_proven_depth[i] == float('inf') and not query_errors[i]:
                queries_checked_this_round += 1
                # *** CALL check_provability_at_depth (no prolog_file needed) ***
                status = check_provability_at_depth(
                    [goal],
                    depth,
                    facts, # Pass facts for post-simplification
                    verbose=2
                )

                if status == 'provable':
                    min_proven_depth[i] = depth
                    proved_queries[str(goal)] = depth
                    queries_newly_proven += 1
                    proven_count_cumulative += 1 # Update cumulative count
                elif status.startswith('error_'):
                    query_errors[i] = True
                    queries_newly_errored += 1
                    error_count_cumulative += 1 # Update cumulative count

        print(f"\rDepth {depth} complete. Checked: {queries_checked_this_round}, Newly Proven: {queries_newly_proven}, Newly Errored: {queries_newly_errored}.       ") # Spaces to clear line

        # Calculate and store cumulative ratio
        # Use updated proven_count_cumulative
        ratios_by_depth[depth] = proven_count_cumulative / num_valid_queries if num_valid_queries > 0 else 0.0

        # Early exit check
        if all((m_depth != float('inf') or err) for m_depth, err in zip(min_proven_depth, query_errors)):
            print(f"\nAll non-error queries resolved or errored by depth {depth}. Stopping.")
            max_depth_check = depth # Adjust max depth for reporting
            break

    # Final summary (remains largely the same)
    total_provable_count = sum(1 for m_depth in min_proven_depth if m_depth != float('inf'))
    total_errors = sum(1 for err in query_errors if err)
    end_time = time.time()

    print(f"\n--- Calculation Summary (Janus Engine) ---")
    print(f"Total valid queries processed: {num_valid_queries}")
    # ... rest of summary print ...
    print(f"Max depth checked: {max_depth_check}")
    print(f"Queries provable: {total_provable_count} ({total_provable_count/num_valid_queries:.2%})" if num_valid_queries else "")
    print(f"Queries resulting in errors: {total_errors} ({total_errors/num_valid_queries:.2%})" if num_valid_queries else "")
    print(f"Execution time: {end_time - start_time:.2f} seconds")


    return ratios_by_depth, proved_queries


# # --- Main Execution Block Example (Modified) ---
# if __name__ == "__main__":

#     max_depth_check = 4

#     # dataset = 'kinship_family'
#     dataset = 'countries_s3'
#     set_file = 'test'
#     root_dir = './data/' + dataset + '/'
#     prolog_knowledge_base_file = root_dir + 'knowledge_base.pl' # Same .pl file needed
#     prolog_knowledge_base_file = root_dir + 'kb_minimal.pl' # Same .pl file needed
#     prolog_knowledge_base_file = root_dir + 'test_consult.pl' # Same .pl file needed
#     prolog_knowledge_base_file = os.path.abspath(prolog_knowledge_base_file) # Ensure absolute path

#     # *** Initialize Janus and Consult the Prolog File ONCE ***
#     try:
#         print(f"Consulting Prolog file: {prolog_knowledge_base_file}")
#         # Ensure the file exists before consulting
#         if not os.path.exists(prolog_knowledge_base_file):
#              print(f"Error: Prolog file not found at {prolog_knowledge_base_file}")
#              exit(1)
#         # Use janus.consult to load the file into the embedded engine
#         consult_result = janus.consult(prolog_knowledge_base_file)
#         # Check result? janus.consult might return True/False or raise exception
#         if consult_result is not True:
#              print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#              print(f"CRITICAL Error: Prolog consult FAILED. Result: {consult_result}")
#              print(f"Check Prolog syntax errors reported above OR ensure the .pl file content is correct.")
#              print(f"Cannot proceed without a valid knowledge base.")
#              print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
#              # Make sure the script STOPS if consult fails
#              exit(1)

#         # This line should only print if consult_result was True
#         print("Prolog file consulted successfully via Janus.")

#     except NameError:
#          print("Error: 'janus' object not available. Import failed?")
#          exit(1)
#     except Exception as e:
#         print(f"Error during janus.consult: {e}")
#         exit(1)


#     # Load facts for Python-based simplification step
#     facts_file = root_dir + 'train.txt'
#     facts_for_simplification = get_queries(facts_file)
#     print(f"Loaded {len(facts_for_simplification)} facts from {facts_file}")

#     # Load queries
#     queries_file = root_dir + set_file + '.txt'
#     print(f"Loading queries from {queries_file}...")
#     if not os.path.exists(queries_file):
#             print(f"Error: Queries file not found at {queries_file}")
#             exit(1)
#     queries = get_queries(queries_file)
#     print(f"Loaded {len(queries)} queries from {queries_file}")
#     queries = queries[:500]

#     # *** CALL the modified function (no prolog_file needed) ***
#     provability_ratios, proved_queries = calculate_provability_ratios_by_depth(
#         queries,
#         max_depth_check=max_depth_check,
#         facts=facts_for_simplification, # Pass facts for simplification
#         is_train_data=('train' in set_file),
#     )

#     # Print results (remains the same)
#     print(f"\n--- Provability Ratios by Depth (up to {max_depth_check}, Janus Engine) ---")
#     if not provability_ratios:
#        print("No ratios calculated.")
#     else:
#         final_denominator = len(queries)
#         if final_denominator == 0: final_denominator = 1
#         last_depth_calculated = max(provability_ratios.keys()) if provability_ratios else 0
#         effective_max_depth = min(max_depth_check, last_depth_calculated) if last_depth_calculated > 0 else 0

#         for depth in range(1, effective_max_depth + 1):
#            ratio = provability_ratios.get(depth, 0.0)
#            count = int(ratio * final_denominator)
#            print(f"Depth <= {depth:<3}: {ratio:.4f} ({count}/{final_denominator})")


#     print(f"\n--- Proven Queries (Janus Engine) ---")
#     sorted_proved = sorted(proved_queries.items(), key=lambda item: (item[1], item[0]))
#     print([(query, depth) for query, depth in sorted_proved])

#     # Optional: Shutdown Janus/Prolog? Not usually needed for scripts.
#     # janus.cleanup() might exist depending on the version/API.











if __name__ == "__main__":

    # *** Temporarily define the path to the ultra-minimal test file ***
    test_prolog_file_name = 'test_consult.pl'
    # Assuming it's in the same directory as the script:
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir += '/data/countries_s3/'
    script_dir = './data/countries_s3/'
    print(f"Script directory: {script_dir}")
    print(f"files in script directory: {os.listdir(script_dir)}")
    test_prolog_file_path = os.path.join(script_dir, test_prolog_file_name)
    # If it's elsewhere, adjust the path accordingly:
    # test_prolog_file_path = os.path.abspath('./path/to/test_consult.pl')

    # *** Initialize Janus and Consult the TEST Prolog File ***
    try:
        print(f"Attempting to consult ULTRA-MINIMAL test file: {test_prolog_file_path}")
        if not os.path.exists(test_prolog_file_path):
            print(f"Error: Test Prolog file not found at {test_prolog_file_path}")
            exit(1)

        # Quick check if file is empty
        if os.path.getsize(test_prolog_file_path) == 0:
             print(f"Error: Test Prolog file '{test_prolog_file_path}' is empty.")
             exit(1)

        # Use janus.consult to load the TEST file
        consult_result = janus.consult(test_prolog_file_path)

        # *** CRITICAL CHECK (Mandatory - Keep Active) ***
        if consult_result is not True:
             print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print(f"CRITICAL Error: ULTRA-MINIMAL test Prolog consult FAILED. Result: {consult_result}")
             print(f"This suggests a fundamental issue with janus.consult, SWI-Prolog, file access, or environment.")
             print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
             # Make sure the script STOPS if consult fails
             exit(1)

        # This line will only be reached if consult_result was True
        print("\n********************************************************")
        print(f"SUCCESS: Ultra-minimal test file '{test_prolog_file_name}' consulted successfully!")
        print("This means janus.consult CAN work in your environment.")
        print("The problem likely lies specifically within the content of your knowledge base file.")
        print("********************************************************\n")

        # *** Exit after the test ***
        print("Exiting script after successful test consult.")
        exit(0)

    except NameError:
        print("Error: 'janus' object not available. Import failed?")
        exit(1)
    except Exception as e:
        print(f"Error during janus.consult of test file: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for consult errors
        exit(1)

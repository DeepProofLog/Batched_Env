import janus_swi as janus
import os
import re
import time # To measure execution time

def check_provability_at_depth(prolog_goal: str, n: int) -> str:
    """
    Checks if a single Prolog goal is provable within depth n.

    Helper function using call_with_depth_limit.

    Args:
        prolog_goal: The cleaned Prolog goal (string, no trailing period).
        n: The maximum depth (integer >= 0).

    Returns:
        'provable', 'not_provable', 'depth_limit_exceeded', or an error string.
    """
    if not prolog_goal:
        return 'error_empty_query'

    # depth_query = f"call_with_depth_limit(({prolog_goal}), {n}, DepthResult)."
    depth_query = (
    f"call_with_time_limit({10}, "
    f"call_with_depth_limit(({prolog_goal}), {n}, DepthResult))"
    )
    label = 'not_provable' # Default assumption

    try:
        result_dict = janus.query_once(depth_query)
        if result_dict:
            prolog_result = result_dict.get('DepthResult')
            if isinstance(prolog_result, int):
                label = 'provable' # Proven within depth n
                # if prolog_result > n:
                    # print(f"Warning: Goal '{prolog_goal}' proven at depth {prolog_result} > max_depth {n}")
            elif prolog_result == 'depth_limit_exceeded':
                label = 'depth_limit_exceeded' # Hit limit, not proven within n
            else:
                # This case might occur if DepthResult is bound to something unexpected
                print(f"Warning: Unexpected result '{prolog_result}' bound to DepthResult for goal '{prolog_goal}' at depth {n}")
                label = 'error_unexpected_result'
        else:
            # call_with_depth_limit failed. Goal not provable within depth n.
            label = 'not_provable'
    except janus.PrologError as e:
         # Catch errors specifically if needed
         print(f"Prolog error processing goal '{prolog_goal}' at depth {n}: {e}")
         label = 'error_prolog' # Generic Prolog error
    except Exception as e:
         # Catch other potential errors (e.g., Python issues in janus)
         print(f"Python/Janus error processing goal '{prolog_goal}' at depth {n}: {e}")
         label = 'error_python'

    return label


def calculate_provability_ratios_by_depth(
    queries: list[str],
    max_depth_check: int,
    prolog_file: str, # IMPORTANT: Point this to the file WITHOUT table directives
    is_training_data: bool = False # Flag to enable leave-one-out behavior
) -> tuple[dict[int, float] | None, dict[str, int] | None]:
    """
    Calculates the ratio of provable queries for each depth from 1 to max_depth_check.
    If is_training_data is True, it temporarily retracts the current query before checking
    its provability and asserts it back afterwards (leave-one-out).

    Args:
        queries: A list of cleaned Prolog query strings (no trailing period).
        max_depth_check: The maximum depth to check.
        prolog_file: Path to the Prolog file (.pl) - WITHOUT table directives.
                     Predicates corresponding to training queries MUST be declared dynamic.
        is_training_data: If True, perform leave-one-out fact retraction/assertion.

    Returns:
        A tuple containing:
        - A dictionary where keys are depths (1 to max_depth_check) and
          values are the ratio (0.0 to 1.0) of queries provable at or
          below that depth. Returns None if setup fails or critical error occurs.
        - A dictionary mapping proven query strings to their minimum proof depth (index starting at 0).
          Returns None on error.
    """
    start_time = time.time()
    if not os.path.exists(prolog_file):
        print(f"ERROR: Prolog file not found: {prolog_file}. CWD: {os.getcwd()}")
        return None, None

    num_valid_queries = len(queries)
    if num_valid_queries == 0:
        print("No valid queries found.")
        return {}, {}

    print(f"Processing {num_valid_queries} valid queries up to depth {max_depth_check}...")
    print(f"Using Prolog file (expecting no tabling): {prolog_file}")
    if is_training_data:
        print("INFO: Running in TRAINING mode (leave-one-out fact retraction enabled).")
        print("      Ensure relevant predicates are declared dynamic in the Prolog file.")
    else:
        print("INFO: Running in standard mode (no fact retraction).")

    # 2. Initialize results storage
    min_proven_depth = [float('inf')] * num_valid_queries
    query_errors = [False] * num_valid_queries
    ratios_by_depth = {}
    proved_queries = {} # Dictionary to store proven queries and their depth index

    try:
        # 3. Consult Prolog file ONCE
        janus.consult(prolog_file)
        print(f"Consulted Prolog file: {prolog_file}")

        # 4. Iterate through depths and queries
        for depth in range(1, max_depth_check + 1):
            proven_in_this_round = 0
            errors_in_this_round = 0
            queries_checked_this_round = 0

            for i, goal in enumerate(queries):
                # Print progress percentage
                print(f"\rProcessing Up to depth {depth}/{max_depth_check}, query {i+1}/{len(queries)}. Proven in this depth {proven_in_this_round}   ", end='')

                if min_proven_depth[i] == float('inf') and not query_errors[i]:
                    queries_checked_this_round += 1
                    goal_term = f"({goal})" # Prepare term for assert/retract

                    retracted_successfully = False # Flag for cleanup logic
                    perform_retract_assert = is_training_data # Decide if retract/assert needed

                    # --- Leave-one-out Logic ---
                    if perform_retract_assert:
                        try:
                            # Ensure the predicate is dynamic (user responsibility in .pl file)
                            retract_query = f"retract({goal_term})."
                            # Use query_once for actions with side-effects, ignore result
                            janus.query_once(retract_query)
                            retracted_successfully = True
                        except janus.PrologError as e:
                            print(f"\nWarning: Prolog error retracting '{goal}': {e}. Marking query as error.")
                            query_errors[i] = True
                            errors_in_this_round += 1
                            # Skip provability check if retraction failed
                            continue # Skip to the next query
                        except Exception as e:
                            print(f"\nWarning: Python error during retract setup for '{goal}': {e}. Marking query as error.")
                            # traceback.print_exc() # Uncomment for detailed Python trace
                            query_errors[i] = True
                            errors_in_this_round += 1
                            continue # Skip to the next query
                    # --- End Leave-one-out Retraction ---

                    # Assume check will proceed unless retraction failed
                    status = 'error_unknown' # Default if skipped somehow

                    # Perform the actual provability check
                    if not query_errors[i]: # Only check if not already marked as error
                         # Ensure check_provability_at_depth is defined correctly elsewhere
                         status = check_provability_at_depth(goal, depth)

                    # --- Leave-one-out Cleanup ---
                    if perform_retract_assert and retracted_successfully:
                        try:
                            assert_query = f"asserta({goal_term})."
                            # Use query_once for actions with side-effects, ignore result
                            janus.query_once(assert_query)
                        except janus.PrologError as e:
                            # This is potentially more problematic for subsequent queries
                            print(f"\nCRITICAL WARNING: Prolog error asserting back '{goal}': {e}. KB might be inconsistent. Marking query as error.")
                            query_errors[i] = True
                            # Count as error even if check succeeded before assert failed
                            if not status.startswith('error_'): errors_in_this_round += 1
                        except Exception as e:
                            print(f"\nCRITICAL WARNING: Python error asserting back '{goal}': {e}. Marking query as error.")
                            # traceback.print_exc() # Uncomment for detailed Python trace
                            query_errors[i] = True
                            if not status.startswith('error_'): errors_in_this_round += 1
                    # --- End Leave-one-out Assertion ---


                    # Process the status from check_provability_at_depth
                    # (only if no critical error happened during retract/assert phases)
                    if not query_errors[i]:
                        if status == 'provable':
                            min_proven_depth[i] = depth
                            # Store depth index (0-based for steps)
                            proved_queries[str(goal)] = depth - 1
                            proven_in_this_round += 1
                        elif status.startswith('error_'):
                            # Mark error if check_provability failed
                            query_errors[i] = True
                            errors_in_this_round += 1
                        # If 'not_provable' or 'depth_limit_exceeded', do nothing, try next depth

            # --- End of Query Loop for this Depth ---
            print() # Newline after progress indicator
            provable_count_at_or_below_d = sum(1 for m_depth in min_proven_depth if m_depth <= depth)
            print(f"Depth {depth}: Checked {queries_checked_this_round} pending queries.")
            print(f"         Newly proven: {proven_in_this_round}. Errors this depth: {errors_in_this_round}.")
            print(f"         Cumulative proven <= depth {depth}: {provable_count_at_or_below_d}/{num_valid_queries}")
            print()

            # Calculate and store cumulative ratio for this depth
            denominator = num_valid_queries
            ratios_by_depth[depth] = provable_count_at_or_below_d / denominator if denominator > 0 else 0.0

            # Optional: Early exit check
            if all( (m_depth != float('inf') or err) for m_depth, err in zip(min_proven_depth, query_errors) ):
                print(f"All non-error queries resolved or errored by depth {depth}. Stopping depth iteration.")
                max_depth_check = depth # Adjust for final report
                break # Stop checking further depths

        # --- End of Depth Loop ---

        # 5. Final Summary Calculation
        total_provable_count = sum(1 for m_depth in min_proven_depth if m_depth != float('inf'))
        total_errors = sum(1 for err in query_errors if err)

    except janus.PrologError as e:
        print(f"\nProlog error during setup (e.g., consulting file '{prolog_file}'): {e}")
        return None, None
    except Exception as e:
        print(f"\nGeneral Python error during setup or execution: {e}")
        traceback.print_exc() # Print stack trace for debugging general errors
        return None, None
    finally:
        # Optional: Clean up Prolog state if needed, e.g., janus.cleanup()
        # Be cautious if janus state is needed elsewhere
        pass

    end_time = time.time()
    # --- Final Summary Print ---
    print(f"\n--- Calculation Summary ---")
    print(f"Total valid queries processed: {num_valid_queries}")
    print(f"Mode: {'Training (Leave-One-Out)' if is_training_data else 'Standard'}")
    print(f"Queries provable within depth {max_depth_check}: {total_provable_count} ({total_provable_count/num_valid_queries:.2%} if num_valid_queries else 0)")
    print(f"Queries that resulted in errors (check/retract/assert): {total_errors} ({total_errors/num_valid_queries:.2%} if num_valid_queries else 0)")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    return ratios_by_depth, proved_queries

# --- Example Usage ---

MAX_DEPTH_TO_ANALYZE = 5 # Keep it reasonable first, might be slow


# 1. Load the rules and facts from the file
dataset = 'wn18rr'
# dataset = 'family'
# dataset = 'countries_s3'

for set_file in ['train']: 
    root_dir = './data/' + dataset + '/'
    queries_file = root_dir + set_file + '.txt'
    is_train_data = 'train' in set_file
    prolog_kb_file = './data/'+dataset+'/'+dataset+'.pl'


    # Check if files exist
    if not os.path.exists(prolog_kb_file):
        print(f"ERROR: Prolog KB file (NO TABLES VERSION) not found at {prolog_kb_file}")
        print("Please create it by copying the original and removing ':- table' directives.")
        exit()
    if not os.path.exists(queries_file):
        print(f"ERROR: Queries file not found at {queries_file}")
        exit()


    # 2. Load queries (same as before)
    try:
        with open(queries_file, 'r') as f:
            raw_queries = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading queries from {queries_file}: {e}")
        exit()

    queries = []
    for i, q in enumerate(raw_queries):
        prolog_goal = q.strip()
        if prolog_goal.endswith('.'):
            prolog_goal = prolog_goal[:-1].strip()
        if prolog_goal:
            queries.append(prolog_goal)
        else:
            print(f"Warning: Skipping empty or invalid query at index {i}: '{q}'")

    # queries = queries[:500]

    # 3. Run the analysis function
    print('\nCurrent working directory:', os.getcwd())
    provability_ratios, proved_queries = calculate_provability_ratios_by_depth(
        queries,
        MAX_DEPTH_TO_ANALYZE,
        prolog_kb_file, # Pointing to the NO TABLES version
        is_training_data = is_train_data
    )

    # 4. Print results (same as before, includes None check)
    if provability_ratios is not None:
        print(f"\n--- Provability Ratios by Depth (up to {MAX_DEPTH_TO_ANALYZE}) ---")
        if not provability_ratios:
            print("No ratios were calculated (possibly due to errors or no depths checked).")
        else:
            final_denominator = len(queries)
            if final_denominator == 0: final_denominator = 1

            # Adjust last depth based on early exit
            last_depth_calculated = max(provability_ratios.keys()) if provability_ratios else 0
            # Ensure max_depth_check reflects actual last calculated depth if early exit occurred
            effective_max_depth = min(MAX_DEPTH_TO_ANALYZE, last_depth_calculated) if last_depth_calculated > 0 else 0

            for depth in range(1, effective_max_depth + 1):
                ratio = provability_ratios.get(depth, 0.0)
                count = int(ratio * final_denominator)
                print(f"Depth <= {depth:<3}: {ratio:.4f} ({count}/{final_denominator})")
    else:
        print("\nAnalysis could not be completed due to setup errors or critical failure.")

    # 5. Save proven queries to a file
    if proved_queries:
        with open(root_dir + set_file + '_depths.txt', 'w') as f:
            for query in queries:

                clean_query = str(query).replace(' ', '')
                if str(query) in proved_queries:
                    f.write(f"{clean_query} {proved_queries[str(query)]}\n")
                else:
                    f.write(f"{clean_query} -1\n")
    print(f"Saved proven queries to {root_dir + set_file + '_depths.txt'}")
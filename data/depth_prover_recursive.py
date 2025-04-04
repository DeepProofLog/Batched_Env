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

    depth_query = f"call_with_depth_limit(({prolog_goal}), {n}, DepthResult)."
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
    prolog_file: str # IMPORTANT: Point this to the file WITHOUT table directives
) -> dict[int, float] | None:
    """
    Calculates the ratio of provable queries for each depth from 1 to max_depth_check.
    Assumes the prolog_file does NOT use tabling to avoid conflicts with call_with_depth_limit.

    Args:
        queries: A list of raw Prolog query strings.
        max_depth_check: The maximum depth to check (e.g., 100).
        prolog_file: Path to the Prolog file (.pl) - WITHOUT table directives.

    Returns:
        A dictionary where keys are depths (1 to max_depth_check) and
        values are the ratio (0.0 to 1.0) of queries provable at or
        below that depth. Returns None if setup fails or critical error occurs.
    """
    start_time = time.time()
    if not os.path.exists(prolog_file):
        raise FileNotFoundError(f"Prolog file not found: {prolog_file}. CWD: {os.getcwd()}")

    # 1. Clean and validate queries upfront
    cleaned_queries = []
    original_indices = []
    for i, q in enumerate(queries):
        prolog_goal = q.strip()
        if prolog_goal.endswith('.'):
            prolog_goal = prolog_goal[:-1].strip()
        if prolog_goal:
            cleaned_queries.append(prolog_goal)
            original_indices.append(i)
        else:
            print(f"Warning: Skipping empty or invalid query at index {i}: '{q}'")

    num_valid_queries = len(cleaned_queries)
    if num_valid_queries == 0:
        print("No valid queries found.")
        return {}

    print(f"Processing {num_valid_queries} valid queries up to depth {max_depth_check}...")
    print(f"Using Prolog file (expecting no tabling): {prolog_file}")


    # 2. Initialize results storage
    min_proven_depth = [float('inf')] * num_valid_queries
    query_errors = [False] * num_valid_queries
    # provable_status_at_max_depth = [False] * num_valid_queries # Can derive from min_proven_depth

    ratios_by_depth = {}

    try:
        # 3. Consult Prolog file ONCE
        janus.consult(prolog_file)
        print(f"Consulted Prolog file: {prolog_file}")

        # 4. Iterate through depths and queries
        for depth in range(1, max_depth_check + 1):
            proven_in_this_round = 0
            errors_in_this_round = 0
            queries_checked_this_round = 0

            # REMOVED: janus.query_once("abolish_all_tables.") - Not needed if tabling is off

            for i, goal in enumerate(cleaned_queries):
                # Print progress percentage instead of a bar
                print(f"\rProcessing Up to depth {depth}({max_depth_check+1}), query {i+1}/{len(cleaned_queries)}. Proven in this depth {proven_in_this_round}  ",end='')
                
                if min_proven_depth[i] == float('inf') and not query_errors[i]:
                    queries_checked_this_round += 1
                    status = check_provability_at_depth(goal, depth) # Use the helper

                    if status == 'provable':
                        min_proven_depth[i] = depth
                        # provable_status_at_max_depth[i] = True # No longer needed
                        proven_in_this_round += 1
                    elif status.startswith('error_'):
                        query_errors[i] = True
                        errors_in_this_round += 1
                    # If 'not_provable' or 'depth_limit_exceeded', do nothing yet
            print()
            print(f"Depth {depth}: Checked {queries_checked_this_round} queries. Newly proven: {proven_in_this_round}. Errors: {errors_in_this_round}. Proven under this depth: {sum(1 for m_depth in min_proven_depth if m_depth <= depth)}")

            # Calculate and store cumulative ratio for this depth
            provable_count_at_or_below_d = sum(1 for m_depth in min_proven_depth if m_depth <= depth)
            denominator = num_valid_queries
            ratios_by_depth[depth] = provable_count_at_or_below_d / denominator if denominator > 0 else 0.0

            # Optional: Early exit
            if all( (m_depth != float('inf') or err) for m_depth, err in zip(min_proven_depth, query_errors) ):
                 print(f"All non-error queries found their minimum provable depth or errored by depth {depth}. Stopping depth iteration.")
                 # Update max_depth_check so the final report only goes up to 'depth'
                 max_depth_check = depth
                 break # Stop checking further depths

        # 5. Final Summary Calculation
        total_provable_count = sum(1 for m_depth in min_proven_depth if m_depth != float('inf'))
        total_errors = sum(1 for err in query_errors if err)

    except janus.PrologError as e:
         print(f"Prolog error during setup (e.g., consulting file '{prolog_file}'): {e}")
         return None
    except Exception as e:
         print(f"General error during setup or execution: {e}")
         return None

    end_time = time.time()
    # --- Final Summary Print --- (copy from previous version)
    print(f"\n--- Calculation Summary ---")
    # ... (rest of summary print) ...
    print(f"Total valid queries processed: {num_valid_queries}")
    print(f"Queries provable within depth {max_depth_check}: {total_provable_count} ({total_provable_count/num_valid_queries:.2%} if num_valid_queries else 0)")
    print(f"Queries that resulted in errors: {total_errors} ({total_errors/num_valid_queries:.2%} if num_valid_queries else 0)")
    print(f"Execution time: {end_time - start_time:.2f} seconds")


    return ratios_by_depth

# --- Example Usage ---

dataset = 'kinship_family'

# 1. Set up paths and parameters
# !!! IMPORTANT: Create this file by copying kinship_family.pl and removing/commenting out all ':- table ...' lines !!!

if dataset == 'kinship_family':
    prolog_kb_file = './data/kinship_family/kinship_family_no_tables.pl'
    # queries_file = './data/kinship_family/valid.txt'
    queries_file = './data/kinship_family/test.txt'

elif dataset == 'countries_s3':
    prolog_kb_file = './data/countries_s3/countries.pl'  # Adjusted for countries_s3 dataset
    # queries_file = './data/countries_s3/train.txt'
    # queries_file = './data/countries_s3/valid.txt'
    queries_file = './data/countries_s3/test.txt'

MAX_DEPTH_TO_ANALYZE = 15 # Keep it reasonable first, might be slow

# Check if files exist
# ... (file existence checks from previous version) ...
if not os.path.exists(prolog_kb_file):
     print(f"ERROR: Prolog KB file (NO TABLES VERSION) not found at {prolog_kb_file}")
     print("Please create it by copying the original and removing ':- table' directives.")
     exit()
if not os.path.exists(queries_file):
     print(f"ERROR: Queries file not found at {queries_file}")
     exit()


# 2. Load queries (same as before)
# ... (query loading logic) ...
cleaned_queries_for_analysis = []
try:
    with open(queries_file, 'r') as f:
        queries_to_analyze = [line.strip() for line in f if line.strip()]
        for q in queries_to_analyze:
            pg = q.strip()
            if pg.endswith('.'): pg = pg[:-1].strip()
            if pg: cleaned_queries_for_analysis.append(pg)
    print(f"Loaded {len(queries_to_analyze)} raw queries, {len(cleaned_queries_for_analysis)} valid queries from {queries_file}")
except Exception as e:
    print(f"Error loading queries from {queries_file}: {e}")
    exit()


# 3. Run the analysis function
print('\nCurrent working directory:', os.getcwd())
provability_ratios = calculate_provability_ratios_by_depth(
    queries_to_analyze,
    MAX_DEPTH_TO_ANALYZE,
    prolog_kb_file # Pointing to the NO TABLES version
)

# 4. Print results (same as before, includes None check)
# ... (results printing logic) ...
if provability_ratios is not None:
    print(f"\n--- Provability Ratios by Depth (up to {MAX_DEPTH_TO_ANALYZE}) ---")
    if not provability_ratios:
        print("No ratios were calculated (possibly due to errors or no depths checked).")
    else:
        final_denominator = len(cleaned_queries_for_analysis)
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

import janus_swi as janus
import os
import re # For cleaning up query strings

def is_provable_at_depth_n(queries: list[str], n: int, prolog_file: str) -> list[str]:
    """
    Checks if a list of Prolog queries are provable within a specific depth limit 'n'.

    Uses SWI-Prolog's call_with_depth_limit/3 predicate via the janus-swi library.

    Args:
        queries: A list of strings, where each string is a Prolog query
                 (e.g., "aunt(1369,1287).", "father(X,Y).").
        n: The maximum proof depth allowed (integer >= 0).
        prolog_file: The path to the Prolog file (.pl) containing the
                     facts and rules needed to answer the queries.

    Returns:
        A list of strings, either 'provable' or 'not_provable', corresponding
        to each query in the input list. 'error' might be returned if a
        Prolog error occurs during query execution.
    """
    if not os.path.exists(prolog_file):
        raise FileNotFoundError(f"Prolog file not found: {prolog_file}. Current working directory: {os.getcwd()}")

    results = []

    try:
        # Initialize Janus (usually only needed once per process)
        # janus.init_swi() # Often implicitly called on first use

        # Consult the Prolog knowledge base.
        # It's often better to consult only once if processing many batches
        # against the same KB. If the KB needs resetting, manage abolishing
        # predicates or restarting the janus engine might be necessary.
        janus.consult(prolog_file)

        for query in queries:
            # Prepare the query for Prolog: remove trailing period and whitespace.
            # Ensure the query is a valid Prolog term.
            prolog_goal = query.strip()
            if prolog_goal.endswith('.'):
                prolog_goal = prolog_goal[:-1].strip()

            if not prolog_goal:
                results.append(('error_empty_query',-1)) # Or handle as needed
                continue

            # Construct the depth-limited query string
            # Format: call_with_depth_limit(:Goal, +Limit, -Result)
            # We bind the result to a variable 'DepthResult'
            depth_query = f"call_with_depth_limit(({prolog_goal}), {n}, DepthResult)."

            label = 'not_provable' # Default assumption
            depth = -1 # Default depth if not proven

            try:
                # Execute the query using query_once (we only need one proof)
                # query_once returns False on failure, or a dict on success.
                result_dict = janus.query_once(depth_query)

                if result_dict:
                    # The call_with_depth_limit goal succeeded.
                    # Check the value of DepthResult.
                    prolog_result = result_dict.get('DepthResult')

                    if isinstance(prolog_result, int):
                        # Succeeded and returned the actual depth (which <= n).
                        label = 'provable'
                        depth = prolog_result
                    elif prolog_result == 'depth_limit_exceeded':
                        # Succeeded because the depth limit 'n' was reached
                        # *before* finding a proof or exhausting the search.
                        # Therefore, it's not provable *at or below* depth n.
                        label = 'not_provable'
                    else:
                        # Unexpected result from call_with_depth_limit
                        print(f"Warning: Unexpected result '{prolog_result}' for query '{query}'")
                        label = 'error_unexpected_result'
                else:
                    # The call_with_depth_limit goal failed entirely.
                    # This means the original goal (prolog_goal) is not provable,
                    # regardless of depth, or at least not within depth n without
                    # hitting the limit explicitly.
                    label = 'not_provable'
                # if label == 'provable':
                #     if depth > n:
                #         print(f"Query '{query}' provable at depth {depth} > max_depth {n}")
                #     else:
                #         print(f"Query '{query}' provable at depth----------------- {depth} <= max_depth {n}")

            except janus.PrologError as e:
                print(f"Prolog error processing query '{query}': {e}")
                # Could be syntax error in goal, predicate undefined, etc.
                label = 'error_prolog'
            except Exception as e:
                 print(f"Python/Janus error processing query '{query}': {e}")
                 label = 'error_python'

            results.append((label, depth)) # Append the label and depth

    except janus.PrologError as e:
         print(f"Prolog error during setup (e.g., consulting file '{prolog_file}'): {e}")
         # Return errors for all queries if setup fails
         return ['error_setup'] * len(queries)
    except Exception as e:
         print(f"General error during setup: {e}")
         return ['error_setup'] * len(queries)

    # janus.cleanup() # Call if you need to release SWI-Prolog resources

    return results

# Example Usage:
# 1. Set up the Prolog file
root = "./data/kinship_family/kinship_family_no_tables.pl"

# 2. Define queries and depth
queries_file = './data/kinship_family/valid.txt'
with open(queries_file, 'r') as f:
    queries_to_test = [line.strip() for line in f if line.strip()]
# Clean up queries (remove trailing periods and whitespace)
queries_to_test = [re.sub(r'\s*\.\s*$', '', query) for query in queries_to_test]
# Remove empty queries
queries_to_test = [query for query in queries_to_test if query]


max_depth = 2 # 11 all are provable, 12 None are provable

# 3. Run the function
print('current working directory:', os.getcwd())
results = is_provable_at_depth_n(queries_to_test, max_depth, root)
labels, depths = zip(*results)


# 4. Print results
print(f"\n--- Results for max_depth = {max_depth} ---")
# for query, label in zip(queries_to_test, labels):
#     print(f"Query: {query:<25} Label: {label}")

print('number of provable queries:', labels.count('provable'))
print('number of not provable queries:', labels.count('not_provable'))
print('number of errors:', labels.count('error'))
print('distribution of depths:')
for i in range(max_depth+2):
    print(f"Depth {i}: {depths.count(i)}")

provable_depths = [d for l, d in results if l == 'provable']
print('Unique depths for provable queries:', set(provable_depths))
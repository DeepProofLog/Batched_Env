from typing import List, Dict, Tuple
import time
from collections import deque
from utils import Term, Rule
from dataset import get_rules_from_rules_file,get_queries
# from python_unification import  get_next_unification_python_with_depth_info


from collections import deque # Efficient queue for BFS


def get_next_unification_python_with_depth_info(
    state: List[Term],
    facts: List[Term],
    rules: List[Rule],
    verbose: int = 0
) -> List[Tuple[List[Term], bool]]: # Returns: List[(next_state, was_rule_used)]
    """
    Processes a state and returns all possible next states, indicating if a rule was used.

    Args:
        state: List of Term objects representing the current state (goals).
        facts: List of Term objects representing known facts.
        rules: List of Rule objects representing inference rules.

    Returns:
        List of tuples: [(next_goal_list, was_rule_used), ...]
        where was_rule_used is True if a rule application generated the next_goal_list,
        False otherwise (e.g., fact unification).
    """

    # --- Handle terminal states ---
    # Return immediately if already True or False, with was_rule_used=False
    if any(term.predicate == 'False' for term in state):
        return [[Term('False', [])], False]
    if all(term.predicate == 'True' for term in state): # Check if ALL are True
         # If state was already empty or only True, it's a success state
         # If it derived from previous step, was_rule_used depends on that step
         # Let's return empty list to signify termination, handled by caller.
         # Or better: return a canonical True state
         return [[Term('True', [])], False] # Signifies success, no rule needed *here*

    # Filter out True terms before processing
    state = [term for term in state if term.predicate != 'True']
    if not state:
         return [[Term('True', [])], False] # Empty state is success

    # --- Variable Renaming (Standardizing Apart) ---
    # (Keep the existing variable renaming logic as it's important for correctness)
    exclude = set()
    for term in state:
        for arg in term.args:
            if isinstance(arg, str) and arg.startswith('_'):
                exclude.add(arg)
    subs = {}
    var_counter = 0
    new_state_goals = []
    for term in state:
        new_args = []
        for arg in term.args:
            if isinstance(arg, str) and arg.isupper():
                if arg not in subs:
                    while f'_{var_counter}' in exclude:
                        var_counter += 1
                    subs[arg] = f'_{var_counter}'
                new_args.append(subs[arg])
            else:
                new_args.append(arg)
        new_state_goals.append(Term(term.predicate, new_args))
    state = new_state_goals # Use the renamed state

    # --- Goal Reordering (Optional, keep if desired) ---
    state.sort(key=lambda term: 1 if all(is_variable(arg) for arg in term.args) else 0)

    # --- Process First Goal ---
    if not state: # Should have been caught earlier, but for safety
        return [[Term('True', [])], False]
    query, *remaining_state = state

    next_states_with_info = [] # This will store List[Tuple[List[Term], bool]]

    # --- Try unifying with facts ---
    if verbose > 0: print(f'\n\n**********\nQuery: {query}, Remaining state: {remaining_state}\n')
    if verbose > 0: print('Unification with facts')
    fact_substitutions = unify_with_facts(query, facts, verbose=0) # Assuming unify_with_facts is correct

    for subs in fact_substitutions:
        new_state = []
        if subs.get('True') == 'True':
            # Query was a ground fact, continue with remaining state
            new_state = remaining_state.copy()
        else:
            # Apply substitutions to remaining state
            new_state = [apply_substitution(term, subs) for term in remaining_state]

        # Simplify state by replacing known facts with True
        simplified_state = []
        all_true = True
        for term in new_state:
             # Check if term is a ground fact present in the facts list
             is_ground = not any(is_variable(arg) for arg in term.args)
             if is_ground and term in facts:
                 simplified_state.append(Term('True', []))
             else:
                 simplified_state.append(term)
                 all_true = False # Found a non-true term

        # Filter out True terms unless it's the only thing left
        final_state = [term for term in simplified_state if term.predicate != 'True']
        if not final_state: # If everything simplified to True
             final_state = [Term('True', [])]

        # Add to results, indicating a rule was NOT used for this step
        next_states_with_info.append((final_state, False))
        if verbose > 1: print(f"  Fact Result: {final_state}")


    # --- Try unifying with rules ---
    if verbose > 0: print('\nUnification with rules')
    rule_results = unify_with_rules(query, rules, verbose=0) # Assuming unify_with_rules is correct

    for i, (body, subs) in enumerate(rule_results):
        # Apply substitutions to remaining state
        new_remaining = [apply_substitution(term, subs) for term in remaining_state]
        # Combine substituted rule body with substituted remaining state
        new_state_from_rule = body + new_remaining

        # --- Apply same simplification logic as for facts ---
        simplified_state = []
        all_true = True
        for term in new_state_from_rule:
             is_ground = not any(is_variable(arg) for arg in term.args)
             if is_ground and term in facts:
                 simplified_state.append(Term('True', []))
             else:
                 simplified_state.append(term)
                 all_true = False

        final_state = [term for term in simplified_state if term.predicate != 'True']
        if not final_state:
             final_state = [Term('True', [])]

        # Add to results, indicating a rule WAS used for this step
        next_states_with_info.append((final_state, True))
        if verbose > 0: print(f"  Rule Result ({rules[i].head}): {final_state}")

    if verbose > 0: print(f'\nNext states generated: {len(next_states_with_info)}\n**********\n')

    # If no unification was possible at all
    if not next_states_with_info:
        return [[Term('False', [])], False] # Return canonical False state

    return next_states_with_info




def check_provability_at_depth_prolog_like(
    initial_goals: List[Term],
    n: int, # Max *rule application* depth
    rules: List[Rule],
    facts: List[Term],
    verbose: bool = False
) -> str:
    """
    Checks if a goal is provable within rule application depth n using BFS.

    Depth increases ONLY when a rule is applied.

    Args:
        initial_goals: The initial query state as a list of Term objects.
        n: The maximum rule application depth (integer >= 0).
        rules: List of Rule objects.
        facts: List of Term objects representing facts.
        verbose: Flag for verbose output.

    Returns:
        'provable', 'not_provable', 'depth_limit_exceeded', or an error string.
    """
    if not initial_goals:
        return 'error_empty_query'

    # State in queue: (goals_list, current_rule_depth)
    # Use deque for efficient BFS popping from the left
    states_to_process = deque([(initial_goals, 0)])

    # Optional: Keep track of visited states to prevent cycles and redundant work
    # Key: tuple(sorted(str(goal) for goal in goals)) + (depth,)
    visited_states = set()
    initial_state_tuple = tuple(sorted(str(g) for g in initial_goals)) + (0,)
    visited_states.add(initial_state_tuple)

    hit_limit = False # Flag if any path requires depth > n

    while states_to_process:
        current_goals, current_depth = states_to_process.popleft()

        if verbose: print(f"Processing State: {current_goals} at Depth: {current_depth}")

        # Get next possible states and whether a rule was used for each
        # Use the modified function
        next_options = get_next_unification_python_with_depth_info(
            current_goals, facts, rules, verbose=verbose
            )

        if not next_options: # Should not happen if function returns False state correctly
            if verbose: print("  No next options returned.")
            continue

        for next_goals, was_rule_used in next_options:

            # --- Check for immediate True/False states ---
            if any(term.predicate == 'False' for term in next_goals):
                 if verbose: print(f"  Branch failed: {next_goals}")
                 continue # This branch fails

            if all(term.predicate == 'True' for term in next_goals):
                 if verbose: print(f"  Branch succeeded: {next_goals}")
                 return 'provable' # Found a proof within limits

            # --- Calculate next depth ---
            next_depth = current_depth + 1 if was_rule_used else current_depth

            # --- Check depth limit ---
            if next_depth > n:
                hit_limit = True # Mark that we needed more depth on at least one path
                if verbose: print(f"  Branch hit depth limit ({next_depth} > {n}): {next_goals}")
                continue # Do not add this state to the queue

            # --- Add valid, non-visited states to queue ---
            # Optional: Visited check
            state_tuple = tuple(sorted(str(g) for g in next_goals)) + (next_depth,)
            if state_tuple in visited_states:
                if verbose: print(f"  State already visited: {next_goals} at depth {next_depth}")
                continue
            visited_states.add(state_tuple)

            if verbose: print(f"  Adding to queue: {next_goals} at depth {next_depth} (Rule used: {was_rule_used})")
            states_to_process.append((next_goals, next_depth))

    # --- Loop finished ---
    # If we exit the loop because the queue is empty:
    if hit_limit:
        # We explored all possibilities within depth n, but some paths needed more.
        # Since no 'provable' was returned, it means no proof was found *within* n rule applications.
        if verbose: print("Queue empty, limit was hit on some paths.")
        return 'depth_limit_exceeded'
    else:
        # Queue is empty, and the limit was never hit. All branches failed.
        if verbose: print("Queue empty, limit never hit.")
        return 'not_provable'

    # Fallback - should not be reached
    return 'error_unexpected_exit'






def calculate_provability_ratios_by_depth(
    queries: list[Term],
    max_depth_check: int,
    rules: list[Rule],
    facts: list[Term],
    is_train_data: bool = False,
) -> dict[int, float] | None:
    """
    Calculates the ratio of provable queries for each depth from 1 to max_depth_check.

    Args:
        queries: A list of query strings.
        max_depth_check: The maximum depth to check (e.g., 100).
        rules: A list of Rule objects.
        facts: A list of Term objects representing facts.

    Returns:
        A dictionary where keys are depths (1 to max_depth_check) and
        values are the ratio (0.0 to 1.0) of queries provable at or
        below that depth. Returns None if setup fails or critical error occurs.
    """
    start_time = time.time()

    num_valid_queries = len(queries)
    if num_valid_queries == 0:
        print("No valid queries found.")
        return {}

    print(f"Proving {num_valid_queries} valid queries up to depth {max_depth_check}...")


    # 2. Initialize results storage
    min_proven_depth = [float('inf')] * num_valid_queries
    query_errors = [False] * num_valid_queries
    ratios_by_depth = {}

    # 4. Iterate through depths and queries
    proved_queries = {} # Dictionary to store proven queries
    for depth in range(1,max_depth_check+1):
        proven_in_this_round = 0
        errors_in_this_round = 0
        queries_checked_this_round = 0

        for i, goal in enumerate(queries):
            # Print progress percentage instead of a bar
            # progress_pct = (depth / max_depth_check) * 100
            print(f"\rProcessing Up to depth {depth}({max_depth_check}), query {i+1}/{len(queries)}. \
                  Proven: {proven_in_this_round}. Checked: {queries_checked_this_round}. Errors: {errors_in_this_round}  ",end='')
            
            if min_proven_depth[i] == float('inf') and not query_errors[i]:
                queries_checked_this_round += 1
                # IMPORTANT, IF CHOOSING PROLOG LIKE DEPTH, INCREASE DEPTH BY 1, IT SKIPS THE FIRST DEPTH 
                # THAT IF FACT MATCHING (BASICALLY IN PROLOG, ONLY RULE MATCHING IS COUNTED FOR DEPTH)
                status = check_provability_at_depth_prolog_like(
                                            [goal],
                                            depth,
                                            rules,
                                            facts=[fact for fact in facts if fact != goal] if is_train_data else facts, 
                                            verbose=0)

                if status == 'provable':
                    min_proven_depth[i] = depth
                    proved_queries[str(goal)] = depth
                    proven_in_this_round += 1
                elif status.startswith('error_'):
                    query_errors[i] = True
                    errors_in_this_round += 1
                # If 'not_provable' or 'depth_limit_exceeded', do nothing yet
        print()
        print(f"Proven so far: {sum(1 for m_depth in min_proven_depth if m_depth <= depth)}")
        print()

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

    end_time = time.time()
    # --- Final Summary Print --- (copy from previous version)
    print(f"\n--- Calculation Summary ---")
    # ... (rest of summary print) ...
    print(f"Total valid queries processed: {num_valid_queries}")
    print(f"Queries provable within depth {max_depth_check}: {total_provable_count} ({total_provable_count/num_valid_queries if num_valid_queries else 0:.2%}")
    print(f"Queries that resulted in errors: {total_errors} ({total_errors/num_valid_queries:.2%} if num_valid_queries else 0)")
    print(f"Execution time: {end_time - start_time:.2f} seconds")


    return ratios_by_depth, proved_queries



if __name__ == "__main__":
    
    max_depth_check = 15  # Adjust as needed

    # 1. Load the rules and facts from the file
    # dataset = 'kinship_family'
    dataset = 'countries_s3'
    set_file = 'test'

    root_dir = './data/' + dataset + '/'
    rules_file = root_dir + 'rules.txt'
    facts_file = root_dir + 'train.txt'
    queries_file = root_dir + set_file + '.txt'
    is_train_data = 'train' in set_file

    # 2. Load rules and facts
    rules = get_rules_from_rules_file(rules_file)
    facts = get_queries(facts_file)

    queries = get_queries(queries_file)
    print(f"Loaded {len(queries)} queries from {queries_file}")
    queries = queries[:500]

    # ('locatedInCR(united_states, americas)', 12)
    queries = [Term('locatedInCR', ['timor_leste', 'asia'])] # Test with a single query

    provability_ratios, proved_queries = calculate_provability_ratios_by_depth(
        queries,
        max_depth_check=max_depth_check,
        rules=rules,
        facts=facts,
        is_train_data=True,
        )
    
    # 4. Print results (includes None check)
    print(f"\n--- Provability Ratios by Depth (up to {max_depth_check}) ---")  # Using hard-coded 10 from max_depth_check
    if not provability_ratios:
        print("No ratios were calculated (possibly due to errors or no depths checked).")
    else:
        final_denominator = len(queries)
        if final_denominator == 0: final_denominator = 1

        # Adjust last depth based on early exit
        last_depth_calculated = max(provability_ratios.keys()) if provability_ratios else 0
        # Ensure max_depth_check reflects actual last calculated depth if early exit occurred
        effective_max_depth = min(max_depth_check, last_depth_calculated) if last_depth_calculated > 0 else 0

        for depth in range(1, effective_max_depth + 1):
            ratio = provability_ratios.get(depth, 0.0)
            count = int(ratio * final_denominator)
            print(f"Depth <= {depth:<3}: {ratio:.4f} ({count}/{final_denominator})")

    print(f"\n--- Proven Queries ---")
    print([(query, depth) for query, depth in proved_queries.items()])
    # # 5. Save proven queries
    # if proved_queries:
    #     with open(queries_file.replace('.txt', f'_depth_{max_depth_check}.txt'), 'w') as f:
    #         for query, depth in proved_queries.items():
    #             clean_query = query.replace(' ', '')
    #             f.write(f"{clean_query}\n")
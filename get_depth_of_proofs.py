from typing import List, Dict, Tuple
from utils import Term, Rule, apply_substitution, is_variable  
from dataset import get_rules_from_file
import os
import time

from typing import List, Dict, Set, Tuple
from collections import deque
from utils import Term, Rule, apply_substitution
from dataset import get_rules_from_file, get_atom_from_string
from python_unification import get_next_unification_python





def check_provability_at_depth(state: List[Term], n: int, rules: List[Rule], facts: List[Term], verbose: bool = False) -> str:
    """
    Checks if a goal is provable within depth n by exploring all branches.

    Explores the proof tree by branching at each step:
    1. Start with the initial state
    2. For each depth level up to n:
       a. Process all current states to get their next possible states
       b. If any branch is proven (contains True), return 'provable'
       c. Remove any branches that are falsified (contain False)
       d. If no branches remain, return 'not_provable'
       e. Otherwise, continue with the remaining branches
    3. If max depth is reached without proving or disproving, return 'depth_limit_exceeded'

    Args:
        state: The initial state as a list of lists of Term objects.
        n: The maximum depth (integer >= 0).
        rules: List of Rule objects.
        facts: List of Term objects representing facts.

    Returns:
        'provable', 'not_provable', 'depth_limit_exceeded', or an error string.
    """

    print(f"\nChecking query {state} at depth {n}...") if verbose else None
    if not state:
        print('error_empty_query') if verbose else None
        return 'error_empty_query'

    current_states = [state]  # List of all active branches we need to explore
    # print()
    for depth in range(n):
        print(f"Depth {depth}")  if verbose else None  
        print(f"Current states: {current_states}") if verbose else None 

        # Show progress for current depth
        # print(f"Processing subdepth {depth+1}/{n}.\r")
        next_generation_states = []
        
        # Process each current state/branch
        for i,current_state in enumerate(current_states):
            # print(f"Processing  state {i}/{len(current_states)}.\r") if verbose else None
            # Get all possible next states for this branch
            branch_next_states = get_next_unification_python(current_state, facts, rules, verbose=0)
            print(f"Branch next states: {branch_next_states}") if verbose else None
            
            # Check if any of these next states represent a successful proof
            if any(any(term.predicate == 'True' for term in branch_state) 
                  for branch_state in branch_next_states):
                print('provable') if verbose else None
                return 'provable'
            
            # Filter out falsified branches
            valid_next_states = [
                branch_state for branch_state in branch_next_states 
                if branch_state and not any(term.predicate == 'False' for term in branch_state)
            ]
            
            # Add all valid next states to the next generation
            next_generation_states.extend(valid_next_states)
        
        # If all branches have been eliminated, the goal is not provable
        if not next_generation_states:
            print('not_provable') if verbose else None
            return 'not_provable'
            
        # If we've reached the maximum depth, we can't determine provability
        if depth == n - 1:
            print('depth_limit_exceeded') if verbose else None
            return 'depth_limit_exceeded'
            
        # Update our list of states to explore in the next iteration
        current_states = next_generation_states
    
    # This line should never be reached if n >= 0
    print('error_should_not_arrive_here') 
    return 'error_should_not_arrive_here'


def calculate_provability_ratios_by_depth(
    queries: list[str],
    max_depth_check: int,
    rules: list[Rule],
    facts: list[Term],
    is_train_data: bool = False,
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
                status = check_provability_at_depth([goal],
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

    max_depth_check = 2  # Adjust as needed

    # 1. Load the rules and facts from the file
    dataset = 'kinship_family'

    if dataset == 'kinship_family':
        prolog_kb_file = './data/kinship_family/kinship_family_no_tables.pl'
        queries_file, is_train_data = './data/kinship_family/train.txt', True
        # queries_file, is_train_data = './data/kinship_family/valid.txt', False
        # queries_file, is_train_data = './data/kinship_family/test.txt', False

    elif dataset == 'countries_s3':
        prolog_kb_file = './data/countries_s3/countries.pl'  # Adjusted for countries_s3 dataset
        queries_file, is_train_data = './data/countries_s3/train.txt', True
        # queries_file = './data/countries_s3/valid.txt'
        # queries_file = './data/countries_s3/test.txt'

    # Check if files exist
    if not os.path.exists(prolog_kb_file):
        print(f"ERROR: Prolog KB file not found at {prolog_kb_file}")
        exit()
    if not os.path.exists(queries_file):
        print(f"ERROR: Queries file not found at {queries_file}")
        exit()

    # 2. Load rules and facts
    facts, rules = get_rules_from_file(prolog_kb_file)
    print(f"Loaded {len(rules)} rules and {len(facts)} facts from {prolog_kb_file}")

    # 3. Load queries
    queries = []
    try:
        with open(queries_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
            queries = [get_atom_from_string(query) for query in queries]
        print(f"Loaded {len(queries)} queries from {queries_file}")
    except Exception as e:
        print(f"Error loading queries from {queries_file}: {e}")
        exit()
    
    # queries = queries[:500]

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

    # 5. Save proven queries
    if proved_queries:
        with open(queries_file.replace('.txt', f'_depth_{max_depth_check}.txt'), 'w') as f:
            for query, depth in proved_queries.items():
                clean_query = query.replace(' ', '')
                f.write(f"{clean_query}\n")
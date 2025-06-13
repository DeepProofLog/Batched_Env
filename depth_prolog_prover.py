import janus_swi as janus
import os
import time
from typing import List, Dict, Tuple

from utils import Term, Rule, get_atom_from_string
from dataset import get_queries
from prolog_unification import get_next_unification_prolog


def check_provability_at_depth(prolog_goal: str, n: int) -> str:
    """
    Checks if a single Prolog goal is provable within depth n.
    
    Returns:
        'provable', 'not_provable', 'depth_limit_exceeded', or an error string.
    """    
    if not prolog_goal:
        return 'error_empty_query'
    # The goal has the structure: pred(arg1,arg2) Change it to pred("arg1","arg2")
    # prolog_goal = prolog_goal.replace('(', '("').replace(',', '","').replace(')', '")')

    depth_query = (
        f"call_with_time_limit({10}, "
        f"call_with_depth_limit(({prolog_goal}), {n}, DepthResult))"
    )

    try:
        result_dict = janus.query_once(depth_query)
        # print(f"Prolog query result for goal '{prolog_goal}' at depth {n}: {result_dict}")
        if result_dict:
            prolog_result = result_dict.get('DepthResult')
            if isinstance(prolog_result, int):
                return 'provable'
            elif prolog_result == 'depth_limit_exceeded':
                return 'depth_limit_exceeded'
            elif result_dict.get('DepthResult') == None and result_dict.get('truth') == False:
                return 'not_provable'
            else:
                print(f"\nWarning: Unexpected result for goal '{prolog_goal}' at depth {n}: {result_dict}")
                return 'error_unexpected_result'
        else:
            return 'not_provable'
            
    except janus.PrologError as e:
        print(f"\nProlog error processing goal '{prolog_goal}' at depth {n}: {e}")
        return 'error_prolog'
    except Exception as e:
        print(f"\nPython/Janus error processing goal '{prolog_goal}' at depth {n}: {e}")
        return 'error_python'


def manual_check_provability_at_depth(state: List[Term], 
                                n: int, 
                                next_var_index: int = None,
                                max_atoms: int = 5, 
                                verbose: bool = False) -> str:
    """
    Checks if a goal is provable within depth n by exploring all branches.
    
    Returns:
        'provable', 'not_provable', 'depth_limit_exceeded', or an error string.
    """
    if verbose:
        print(f"\nChecking query {state} at depth {n}...")

    # print(f"\nChecking query {state} at depth {n}...")
    current_states = [state]
        
    for depth in range(n):
        if verbose:
            print(f"Depth {depth}")
            print(f"Current states: {current_states}")

        next_generation_states = []
        
        for current_state in current_states:
            branch_next_states, next_var_index = get_next_unification_prolog(
                current_state,
                next_var_index=next_var_index,
                verbose=verbose,
                )
            
            if verbose:
                print(f"Branch next states: {branch_next_states}")
            
            # Check for successful proof
            if any(all(term.predicate == 'True' for term in branch_state) 
                  for branch_state in branch_next_states):
                if verbose: print('provable')
                return 'provable', next_var_index
            
            # Filter out falsified and oversized branches
            valid_next_states = []
            for branch_state in branch_next_states:
                if (branch_state and 
                    not any(term.predicate == 'False' for term in branch_state) and
                    len(branch_state) <= max_atoms):
                    valid_next_states.append(branch_state)
                # elif len(branch_state) > max_atoms:
                #     print(f"Skipping oversized branch: {branch_state} (size {len(branch_state)})")
            next_generation_states.extend(valid_next_states)
        
        # Check termination conditions
        if not next_generation_states:
            if verbose:
                print('\nnot_provable')
            return 'not_provable', next_var_index
            
        if depth == n - 1:
            if verbose:
                print('\ndepth_limit_exceeded')
            return 'depth_limit_exceeded', next_var_index
            
        current_states = next_generation_states
    
    return 'error_should_not_arrive_here', next_var_index


def calculate_provability_ratios_by_depth(
    queries: List[Term],
    max_depth_check: int,
    prolog_file: str,
    is_training_data: bool = False,
    designed_prover: bool = False
) -> Tuple[Dict[int, float], Dict[str, int]]:
    """
    Calculates the ratio of provable queries for each depth from 1 to max_depth_check.
    
    Returns:
        Tuple of (ratios_by_depth, proved_queries)
    """
    start_time = time.time()
    num_queries = len(queries)
    
    if num_queries == 0:
        print("No valid queries found.")
        return {}, {}

    min_proven_depth = [float('inf')] * num_queries
    query_errors = [False] * num_queries
    ratios_by_depth = {}
    proved_queries = {}
    proven_by_depth = {}
    next_var_index = 1

    # Consult Prolog file once
    janus.consult(prolog_file)
    print(f"Consulted Prolog file: {prolog_file}")

    for depth in range(1, max_depth_check + 1):
        proven_in_this_round = 0
        errors_in_this_round = 0
        queries_checked_this_round = 0

        for i, goal in enumerate(queries):
            goal_str = goal.prolog_str()
            print(f"\rDepth {depth}/{max_depth_check}, query {i+1}/{num_queries} available queries. "
                    f"Proven: {proven_in_this_round}", end='')

            # Skip already resolved queries
            if min_proven_depth[i] != float('inf') or query_errors[i]:
                continue
            queries_checked_this_round += 1
            retracted_successfully = False

            # Leave-one-out logic for training data
            if is_training_data:
                try:
                    retract_query = f"retract({goal_str})."
                    janus.query_once(retract_query)
                    retracted_successfully = True
                except (janus.PrologError, Exception) as e:
                    print(f"\nWarning: Error retracting '{goal_str}': {e}")
                    query_errors[i] = True
                    errors_in_this_round += 1
                    continue

            # Check provability
            if not query_errors[i]:
                if designed_prover:
                    status, next_var_index = manual_check_provability_at_depth([goal], depth, next_var_index)
                else:   
                    status = check_provability_at_depth(goal_str, depth)
            # Restore fact for training data
            if is_training_data and retracted_successfully:
                try:
                    assert_query = f"asserta({goal_str})."
                    janus.query_once(assert_query)
                except (janus.PrologError, Exception) as e:
                    print(f"\nCRITICAL WARNING: Error asserting back '{goal_str}': {e}")
                    query_errors[i] = True
                    if not status.startswith('error_'):
                        errors_in_this_round += 1

            # Process status
            if not query_errors[i]:
                if status == 'provable':
                    min_proven_depth[i] = depth
                    proved_queries[str(goal)] = depth - 1
                    proven_in_this_round += 1
                elif status.startswith('error_'):
                    query_errors[i] = True
                    errors_in_this_round += 1

        # Calculate cumulative ratio
        provable_count = sum(1 for depth_val in min_proven_depth if depth_val <= depth)
        ratios_by_depth[depth] = provable_count / num_queries if num_queries > 0 else 0.0
        proven_by_depth[depth] = proven_in_this_round

        print(f"\nDepth {depth}: Checked {queries_checked_this_round} queries.")
        print(f"         Errors: {errors_in_this_round}. Newly proven: {proven_in_this_round}. ")
        print(f"         Cumulative proven <= depth {depth}: {provable_count}/{num_queries}\n")

        # Early exit if all queries resolved
        if all((depth_val != float('inf') or err) 
                for depth_val, err in zip(min_proven_depth, query_errors)):
            print(f"All queries resolved by depth {depth}. Stopping.")
            break

    # Summary
    total_provable = sum(1 for depth_val in min_proven_depth if depth_val != float('inf'))
    total_errors = sum(query_errors)
    end_time = time.time()

    print(f"\n--- Summary ---")
    print(f"Total queries: {num_queries}")
    print(f"Provable: {total_provable} ({total_provable/num_queries:.2%})")
    print(f"Errors: {total_errors} ({total_errors/num_queries:.2%})")
    print(f"Time: {end_time - start_time:.2f} seconds")

    return proven_by_depth, proved_queries


def load_queries(dataset_name: str, set_file: str, data_path: str) -> Tuple[List[Term], str]:
    """
    Load queries from file and return the prolog KB file path.
    
    Returns:
        Tuple of (queries, prolog_kb_file_path)
    """
    root_dir = data_path + dataset_name + '/'
    queries_file = root_dir + set_file + '.txt'
    prolog_kb_file = root_dir + dataset_name + '.pl'
    
    # Check if files exist
    if not os.path.exists(prolog_kb_file):
        raise FileNotFoundError(f"Prolog KB file not found: {prolog_kb_file}")
    if not os.path.exists(queries_file):
        raise FileNotFoundError(f"Queries file not found: {queries_file}")
    
    # Load and clean queries
    queries = get_queries(queries_file)

    print(f"Loaded {len(queries)} queries from {queries_file}")
    return queries, prolog_kb_file


if __name__ == "__main__":
    max_depth_check = 4
    dataset_name = 'wn18rr'
    data_path = './data/'
    designed_prover = True

    for set_file in ['valid','test']:
        try:
            queries, prolog_kb_file = load_queries(dataset_name, set_file, data_path)
            is_train_data = 'train' in set_file
            
            # Limit queries for testing
            # queries = queries[:500]
            
            print(f"\nCurrent working directory: {os.getcwd()}")
            print(f"Processing {len(queries)} queries from {dataset_name} for set {set_file}")
            print(f"Mode: {'Training (Leave-One-Out)' if is_train_data else 'Standard'}")
            
            proven_by_depth, proved_queries = calculate_provability_ratios_by_depth(
                queries, max_depth_check, prolog_kb_file, is_train_data, designed_prover
            )
            print(f"\nProven by depth: {proven_by_depth}. Total proved queries: {len(proved_queries)}/ {len(queries)}")
            # # Save results
            # if proved_queries:
            #     root_dir = data_path + dataset_name + '/'
            #     output_file = root_dir + set_file + '_depths.txt'
            #     with open(output_file, 'w') as f:
            #         for query in queries:
            #             clean_query = str(query).replace(' ', '')
            #             depth = proved_queries.get(str(query), -1)
            #             f.write(f"{clean_query} {depth}\n")
            #     print(f"Saved results to {output_file}")
                
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error: {e}")
            continue
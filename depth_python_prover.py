from typing import List, Dict, Tuple
import time
import random
import numpy as np # <-- ADD THIS IMPORT
from dataset import DataHandler
from index_manager import IndexManager
from utils import Term, Rule
from python_unification import get_next_unification_python


# (The check_provability_random_walk function and others remain the same)
def check_provability_random_walk(state: List[Term],
                                  n: int,
                                  rules: List[Rule],
                                  facts: List[Term],
                                  index_manager: IndexManager = None,
                                  is_train_data: bool = False,
                                  max_atoms: int = 20) -> str:
    """
    Checks if a goal is provable by following a single random path up to depth n.
    """
    current_state = state
    for depth in range(n):
        branch_next_states, _ = get_next_unification_python(
            current_state,
            facts_set=facts,
            facts_indexed=index_manager.fact_index,
            rules=rules,
            excluded_fact=state[0] if is_train_data else None,
            verbose=0,
            next_var_index=index_manager.variable_start_index,
        )
        if any(all(term.predicate == 'True' for term in branch_state)
               for branch_state in branch_next_states):
            return 'provable'
        valid_next_states = [
            branch_state for branch_state in branch_next_states
            if (branch_state and
                not any(term.predicate == 'False' for term in branch_state) and
                len(branch_state) <= max_atoms)
        ]
        if not valid_next_states:
            return 'not_provable'
        current_state = random.choice(valid_next_states)
    return 'depth_limit_exceeded'


def run_random_agent_on_queries(
    queries: List[Term],
    max_depth: int,
    num_trials: int,
    rules: List[Rule],
    facts: set,
    index_manager: IndexManager = None,
    is_train_data: bool = False,
    max_atoms: int = 20
):
    """
    Calculates the success rate of a random agent over multiple independent trials.
    In each trial, the agent attempts to solve ALL queries.
    """
    start_time = time.time()
    num_queries = len(queries)
    
    print(f"Testing a random agent on {num_queries} queries over {num_trials} independent trials.")
    print(f"Parameters: max_depth={max_depth}\n")

    proven_counts_per_trial = []

    # Loop through trials
    for trial_num in range(1, num_trials + 1):
        proven_in_this_trial = 0
        
        # In each trial, attempt every query
        for goal in queries:
            status = check_provability_random_walk(
                [goal], max_depth, rules, facts,
                index_manager=index_manager,
                is_train_data=is_train_data,
                max_atoms=max_atoms
            )
            if status == 'provable':
                proven_in_this_trial += 1
        
        # Record the number of successes for this independent trial
        proven_counts_per_trial.append(proven_in_this_trial)
        
        success_rate = (proven_in_this_trial / num_queries) * 100
        print(f"Trial {trial_num:>2}/{num_trials}: {proven_in_this_trial:>3}/{num_queries} queries proven ({success_rate:.2f}%)")

    end_time = time.time()
    
    # Calculate final summary statistics based on the independent trials
    avg_proven_count = np.mean(proven_counts_per_trial)
    std_dev_proven_count = np.std(proven_counts_per_trial)
    avg_success_rate = (avg_proven_count / num_queries)
    
    min_success_rate = (min(proven_counts_per_trial) / num_queries)
    max_success_rate = (max(proven_counts_per_trial) / num_queries)

    print(f"\n--- Random Agent Summary (Independent Trials) ---")
    print(f"Total queries per trial: {num_queries}")
    print(f"Number of trials: {num_trials}")
    print(f"Average Success Rate: {avg_success_rate:.2%}")
    print(f"Success Rate Range: {min_success_rate:.2%} - {max_success_rate:.2%}")
    print(f"Avg. Proven Queries per Trial: {avg_proven_count:.2f} ± {std_dev_proven_count:.2f}")
    print(f"Total time: {end_time - start_time:.2f} seconds")

    return proven_counts_per_trial


def check_provability_at_depth(state: List[Term],
                                n: int,
                                rules: List[Rule],
                                facts: List[Term],
                                index_manager: IndexManager = None,
                                is_train_data: bool = False,
                                max_atoms: int = 20,
                                verbose: bool = False) -> str:
    """
    Checks if a goal is provable within depth n by exploring all branches.
    
    Returns:
        'provable', 'not_provable', 'depth_limit_exceeded', or an error string.
    """
    if verbose:
        print(f"\nChecking query {state} at depth {n}...")

    current_states = [state]
    
    for depth in range(n):
        if verbose:
            print(f"Depth {depth}")
            print(f"Current states: {current_states}")

        next_generation_states = []
        
        for current_state in current_states:
            branch_next_states, _ = get_next_unification_python(
                current_state,
                facts_set=facts,
                facts_indexed=index_manager.fact_index,
                rules=rules,
                excluded_fact=state[0] if is_train_data else None,
                verbose=0,
                next_var_index=index_manager.variable_start_index,
            )
            
            if verbose:
                print(f"Branch next states: {branch_next_states}")
            
            # Check for successful proof
            if any(all(term.predicate == 'True' for term in branch_state)
                  for branch_state in branch_next_states):
                if verbose: print('provable')
                return 'provable'
            
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
            return 'not_provable'
            
        if depth == n - 1:
            if verbose:
                print('\ndepth_limit_exceeded')
            return 'depth_limit_exceeded'
            
        current_states = next_generation_states
    
    return 'error_should_not_arrive_here'


def calculate_provability_ratios_by_depth(
    queries: List[Term],
    max_depth_check: int,
    rules: List[Rule],
    facts: set,
    index_manager: IndexManager = None,
    is_train_data: bool = False,
    max_atoms: int = 20
) -> Tuple[Dict[int, float], Dict[str, int]]:
    """
    Calculates the ratio of provable queries for each depth from 1 to max_depth_check.
    
    Returns:
        Tuple of (ratios_by_depth, proved_queries)
    """
    start_time = time.time()
    num_queries = len(queries)
    
    print(f"Proving {num_queries} queries up to depth {max_depth_check}...")

    min_proven_depth = [float('inf')] * num_queries
    query_errors = [False] * num_queries
    ratios_by_depth = {}
    proved_queries = {}
    proven_by_depth = {}

    for depth in range(1, max_depth_check + 1):
        proven_in_this_round = 0
        errors_in_this_round = 0
        queries_checked_this_round = 0

        for i, goal in enumerate(queries):
            print(f"\rDepth {depth}/{max_depth_check}, query {i+1}/{num_queries}. "
                    f"Proven: {proven_in_this_round}", end='')
            
            # Skip already proven or errored queries
            if min_proven_depth[i] != float('inf') or query_errors[i]:
                continue
                
            queries_checked_this_round += 1
            status = check_provability_at_depth(
                [goal], depth, rules, facts=facts,
                index_manager=index_manager,
                is_train_data=is_train_data,
                max_atoms=max_atoms, verbose=False
            )

            if status == 'provable':
                min_proven_depth[i] = depth
                proved_queries[str(goal)] = depth
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


def load_queries(dataset_name: str, set_file: str, data_path: str, root_dir: str) -> Tuple[List[Term], List[Rule], List[Term], IndexManager]:
    """Load queries, rules, and facts from the specified dataset."""
    
    data_handler = DataHandler(
        dataset_name=dataset_name,
        base_path=data_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        corruption_mode='dynamic'
    )

    rules = data_handler.rules
    facts_set = set(data_handler.facts)

    index_manager = IndexManager(
        data_handler.constants,
        data_handler.predicates,
        max_total_vars=100,
        constants_images=set(),
        constant_images_no=0,
        rules=data_handler.rules,
        max_arity=data_handler.max_arity,
        device='cpu',
        padding_atoms=4
    )
    index_manager.build_fact_index(data_handler.facts)

    # Load queries based on set_file
    query_map = {
        'train': data_handler.train_queries,
        'valid': data_handler.valid_queries,
        'test': data_handler.test_queries
    }
    queries = query_map.get(set_file, [])
    
    print(f"Loaded {len(queries)} queries from {set_file}.txt")
    return queries, rules, facts_set, index_manager


if __name__ == "__main__":
    max_atoms = 5
    dataset_name = 'countries_s3'
    data_path = "./data/"
    root_dir = data_path + dataset_name + '/'

    mode = 'depth_check' # 'random_agent' or 'depth_check'

    for set_file in ['valid']: #['train','valid','test']:
        queries, rules, facts_set, index_manager = load_queries(
            dataset_name, set_file, data_path, root_dir
        )

        # Limit queries for testing
        # queries = queries[:500]

        print(f"Loaded {len(queries)} queries from {dataset_name} for set {set_file}")
        
        if mode == 'depth_check':
            max_depth_check = 4
            proven_by_depth, proved_queries = calculate_provability_ratios_by_depth(
                queries, max_depth_check, rules, facts_set, 
                index_manager, is_train_data=(set_file == 'train'), 
                max_atoms=max_atoms
            )
            print(f"\nProven by depth: {proven_by_depth}. Total proved queries: {len(proved_queries)}/ {len(queries)}")

            # print(f"\n--- Proven Queries ---")
            # print([(query, depth) for query, depth in proved_queries.items()])

            # # Save results
            # if proved_queries:
            #     output_file = root_dir + set_file + '_depths.txt'
            #     with open(output_file, 'w') as f:
            #         for query in queries:
            #             clean_query = str(query).replace(' ', '')
            #             depth = proved_queries.get(str(query), -1)
            #             f.write(f"{clean_query} {depth}\n")
            #     print(f"Saved results to {output_file}")

        elif mode != 'random_agent':
            num_random_trials = 10 # Number of random walks per query. ONLY FOR RANDOM AGENT (run_random_agent_on_queries)
            run_random_agent_on_queries(
                queries=queries, 
                max_depth=max_depth_check, 
                num_trials=num_random_trials,
                rules=rules, 
                facts=facts_set, 
                index_manager=index_manager, 
                is_train_data=(set_file == 'train'), 
                max_atoms=max_atoms
            )
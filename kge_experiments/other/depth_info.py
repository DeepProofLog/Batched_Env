from typing import List, Dict, Tuple
import time
import random
import numpy as np
from str_based.str_dataset import DataHandler
from str_based.str_index_manager import IndexManager
from str_based.str_utils import Term, Rule
from str_based.str_unification import get_next_unification_python
from str_based.str_dataset import get_filtered_queries

def check_provability_random_walk(state: List[Term],
                                  n: int,
                                  rules: List[Rule],
                                  facts: List[Term],
                                  index_manager: IndexManager = None,
                                  is_train_data: bool = False,
                                  max_atoms: int = 20) -> str:
    """
    Checks if a goal is provable by following a single random path up to depth n.
    Returns the status ('provable', 'not_provable', 'depth_limit_exceeded').
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

def print_branching_factor_analysis(pos_factors_tuples: List[Tuple[int, int]], 
                                    neg_factors_tuples: List[Tuple[int, int]]):
    """
    Calculates and prints a detailed analysis comparing branching factors
    for all positive vs. all negative queries.
    Factors >= 10 are bucketed in the distribution table.
    """
    print("\n" + "="*25 + " Overall Branching Factor Analysis " + "="*25)

    # Extract just the factor values for overall stats
    pos_factors = [factor for step, factor in pos_factors_tuples]
    neg_factors = [factor for step, factor in neg_factors_tuples]

    # --- Analysis for POSITIVE queries ---
    print("\n--- Statistics for ALL POSITIVE Queries (Provable) ---")
    if not pos_factors:
        print("No branching factor data collected for this category.")
    else:
        total_steps = len(pos_factors)
        print(f"Total Steps Analyzed: {total_steps}")
        print(f"Mean / Median: {np.mean(pos_factors):.2f} / {np.median(pos_factors):.1f}. Min / Max: {np.min(pos_factors)} / {np.max(pos_factors)}")
        p25, p75, p95 = np.percentile(pos_factors, [25, 75, 95])
        print(f"Percentiles: 25th={p25:.1f}, 75th={p75:.1f}, 95th={p95:.1f}")

    # --- Analysis for NEGATIVE queries ---
    print("\n--- Statistics for ALL NEGATIVE Queries (Non-Provable) ---")
    if not neg_factors:
        print("No branching factor data collected for this category.")
    else:
        total_steps = len(neg_factors)
        print(f"Total Steps Analyzed: {total_steps}")
        print(f"Mean / Median: {np.mean(neg_factors):.2f} / {np.median(neg_factors):.1f}. Min / Max: {np.min(neg_factors)} / {np.max(neg_factors)}")
        p25, p75, p95 = np.percentile(neg_factors, [25, 75, 95])
        print(f"Percentiles: 25th={p25:.1f}, 75th={p75:.1f}, 95th={p95:.1f}")

    # --- Combined Distribution with Bucketing ---
    print("\n--- Combined Distribution (Factor -> Count) ---")
    pos_unique, pos_counts = np.unique(pos_factors, return_counts=True)
    neg_unique, neg_counts = np.unique(neg_factors, return_counts=True)
    pos_dist = dict(zip(pos_unique, pos_counts))
    neg_dist = dict(zip(neg_unique, neg_counts))

    def get_bucket(factor):
        if factor < 10:
            return str(factor)
        start = 10 + 5 * ((factor - 10) // 5)
        return f"{start}-{start + 4}"

    pos_bucketed, neg_bucketed, all_buckets = {}, {}, set()
    for dist, bucketed in [(pos_dist, pos_bucketed), (neg_dist, neg_bucketed)]:
        for factor, count in dist.items():
            bucket = get_bucket(factor)
            bucketed[bucket] = bucketed.get(bucket, 0) + count
            all_buckets.add(bucket)

    def sort_key(bucket_str):
        return int(bucket_str.split('-')[0]) if '-' in bucket_str else int(bucket_str)
    
    sorted_buckets = sorted(list(all_buckets), key=sort_key)
    
    print(f"{'Factor':>8} | {'Positive (Provable)':>25} | {'Negative (Non-Provable)':>28}")
    print("-" * 68)

    total_pos = sum(pos_dist.values()) if pos_dist else 1
    total_neg = sum(neg_dist.values()) if neg_dist else 1

    for bucket in sorted_buckets:
        pos_count = pos_bucketed.get(bucket, 0)
        neg_count = neg_bucketed.get(bucket, 0)
        pos_perc = (pos_count / total_pos) * 100
        neg_perc = (neg_count / total_neg) * 100
        pos_str = f"{pos_count:<6} ({pos_perc:5.1f}%)"
        neg_str = f"{neg_count:<6} ({neg_perc:5.1f}%)"
        print(f"{bucket:>8} | {pos_str:>25} | {neg_str:>28}")
    print("="*75)


def print_bf_analysis_by_resolution_depth(
    pos_factors_by_idx: Dict[int, List[Tuple[int, int]]],
    min_proven_depth: List[float],
    max_depth: int
):
    """
    Prints a detailed BF analysis for positive queries, grouped by the depth
    at which they were proven. Depths with no proven queries are skipped.
    """
    print("\n" + "="*19 + " Branching Factor Analysis by Resolution Depth " + "="*19)

    def _print_stats_for_tuples(tuples: List[Tuple[int, int]]):
        """Helper to print overall and per-step stats for a list of factor tuples."""
        factors = [f for s, f in tuples]
        
        # Overall stats
        print(f"Total Steps Analyzed: {len(factors)}")
        print(f"Mean / Median: {np.mean(factors):.2f} / {np.median(factors):.1f}. Min / Max: {np.min(factors)} / {np.max(factors)}")
        p25_o, p75_o, p95_o = np.percentile(factors, [25, 75, 95])
        print(f"Percentiles: 25th={p25_o:.1f}, 75th={p75_o:.1f}, 95th={p95_o:.1f}")

        # Per-step analysis
        factors_per_step = {}
        for step, factor in tuples:
            if step not in factors_per_step:
                factors_per_step[step] = []
            factors_per_step[step].append(factor)
        
        print("BF per Step:")
        for step in sorted(factors_per_step.keys()):
            step_factors = factors_per_step[step]
            mean_bf = np.mean(step_factors)
            median_bf = np.median(step_factors)
            min_bf = np.min(step_factors)
            max_bf = np.max(step_factors)
            p25, p75, p95 = np.percentile(step_factors, [25, 75, 95])
            print(f"  - Step {step}: Mean {mean_bf:.2f}, Median {median_bf:.1f}, Min {min_bf}, Max {max_bf}. Percentiles: 25th={p25:.1f}, 75th={p75:.1f}, 95th={p95:.1f}")
    for d in range(1, max_depth + 1):
        # Find positive queries resolved at this specific depth
        pos_indices_at_d = [i for i, proof_depth in enumerate(min_proven_depth) if proof_depth == d]
        num_pos_at_d = len(pos_indices_at_d)

        # If no queries were proven at this depth, skip the entire section
        if num_pos_at_d == 0:
            continue
        
        print(f"\n{'='*10} Analysis for {num_pos_at_d} Queries Resolved at Depth {d} {'='*10}")        
        pos_tuples_at_d = []
        for i in pos_indices_at_d:
            pos_tuples_at_d.extend(pos_factors_by_idx.get(i, []))
        _print_stats_for_tuples(pos_tuples_at_d)

    print("\n" + "="*75)


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
    Calculates success rate for a random agent over multiple independent trials.
    Branching factor analysis has been moved to the exhaustive search function.
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
        
        proven_counts_per_trial.append(proven_in_this_trial)
        success_rate = (proven_in_this_trial / num_queries) * 100
        print(f"Trial {trial_num:>2}/{num_trials}: {proven_in_this_trial:>3}/{num_queries} queries proven ({success_rate:.2f}%)")

    end_time = time.time()
    
    # --- Main Summary Statistics Calculation ---
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
    
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")

    return proven_counts_per_trial


def check_provability_at_depth(state: List[Term],
                               n: int,
                               rules: List[Rule],
                               facts: List[Term],
                               index_manager: IndexManager = None,
                               is_train_data: bool = False,
                               max_atoms: int = 20,
                               verbose: bool = False) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Checks if a goal is provable within depth n by exploring all branches.
    Also returns all branching factors encountered during the exploration.
    
    Returns:
        Tuple of (status, branching_factors).
        Status can be 'provable', 'not_provable', 'depth_limit_exceeded'.
        branching_factors is a list of (step, factor) tuples.
    """
    if verbose:
        print(f"\nChecking query {state} at depth {n}...")

    current_states = [state]
    branching_factors = []
    
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
            
            branching_factors.append((depth, len(branch_next_states)))
            
            if verbose:
                print(f"Branch next states: {branch_next_states}")
            
            # Check for successful proof
            if any(all(term.predicate == 'True' for term in branch_state)
                   for branch_state in branch_next_states):
                if verbose: print('provable')
                return 'provable', branching_factors
            
            # Filter out falsified and oversized branches
            valid_next_states = []
            for branch_state in branch_next_states:
                if (branch_state and
                    not any(term.predicate == 'False' for term in branch_state) and
                    len(branch_state) <= max_atoms):
                    valid_next_states.append(branch_state)
            next_generation_states.extend(valid_next_states)
        
        # Check termination conditions
        if not next_generation_states:
            if verbose: print('\nnot_provable')
            return 'not_provable', branching_factors
            
        if depth == n - 1:
            if verbose: print('\ndepth_limit_exceeded')
            return 'depth_limit_exceeded', branching_factors
            
        current_states = next_generation_states
    
    return 'error_should_not_arrive_here', branching_factors


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
    Calculates the ratio of provable queries for each depth from 1 to max_depth_check
    and performs a detailed branching factor analysis for provable vs. non-provable queries.
    """
    start_time = time.time()
    num_queries = len(queries)
    
    print(f"Proving {num_queries} queries up to depth {max_depth_check}...")

    min_proven_depth = [float('inf')] * num_queries
    query_errors = [False] * num_queries
    proved_queries = {}
    proven_by_depth = {}

    # Store factors for analysis
    positive_factors_by_query_idx = {}
    negative_factors_by_query_idx = {}

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
            status, factors = check_provability_at_depth(
                [goal], depth, rules, facts=facts,
                index_manager=index_manager,
                is_train_data=is_train_data,
                max_atoms=max_atoms, verbose=False
            )

            if status == 'provable':
                min_proven_depth[i] = depth
                proved_queries[str(goal)] = depth
                proven_in_this_round += 1
                # Store factors from the first successful proof run
                positive_factors_by_query_idx[i] = factors
            elif status.startswith('error_'):
                query_errors[i] = True
                errors_in_this_round += 1
            else: # not_provable or depth_limit_exceeded
                # Store factors from the last (deepest) unsuccessful run
                negative_factors_by_query_idx[i] = factors

        # Calculate cumulative proven count for reporting
        provable_count = sum(1 for d in min_proven_depth if d <= depth)
        proven_by_depth[depth] = proven_in_this_round

        print(f"\nDepth {depth}: Checked {queries_checked_this_round} queries.")
        print(f"        Errors: {errors_in_this_round}. Newly proven: {proven_in_this_round}. ")
        print(f"        Cumulative proven <= depth {depth}: {provable_count}/{num_queries}\n")

        # Early exit if all queries resolved
        if all((d != float('inf') or err) for d, err in zip(min_proven_depth, query_errors)):
            print(f"All queries resolved by depth {depth}. Stopping.")
            break
            
    # --- Aggregate factors for final analysis ---
    all_positive_factors = []
    for factors in positive_factors_by_query_idx.values():
        all_positive_factors.extend(factors)

    # Re-build negative_factors_by_query_idx to only include true negatives
    final_negative_factors_by_idx = {
        i: factors for i, factors in negative_factors_by_query_idx.items()
        if min_proven_depth[i] == float('inf')
    }
    all_negative_factors = []
    for factors in final_negative_factors_by_idx.values():
        all_negative_factors.extend(factors)

    # --- Print Analyses ---
    print_branching_factor_analysis(all_positive_factors, all_negative_factors)
    
    print_bf_analysis_by_resolution_depth(
        positive_factors_by_query_idx,
        min_proven_depth,
        max_depth_check
    )

    # --- Final Summary ---
    total_provable = len(proved_queries)
    total_errors = sum(query_errors)
    end_time = time.time()

    print(f"\n--- Proof Summary ---")
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
    rules = index_manager.rules_by_pred

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
    max_atoms = 6
    max_depth_check = 7
    dataset_name = 'family'
    data_path = "./data/"
    root_dir = data_path + dataset_name + '/'

    # Set mode to 'depth_check' to run the exhaustive search with branching factor analysis
    # Set mode to 'random_agent' to run the simplified random walk simulation
    mode = 'depth_check' 

    for set_file in ['train']:
        queries, rules, facts_set, index_manager = load_queries(
            dataset_name, set_file, data_path, root_dir
        )

        filter_depth = False
        if filter_depth:
            filter_depth = {3}
            print(f'Filtering queries to depth {filter_depth}')
            queries = get_filtered_queries(root_dir+set_file+'_depths.txt', filter_depth, set_file)

        # You can uncomment the line below to test with a smaller subset of queries
        # queries = queries[:10]

        print(f"Loaded {len(queries)} queries from {dataset_name} for set {set_file}")
        
        if mode == 'depth_check':
            proven_by_depth, proved_queries = calculate_provability_ratios_by_depth(
                queries, max_depth_check, rules, facts_set, 
                index_manager, is_train_data=(set_file == 'train'), 
                max_atoms=max_atoms
            )
            print(f"\nProven by depth: {proven_by_depth}. Total proved queries: {len(proved_queries)}/ {len(queries)} (% {len(proved_queries)/len(queries)*100:.2f}%)")

            # # Save results
            # if proved_queries:
            #     output_file = root_dir + set_file + '_depths.txt'
            #     with open(output_file, 'w') as f:
            #         for query in queries:
            #             clean_query = str(query).replace(' ', '')
            #             depth = proved_queries.get(str(query), -1)
            #             f.write(f"{clean_query} {depth}\n")
            #     print(f"Saved results to {output_file}")

        elif mode == 'random_agent':           
            num_random_trials = 100 # Number of random walks per query
            print('Starting random agent evaluation...')
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
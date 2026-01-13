"""
Generate depth files for datasets.
Runs exhaustive search to find minimum proof depth for each query.
Output format: "query depth" where depth=-1 means non-provable.
"""
from typing import List, Dict, Tuple, Set, Optional, FrozenSet
import time
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sb3.sb3_dataset import DataHandler
from sb3.sb3_index_manager import IndexManager
from sb3.sb3_utils import Term, Rule
from sb3.sb3_unification import get_next_unification_python


def check_provability_at_depth(
    state: List[Term],
    n: int,
    rules: Dict,
    facts: FrozenSet[Term],
    index_manager: IndexManager,
    is_train_data: bool = False,
    max_atoms: int = 20,
    verbose: bool = False
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Checks if a goal is provable within depth n by exploring all branches.

    Returns:
        Tuple of (status, branching_factors).
        Status can be 'provable', 'not_provable', 'depth_limit_exceeded'.
    """
    if verbose:
        print(f"\nChecking query {state} at depth {n}...")

    current_states = [state]
    branching_factors = []

    for depth in range(n):
        if verbose:
            print(f"Depth {depth}, states: {len(current_states)}")

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
            if verbose: print('not_provable')
            return 'not_provable', branching_factors

        if depth == n - 1:
            if verbose: print('depth_limit_exceeded')
            return 'depth_limit_exceeded', branching_factors

        current_states = next_generation_states

    return 'error_should_not_arrive_here', branching_factors


def load_queries(
    dataset_name: str,
    split: str,
    data_path: str
) -> Tuple[List[Term], Dict, FrozenSet[Term], IndexManager]:
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

    facts_set = frozenset(data_handler.facts)

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

    # Load queries based on split
    query_map = {
        'train': data_handler.train_queries,
        'valid': data_handler.valid_queries,
        'test': data_handler.test_queries
    }
    queries = query_map.get(split, [])

    print(f"Loaded {len(queries)} queries from {split}.txt")
    return queries, rules, facts_set, index_manager


def generate_depths_for_dataset(
    dataset_name: str,
    splits: List[str],
    data_path: str,
    max_depth_check: int = 7,
    max_atoms: int = 20,
):
    """
    Generate depth files for a dataset.

    Args:
        dataset_name: Name of dataset (e.g., 'nations', 'umls')
        splits: List of splits to process (e.g., ['train', 'valid', 'test'])
        data_path: Base path to data directory
        max_depth_check: Maximum depth to search
        max_atoms: Maximum atoms allowed in a state
    """
    root_dir = os.path.join(data_path, dataset_name)

    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Max depth: {max_depth_check}, Max atoms: {max_atoms}")
    print(f"{'='*60}\n")

    for split in splits:
        print(f"\n--- Processing {split} split ---")
        start_time = time.time()

        # Load queries
        queries, rules, facts_set, index_manager = load_queries(
            dataset_name, split, data_path
        )

        if not queries:
            print(f"No queries found for {split}, skipping.")
            continue

        num_queries = len(queries)
        is_train = (split == 'train')

        # Track results
        proved_queries: Dict[str, int] = {}
        min_proven_depth = [float('inf')] * num_queries
        query_errors = [False] * num_queries

        # Process each depth level
        for depth in range(1, max_depth_check + 1):
            proven_in_round = 0
            queries_checked = 0

            for i, goal in enumerate(queries):
                # Skip already proven or errored
                if min_proven_depth[i] != float('inf') or query_errors[i]:
                    continue

                queries_checked += 1
                status, _ = check_provability_at_depth(
                    [goal], depth, rules, facts=facts_set,
                    index_manager=index_manager,
                    is_train_data=is_train,
                    max_atoms=max_atoms,
                    verbose=False
                )

                if status == 'provable':
                    min_proven_depth[i] = depth
                    proved_queries[str(goal)] = depth
                    proven_in_round += 1
                elif status.startswith('error_'):
                    query_errors[i] = True

                # Progress update
                if (i + 1) % 50 == 0 or i == num_queries - 1:
                    cumulative = sum(1 for d in min_proven_depth if d <= depth)
                    print(f"\rDepth {depth}/{max_depth_check}: {i+1}/{num_queries} queries, "
                          f"cumulative proven: {cumulative}", end='', flush=True)

            cumulative_proven = sum(1 for d in min_proven_depth if d <= depth)
            print(f"\nDepth {depth}: checked {queries_checked}, newly proven: {proven_in_round}, "
                  f"cumulative: {cumulative_proven}/{num_queries}")

            # Early exit if all resolved
            if all((d != float('inf') or err) for d, err in zip(min_proven_depth, query_errors)):
                print(f"All queries resolved by depth {depth}.")
                break

        # Save results
        output_file = os.path.join(root_dir, f'{split}_depths.txt')
        with open(output_file, 'w') as f:
            for i, goal in enumerate(queries):
                clean_query = str(goal).replace(' ', '')
                depth_val = proved_queries.get(str(goal), -1)
                f.write(f"{clean_query} {depth_val}\n")

        elapsed = time.time() - start_time
        total_proven = len(proved_queries)
        total_errors = sum(query_errors)

        print(f"\n{split} Summary:")
        print(f"  Total queries: {num_queries}")
        print(f"  Provable: {total_proven} ({total_proven/num_queries:.1%})")
        print(f"  Non-provable: {num_queries - total_proven - total_errors}")
        print(f"  Errors: {total_errors}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    # Configuration
    datasets = ['nations'] # 'umls'
    splits = ['train', 'valid', 'test']
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'

    # Dataset-specific settings
    dataset_configs = {
        'nations': {'max_depth_check': 7, 'max_atoms': 20},
        'umls': {'max_depth_check': 7, 'max_atoms': 20},
    }

    for dataset in datasets:
        config = dataset_configs.get(dataset, {'max_depth_check': 7, 'max_atoms': 20})
        generate_depths_for_dataset(
            dataset_name=dataset,
            splits=splits,
            data_path=data_path,
            **config
        )

    print("\n" + "="*60)
    print("All datasets processed!")
    print("="*60)

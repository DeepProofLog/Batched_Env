"""
Optimized depth generation using SB3 unification engine.
Uses true BFS (no depth restart) with state deduplication.

Key optimizations:
- Single BFS pass per query (no restarting from scratch at each depth)
- State deduplication via frozenset hashing
- Early termination on proof
- deque for O(1) popleft
- Local variable caching
"""
from typing import List, Dict, Tuple, Set, Optional, FrozenSet
from collections import deque
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sb3.sb3_dataset import DataHandler
from sb3.sb3_index_manager import IndexManager
from sb3.sb3_utils import Term, Rule
from sb3.sb3_unification import get_next_unification_python


def check_provability_bfs(
    initial_query: Term,
    rules: Dict,
    facts: FrozenSet[Term],
    index_manager: IndexManager,
    is_train_data: bool = False,
    max_depth: int = 7,
    max_atoms: int = 20,
) -> int:
    """
    Check provability using true BFS with deduplication.
    Returns minimum proof depth, or -1 if not provable within max_depth.
    """
    # Depth 0: check if query is a direct fact
    if not is_train_data and initial_query in facts:
        return 0

    # Cache for speed
    excluded_fact = initial_query if is_train_data else None
    fact_index = index_manager.fact_index
    var_start = index_manager.variable_start_index
    _frozenset = frozenset  # Local reference

    # BFS with deque for O(1) popleft
    initial_state = [initial_query]
    frontier = deque([(initial_state, 0)])

    # Track visited states
    visited = {_frozenset(initial_state)}
    visited_add = visited.add  # Cache method

    while frontier:
        current_state, current_depth = frontier.popleft()

        if current_depth >= max_depth:
            continue

        # Get all successor states
        next_states, _ = get_next_unification_python(
            current_state,
            facts_set=facts,
            facts_indexed=fact_index,
            rules=rules,
            excluded_fact=excluded_fact,
            verbose=0,
            next_var_index=var_start,
        )

        next_depth = current_depth + 1

        for next_state in next_states:
            # Quick check for empty state
            if not next_state:
                continue

            # Check predicates
            first_pred = next_state[0].predicate

            # Check for proof (all atoms are True)
            if first_pred == 'True' and all(t.predicate == 'True' for t in next_state):
                return next_depth

            # Skip if any atom is False (dead end)
            if first_pred == 'False' or any(t.predicate == 'False' for t in next_state):
                continue

            # Skip if too many atoms
            if len(next_state) > max_atoms:
                continue

            # Deduplicate
            state_key = _frozenset(next_state)
            if state_key in visited:
                continue
            visited_add(state_key)

            frontier.append((next_state, next_depth))

    return -1


def load_data(dataset_name: str, data_path: str):
    """Load dataset and create index manager."""
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

    return data_handler, facts_set, index_manager, rules


def generate_depths_for_dataset(
    dataset_name: str,
    splits: List[str],
    data_path: str,
    max_depth_check: int = 7,
    max_atoms: int = 20,
    incremental: bool = False,
):
    """Generate depth files for a dataset.

    Args:
        incremental: If True, process depth-by-depth with saves at each level.
    """
    root_dir = os.path.join(data_path, dataset_name)

    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Max depth: {max_depth_check}, Max atoms: {max_atoms}")
    print(f"Mode: {'incremental' if incremental else 'full BFS'}")
    print(f"{'='*60}\n")

    # Load data once
    data_handler, facts_set, index_manager, rules = load_data(dataset_name, data_path)

    for split in splits:
        print(f"\n--- Processing {split} split ---")
        start_time = time.time()

        # Get queries
        query_map = {
            'train': data_handler.train_queries,
            'valid': data_handler.valid_queries,
            'test': data_handler.test_queries
        }
        queries = query_map.get(split, [])

        if not queries:
            print(f"No queries found for {split}, skipping.")
            continue

        num_queries = len(queries)
        is_train = (split == 'train')
        output_file = os.path.join(root_dir, f'{split}_depths.txt')

        if incremental:
            _generate_incremental(
                queries, rules, facts_set, index_manager,
                is_train, max_depth_check, max_atoms,
                output_file, start_time
            )
        else:
            _generate_full_bfs(
                queries, rules, facts_set, index_manager,
                is_train, max_depth_check, max_atoms,
                output_file, start_time
            )


def _generate_full_bfs(queries, rules, facts_set, index_manager,
                       is_train, max_depth, max_atoms, output_file, start_time):
    """Original full BFS mode - process all queries to max depth."""
    num_queries = len(queries)
    depths = []
    proven_count = 0

    for i, query in enumerate(queries):
        depth = check_provability_bfs(
            query, rules, facts_set, index_manager,
            is_train_data=is_train,
            max_depth=max_depth,
            max_atoms=max_atoms,
        )
        depths.append(depth)
        if depth >= 0:
            proven_count += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            qps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"\r  {i+1}/{num_queries}, proven: {proven_count}, {qps:.1f} q/s", end='', flush=True)

    print()
    _save_and_report(queries, depths, output_file, start_time)


def _generate_incremental(queries, rules, facts_set, index_manager,
                          is_train, max_depth, max_atoms, output_file, start_time):
    """Incremental mode - process depth-by-depth with saves at each level."""
    num_queries = len(queries)
    depths = [-1] * num_queries  # -1 = not yet proven
    unproven_indices = set(range(num_queries))

    # Check depth 0 (direct facts) first
    if not is_train:
        for i in list(unproven_indices):
            if queries[i] in facts_set:
                depths[i] = 0
                unproven_indices.discard(i)
        if num_queries - len(unproven_indices) > 0:
            print(f"Depth 0: {num_queries - len(unproven_indices)} direct facts")

    # Process depth by depth
    for target_depth in range(1, max_depth + 1):
        if not unproven_indices:
            print(f"All queries proven! Stopping at depth {target_depth - 1}")
            break

        depth_start = time.time()
        proven_this_depth = 0
        checked = 0

        for i in list(unproven_indices):
            query = queries[i]
            result = check_provability_bfs(
                query, rules, facts_set, index_manager,
                is_train_data=is_train,
                max_depth=target_depth,
                max_atoms=max_atoms,
            )
            checked += 1

            if result >= 0:
                depths[i] = result
                unproven_indices.discard(i)
                proven_this_depth += 1

            # Progress within depth
            if checked % 100 == 0:
                elapsed = time.time() - depth_start
                qps = checked / elapsed if elapsed > 0 else 0
                print(f"\r  Depth {target_depth}: {checked}/{len(unproven_indices) + checked}, "
                      f"+{proven_this_depth} proven, {qps:.1f} q/s", end='', flush=True)

        depth_elapsed = time.time() - depth_start
        total_proven = num_queries - len(unproven_indices)

        # Report for this depth
        print(f"\r  Depth {target_depth}: +{proven_this_depth} proven | "
              f"Total: {total_proven}/{num_queries} ({total_proven/num_queries:.1%}) | "
              f"Time: {depth_elapsed:.1f}s")

        # Save intermediate results
        _save_depths(queries, depths, output_file)
        print(f"    Saved to: {output_file}")

        # Hint if no progress
        if proven_this_depth == 0:
            print(f"    [No new proofs at depth {target_depth} - consider stopping if this continues]")

    print()
    _save_and_report(queries, depths, output_file, start_time)


def _save_depths(queries, depths, output_file):
    """Save current depth results to file."""
    with open(output_file, 'w') as f:
        for query, depth in zip(queries, depths):
            clean_query = str(query).replace(' ', '')
            f.write(f"{clean_query} {depth}\n")


def _save_and_report(queries, depths, output_file, start_time):
    """Save final results and print summary."""
    num_queries = len(queries)
    _save_depths(queries, depths, output_file)

    elapsed = time.time() - start_time
    proven_count = sum(1 for d in depths if d >= 0)

    # Depth distribution
    depth_counts = {}
    for d in depths:
        depth_counts[d] = depth_counts.get(d, 0) + 1

    print(f"Summary:")
    print(f"  Total queries: {num_queries}")
    print(f"  Provable: {proven_count} ({proven_count/num_queries:.1%})")
    print(f"  Non-provable: {num_queries - proven_count}")
    print(f"  Depth distribution: {dict(sorted(depth_counts.items()))}")
    print(f"  Time: {elapsed:.1f}s ({num_queries/elapsed:.1f} q/s)")
    print(f"  Saved to: {output_file}")


# Keep old function for compatibility/testing
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
    """Legacy function - check provability at specific depth."""
    excluded_fact = state[0] if is_train_data else None

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
                excluded_fact=excluded_fact,
                verbose=0,
                next_var_index=index_manager.variable_start_index,
            )

            branching_factors.append((depth, len(branch_next_states)))

            if any(all(term.predicate == 'True' for term in branch_state)
                   for branch_state in branch_next_states):
                if verbose: print('provable')
                return 'provable', branching_factors

            valid_next_states = []
            for branch_state in branch_next_states:
                if (branch_state and
                    not any(term.predicate == 'False' for term in branch_state) and
                    len(branch_state) <= max_atoms):
                    valid_next_states.append(branch_state)
            next_generation_states.extend(valid_next_states)

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
    """Load queries - legacy function for compatibility."""
    data_handler, facts_set, index_manager, rules = load_data(dataset_name, data_path)

    query_map = {
        'train': data_handler.train_queries,
        'valid': data_handler.valid_queries,
        'test': data_handler.test_queries
    }
    queries = query_map.get(split, [])

    print(f"Loaded {len(queries)} queries from {split}.txt")
    return queries, rules, facts_set, index_manager


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate depth files using optimized SB3 BFS')
    parser.add_argument('--datasets', nargs='+', default=['wn18rr'],
                        help='Datasets to process')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        help='Splits to process')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum proof depth')
    parser.add_argument('--max_atoms', type=int, default=20,
                        help='Maximum atoms per state')
    parser.add_argument('--incremental', default=True,
                        help='Process depth-by-depth with saves at each level (allows early stopping)')
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'

    for dataset in args.datasets:
        generate_depths_for_dataset(
            dataset_name=dataset,
            splits=args.splits,
            data_path=data_path,
            max_depth_check=args.max_depth,
            max_atoms=args.max_atoms,
            incremental=args.incremental,
        )

    print("\n" + "="*60)
    print("All datasets processed!")
    print("="*60)

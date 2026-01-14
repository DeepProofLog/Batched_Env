"""
Fast depth generation using integer-based unification.

Usage:
    python generate_depths_fast.py --datasets family --splits valid --max_depth 5

Compares results against SB3 for correctness verification.
"""
from typing import List, Dict, Tuple
from collections import deque
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sb3.sb3_dataset import DataHandler
from fast_unifier import FastUnifier


def load_data(dataset_name: str, data_path: str) -> Tuple[DataHandler, FastUnifier]:
    """Load dataset and create fast unifier."""
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

    unifier = FastUnifier(data_handler)
    return data_handler, unifier


def generate_depths_for_dataset(
    dataset_name: str,
    splits: List[str],
    data_path: str,
    max_depth_check: int = 7,
    max_atoms: int = 20,
    incremental: bool = False,
    verify: bool = False,
):
    """Generate depth files for a dataset using fast unifier."""
    root_dir = os.path.join(data_path, dataset_name)

    print(f"\n{'='*60}")
    print(f"FAST UNIFIER - Processing dataset: {dataset_name}")
    print(f"Max depth: {max_depth_check}, Max atoms: {max_atoms}")
    print(f"Mode: {'incremental' if incremental else 'full BFS'}")
    print(f"{'='*60}\n")

    # Load data and create unifier
    data_handler, unifier = load_data(dataset_name, data_path)

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
        output_file = os.path.join(root_dir, f'{split}_depths_fast.txt')

        if incremental:
            depths = _generate_incremental(
                queries, unifier, is_train,
                max_depth_check, max_atoms, output_file, start_time
            )
        else:
            depths = _generate_full_bfs(
                queries, unifier, is_train,
                max_depth_check, max_atoms, start_time
            )

        # Save results
        _save_depths(queries, depths, output_file)

        elapsed = time.time() - start_time
        proven_count = sum(1 for d in depths if d >= 0)

        # Depth distribution
        depth_counts: Dict[int, int] = {}
        for d in depths:
            depth_counts[d] = depth_counts.get(d, 0) + 1

        print(f"\n{split} Summary:")
        print(f"  Total queries: {num_queries}")
        print(f"  Provable: {proven_count} ({proven_count/num_queries:.1%})")
        print(f"  Non-provable: {num_queries - proven_count}")
        print(f"  Depth distribution: {dict(sorted(depth_counts.items()))}")
        print(f"  Time: {elapsed:.1f}s ({num_queries/elapsed:.1f} q/s)")
        print(f"  Saved to: {output_file}")

        # Verify against SB3 if requested
        if verify:
            _verify_against_sb3(
                dataset_name, split, queries, depths,
                data_path, max_depth_check, max_atoms
            )


def _generate_full_bfs(
    queries, unifier: FastUnifier, is_train: bool,
    max_depth: int, max_atoms: int, start_time: float
) -> List[int]:
    """Full BFS mode - process all queries to max depth."""
    num_queries = len(queries)
    depths: List[int] = []
    proven_count = 0

    for i, query in enumerate(queries):
        depth = unifier.check_provability(
            query,
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
    return depths


def _generate_incremental(
    queries, unifier: FastUnifier, is_train: bool,
    max_depth: int, max_atoms: int, output_file: str, start_time: float
) -> List[int]:
    """Incremental mode - process depth-by-depth with saves."""
    num_queries = len(queries)
    depths = [-1] * num_queries
    unproven_indices = set(range(num_queries))

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
            result = unifier.check_provability(
                query,
                is_train_data=is_train,
                max_depth=target_depth,
                max_atoms=max_atoms,
            )
            checked += 1

            if result >= 0:
                depths[i] = result
                unproven_indices.discard(i)
                proven_this_depth += 1

            if checked % 100 == 0:
                elapsed = time.time() - depth_start
                qps = checked / elapsed if elapsed > 0 else 0
                print(f"\r  Depth {target_depth}: {checked}/{len(unproven_indices) + checked}, "
                      f"+{proven_this_depth} proven, {qps:.1f} q/s", end='', flush=True)

        depth_elapsed = time.time() - depth_start
        total_proven = num_queries - len(unproven_indices)

        print(f"\r  Depth {target_depth}: +{proven_this_depth} proven | "
              f"Total: {total_proven}/{num_queries} ({total_proven/num_queries:.1%}) | "
              f"Time: {depth_elapsed:.1f}s")

        # Save intermediate results
        _save_depths(queries, depths, output_file)
        print(f"    Saved to: {output_file}")

        if proven_this_depth == 0:
            print(f"    [No new proofs at depth {target_depth}]")

    return depths


def _save_depths(queries, depths: List[int], output_file: str):
    """Save depth results to file."""
    with open(output_file, 'w') as f:
        for query, depth in zip(queries, depths):
            clean_query = str(query).replace(' ', '')
            f.write(f"{clean_query} {depth}\n")


def _verify_against_sb3(
    dataset_name: str, split: str, queries, depths: List[int],
    data_path: str, max_depth: int, max_atoms: int
):
    """Verify results against SB3 unifier."""
    print(f"\n  Verifying against SB3...")

    # Import SB3 unifier
    from generate_depths_sb3 import check_provability_bfs, load_data as load_data_sb3

    # Load SB3 data
    _, facts_set, index_manager, rules = load_data_sb3(dataset_name, data_path)
    is_train = (split == 'train')

    # Compare results
    match = 0
    mismatch = 0
    mismatches: List[Tuple[str, int, int]] = []

    for i, query in enumerate(queries):
        sb3_depth = check_provability_bfs(
            query, rules, facts_set, index_manager,
            is_train_data=is_train,
            max_depth=max_depth,
            max_atoms=max_atoms,
        )
        fast_depth = depths[i]

        if sb3_depth == fast_depth:
            match += 1
        else:
            mismatch += 1
            if len(mismatches) < 5:
                mismatches.append((str(query), sb3_depth, fast_depth))

    if mismatches:
        print(f"  MISMATCHES ({mismatch}):")
        for q, sb3_d, fast_d in mismatches:
            print(f"    {q}: SB3={sb3_d}, Fast={fast_d}")

    pct = match / len(queries) * 100
    status = "PASS" if pct == 100 else "FAIL"
    print(f"  Verification: {match}/{len(queries)} ({pct:.1f}%) [{status}]")


def compare_speed(
    dataset_name: str,
    split: str,
    data_path: str,
    num_queries: int = 600,
    max_depth: int = 4,
    max_atoms: int = 20,
):
    """Compare speed of fast unifier vs SB3."""
    print(f"\n{'='*60}")
    print(f"SPEED COMPARISON: {dataset_name} {split}")
    print(f"Queries: {num_queries}, Max depth: {max_depth}")
    print(f"{'='*60}\n")

    # Load fast unifier
    data_handler, unifier = load_data(dataset_name, data_path)

    query_map = {
        'train': data_handler.train_queries,
        'valid': data_handler.valid_queries,
        'test': data_handler.test_queries
    }
    queries = query_map.get(split, [])[:num_queries]
    is_train = (split == 'train')

    # Fast unifier
    print("[FAST UNIFIER]")
    start = time.time()
    fast_results = {}
    for query in queries:
        depth = unifier.check_provability(
            query, is_train_data=is_train,
            max_depth=max_depth, max_atoms=max_atoms
        )
        if depth >= 0:
            fast_results[str(query)] = depth
    fast_time = time.time() - start
    print(f"  Time: {fast_time:.2f}s, {len(queries)/fast_time:.1f} q/s, proven: {len(fast_results)}")

    # SB3 unifier
    print("\n[SB3 UNIFIER]")
    from generate_depths_sb3 import check_provability_bfs, load_data as load_data_sb3
    _, facts_set, index_manager, rules = load_data_sb3(dataset_name, data_path)

    start = time.time()
    sb3_results = {}
    for query in queries:
        depth = check_provability_bfs(
            query, rules, facts_set, index_manager,
            is_train_data=is_train,
            max_depth=max_depth, max_atoms=max_atoms
        )
        if depth >= 0:
            sb3_results[str(query)] = depth
    sb3_time = time.time() - start
    print(f"  Time: {sb3_time:.2f}s, {len(queries)/sb3_time:.1f} q/s, proven: {len(sb3_results)}")

    # Comparison
    print(f"\n{'='*60}")
    speedup = sb3_time / fast_time if fast_time > 0 else 0
    print(f"SPEEDUP: {speedup:.2f}x")

    # Verify correctness
    match = sum(1 for q in queries if fast_results.get(str(q), -1) == sb3_results.get(str(q), -1))
    print(f"CORRECTNESS: {match}/{len(queries)} ({match/len(queries):.1%})")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate depth files using fast unifier')
    parser.add_argument('--datasets', nargs='+', default=['family'],
                        help='Datasets to process')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        help='Splits to process')
    parser.add_argument('--max_depth', type=int, default=7,
                        help='Maximum proof depth')
    parser.add_argument('--max_atoms', type=int, default=20,
                        help='Maximum atoms per state')
    parser.add_argument('--incremental', action='store_true',
                        help='Process depth-by-depth with saves')
    parser.add_argument('--verify', action='store_true',
                        help='Verify results against SB3')
    parser.add_argument('--compare', action='store_true',
                        help='Run speed comparison against SB3')
    parser.add_argument('--num_queries', type=int, default=600,
                        help='Number of queries for comparison')
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'

    if args.compare:
        for dataset in args.datasets:
            for split in args.splits:
                compare_speed(
                    dataset_name=dataset,
                    split=split,
                    data_path=data_path,
                    num_queries=args.num_queries,
                    max_depth=args.max_depth,
                    max_atoms=args.max_atoms,
                )
    else:
        for dataset in args.datasets:
            generate_depths_for_dataset(
                dataset_name=dataset,
                splits=args.splits,
                data_path=data_path,
                max_depth_check=args.max_depth,
                max_atoms=args.max_atoms,
                incremental=args.incremental,
                verify=args.verify,
            )

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

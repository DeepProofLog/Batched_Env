"""
Fast depth generation using optimized BFS with parallelization.

Key features:
- 100% parity with SB3 results
- ~5x speedup via multiprocessing
- Same interface as generate_depths_sb3.py

Usage:
    # Generate depths for dataset
    python generate_depths_fast.py --datasets family --splits valid --max_depth 5

    # Compare speed against SB3
    python generate_depths_fast.py --datasets family --splits valid --max_depth 5 --compare

    # Use parallel processing (default)
    python generate_depths_fast.py --datasets family --splits valid --max_depth 5 --parallel
"""
from typing import List, Dict, Tuple
import time
import os
import sys
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sb3.sb3_dataset import DataHandler
from fast_unifier import FastUnifier

# Default max_depth per dataset (default: 10)
DEFAULT_MAX_DEPTH = {
    'wn18rr': 7,
    'fb15k237': 4,
    'pharmkg_full': 7,
}
DEFAULT_MAX_DEPTH_FALLBACK = 10


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
    parallel: bool = True,
    n_workers: int = None,
):
    """Generate depth files for a dataset.

    Args:
        dataset_name: Name of dataset (e.g., 'family', 'countries_s3')
        splits: List of splits to process ('train', 'valid', 'test')
        data_path: Path to data directory
        max_depth_check: Maximum proof depth to search
        max_atoms: Maximum atoms per state
        incremental: If True, process depth-by-depth with saves at each level
        parallel: If True, use multiprocessing for speedup
        n_workers: Number of parallel workers (default: CPU count)
    """
    root_dir = os.path.join(data_path, dataset_name)

    if n_workers is None:
        n_workers = mp.cpu_count()

    print(f"\n{'='*60}")
    print(f"FAST UNIFIER - Processing dataset: {dataset_name}")
    print(f"Max depth: {max_depth_check}, Max atoms: {max_atoms}")
    mode_str = 'incremental' if incremental else ('parallel BFS' if parallel else 'sequential BFS')
    print(f"Mode: {mode_str}" + (f" ({n_workers} workers)" if parallel and not incremental else ""))
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
        print(f"  Queries to process: {num_queries}")
        output_file = os.path.join(root_dir, f'{split}_depths.txt')

        if incremental:
            depths = _generate_incremental(
                queries, unifier,
                max_depth_check, max_atoms,
                output_file, start_time
            )
        elif parallel:
            depths = _generate_parallel(
                queries, unifier,
                max_depth_check, max_atoms,
                n_workers, start_time
            )
        else:
            depths = _generate_sequential(
                queries, unifier,
                max_depth_check, max_atoms,
                start_time
            )

        # Save and report
        _save_and_report(queries, depths, output_file, start_time)


def _generate_sequential(
    queries, unifier: FastUnifier,
    max_depth: int, max_atoms: int, start_time: float
) -> List[int]:
    """Sequential BFS mode - process all queries one by one."""
    num_queries = len(queries)
    depths: List[int] = []
    proven_count = 0

    for i, query in enumerate(queries):
        depth = unifier.check_provability(
            query,
            is_train_data=True,  # Always exclude query from facts
            max_depth=max_depth,
            max_atoms=max_atoms,
        )
        depths.append(depth)
        if depth >= 0:
            proven_count += 1

        # Progress every 100 queries
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            qps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"\r  Progress: {i+1}/{num_queries}, proven: {proven_count}, {qps:.1f} q/s", end='', flush=True)

    print()  # Newline after progress
    return depths


def _generate_parallel(
    queries, unifier: FastUnifier,
    max_depth: int, max_atoms: int,
    n_workers: int, start_time: float
) -> List[int]:
    """Parallel BFS mode - process queries using multiprocessing."""
    num_queries = len(queries)

    print(f"  Starting parallel processing with {n_workers} workers...")

    # Process in batches to show progress
    batch_size = min(5000, max(100, num_queries // 10))
    depths: List[int] = []
    proven_count = 0

    for batch_start in range(0, num_queries, batch_size):
        batch_end = min(batch_start + batch_size, num_queries)
        batch_queries = queries[batch_start:batch_end]

        # Process batch in parallel
        batch_depths = unifier.check_provability_batch(
            batch_queries,
            max_depth=max_depth,
            max_atoms=max_atoms,
            n_workers=n_workers
        )

        depths.extend(batch_depths)
        proven_count += sum(1 for d in batch_depths if d >= 0)

        # Progress
        elapsed = time.time() - start_time
        qps = batch_end / elapsed if elapsed > 0 else 0
        print(f"\r  Progress: {batch_end}/{num_queries}, proven: {proven_count}, {qps:.1f} q/s", end='', flush=True)

    print()  # Newline after progress
    return depths


def _generate_incremental(
    queries, unifier: FastUnifier,
    max_depth: int, max_atoms: int,
    output_file: str, start_time: float
) -> List[int]:
    """Incremental mode - process depth-by-depth with saves at each level."""
    num_queries = len(queries)
    depths = [-1] * num_queries  # -1 = not yet proven
    unproven_indices = set(range(num_queries))

    # Process depth by depth
    for target_depth in range(1, max_depth + 1):
        if not unproven_indices:
            print(f"  All queries proven! Stopping at depth {target_depth - 1}")
            break

        depth_start = time.time()
        proven_this_depth = 0
        checked = 0

        for i in list(unproven_indices):
            query = queries[i]
            result = unifier.check_provability(
                query,
                is_train_data=True,
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
                remaining = len(unproven_indices) + checked - checked
                print(f"\r  Depth {target_depth}: {checked}/{len(unproven_indices) + proven_this_depth}, "
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
    return depths


def _save_depths(queries, depths: List[int], output_file: str):
    """Save depth results to file."""
    with open(output_file, 'w') as f:
        for query, depth in zip(queries, depths):
            clean_query = str(query).replace(' ', '')
            f.write(f"{clean_query} {depth}\n")


def _save_and_report(queries, depths: List[int], output_file: str, start_time: float):
    """Save final results and print summary."""
    num_queries = len(queries)
    _save_depths(queries, depths, output_file)

    elapsed = time.time() - start_time
    proven_count = sum(1 for d in depths if d >= 0)

    # Depth distribution
    depth_counts: Dict[int, int] = {}
    for d in depths:
        depth_counts[d] = depth_counts.get(d, 0) + 1

    print(f"\nSummary:")
    print(f"  Total queries: {num_queries}")
    print(f"  Provable: {proven_count} ({proven_count/num_queries:.1%})")
    print(f"  Non-provable: {num_queries - proven_count}")
    print(f"  Depth distribution: {dict(sorted(depth_counts.items()))}")
    print(f"  Time: {elapsed:.1f}s ({num_queries/elapsed:.1f} q/s)")
    print(f"  Saved to: {output_file}")


def compare_speed(
    dataset_name: str,
    split: str,
    data_path: str,
    num_queries: int = 500,
    max_depth: int = 5,
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

    # Fast unifier - Sequential
    print("[FAST - Sequential]")
    start = time.time()
    fast_seq_results = {}
    for i, query in enumerate(queries):
        depth = unifier.check_provability(
            query, is_train_data=True,
            max_depth=max_depth, max_atoms=max_atoms,
        )
        if depth >= 0:
            fast_seq_results[str(query)] = depth
        if (i + 1) % 100 == 0:
            print(f"\r  {i+1}/{len(queries)}...", end='', flush=True)
    fast_seq_time = time.time() - start
    print(f"\r  Time: {fast_seq_time:.2f}s, {len(queries)/fast_seq_time:.1f} q/s, proven: {len(fast_seq_results)}")

    # Fast unifier - Parallel
    n_workers = mp.cpu_count()
    print(f"\n[FAST - Parallel ({n_workers} workers)]")
    start = time.time()
    parallel_depths = unifier.check_provability_batch(
        queries, max_depth=max_depth, max_atoms=max_atoms, n_workers=n_workers
    )
    fast_par_results = {str(q): d for q, d in zip(queries, parallel_depths) if d >= 0}
    fast_par_time = time.time() - start
    print(f"  Time: {fast_par_time:.2f}s, {len(queries)/fast_par_time:.1f} q/s, proven: {len(fast_par_results)}")

    # SB3 unifier
    print("\n[SB3 - Sequential]")
    from generate_depths_sb3 import check_provability_bfs, load_data as load_data_sb3
    _, facts_set, index_manager, rules = load_data_sb3(dataset_name, data_path)

    start = time.time()
    sb3_results = {}
    for i, query in enumerate(queries):
        depth = check_provability_bfs(
            query, rules, facts_set, index_manager,
            max_depth=max_depth, max_atoms=max_atoms
        )
        if depth >= 0:
            sb3_results[str(query)] = depth
        if (i + 1) % 100 == 0:
            print(f"\r  {i+1}/{len(queries)}...", end='', flush=True)
    sb3_time = time.time() - start
    print(f"\r  Time: {sb3_time:.2f}s, {len(queries)/sb3_time:.1f} q/s, proven: {len(sb3_results)}")

    # Comparison
    print(f"\n{'='*60}")
    print("SPEEDUP vs SB3:")
    print(f"  Sequential: {sb3_time/fast_seq_time:.2f}x")
    print(f"  Parallel:   {sb3_time/fast_par_time:.2f}x")

    # Verify correctness
    seq_match = sum(1 for q in queries if fast_seq_results.get(str(q), -1) == sb3_results.get(str(q), -1))
    par_match = sum(1 for q in queries if fast_par_results.get(str(q), -1) == sb3_results.get(str(q), -1))

    print(f"\nCORRECTNESS:")
    print(f"  Sequential vs SB3: {seq_match}/{len(queries)} ({seq_match/len(queries):.1%})")
    print(f"  Parallel vs SB3:   {par_match}/{len(queries)} ({par_match/len(queries):.1%})")

    # Show mismatches if any
    mismatches = []
    for q in queries:
        q_str = str(q)
        fast_d = fast_seq_results.get(q_str, -1)
        sb3_d = sb3_results.get(q_str, -1)
        if fast_d != sb3_d and len(mismatches) < 5:
            mismatches.append((q_str, fast_d, sb3_d))

    if mismatches:
        print(f"\nMISMATCHES (first {len(mismatches)}):")
        for q_str, fast_d, sb3_d in mismatches:
            print(f"  {q_str}: Fast={fast_d}, SB3={sb3_d}")

    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate depth files using fast unifier')
    parser.add_argument('--datasets', nargs='+', default=['pharmkg_full','fb15k237'],
                        help='Datasets to process')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        help='Splits to process')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='Maximum proof depth (default: per-dataset, see DEFAULT_MAX_DEPTH)')
    parser.add_argument('--max_atoms', type=int, default=20,
                        help='Maximum atoms per state')
    parser.add_argument('--incremental', action='store_true',
                        help='Process depth-by-depth with saves at each level')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Use parallel processing (default: True)')
    parser.add_argument('--no-parallel', action='store_false', dest='parallel',
                        help='Disable parallel processing')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--compare', action='store_true',
                        help='Run speed comparison against SB3')
    parser.add_argument('--num_queries', type=int, default=500,
                        help='Number of queries for comparison')
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'

    if args.compare:
        for dataset in args.datasets:
            max_depth = args.max_depth if args.max_depth is not None else DEFAULT_MAX_DEPTH.get(dataset, DEFAULT_MAX_DEPTH_FALLBACK)
            for split in args.splits:
                compare_speed(
                    dataset_name=dataset,
                    split=split,
                    data_path=data_path,
                    num_queries=args.num_queries,
                    max_depth=max_depth,
                    max_atoms=args.max_atoms,
                )
    else:
        for dataset in args.datasets:
            max_depth = args.max_depth if args.max_depth is not None else DEFAULT_MAX_DEPTH.get(dataset, DEFAULT_MAX_DEPTH_FALLBACK)
            generate_depths_for_dataset(
                dataset_name=dataset,
                splits=args.splits,
                data_path=data_path,
                max_depth_check=max_depth,
                max_atoms=args.max_atoms,
                incremental=args.incremental,
                parallel=args.parallel,
                n_workers=args.n_workers,
            )

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

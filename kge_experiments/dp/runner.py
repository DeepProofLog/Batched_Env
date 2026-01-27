#!/usr/bin/env python
"""
Standalone DP Prover evaluation script.

Usage:
    python -m kge_experiments.dp.runner --dataset countries_s3
    python -m kge_experiments.dp.runner --dataset countries_s3 --profile
    python -m kge_experiments.dp.runner --dataset countries_s3 --verbose

Expected output for countries_s3:
    MRR: 1.000
    Hits@1: 1.000
    All 24 test queries proven
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Optional

import torch

# Set up path to import from project root
import sys
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Also add kge_experiments for relative imports within that package
kge_root = str(Path(__file__).parent.parent)
if kge_root not in sys.path:
    sys.path.insert(0, kge_root)

from data_handler import DataHandler
from index_manager import IndexManager
from tensor.tensor_sampler import Sampler
from kge_experiments.dp.prover import DPProver, DPProverFast
from kge_experiments.dp.ranking import DPRankingCallback, evaluate_dp_mrr


def load_data(dataset: str, device: torch.device) -> DataHandler:
    """Load dataset using DataHandler."""
    # Base path is the data directory
    base_path = str(Path(__file__).parent.parent / "data")

    print(f"Loading dataset: {dataset}")
    print(f"Base path: {base_path}")

    # Determine facts file based on dataset
    # For most datasets, train.txt contains the facts
    # Some datasets have separate facts.txt
    facts_file = "train.txt"  # Default: use train.txt as facts

    dh = DataHandler(
        dataset_name=dataset,
        base_path=base_path,
        facts_file=facts_file,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        filter_queries_by_rules=True,
    )

    # Create IndexManager and materialize indices
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=4096,
        max_arity=dh.max_arity,
        padding_atoms=10,
        device=device,
        rules=dh.rules,
    )

    dh.materialize_indices(im=im, device=device)
    dh.index_manager = im

    return dh


def create_prover(
    dh: DataHandler,
    max_depth: int = 20,
    cache_capacity: int = 100000,
    fast: bool = True,
) -> DPProver:
    """Create DPProver from DataHandler."""
    im = dh.index_manager

    ProverClass = DPProverFast if fast else DPProver
    prover = ProverClass(
        facts=im.facts_idx,
        rules_heads=im.rules_heads_idx,
        rules_bodies=im.rules_idx,
        rule_lens=im.rule_lens,
        constant_no=im.constant_no,
        padding_idx=im.padding_idx,
        max_depth=max_depth,
        cache_capacity=cache_capacity,
        device=im.device,
    )

    return prover


def evaluate_all_queries(
    prover: DPProver,
    dh: DataHandler,
    verbose: bool = False,
) -> dict:
    """
    Evaluate all test queries for provability.

    Returns:
        dict with 'total', 'proven', 'depths', 'proven_rate'
    """
    test_split = dh.get_materialized_split('test')
    test_queries = test_split.queries  # [N, 1, 3]
    queries = test_queries.squeeze(1)  # [N, 3]

    N = queries.shape[0]
    print(f"\nEvaluating {N} test queries...")

    # Prove all queries
    t0 = time.perf_counter()
    proven, depths = prover.prove_batch(queries)
    t1 = time.perf_counter()

    proven_count = proven.sum().item()
    proven_rate = proven_count / N if N > 0 else 0

    print(f"Proved {proven_count}/{N} queries ({proven_rate:.1%})")
    print(f"Time: {t1-t0:.3f}s ({N/(t1-t0):.1f} queries/sec)")

    if verbose:
        print("\nQuery results:")
        im = dh.index_manager
        for i in range(N):
            q = queries[i]
            pred_idx = q[0].item()
            arg0_idx = q[1].item()
            arg1_idx = q[2].item()

            # Handle list vs dict indexing
            if isinstance(im.idx2predicate, list):
                pred_str = im.idx2predicate[pred_idx] if pred_idx < len(im.idx2predicate) else f"pred_{pred_idx}"
            else:
                pred_str = im.idx2predicate.get(pred_idx, f"pred_{pred_idx}")

            if isinstance(im.idx2constant, list):
                arg0_str = im.idx2constant[arg0_idx] if arg0_idx < len(im.idx2constant) else f"ent_{arg0_idx}"
                arg1_str = im.idx2constant[arg1_idx] if arg1_idx < len(im.idx2constant) else f"ent_{arg1_idx}"
            else:
                arg0_str = im.idx2constant.get(arg0_idx, f"ent_{arg0_idx}")
                arg1_str = im.idx2constant.get(arg1_idx, f"ent_{arg1_idx}")

            status = "PROVEN" if proven[i] else "NOT PROVEN"
            depth_str = f"depth={depths[i].item()}" if proven[i] else ""
            print(f"  {pred_str}({arg0_str}, {arg1_str}): {status} {depth_str}")

    return {
        "total": N,
        "proven": proven_count,
        "proven_rate": proven_rate,
        "depths": depths.tolist(),
        "time": t1 - t0,
    }


def evaluate_mrr(
    prover: DPProver,
    dh: DataHandler,
    im: IndexManager,
    n_corruptions: int = 50,
    verbose: bool = False,
) -> dict:
    """
    Evaluate MRR using corruption-based ranking.

    Returns:
        dict with 'mrr', 'hits@1', 'hits@3', 'hits@10', etc.
    """
    test_split = dh.get_materialized_split('test')
    test_queries = test_split.queries.squeeze(1)  # [N, 3]

    # Build all known triples for filtering
    all_known = dh.all_known_triples_idx  # [T, 3]

    # Create sampler for corruption generation
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=all_known,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=prover.device,
        default_mode='tail',
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )

    print(f"\nEvaluating MRR with {n_corruptions} corruptions per query...")

    t0 = time.perf_counter()
    results = evaluate_dp_mrr(
        prover=prover,
        queries=test_queries,
        sampler=sampler,
        n_corruptions=n_corruptions,
        corruption_modes=('head', 'tail'),
        verbose=verbose,
    )
    t1 = time.perf_counter()

    print(f"\nResults:")
    print(f"  MRR: {results['mrr']:.4f}")
    print(f"  Hits@1: {results['hits@1']:.4f}")
    print(f"  Hits@3: {results['hits@3']:.4f}")
    print(f"  Hits@10: {results['hits@10']:.4f}")
    print(f"  Positive provability: {results['positive_provability']:.1%}")
    print(f"  Corruption provability: {results['corruption_provability']:.1%}")
    print(f"  Time: {t1-t0:.3f}s")

    results["eval_time"] = t1 - t0
    return results


def run_profiling(
    prover: DPProver,
    dh: DataHandler,
    n_iterations: int = 10,
) -> dict:
    """
    Profile prover performance.

    Returns timing statistics for batch proving.
    """
    test_split = dh.get_materialized_split('test')
    test_queries = test_split.queries.squeeze(1)  # [N, 3]
    N = test_queries.shape[0]

    print(f"\nProfiling {n_iterations} iterations of proving {N} queries...")

    # Warmup
    prover.clear_cache()
    prover.prove_batch(test_queries)

    # Timed iterations
    times = []
    for i in range(n_iterations):
        prover.clear_cache()
        t0 = time.perf_counter()
        prover.prove_batch(test_queries)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    std_time = times_tensor.std().item()

    print(f"\nProfiling results:")
    print(f"  Mean time: {mean_time*1000:.2f}ms")
    print(f"  Std time: {std_time*1000:.2f}ms")
    print(f"  Queries/sec: {N/mean_time:.1f}")

    # Check cache stats
    stats = prover.cache_stats()
    print(f"\nCache stats (last iteration):")
    print(f"  Capacity: {stats['capacity']}")
    print(f"  Occupied: {stats['occupied']}")
    print(f"  Load factor: {stats['load_factor']:.2%}")

    return {
        "mean_time_ms": mean_time * 1000,
        "std_time_ms": std_time * 1000,
        "queries_per_sec": N / mean_time,
        "cache_stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DP Prover evaluation for link prediction"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="countries_s3",
        help="Dataset name (default: countries_s3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=20,
        help="Maximum proof depth (default: 20)",
    )
    parser.add_argument(
        "--cache_capacity",
        type=int,
        default=100000,
        help="Memoization cache capacity (default: 100000)",
    )
    parser.add_argument(
        "--n_corruptions",
        type=int,
        default=50,
        help="Number of corruptions for MRR (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profiling",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--skip_mrr",
        action="store_true",
        help="Skip MRR evaluation",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    dh = load_data(args.dataset, device)

    # Print dataset info
    im = dh.index_manager
    test_split = dh.get_materialized_split('test')
    print(f"\nDataset: {args.dataset}")
    print(f"  Facts: {im.facts_idx.shape[0]}")
    print(f"  Rules: {im.rules_idx.shape[0]}")
    print(f"  Constants: {im.constant_no}")
    print(f"  Test queries: {test_split.queries.shape[0]}")

    # Create prover
    prover = create_prover(
        dh,
        max_depth=args.max_depth,
        cache_capacity=args.cache_capacity,
    )

    # Evaluate query provability
    query_results = evaluate_all_queries(prover, dh, verbose=args.verbose)

    # Evaluate MRR
    if not args.skip_mrr:
        mrr_results = evaluate_mrr(
            prover, dh, im,
            n_corruptions=args.n_corruptions,
            verbose=args.verbose,
        )

    # Run profiling if requested
    if args.profile:
        profile_results = run_profiling(prover, dh)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Test queries: {query_results['total']}")
    print(f"Queries proven: {query_results['proven']} ({query_results['proven_rate']:.1%})")

    if not args.skip_mrr:
        print(f"MRR: {mrr_results['mrr']:.4f}")
        print(f"Hits@1: {mrr_results['hits@1']:.4f}")

    if query_results['proven_rate'] == 1.0:
        print("\nAll test queries are provable!")

    return 0


if __name__ == "__main__":
    exit(main())

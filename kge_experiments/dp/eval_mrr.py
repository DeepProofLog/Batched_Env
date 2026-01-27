#!/usr/bin/env python
"""
MRR Evaluation script for DP Prover.

Usage:
    python -m kge_experiments.dp.eval_mrr --dataset family --n_queries 500 --n_corruptions 100
    python -m kge_experiments.dp.eval_mrr --dataset wn18rr --max_depth 2 --n_queries 200
    python -m kge_experiments.dp.eval_mrr --dataset countries_s3
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import sys

import torch

# Set up paths
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
kge_root = str(Path(__file__).parent.parent)
if kge_root not in sys.path:
    sys.path.insert(0, kge_root)

from data_handler import DataHandler
from index_manager import IndexManager
from tensor.tensor_sampler import Sampler
from kge_experiments.dp.prover_multiproc import DPProverMultiProc
from kge_experiments.dp.prover_parallel_bfs import DPProverParallelBFS
from kge_experiments.dp.ranking import evaluate_dp_mrr


def main():
    parser = argparse.ArgumentParser(description="DP Prover MRR Evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., family, wn18rr, countries_s3)")
    parser.add_argument("--max_depth", type=int, default=6,
                        help="Maximum proof depth (default: 6)")
    parser.add_argument("--n_queries", type=int, default=None,
                        help="Number of test queries (default: all)")
    parser.add_argument("--n_corruptions", type=int, default=50,
                        help="Corruptions per query per mode (default: 50)")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count)")
    parser.add_argument("--single_process", action="store_true",
                        help="Use single-process prover (slower but no multiprocessing)")
    parser.add_argument("--modes", type=str, default="head,tail",
                        help="Corruption modes, comma-separated (default: head,tail)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load dataset
    base_path = str(Path(__file__).parent.parent / "data")
    print(f"Loading dataset: {args.dataset}")

    dh = DataHandler(
        dataset_name=args.dataset,
        base_path=base_path,
        facts_file="train.txt",
        filter_queries_by_rules=True,
    )

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

    print(f"Facts: {im.facts_idx.shape[0]}, Rules: {im.rules_idx.shape[0]}")
    print(f"Constants: {im.constant_no}, Predicates: {im.predicate_no}")

    # Create prover
    if args.single_process:
        print("Using single-process prover (DPProverParallelBFS)")
        prover = DPProverParallelBFS.from_index_manager(
            im, max_depth=args.max_depth
        )
    else:
        import multiprocessing as mp
        n_workers = args.n_workers or min(mp.cpu_count(), 16)
        print(f"Using multiprocessing prover with {n_workers} workers")
        prover = DPProverMultiProc.from_index_manager(
            im, max_depth=args.max_depth, n_workers=n_workers
        )

    # Create sampler
    sampler = Sampler.from_data(
        all_known_triples_idx=im.facts_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
    )

    # Get test queries
    test_split = dh.get_materialized_split("test")
    test_queries = test_split.queries.squeeze(1).to(device)

    n_queries = args.n_queries or test_queries.shape[0]
    n_queries = min(n_queries, test_queries.shape[0])
    test_sample = test_queries[:n_queries]

    corruption_modes = [m.strip() for m in args.modes.split(",")]

    print()
    print("=" * 60)
    print(f"MRR Evaluation: {args.dataset}")
    print(f"  Queries: {n_queries}")
    print(f"  Corruptions per mode: {args.n_corruptions}")
    print(f"  Modes: {corruption_modes}")
    print(f"  Max depth: {args.max_depth}")
    print("=" * 60)

    t0 = time.time()
    results = evaluate_dp_mrr(
        prover=prover,
        queries=test_sample,
        sampler=sampler,
        n_corruptions=args.n_corruptions,
        corruption_modes=corruption_modes,
        verbose=True,
    )
    t1 = time.time()

    print()
    print(f"Total time: {t1 - t0:.1f}s")
    print(f"Queries/sec: {n_queries / (t1 - t0):.2f}")

    return results


if __name__ == "__main__":
    main()

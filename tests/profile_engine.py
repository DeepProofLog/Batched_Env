"""
Performance profiling for tensor vs SB3 unification engines.

This script profiles both engines to identify performance bottlenecks
and compare their execution times.

Usage:
    python tests/profile_engine.py --dataset countries_s3 --n-queries 200
    python tests/profile_engine.py --profile  # Run with cProfile
"""
import os
import sys
import time
import random
import argparse
import cProfile
import pstats
import io
from types import SimpleNamespace
from typing import List, Tuple, Dict

import torch

# Setup paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SB3_ROOT = os.path.join(ROOT, 'sb3')
TEST_ENVS_ROOT = os.path.join(ROOT, 'test_envs')

sys.path.insert(0, ROOT)
sys.path.insert(0, SB3_ROOT)
sys.path.insert(0, TEST_ENVS_ROOT)

from test_engine_sb3 import setup_sb3_engine, run_sb3_engine
from test_engine_tensor import setup_tensor_engine, run_tensor_engine
from sb3.sb3_dataset import DataHandler


def create_config() -> SimpleNamespace:
    """Create default configuration for profiling."""
    return SimpleNamespace(
        dataset="countries_s3",
        n_queries=200,
        deterministic=True,
        max_depth=20,
        seed=42,
        verbose=False,
        debug=False,
        padding_atoms=6,
        padding_states=40,
        max_derived_per_state=40,
        skip_unary_actions=True,
        end_proof_action=False,
        memory_pruning=True,
        use_exact_memory=True,
        reward_type=0,
        prover_verbose=0,
        max_total_runtime_vars=1_000_000,
        device='cpu',
        collect_action_stats=True,
    )


def prepare_queries(
    dataset: str = "countries_s3",
    base_path: str = "./data/",
    n_queries: int = None,
    seed: int = 42
) -> List[Tuple[str, Tuple[str, str, str]]]:
    """Prepare list of queries from dataset."""
    dh = DataHandler(
        dataset_name=dataset,
        base_path=base_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )
    
    all_queries = []
    for q in dh.train_queries:
        all_queries.append(('train', (q.predicate, q.args[0], q.args[1])))
    for q in dh.valid_queries:
        all_queries.append(('valid', (q.predicate, q.args[0], q.args[1])))
    for q in dh.test_queries:
        all_queries.append(('test', (q.predicate, q.args[0], q.args[1])))
    
    rng = random.Random(seed)
    rng.shuffle(all_queries)
    
    if n_queries is not None:
        all_queries = all_queries[:n_queries]
    
    return all_queries


def time_engine(name: str, setup_func, run_func, queries, config, warmup: int = 1) -> Dict:
    """Time a single engine with optional warmup."""
    print(f"\nSetup {name} engine...")
    engine_data = setup_func(dataset=config.dataset, config=config)
    
    # Warmup runs
    if warmup > 0:
        print(f"  Warmup ({warmup} run(s))...")
        warmup_queries = queries[:min(10, len(queries))]
        for _ in range(warmup):
            run_func(warmup_queries, engine_data, config)
    
    # Timed run
    print(f"  Running {name} engine on {len(queries)} queries...")
    t0 = time.perf_counter()
    results = run_func(queries, engine_data, config)
    elapsed = time.perf_counter() - t0
    
    print(f"  {name}: {results['successful']}/{results['total_queries']} successful")
    print(f"  {name} time: {elapsed:.3f}s ({1000*elapsed/len(queries):.2f}ms/query)")
    
    return {
        'name': name,
        'time': elapsed,
        'results': results,
        'queries_per_sec': len(queries) / elapsed if elapsed > 0 else 0,
        'ms_per_query': 1000 * elapsed / len(queries) if len(queries) > 0 else 0,
    }


def profile_engine(name: str, setup_func, run_func, queries, config, top_n: int = 30) -> str:
    """Profile a single engine and return stats string."""
    print(f"\nProfiling {name} engine...")
    engine_data = setup_func(dataset=config.dataset, config=config)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    results = run_func(queries, engine_data, config)
    
    profiler.disable()
    
    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(top_n)
    
    return s.getvalue()


def run_comparison(config: SimpleNamespace, queries: List) -> Dict:
    """Run timing comparison between SB3 and tensor engines."""
    print(f"\n{'='*70}")
    print(f"ENGINE PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"Dataset: {config.dataset}")
    print(f"Queries: {len(queries)}")
    print(f"Max depth: {config.max_depth}")
    print(f"Deterministic: {config.deterministic}")
    print(f"{'='*70}")
    
    # Time both engines
    sb3_stats = time_engine("SB3", setup_sb3_engine, run_sb3_engine, queries, config)
    tensor_stats = time_engine("Tensor", setup_tensor_engine, run_tensor_engine, queries, config)
    
    # Calculate speedup
    if tensor_stats['time'] > 0:
        speedup = sb3_stats['time'] / tensor_stats['time']
    else:
        speedup = float('inf')
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"SB3 engine:")
    print(f"  Total time:     {sb3_stats['time']:.3f}s")
    print(f"  Per query:      {sb3_stats['ms_per_query']:.2f}ms")
    print(f"  Queries/sec:    {sb3_stats['queries_per_sec']:.1f}")
    print(f"  Success rate:   {sb3_stats['results']['successful']}/{sb3_stats['results']['total_queries']}")
    print()
    print(f"Tensor engine:")
    print(f"  Total time:     {tensor_stats['time']:.3f}s")
    print(f"  Per query:      {tensor_stats['ms_per_query']:.2f}ms")
    print(f"  Queries/sec:    {tensor_stats['queries_per_sec']:.1f}")
    print(f"  Success rate:   {tensor_stats['results']['successful']}/{tensor_stats['results']['total_queries']}")
    print()
    print(f"Speedup (SB3/Tensor): {speedup:.2f}x")
    if speedup < 1:
        print(f"  ⚠️  Tensor engine is {1/speedup:.1f}x SLOWER than SB3")
    else:
        print(f"  ✓ Tensor engine is {speedup:.1f}x faster than SB3")
    print(f"{'='*70}")
    
    return {
        'sb3': sb3_stats,
        'tensor': tensor_stats,
        'speedup': speedup,
    }


def benchmark_core_unification(config: SimpleNamespace, queries: List, n_iterations: int = 100) -> Dict:
    """
    Benchmark just the core get_derived_states operation.
    
    This measures what would actually run on GPU in production,
    without the test harness overhead (string conversions, trace building).
    """
    print(f"\n{'='*70}")
    print(f"CORE UNIFICATION BENCHMARK (get_derived_states only)")
    print(f"{'='*70}")
    print(f"Dataset: {config.dataset}")
    print(f"Batch size: {len(queries)}")
    print(f"Iterations: {n_iterations}")
    print(f"{'='*70}")
    
    # Setup tensor engine
    print("\nSetting up tensor engine...")
    from test_engine_tensor import setup_tensor_engine
    dh_non, im_non, engine, debug_helper, next_var_start = setup_tensor_engine(
        dataset=config.dataset, config=config
    )
    
    B = len(queries)
    device = engine.device
    pad = engine.padding_idx
    initial_max_atoms = 20
    
    # Build initial states batch
    current_states = torch.full((B, initial_max_atoms, 3), pad, dtype=torch.long, device=device)
    excluded_queries = torch.full((B, initial_max_atoms, 3), pad, dtype=torch.long, device=device)
    
    for i, (split, (p, h, t)) in enumerate(queries):
        query_tensor = im_non.atom_to_tensor(p, h, t)
        current_states[i, 0] = query_tensor
        if split == 'train':
            excluded_queries[i, 0] = query_tensor
    
    next_var_tracker = torch.full((B,), next_var_start, dtype=torch.long, device=device)
    
    # Warmup
    print("Warmup...")
    for _ in range(5):
        _, _, _ = engine.get_derived_states(
            current_states, next_var_tracker,
            excluded_queries=excluded_queries, verbose=0
        )
    
    # Benchmark
    print(f"Running {n_iterations} iterations...")
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        derived, counts, updated_next = engine.get_derived_states(
            current_states, next_var_tracker,
            excluded_queries=excluded_queries, verbose=0
        )
    elapsed = time.perf_counter() - t0
    
    per_call_ms = 1000 * elapsed / n_iterations
    per_query_us = 1_000_000 * elapsed / (n_iterations * B)
    queries_per_sec = (n_iterations * B) / elapsed
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total time:           {elapsed:.3f}s")
    print(f"Per call (batch):     {per_call_ms:.3f}ms")
    print(f"Per query:            {per_query_us:.1f}μs")
    print(f"Throughput:           {queries_per_sec:.0f} queries/sec")
    print(f"{'='*70}")
    
    return {
        'elapsed': elapsed,
        'per_call_ms': per_call_ms,
        'per_query_us': per_query_us,
        'queries_per_sec': queries_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description='Profile unification engines')
    parser.add_argument('--dataset', type=str, default='countries_s3',
                        help='Dataset name (default: countries_s3)')
    parser.add_argument('--n-queries', type=int, default=200,
                        help='Number of queries to test (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--profile', action='store_true',
                        help='Run cProfile on both engines')
    parser.add_argument('--profile-tensor-only', action='store_true',
                        help='Run cProfile only on tensor engine')
    parser.add_argument('--top-n', type=int, default=40,
                        help='Number of top functions to show in profile (default: 40)')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Number of warmup runs (default: 1)')
    parser.add_argument('--core-only', action='store_true',
                        help='Benchmark just the core get_derived_states operation')
    
    args = parser.parse_args()
    
    # Create config
    config = create_config()
    config.dataset = args.dataset
    config.n_queries = args.n_queries
    config.seed = args.seed
    
    # Prepare queries
    print(f"Preparing queries from {args.dataset}...")
    queries = prepare_queries(
        dataset=config.dataset,
        n_queries=config.n_queries,
        seed=config.seed
    )
    print(f"Prepared {len(queries)} queries")
    
    if args.core_only:
        # Just benchmark the core unification step
        benchmark_core_unification(config, queries)
    elif args.profile or args.profile_tensor_only:
        # Profile mode
        if not args.profile_tensor_only:
            print("\n" + "="*70)
            print("SB3 ENGINE PROFILE")
            print("="*70)
            sb3_profile = profile_engine("SB3", setup_sb3_engine, run_sb3_engine, 
                                         queries, config, args.top_n)
            print(sb3_profile)
        
        print("\n" + "="*70)
        print("TENSOR ENGINE PROFILE")
        print("="*70)
        tensor_profile = profile_engine("Tensor", setup_tensor_engine, run_tensor_engine,
                                        queries, config, args.top_n)
        print(tensor_profile)
    else:
        # Timing comparison mode
        run_comparison(config, queries)


if __name__ == "__main__":
    main()

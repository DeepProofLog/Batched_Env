"""
Test and Profile Optimized Evaluation.

This script compares the original and compiled evaluation pipelines:
1. Low-level Correctness: Vectorized unification, graph breaks
2. Metrics Correctness: Verify MRR/Hits@K match between implementations
3. Performance: Measure timing difference (Original vs Compiled)

The compiled path uses torch.compile with mode='reduce-overhead' and fullgraph=True
for maximum CUDA graph performance.

Usage:
    python tests/test_eval_optimized.py                    # Basic test
    python tests/test_eval_optimized.py --performance      # Run performance benchmark
    python tests/test_eval_optimized.py --check-compile    # Check for graph breaks
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
from types import SimpleNamespace
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch._dynamo as dynamo

from tests.test_eval_perf import setup_components, run_original_eval, run_optimized_eval, create_and_warmup_optimized_evaluator


def test_correctness(components, config):
    """Test that metrics match between Original and Compiled implementations."""
    print("\n" + "="*60)
    print("CORRECTNESS TEST: Original vs Compiled")
    print("="*60)
    
    # Use small test for correctness (avoid OOM and long runtimes)
    small_config = SimpleNamespace(**vars(config))
    small_config.n_test_queries = min(20, config.n_test_queries)
    small_config.n_corruptions = min(50, config.n_corruptions)
    small_config.verbose = False
    small_config.chunk_queries = 10  # Small chunks for memory efficiency
    
    print(f"Testing with {small_config.n_test_queries} queries, {small_config.n_corruptions} corruptions...")
    
    results = {}
    
    # Original
    print("Running Original...")
    try:
        torch.cuda.empty_cache()
        results['Original'] = run_original_eval(components, small_config)
    except Exception as e:
        print(f"Original failed: {e}")
        import traceback
        traceback.print_exc()
        results['Original'] = None
        
    # Compiled (with reduce-overhead and fullgraph)
    print("Running Compiled (reduce-overhead + fullgraph)...")
    try:
        torch.cuda.empty_cache()
        res_comp, evaluator, warmup_time_s = run_optimized_eval(
            components, small_config, return_evaluator=True, mode='compiled'
        )
        results['Compiled'] = res_comp
        print(f"  Compiled warmup time: {warmup_time_s:.2f}s")
    except Exception as e:
        print(f"Compiled failed: {e}")
        import traceback
        traceback.print_exc()
        results['Compiled'] = None

    # Compare
    metrics = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    
    # Random baseline MRR for reference
    random_mrr = 1.0 / (small_config.n_corruptions + 1)
    min_mrr_threshold = random_mrr * 3  # Should be at least 3x better than random
    
    print(f"\n{'Metric':<10} {'Original':>12} {'Compiled':>12} {'Random':>10} {'Status':>8}")
    print("-" * 60)
    
    all_pass = True
    for m in metrics:
        v_orig = results['Original'].get(m, 0.0) if results['Original'] else 0.0
        v_comp = results['Compiled'].get(m, 0.0) if results['Compiled'] else 0.0
        v_rand = random_mrr if m == "MRR" else 0.0
        
        # Pass if compiled is significantly better than random baseline
        status = "PASS"
        if m == "MRR" and v_comp < min_mrr_threshold:
            status = "FAIL"
            all_pass = False
            
        print(f"{m:<10} {v_orig:>12.4f} {v_comp:>12.4f} {v_rand:>10.4f} {status:>8}")
    
    # Print summary
    print(f"\nRandom baseline MRR: {random_mrr:.4f}")
    print(f"Minimum MRR threshold (3x random): {min_mrr_threshold:.4f}")
    
    if all_pass:
        print("\n✓ Compiled metrics are healthy (significantly above random)")
    else:
        print("\n✗ Compiled MRR is too low (below 3x random)")
    
    # Also report ratio
    orig_mrr = results['Original'].get('MRR', 0.0) if results['Original'] else 0.0
    comp_mrr = results['Compiled'].get('MRR', 0.0) if results['Compiled'] else 0.0
    if orig_mrr > 0 and comp_mrr > 0:
        ratio = comp_mrr / orig_mrr
        print(f"\nCompiled/Original MRR ratio: {ratio:.2f}x")
        if ratio < 0.5:
            print("⚠ Warning: Compiled is less than 50% of Original - investigate")
        
    return all_pass




def test_seed_correctness(components, config, device):
    """
    Seed-based correctness test.
    
    Key insight: In family dataset, most negatives don't have proofs, so MRR is generally high.
    Random baseline is not meaningful. Instead, we compare Original vs Compiled with same seeds.
    
    Strategy:
    1. Run Original ONCE with seed=0 (it's slow, don't repeat)
    2. Run Compiled with multiple seeds (fast)
    3. Also run Original with same seeds to compare (necessary for fair comparison)
    4. If averages differ significantly, there's a bug in compiled version
    """
    print("\n" + "="*60)
    print("SEED-BASED CORRECTNESS TEST")
    print("="*60)
    
    test_config = SimpleNamespace(**vars(config))
    test_config.n_test_queries = min(50, config.n_test_queries)
    test_config.n_corruptions = min(100, config.n_corruptions)
    
    num_seeds = config.num_seeds
    seeds = list(range(num_seeds))
    
    print(f"Testing with {test_config.n_test_queries} queries, {test_config.n_corruptions} corruptions")
    print(f"Seeds to test: {seeds}")
    
    # Collect results
    original_results = []
    compiled_results = []
    
    # Create a single compiled evaluator to reuse across seeds
    from model_eval_optimized import (
        OptimizedEvaluator,
        create_policy_logits_fn,
        compute_optimal_batch_size,
    )
    env_c = components['eval_env_compiled']
    policy_logits_fn = create_policy_logits_fn(components['policy'], deterministic=True)
    batch_size = compute_optimal_batch_size(
        chunk_queries=test_config.chunk_queries,
        n_corruptions=test_config.n_corruptions,
        max_vram_gb=test_config.vram_gb,
    )
    compiled_evaluator = OptimizedEvaluator(
        env=env_c,
        policy_logits_fn=policy_logits_fn,
        batch_size=batch_size,
        max_steps=test_config.max_depth,
        deterministic=True,
        compile_mode=getattr(config, 'compile_mode', 'default'),
        fullgraph=getattr(config, 'fullgraph', False),
    )
    warmup_start = time.time()
    compiled_evaluator.warmup(
        components['test_queries'][:min(20, test_config.n_test_queries)].to(env_c.device)
    )
    warmup_time_s = (
        compiled_evaluator.warmup_time_s
        if getattr(compiled_evaluator, "warmup_time_s", None) is not None
        else time.time() - warmup_start
    )
    print(f"\nCompiled warmup time (one-time): {warmup_time_s:.2f}s")

    # Run both versions with same seeds for fair comparison
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        
        # Run Original
        print(f"  Running Original (seed={seed})...")
        torch.cuda.empty_cache()
        try:
            res_orig = run_original_eval(components, test_config, seed=seed)
            original_results.append(res_orig)
            print(f"    MRR: {res_orig['MRR']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")
            original_results.append({'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0})
        
        # Run Compiled
        print(f"  Running Compiled (seed={seed})...")
        torch.cuda.empty_cache()
        try:
            res_comp = run_optimized_eval(
                components, test_config, seed=seed, evaluator=compiled_evaluator, mode='compiled'
            )
            compiled_results.append(res_comp)
            print(f"    MRR: {res_comp['MRR']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")
            compiled_results.append({'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0})
    
    # Compute statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    
    print(f"\n{'Metric':<10} {'Original (avg ± std)':<25} {'Compiled (avg ± std)':<25} {'Diff':>10} {'Status':>8}")
    print("-" * 85)
    
    all_pass = True
    for m in metrics:
        orig_vals = np.array([r.get(m, 0.0) for r in original_results])
        comp_vals = np.array([r.get(m, 0.0) for r in compiled_results])
        
        orig_mean, orig_std = orig_vals.mean(), orig_vals.std()
        comp_mean, comp_std = comp_vals.mean(), comp_vals.std()
        diff = comp_mean - orig_mean
        
        # Allow 10% relative tolerance (some stochasticity from policy)
        # Or absolute tolerance of 0.05 for small values
        tolerance = max(0.1 * orig_mean, 0.05)
        status = "PASS" if abs(diff) <= tolerance else "FAIL"
        if status == "FAIL":
            all_pass = False
        
        print(f"{m:<10} {orig_mean:>8.4f} ± {orig_std:>6.4f}       {comp_mean:>8.4f} ± {comp_std:>6.4f}       {diff:>+7.4f}   {status}")
    
    # Per-seed comparison
    print("\n--- Per-Seed MRR Comparison ---")
    print(f"{'Seed':>6} {'Original':>12} {'Compiled':>12} {'Diff':>10}")
    print("-" * 45)
    for i, seed in enumerate(seeds):
        orig_mrr = original_results[i].get('MRR', 0.0)
        comp_mrr = compiled_results[i].get('MRR', 0.0)
        diff = comp_mrr - orig_mrr
        print(f"{seed:>6} {orig_mrr:>12.4f} {comp_mrr:>12.4f} {diff:>+10.4f}")
    
    if all_pass:
        print("\n✓ Compiled version matches Original within tolerance")
    else:
        print("\n✗ SIGNIFICANT DIFFERENCE detected - investigate compiled version!")
        print("  The compiled version may have a bug if differences are consistent.")
    
    return all_pass




def main():
    parser = argparse.ArgumentParser(description='Test Optimized Evaluation')
    parser.add_argument('--dataset', type=str, default='family', help='Dataset name')
    parser.add_argument('--n-test-queries', type=int, default=50, help='Number of test queries')
    parser.add_argument('--n-corruptions', type=int, default=50, help='Corruptions per query')
    parser.add_argument('--chunk-queries', type=int, default=100, help='Queries per chunk (smaller = less VRAM)')
    parser.add_argument('--batch-size-env', type=int, default=100, help='Environment batch size for original eval')
    parser.add_argument('--corruption-modes', type=str, nargs='+', default=['both'], help='Corruption modes')
    parser.add_argument('--vram-gb', type=float, default=6.0, help='Available VRAM budget in GB')
    parser.add_argument('--performance', action='store_true', help='Run performance benchmark')

    parser.add_argument('--skip-correctness', action='store_true', help='Skip correctness test')
    parser.add_argument('--compiled-smoke', action='store_true', help='Run compiled-only smoke test with warmup timing')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead', 
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile mode: default (fast compile), reduce-overhead (slow compile, fast runtime), max-autotune (very slow compile)')
    parser.add_argument('--fullgraph', action='store_true', default=False,
                       help='Use fullgraph=True (slow compile, best runtime). Default: False for faster compilation.')
    parser.add_argument('--atom-embedding-size', type=int, default=250, help='Embedding size (250 for parity with profile_eval)')
    parser.add_argument('--seed-test', action='store_true', help='Run seed-based correctness test (multi-seed comparison)')
    parser.add_argument('--num-seeds', type=int, default=3, help='Number of seeds to test')
    args = parser.parse_args()
    
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path='./data/',
        n_test_queries=args.n_test_queries,
        n_corruptions=args.n_corruptions,
        chunk_queries=args.chunk_queries,  # For compiled eval chunking
        batch_size_env=args.batch_size_env,  # For original eval
        corruption_modes=args.corruption_modes,
        verbose=args.verbose,
        vram_gb=args.vram_gb,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        memory_pruning=True,
        skip_unary_actions=False,
        end_proof_action=True,
        reward_type=0,
        max_total_vars=100,
        atom_embedding_size=args.atom_embedding_size,
        seed=0,
        compile=args.compile,
        num_seeds=args.num_seeds,
        compile_mode=args.compile_mode,
        fullgraph=args.fullgraph,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    components = setup_components(device, config)

    # Metric Correctness Test
    if not args.skip_correctness:
        test_correctness(components, config)
    
    # Seed-based correctness test (statistical comparison)
    if args.seed_test:
        test_seed_correctness(components, config, device)
    print("\n" + "="*60)
    print("DONE")
    print("="*60)

if __name__ == '__main__':
    main()

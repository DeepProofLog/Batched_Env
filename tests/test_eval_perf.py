"""
Performance Benchmark for Optimized Evaluation.

This script benchmarks the performance difference between:
1. Original Evaluation (Python/Stack-based)
2. Eager Optimized (Vectorized, No Compile)
3. Compiled Optimized (Vectorized + Compile)

It reproduces the "Performance Summary" table from docs/eval_optimization.md.

Usage:
    python tests/test_eval_perf.py --dataset family --modes compiled --warmup-only
    python tests/test_eval_perf.py --dataset family --modes original eager compiled --n-test-queries 100
    python tests/test_eval_perf.py --check-compile  # Check for graph breaks
"""

import os
import sys
import argparse
import time
import torch
import torch._dynamo as dynamo
import numpy as np
from types import SimpleNamespace

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import setup and runners from the main test file
from tests.test_eval_mrr import setup_components, run_original_eval, run_optimized_eval, create_and_warmup_optimized_evaluator

def check_graph_breaks(vec_engine, device, config):
    """Check for graph breaks in the compiled function."""
    print("\n" + "="*60)
    print("CHECKING FOR GRAPH BREAKS")
    print("="*60)
    
    # Create sample inputs
    B = 10
    A = config.padding_atoms
    
    current_states = torch.zeros(B, A, 3, dtype=torch.long, device=device)
    current_states[:, 0, :] = 1  # Some dummy values
    next_var_indices = torch.full((B,), 1000, dtype=torch.long, device=device)
    excluded = torch.zeros(B, 1, 3, dtype=torch.long, device=device)
    
    # Explain the function
    try:
        explanation = dynamo.explain(vec_engine.get_derived_states_compiled)(
            current_states, next_var_indices, excluded
        )
        
        print(f"\nGraph Count: {explanation.graph_count}")
        print(f"Graph Break Count: {explanation.graph_break_count}")
        
        if explanation.break_reasons:
            print("\nBreak Reasons:")
            for i, reason in enumerate(explanation.break_reasons):
                print(f"  {i+1}. {reason}")
        else:
            print("\nNo graph breaks! ✓")
        
        return explanation.graph_break_count == 0
        
    except Exception as e:
        print(f"Error during explain: {e}")
        return False

def run_performance_test(components, config, modes, warmup_only):
    """
    Run performance benchmark for specified modes.
    
    Args:
        modes: List of modes to test ('original', 'eager', 'compiled')
        warmup_only: If True, only measure warmup/compile time (for compiled).
                     Evaluation will run on a minimal set of queries.
    """
    print("\n" + "="*60)
    print(f"PERFORMANCE BENCHMARK (Warmup Only: {warmup_only})")
    print(f"Modes: {modes}")
    print(f"Config: {config.n_test_queries} queries, {config.n_corruptions} corruptions")
    print("="*60)
    
    # Define table header
    print(f"\n{'Mode':<20} {'Compile (s)':>12} {'Runtime (s)':>12} {'Runtime (ms/q)':>18} {'Total Time (s)':>15}")
    print("-" * 85)
    
    for mode in modes:
        # Prepare run config
        run_config = SimpleNamespace(**vars(config))
        run_config.verbose = False
        
        if warmup_only:
            run_config.n_test_queries = min(config.chunk_queries)
            
        warmup_s = 0.0
        eval_ms_per_q = 0.0
        total_s = 0.0
        
        try:
            start_t = time.time()
            
            if mode == 'original':
                if warmup_only:
                    # Original has no warmup
                    pass
                else:
                    print(f"  Running {mode} eval...", flush=True)
                    run_original_eval(components, run_config)
                    
            elif mode == 'eager':
                if warmup_only:
                    # Just run warmup for eager
                    _, warmup_s = create_and_warmup_evaluator(components, run_config, mode='eager')
                    print(f"  Warmup finished in {warmup_s:.4f}s", flush=True)
                else:
                    print(f"  Running {mode} warmup...", flush=True)
                    # Run with return_evaluator to get warmup time (same pattern as compiled)
                    _, _, start_warmup_s = run_optimized_eval(
                        components, run_config, return_evaluator=True, mode='eager'
                    )
                    warmup_s = start_warmup_s
                    print(f"  Warmup finished in {warmup_s:.4f}s", flush=True)
                    print(f"  Running {mode} eval...", flush=True)

            elif mode == 'compiled':
                if warmup_only:
                    # Just run warmup
                    evaluator, warmup_s = create_and_warmup_optimized_evaluator(components, run_config)
                    print(f"  Warmup finished in {warmup_s:.4f}s", flush=True)
                else:
                    print(f"  Running {mode} warmup...", flush=True)
                    # First: warmup only
                    evaluator, warmup_s = create_and_warmup_optimized_evaluator(components, run_config)
                    print(f"  Warmup finished in {warmup_s:.4f}s", flush=True)
                    
                    # Second: timed evaluation run (separate from warmup)
                    print(f"  Running {mode} eval...", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    eval_start = time.time()
                    from model_eval_optimized import eval_corruptions_optimized
                    effective_chunk_queries = min(int(run_config.chunk_queries), int(run_config.n_test_queries))
                    queries = components['test_queries'][:run_config.n_test_queries].to(evaluator.env.device)
                    _ = eval_corruptions_optimized(
                        evaluator=evaluator,
                        queries=queries,
                        sampler=components['sampler'],
                        n_corruptions=run_config.n_corruptions,
                        corruption_modes=tuple(run_config.corruption_modes),
                        chunk_queries=effective_chunk_queries,
                        verbose=False,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    eval_s = time.time() - eval_start
                    # Override total_s calculation below
                    total_s = warmup_s + eval_s

            # For compiled/fast mode, total_s is set above; for others, calculate from start_t
            if mode not in ('compiled') or warmup_only:
                total_s = time.time() - start_t
                eval_s = max(0.0, total_s - warmup_s)
            # else: eval_s and total_s already set for compiled mode
            
            # Calculate metrics
            n_run = run_config.n_test_queries
            
            if n_run > 0:
                eval_ms_per_q = (eval_s / n_run) * 1000.0
            
            print(f"  Eval finished in {eval_s:.4f}s", flush=True)
            
            # Display summary row
            result_line = f"{mode:<20} {warmup_s:>12.4f} {eval_s:>12.4f} {eval_ms_per_q:>18.2f} {total_s:>15.4f}"
            print(result_line)
            
            # Append result to file
            with open("test_eval_perf.txt", "a") as f:
                f.write(result_line + "\n")
            
        except Exception as e:
            print(f"{mode:<20} {'FAILED':>12} {'FAILED':>12} {'FAILED':>18} {str(e):>15}")
            # import traceback
            # traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Test Evaluation Performance')
    parser.add_argument('--dataset', type=str, default='family', help='Dataset name')
    parser.add_argument('--n-test-queries', type=int, default=50, help='Number of test queries')
    parser.add_argument('--n-corruptions', type=int, default=50)
    parser.add_argument('--chunk-queries', type=int, default=100)
    parser.add_argument('--modes', nargs='+', default=['compiled'], 
                       choices=['original', 'eager', 'compiled'],
                       help='Modes to benchmark. compiled=single-step compile (faster warmup).')
    parser.add_argument('--warmup-only', action='store_true', help='Only measure warmup/compile time')
    parser.add_argument('--check-compile', action='store_true', help='Check for graph breaks')
    
    parser.add_argument('--vram-gb', type=float, default=6.0, help='Available VRAM budget in GB')
    parser.add_argument('--batch-size-env', type=int, default=100)
    parser.add_argument('--corruption-modes', type=str, nargs='+', default=['both'])
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                       help='torch.compile mode: default, reduce-overhead, max-autotune')
    parser.add_argument('--fixed-batch-size', type=int, default=None,
                       help='Fixed batch size for compilation. If None, uses adaptive sizing (smaller for fewer queries = faster compile).')
    

    args = parser.parse_args()
    
    # Initialize result file
    with open("test_eval_perf.txt", "w") as f:
        f.write("Performance Benchmark Results\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*85 + "\n")
        f.write(f"{'Mode':<20} {'Compile (s)':>12} {'Runtime (s)':>12} {'Runtime (ms/q)':>18} {'Total Time (s)':>15}\n")
        f.write("-" * 85 + "\n")
    
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path='./data/',
        n_test_queries=args.n_test_queries,
        n_corruptions=args.n_corruptions,
        chunk_queries=args.chunk_queries,
        batch_size_env=args.batch_size_env,
        corruption_modes=args.corruption_modes,
        verbose=True,
        vram_gb=args.vram_gb,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        memory_pruning=True,
        skip_unary_actions=False,
        end_proof_action=True,
        reward_type=0,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=42,
        compile=True, # Will be ignored/overridden by mode logic in run_perf_test or run_optimized_eval wrapper? 
                      # Actually, run_optimized_eval handles enable/disable of compilation based on mode.
                      # Ideally we set this to True generally so components like Unification are ready for compilation if needed.
        compile_mode=args.compile_mode,
        fullgraph=True, # Default to True, but run_optimized_eval controls usage
        fixed_batch_size=args.fixed_batch_size,  # Fixed batch size to prevent recompilation
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    components = setup_components(device, config)
    
    if args.check_compile:
        check_graph_breaks(components['vec_engine'], device, config)
    else:
        run_performance_test(components, config, args.modes, args.warmup_only)

if __name__ == '__main__':
    main()

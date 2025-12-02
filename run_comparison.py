# "/home/castellanoontiv/miniconda3/bin/conda", "run", "-n", "rl", 
"""
Simple script to run batched and SB3 versions separately and compare their outputs.

This avoids import conflicts by running them as separate processes.
"""

import time
import os
import sys

import argparse

import runner_shared as tensor_runner
import sb3.sb3_runner_shared as sb3_runner


def _run_with_smoke_flag(fn, smoke: bool, extra_args=None):
    """
    Call a runner main() while temporarily injecting a --smoke flag so both
    sides use identical minimal configs when requested.
    """
    argv_backup = sys.argv[:]
    try:
        sys.argv = [argv_backup[0]]
        if smoke:
            sys.argv.append("--smoke")
        if extra_args:
            sys.argv.extend(extra_args)
        return fn()
    finally:
        sys.argv = argv_backup


def run_tensor(smoke: bool = False, extra_args=None):
    """Run batched version."""
    print("="*80)
    print("Running BATCHED version...")
    print("="*80)
    
    result = _run_with_smoke_flag(tensor_runner.main, smoke, extra_args)
    return result

def run_sb3(smoke: bool = False, extra_args=None):
    """Run SB3 version."""
    print("\n" + "="*80)
    print("Running SB3 version...")
    print("="*80)

    result = _run_with_smoke_flag(sb3_runner.main, smoke, extra_args)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run batched and SB3 comparison")
    parser.add_argument("--smoke", action="store_true", help="Run minimal rollout for parity debugging")
    parser.add_argument("--n_queries", type=int, default=None, help="Limit train/valid/test queries to this number")
    parser.add_argument("--n_steps", type=int, default=None, help="Override rollout steps")
    parser.add_argument("--batch_size_env", type=int, default=None, help="Env batch size / n_envs (applied to both)")
    parser.add_argument("--n_envs", type=int, default=None, help="Alias for batch_size_env")
    parser.add_argument("--trace_dir", type=str, default=None, help="Optional directory to dump rollout traces")
    args, unknown = parser.parse_known_args()

    extra_args = []
    if args.n_queries is not None:
        extra_args += ["--n_queries", str(args.n_queries)]
        if args.n_queries == 1:
            print("Forcing batch_size_env=1 and n_envs=1 because n_queries=1")
            args.batch_size_env = 1
            args.n_envs = 1
    if args.n_steps is not None:
        extra_args += ["--n_steps", str(args.n_steps)]
    batch_env = args.batch_size_env if args.batch_size_env is not None else args.n_envs
    if batch_env is not None:
        extra_args += ["--batch_size_env", str(batch_env), "--n_envs", str(batch_env)]
    if args.trace_dir:
        extra_args += ["--trace_dir", args.trace_dir]
    extra_args += unknown  # forward any other flags

    print("="*80)
    print("COMPARISON SCRIPT: Batched vs SB3")
    print("="*80)
    print("This script runs both versions separately with identical parameters:")
    print()
    smoke = args.smoke
    
    # Run batched
    batched_ok = run_tensor(smoke=smoke, extra_args=extra_args)
    
    print("\n✓ Batched version completed")
    # Free any large objects before entering SB3 path to reduce peak memory
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    # Run SB3
    sb3_ok = run_sb3(smoke=smoke, extra_args=extra_args)

    print("\n✓ SB3 version completed")
    
    print("\n" + "="*80)
    print("Both versions completed successfully!")
    print("Check the output logs to compare metrics.")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    main()

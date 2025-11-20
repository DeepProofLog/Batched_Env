#!/usr/bin/env python3
"""
Simple script to run batched and SB3 versions separately and compare their outputs.

This avoids import conflicts by running them as separate processes.
"""

import subprocess
import time
import os
import sys

def run_batched():
    """Run batched version."""
    print("="*80)
    print("Running BATCHED version...")
    print("="*80)
    
    cmd = [
        "/home/castellanoontiv/miniconda3/bin/conda", "run", "-n", "rl", "--no-capture-output",
        "python", "runner_new.py",
        "--timesteps_train", "512",
        "--n_steps", "128",
        "--batch_size_env", "1",
        "--seed", "0",
        "--verbose_env", "0"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/home/castellanoontiv/Batched_env", capture_output=False)
    return result.returncode == 0

def run_sb3():
    """Run SB3 version."""
    print("\n" + "="*80)
    print("Running SB3 version...")
    print("="*80)
    
    cmd = [
        "/home/castellanoontiv/miniconda3/bin/conda", "run", "-n", "rl", "--no-capture-output",
        "python", "sb3_runner_simple.py",
        "--n_envs", "1",
        "--n_steps", "128",
        "--timesteps_train", "512",
        "--seed", "0"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/home/castellanoontiv/Batched_env/sb3", capture_output=False)
    return result.returncode == 0

def main():
    print("="*80)
    print("COMPARISON SCRIPT: Batched vs SB3")
    print("="*80)
    print("This script runs both versions separately with identical parameters:")
    print("  - timesteps_train: 512")
    print("  - n_steps: 128")
    print("  - batch_size/n_envs: 1")
    print("  - seed: 0")
    print()
    
    # Run batched
    batched_ok = run_batched()
    
    if not batched_ok:
        print("\n❌ Batched version failed!")
        return 1
    
    print("\n✓ Batched version completed")
    
    # Run SB3
    sb3_ok = run_sb3()
    
    if not sb3_ok:
        print("\n❌ SB3 version failed!")
        return 1
    
    print("\n✓ SB3 version completed")
    
    print("\n" + "="*80)
    print("Both versions completed successfully!")
    print("Check the output logs to compare metrics.")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
Script Parity Tests (In-Process).

Tests verifying that train_parity.py and sb3/sb3_train_parity.py produce IDENTICAL results
when run with the same configuration.

This test validates end-to-end parity by:
1. Running sb3_train_parity.run_experiment() 
2. Running train_parity.run_experiment()  
3. Comparing comprehensive metrics (MRR, Hits@1/3/10, checksums, etc.)

Usage:
    python tests/parity/test_script_parity.py
    python tests/parity/test_script_parity.py --total-timesteps 200 --n-epochs 4
"""
import os
import sys
from pathlib import Path
from dataclasses import asdict

# Paths
ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))
if str(ROOT / "tests" / "parity") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests" / "parity"))

# Set environment variables before importing torch
os.environ['USE_FAST_CATEGORICAL'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import numpy as np

from tests.test_utils.parity_config import ParityConfig, TOLERANCE, create_parser, config_from_args

# Import train functions directly from train files - must be after setting env vars
from tensor.tensor_train_parity import run_experiment as tensor_run_experiment, TrainParityConfig as TensorConfig
from sb3.sb3_train_parity import run_experiment as sb3_run_experiment, TrainParityConfig as SB3Config


def parity_config_to_train_config(parity_config: ParityConfig) -> TensorConfig:
    """Convert ParityConfig to TrainParityConfig for train_parity.py."""
    return TensorConfig(
        dataset=parity_config.dataset,
        data_path=parity_config.data_path,
        train_file=parity_config.train_file,
        valid_file=parity_config.valid_file,
        test_file=parity_config.test_file,
        rules_file=parity_config.rules_file,
        facts_file=parity_config.facts_file,
        padding_atoms=parity_config.padding_atoms,
        padding_states=parity_config.padding_states,
        max_steps=parity_config.max_depth,
        use_exact_memory=parity_config.use_exact_memory,
        sample_deterministic_per_env=parity_config.sample_deterministic_per_env,
        n_envs=parity_config.n_envs,
        n_steps=parity_config.n_steps,
        n_epochs=parity_config.n_epochs,
        batch_size=parity_config.batch_size,
        learning_rate=parity_config.learning_rate,
        gamma=parity_config.gamma,
        clip_range=parity_config.clip_range,
        ent_coef=parity_config.ent_coef,
        target_kl=parity_config.target_kl,
        total_timesteps=parity_config.total_timesteps,
        n_corruptions=parity_config.n_corruptions,
        train_neg_ratio=parity_config.negative_ratio,
        corruption_scheme=[parity_config.corruption_mode],
        atom_embedding_size=parity_config.embed_dim,
        seed=parity_config.seed,
        device=parity_config.device,
        verbose=parity_config.verbose,
        parity=parity_config.parity,
        # Note: sampler_default_mode defaults to 'both' in TrainParityConfig,
        # allowing the sampler to generate all corruption types.
    )


def compare_results(sb3_results: dict, tensor_results: dict) -> bool:
    """Compare metrics between SB3 and Tensor implementations."""
    print("\n" + "=" * 90)
    print("METRIC COMPARISONS")
    print("=" * 90)
    print(f"{'Metric':<30} {'SB3':>14} {'Tensor':>14} {'Diff':>12} {'Tol':>8} {'Status':>6}")
    print("-" * 90)
    
    # Define tolerance categories for different metric types
    tolerances = {
        # Exact match (integers)
        'index_manager_constants': 0,
        'index_manager_predicates': 0,
        # Strict tolerance for checksums and eval metrics
        'embedder_checksum': TOLERANCE,
        'policy_checksum_init': TOLERANCE,
        'MRR': TOLERANCE,
        'Hits@1': TOLERANCE,
        'Hits@3': TOLERANCE,
        'Hits@10': TOLERANCE,
        # Looser tolerance for training metrics (can diverge during optimization)
        'policy_checksum_trained': 10.0,
        'policy_loss': 0.1,
        'value_loss': 0.5,
        'entropy': 0.1,
        'approx_kl': 0.001,  # Strict tolerance for KL divergence parity
        'clip_fraction': 0.1,
    }
    
    all_passed = True
    passed_count = 0
    failed_count = 0
    
    # Iterate over all keys in sb3 results
    for key in sb3_results.keys():
        sb3_val = sb3_results.get(key, 0.0)
        tensor_val = tensor_results.get(key, 0.0)
        diff = abs(sb3_val - tensor_val)
        tol = tolerances.get(key, TOLERANCE)  # Default to strict tolerance
        
        # Integer comparison for exact match
        if tol == 0:
            passed = sb3_val == tensor_val
        else:
            passed = diff <= tol
        
        status = "✅" if passed else "❌"
        
        if passed:
            passed_count += 1
        else:
            failed_count += 1
            all_passed = False
        
        # Format values appropriately
        if isinstance(sb3_val, int) or tol == 0:
            print(f"{key:<30} {sb3_val:>14} {tensor_val:>14} {diff:>12} {tol:>8} {status:>6}")
        else:
            print(f"{key:<30} {sb3_val:>14.4f} {tensor_val:>14.4f} {diff:>12.6f} {tol:>8.4f} {status:>6}")
    
    print("-" * 90)
    print(f"Total: {passed_count} passed, {failed_count} failed | STRICT TOLERANCE: {TOLERANCE}")
    print("=" * 90)
    
    return all_passed


def test_script_parity(config: ParityConfig = None):
    """
    Run both training pipelines in-process and compare metrics.
    
    Args:
        config: ParityConfig with test parameters. If None, uses defaults.
    """
    if config is None:
        config = ParityConfig()
    
    # Convert to train_parity config format
    train_config = parity_config_to_train_config(config)
    
    print("\n" + "=" * 80)
    print("SCRIPT PARITY TEST (In-Process)")
    print("=" * 80)
    print(f"Dataset: {config.dataset}")
    print(f"N envs: {config.n_envs}, N steps: {config.n_steps}")
    print(f"Timesteps: {config.total_timesteps}, Epochs: {config.n_epochs}")
    print(f"Seed: {config.seed}")
    print(f"Device: {config.device}")
    print("=" * 80)
    
    # Run SB3 training
    print("\n" + "-" * 40)
    print(">>> Running SB3 Training...")
    print("-" * 40)
    sb3_results = sb3_run_experiment(train_config)
    
    # Run Tensor training
    print("\n" + "-" * 40)
    print(">>> Running Tensor Training...")
    print("-" * 40)
    tensor_results = tensor_run_experiment(train_config)
    
    # Compare results
    all_passed = compare_results(sb3_results, tensor_results)
    
    # Key assertions
    sb3_mrr = sb3_results.get('MRR', 0.0)
    tensor_mrr = tensor_results.get('MRR', 0.0)
    
    assert sb3_mrr > 0, f"SB3 run failed to produce MRR (got {sb3_mrr})"
    assert tensor_mrr > 0, f"Tensor run failed to produce MRR (got {tensor_mrr})"
    
    mrr_diff = abs(sb3_mrr - tensor_mrr)
    assert mrr_diff <= TOLERANCE, \
        f"MRR mismatch: SB3={sb3_mrr:.4f}, Tensor={tensor_mrr:.4f}, diff={mrr_diff:.6f} > TOLERANCE={TOLERANCE}"
    
    if all_passed:
        print("\n✅ SUCCESS: Script parity within expected tolerances!")
    else:
        print("\n❌ FAILED: Script parity failed!")
    
    return all_passed


if __name__ == "__main__":
    # Parse command-line arguments using shared parity config
    parser = create_parser(description="Script Parity Test")
    args = parser.parse_args()
    
    # Create config from parsed args (uses ParityConfig defaults if not overridden)
    config = config_from_args(args)
    
    # Run test
    success = test_script_parity(config)
    sys.exit(0 if success else 1)

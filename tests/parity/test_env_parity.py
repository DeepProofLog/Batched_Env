"""
Environment Parity Tests.

Tests verifying that the tensor-based BatchedEnv produces EXACTLY the same 
behavior as the SB3 string-based LogicEnv_gym.

This test module mirrors the structure of test_unification_parity.py but focuses
on full environment parity, not just the unification engine.

Usage:
    pytest tests/parity/test_env_parity.py -v
    pytest tests/parity/test_env_parity.py -v -k "countries"
    pytest tests/parity/test_env_parity.py -v -k "family"
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Tuple
import random

import pytest
import torch
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"
TEST_ENVS_ROOT = ROOT / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(2, str(TEST_ENVS_ROOT))

# Import environment test utilities (use underscore prefix to avoid pytest autodiscovery)
from test_env_sb3 import setup_sb3_env, run_sb3_env as _run_sb3_env
from test_env_tensor import setup_tensor_env, run_tensor_env as _run_tensor_env


# ============================================================================
# Configuration
# ============================================================================

def create_default_parity_config() -> SimpleNamespace:
    """Create default configuration for parity tests."""
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
    )


def clone_config(config: SimpleNamespace) -> SimpleNamespace:
    """Create a shallow clone of a SimpleNamespace."""
    return SimpleNamespace(**vars(config))


# ============================================================================
# Query Preparation
# ============================================================================

def prepare_queries(
    dataset: str = "countries_s3",
    base_path: str = "./data/",
    n_queries: int = None,
    seed: int = 42
) -> List[Tuple[str, Tuple[str, str, str]]]:
    """
    Prepare list of queries from dataset.
    
    Args:
        dataset: Dataset name
        base_path: Base path to data directory
        n_queries: If specified, sample this many queries; otherwise use all
        seed: Random seed for sampling
        
    Returns:
        List of (split, (predicate, head, tail)) tuples
    """
    from sb3.sb3_dataset import DataHandler
    
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
    
    # Collect all queries
    all_queries = []
    for q in dh.train_queries:
        all_queries.append(('train', (q.predicate, q.args[0], q.args[1])))
    for q in dh.valid_queries:
        all_queries.append(('valid', (q.predicate, q.args[0], q.args[1])))
    for q in dh.test_queries:
        all_queries.append(('test', (q.predicate, q.args[0], q.args[1])))
    
    # Shuffle and take first n
    rng = random.Random(seed)
    rng.shuffle(all_queries)
    
    if n_queries is not None:
        all_queries = all_queries[:n_queries]
    
    return all_queries


# ============================================================================
# Environment Runner
# ============================================================================

def run_env(
    name: str,
    setup_func,
    run_func,
    queries: List[Tuple[str, Tuple[str, str, str]]],
    config: SimpleNamespace
) -> Dict:
    """
    Run a single environment configuration.
    
    Args:
        name: Environment name for logging
        setup_func: Function to setup environment
        run_func: Function to run environment
        queries: List of queries
        config: Configuration
        
    Returns:
        Results dict with traces
    """
    setup_kwargs = {
        "dataset": config.dataset,
        "config": config,
    }
    
    env_data = setup_func(**setup_kwargs)
    results = run_func(queries, env_data, config)
    
    return results


def run_both_envs(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    config: SimpleNamespace
) -> Tuple[Dict, Dict]:
    """
    Run both SB3 and tensor environments on the same queries.
    
    Returns:
        (sb3_results, tensor_results)
    """
    # Run SB3 env
    sb3_results = run_env(
        name="sb3_env",
        setup_func=setup_sb3_env,
        run_func=_run_sb3_env,
        queries=queries,
        config=config,
    )
    
    # Run tensor env
    tensor_results = run_env(
        name="tensor_env",
        setup_func=setup_tensor_env,
        run_func=_run_tensor_env,
        queries=queries,
        config=config,
    )
    
    return sb3_results, tensor_results


# ============================================================================
# Trace Comparison
# ============================================================================

def compare_trace_step(step_tensor: Dict, step_sb3: Dict, step_idx: int) -> Tuple[bool, str]:
    """
    Compare a single step from both traces.
    Returns (match, error_message).
    """
    # Compare state canonical strings
    if step_tensor.get('state') != step_sb3.get('state'):
        return False, f"Step {step_idx}: state mismatch:\n  Tensor: {step_tensor.get('state')}\n  SB3:    {step_sb3.get('state')}"
    
    # Compare number of derived states (actions)
    if step_tensor.get('num_actions') != step_sb3.get('num_actions'):
        return False, f"Step {step_idx}: num_actions mismatch: {step_tensor.get('num_actions')} vs {step_sb3.get('num_actions')}"
    
    # Compare derived states list (already in canonical order from envs)
    tensor_derived = step_tensor.get('derived_states', [])
    sb3_derived = step_sb3.get('derived_states', [])
    if tensor_derived != sb3_derived:
        return False, f"Step {step_idx}: derived_states mismatch:\n  Tensor: {tensor_derived[:3]}...\n  SB3:    {sb3_derived[:3]}..."
    
    # Compare chosen action
    if step_tensor.get('action') != step_sb3.get('action'):
        return False, f"Step {step_idx}: action mismatch: {step_tensor.get('action')} vs {step_sb3.get('action')}"
    
    # Compare done flag
    if step_tensor.get('done') != step_sb3.get('done'):
        return False, f"Step {step_idx}: done mismatch: {step_tensor.get('done')} vs {step_sb3.get('done')}"
    
    return True, ""


def compare_full_traces(trace_tensor: List[Dict], trace_sb3: List[Dict]) -> Tuple[bool, str]:
    """
    Compare full traces step-by-step.
    Returns (match, error_message).
    """
    if len(trace_tensor) != len(trace_sb3):
        return False, f"Trace length mismatch: {len(trace_tensor)} vs {len(trace_sb3)}"
    
    for i, (step_t, step_s) in enumerate(zip(trace_tensor, trace_sb3)):
        match, error = compare_trace_step(step_t, step_s, i)
        if not match:
            return False, error
    
    return True, ""


def compare_all_traces(
    sb3_traces: List[Dict],
    tensor_traces: List[Dict],
    queries: List[Tuple[str, Tuple[str, str, str]]]
) -> Tuple[int, List[Tuple[int, str, str]]]:
    """
    Compare all traces between SB3 and tensor environments.
    
    Returns:
        (num_matches, list of (query_idx, query_str, error_msg) for mismatches)
    """
    mismatches = []
    matches = 0
    
    for i, (sb3_trace, tensor_trace, query) in enumerate(zip(sb3_traces, tensor_traces, queries)):
        split, (pred, head, tail) = query
        query_str = f"{pred}({head}, {tail}) [{split}]"
        
        sb3_trace_steps = sb3_trace.get('trace', [])
        tensor_trace_steps = tensor_trace.get('trace', [])
        
        match, error = compare_full_traces(tensor_trace_steps, sb3_trace_steps)
        if match:
            matches += 1
        else:
            mismatches.append((i, query_str, error))
    
    return matches, mismatches


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def base_config():
    """Base configuration for all tests."""
    return create_default_parity_config()


# ============================================================================
# Parity Tests - Parametrized by dataset and n_queries
# ============================================================================

class TestEnvParity:
    """Test that tensor and SB3 environments produce EXACTLY the same traces."""
    
    @pytest.mark.parametrize("dataset,n_queries", [
        ("countries_s3", 10),
        ("countries_s3", 200),
        ("countries_s3", 800),
        ("family", 10),
        ("family", 200),
        ("family", 800),
    ])
    def test_env_parity_sequential(self, dataset: str, n_queries: int, base_config):
        """
        Test that tensor and SB3 environments produce identical traces in sequential mode.
        
        Verifies:
        - Same states at each step
        - Same derived states (actions)
        - Same action choices
        - Same success/failure outcomes
        """
        # Clone and customize config
        config = clone_config(base_config)
        config.dataset = dataset
        config.n_queries = n_queries
        
        # Set seeds for reproducibility
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Prepare queries
        queries = prepare_queries(
            dataset=config.dataset,
            n_queries=config.n_queries,
            seed=config.seed
        )
        
        print(f"\n{'='*60}")
        print(f"Testing env parity (sequential): {dataset} with {n_queries} queries")
        print(f"{'='*60}")
        
        # Run both environments
        sb3_results, tensor_results = run_both_envs(queries, config)
        
        # Basic result comparison
        assert sb3_results['total_queries'] == tensor_results['total_queries'], \
            f"Query count mismatch: SB3={sb3_results['total_queries']}, Tensor={tensor_results['total_queries']}"
        
        print(f"SB3 env:    {sb3_results['successful']}/{sb3_results['total_queries']} successful")
        print(f"Tensor env: {tensor_results['successful']}/{tensor_results['total_queries']} successful")
        
        # Compare traces
        sb3_traces = sb3_results.get('traces', [])
        tensor_traces = tensor_results.get('traces', [])
        
        assert len(sb3_traces) == len(tensor_traces), \
            f"Trace count mismatch: SB3={len(sb3_traces)}, Tensor={len(tensor_traces)}"
        
        matches, mismatches = compare_all_traces(sb3_traces, tensor_traces, queries)
        
        print(f"Trace matches: {matches}/{len(queries)}")
        
        if mismatches:
            error_msg = f"Found {len(mismatches)} trace mismatches out of {len(queries)} queries:\n"
            for idx, query_str, error in mismatches[:5]:  # Show first 5
                error_msg += f"  Query {idx} {query_str}:\n    {error}\n"
            if len(mismatches) > 5:
                error_msg += f"  ... and {len(mismatches) - 5} more mismatches\n"
            pytest.fail(error_msg)
        
        # Verify success rates match exactly
        assert sb3_results['successful'] == tensor_results['successful'], \
            f"Success count mismatch: SB3={sb3_results['successful']}, Tensor={tensor_results['successful']}"
        
        print(f"✓ All {n_queries} traces match exactly for {dataset}")

    @pytest.mark.parametrize("dataset,n_queries", [
        ("countries_s3", 10),
        ("countries_s3", 200),
        ("countries_s3", 800),
        ("family", 10),
        ("family", 200),
        ("family", 800),
    ])
    def test_env_parity_batched(self, dataset: str, n_queries: int, base_config):
        """
        Test that tensor env in batched mode matches SB3 sequential mode.
        
        - SB3: processes queries sequentially (no true batching)
        - Tensor: processes all queries in parallel using batch_size
        
        Both should produce identical traces.
        """
        # Clone and customize config
        config = clone_config(base_config)
        config.dataset = dataset
        config.n_queries = n_queries
        
        # Set seeds for reproducibility
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Prepare queries
        queries = prepare_queries(
            dataset=config.dataset,
            n_queries=config.n_queries,
            seed=config.seed
        )
        
        print(f"\n{'='*60}")
        print(f"Testing env parity (batched): {dataset} with {n_queries} queries")
        print(f"{'='*60}")
        
        # Run both environments (tensor env will use true batching)
        sb3_results, tensor_results = run_both_envs(queries, config)
        
        # Basic result comparison
        assert sb3_results['total_queries'] == tensor_results['total_queries'], \
            f"Query count mismatch: SB3={sb3_results['total_queries']}, Tensor={tensor_results['total_queries']}"
        
        print(f"SB3 env:    {sb3_results['successful']}/{sb3_results['total_queries']} successful")
        print(f"Tensor env: {tensor_results['successful']}/{tensor_results['total_queries']} successful")
        
        # Compare traces
        sb3_traces = sb3_results.get('traces', [])
        tensor_traces = tensor_results.get('traces', [])
        
        assert len(sb3_traces) == len(tensor_traces), \
            f"Trace count mismatch: SB3={len(sb3_traces)}, Tensor={len(tensor_traces)}"
        
        matches, mismatches = compare_all_traces(sb3_traces, tensor_traces, queries)
        
        print(f"Trace matches: {matches}/{len(queries)}")
        
        if mismatches:
            error_msg = f"Found {len(mismatches)} trace mismatches out of {len(queries)} queries:\n"
            for idx, query_str, error in mismatches[:5]:
                error_msg += f"  Query {idx} {query_str}:\n    {error}\n"
            if len(mismatches) > 5:
                error_msg += f"  ... and {len(mismatches) - 5} more mismatches\n"
            pytest.fail(error_msg)
        
        # Verify success rates match exactly
        assert sb3_results['successful'] == tensor_results['successful'], \
            f"Success count mismatch: SB3={sb3_results['successful']}, Tensor={tensor_results['successful']}"
        
        print(f"✓ All {n_queries} traces match exactly for {dataset} (batched mode)")


# ============================================================================
# CLI Runner (for manual testing)
# ============================================================================

def run_parity_tests(
    dataset: str = "countries_s3",
    n_queries: int = 200,
    seed: int = 42,
    verbose: bool = False,
    memory_pruning: bool = True,
    skip_unary_actions: bool = True
) -> Tuple[bool, Dict]:
    """
    Run parity tests programmatically.
    
    Returns:
        (all_passed, results_dict)
    """
    config = create_default_parity_config()
    config.dataset = dataset
    config.n_queries = n_queries
    config.seed = seed
    config.verbose = verbose
    config.memory_pruning = memory_pruning
    config.skip_unary_actions = skip_unary_actions
    
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    print(f"\n{'='*80}")
    print(f"ENVIRONMENT PARITY TEST")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Queries: {n_queries}")
    print(f"Seed: {seed}")
    print(f"Memory pruning: {memory_pruning}")
    print(f"Skip unary actions: {skip_unary_actions}")
    print(f"{'='*80}\n")
    
    queries = prepare_queries(
        dataset=config.dataset,
        n_queries=config.n_queries,
        seed=config.seed
    )
    
    print(f"Prepared {len(queries)} queries\n")
    
    # Run both environments
    print("Running SB3 env...")
    sb3_results = run_env(
        name="sb3_env",
        setup_func=setup_sb3_env,
        run_func=_run_sb3_env,
        queries=queries,
        config=config,
    )
    print(f"  ✓ {sb3_results['successful']}/{sb3_results['total_queries']} successful")
    
    print("Running tensor env...")
    tensor_results = run_env(
        name="tensor_env",
        setup_func=setup_tensor_env,
        run_func=_run_tensor_env,
        queries=queries,
        config=config,
    )
    print(f"  ✓ {tensor_results['successful']}/{tensor_results['total_queries']} successful")
    
    # Compare traces
    print("\nComparing traces...")
    sb3_traces = sb3_results.get('traces', [])
    tensor_traces = tensor_results.get('traces', [])
    
    matches, mismatches = compare_all_traces(sb3_traces, tensor_traces, queries)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Trace matches: {matches}/{len(queries)}")
    print(f"Trace mismatches: {len(mismatches)}/{len(queries)}")
    
    if mismatches:
        print(f"\nFirst 5 mismatches:")
        for idx, query_str, error in mismatches[:5]:
            print(f"  Query {idx} {query_str}:")
            print(f"    {error}")
    
    all_passed = len(mismatches) == 0
    
    if all_passed:
        print(f"\n✓ ALL TRACES MATCH - Environments are EXACTLY equivalent")
    else:
        print(f"\n✗ TRACES DIFFER - Found {len(mismatches)} mismatches")
    
    print(f"{'='*80}\n")
    
    return all_passed, {
        'sb3': sb3_results,
        'tensor': tensor_results,
        'matches': matches,
        'mismatches': mismatches,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test environment parity')
    parser.add_argument('--dataset', type=str, default='countries_s3',
                        help='Dataset name (default: countries_s3)')
    parser.add_argument('--n-queries', type=int, default=200,
                        help='Number of queries to test (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-memory-pruning', action='store_true',
                        help='Disable memory pruning (default: enabled)')
    parser.add_argument('--no-skip-unary', action='store_true',
                        help='Disable skipping unary actions (default: enabled)')
    
    args = parser.parse_args()
    
    passed, results = run_parity_tests(
        dataset=args.dataset,
        n_queries=args.n_queries,
        seed=args.seed,
        verbose=args.verbose,
        memory_pruning=not args.no_memory_pruning,
        skip_unary_actions=not args.no_skip_unary
    )
    
    sys.exit(0 if passed else 1)

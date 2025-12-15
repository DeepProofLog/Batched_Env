"""
Unification Engine Parity Tests.

Tests verifying that the tensor-based UnificationEngine produces
EXACTLY the same results as the SB3 string-based unification engine.

This test module mirrors the structure of test_all_configs.py but focuses
only on engine parity, not environments or rollout collection.

Usage:
    pytest tests/parity/test_unification_parity.py -v
    pytest tests/parity/test_unification_parity.py -v -k "countries"
    pytest tests/parity/test_unification_parity.py -v -k "family"
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
TEST_ENVS_ROOT = ROOT / "tests" / "other" / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(2, str(TEST_ENVS_ROOT))

# Import engine test utilities from test_envs
from test_engine_sb3 import setup_sb3_engine, run_sb3_engine
from test_engine_tensor import setup_tensor_engine, run_tensor_engine


# ============================================================================
# Default Configuration
# ============================================================================

def create_default_config() -> SimpleNamespace:
    """Default parameters used across unification parity tests."""
    return SimpleNamespace(
        # Dataset/files
        dataset="countries_s3",
        data_path="./data/",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
        
        # Query selection
        n_queries=200,
        
        # Engine behavior
        max_depth=20,
        padding_atoms=6,
        padding_states=40,
        max_derived_per_state=40,
        skip_unary_actions=True,
        memory_pruning=True,
        use_exact_memory=True,
        reward_type=0,
        prover_verbose=0,
        max_total_runtime_vars=1000,
        collect_action_stats=True,
        
        # Seeds / device / logging
        seed=42,
        device="cpu",
        verbose=False,
        debug=False,
    )


def clone_config(config: SimpleNamespace) -> SimpleNamespace:
    """Create a shallow clone of a SimpleNamespace."""
    return SimpleNamespace(**vars(config))


# ============================================================================
# Query Preparation (from test_all_configs)
# ============================================================================

def prepare_queries(config: SimpleNamespace) -> List[Tuple[str, Tuple[str, str, str]]]:
    """
    Prepare list of queries from dataset.
    
    Args:
        config: Test configuration
        
    Returns:
        List of (split, (predicate, head, tail)) tuples
    """
    from sb3.sb3_dataset import DataHandler
    
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
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
    rng = random.Random(config.seed)
    rng.shuffle(all_queries)
    
    if config.n_queries is not None:
        all_queries = all_queries[:config.n_queries]
    
    return all_queries


# ============================================================================
# Engine Runner
# ============================================================================

def run_engine(
    name: str,
    setup_func,
    test_func,
    queries: List[Tuple[str, Tuple[str, str, str]]],
    config: SimpleNamespace
) -> Dict:
    """
    Run a single engine configuration.
    
    Args:
        name: Engine name for logging
        setup_func: Function to setup engine
        test_func: Function to test engine
        queries: List of queries
        config: Configuration
        
    Returns:
        Results dict with traces
    """
    setup_kwargs = {
        "config": config,
    }
    
    engine_data = setup_func(**setup_kwargs)
    results = test_func(queries, engine_data, config)
    
    return results


def run_both_engines(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    config: SimpleNamespace
) -> Tuple[Dict, Dict]:
    """
    Run both SB3 and tensor engines on the same queries.
    
    Returns:
        (sb3_results, tensor_results)
    """
    # Run SB3 engine
    sb3_results = run_engine(
        name="sb3_engine",
        setup_func=setup_sb3_engine,
        test_func=run_sb3_engine,
        queries=queries,
        config=config,
    )
    
    # Run tensor engine
    tensor_results = run_engine(
        name="tensor_engine",
        setup_func=setup_tensor_engine,
        test_func=run_tensor_engine,
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
    
    # Compare derived states list (already in canonical order from engines)
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
    Compare all traces between SB3 and tensor engines.
    
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
    return create_default_config()


# ============================================================================
# Parity Tests - Parametrized by dataset and n_queries
# ============================================================================

class TestUnificationEngineParity:
    """Test that tensor and SB3 engines produce EXACTLY the same traces."""
    
    @pytest.mark.parametrize("dataset,n_queries", [
        ("countries_s3", 50),
        ("countries_s3", 200),
        ("family", 50),
        ("family", 200),
    ])
    def test_engine_traces_match(self, dataset: str, n_queries: int, base_config):
        """
        Test that tensor and SB3 engines produce identical traces.
        
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
        queries = prepare_queries(config)
        
        print(f"\n{'='*60}")
        print(f"Testing engine parity: {dataset} with {n_queries} queries")
        print(f"{'='*60}")
        
        # Run both engines
        sb3_results, tensor_results = run_both_engines(queries, config)
        
        # Basic result comparison
        assert sb3_results['total_queries'] == tensor_results['total_queries'], \
            f"Query count mismatch: SB3={sb3_results['total_queries']}, Tensor={tensor_results['total_queries']}"
        
        print(f"SB3 engine:    {sb3_results['successful']}/{sb3_results['total_queries']} successful")
        print(f"Tensor engine: {tensor_results['successful']}/{tensor_results['total_queries']} successful")
        
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

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_success_failure_parity(self, dataset: str, base_config):
        """
        Test that success/failure outcomes match exactly for all queries.
        """
        config = clone_config(base_config)
        config.dataset = dataset
        config.n_queries = 100
        
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        queries = prepare_queries(config)
        
        sb3_results, tensor_results = run_both_engines(queries, config)
        
        sb3_traces = sb3_results.get('traces', [])
        tensor_traces = tensor_results.get('traces', [])
        
        outcome_mismatches = []
        for i, (sb3_t, tensor_t, query) in enumerate(zip(sb3_traces, tensor_traces, queries)):
            sb3_success = sb3_t.get('success', False)
            tensor_success = tensor_t.get('success', False)
            
            if sb3_success != tensor_success:
                split, (pred, head, tail) = query
                outcome_mismatches.append((i, f"{pred}({head},{tail})", sb3_success, tensor_success))
        
        if outcome_mismatches:
            error_msg = f"Found {len(outcome_mismatches)} outcome mismatches:\n"
            for idx, q, sb3_s, tensor_s in outcome_mismatches[:10]:
                error_msg += f"  Query {idx} {q}: SB3={sb3_s}, Tensor={tensor_s}\n"
            pytest.fail(error_msg)
        
        print(f"\n✓ All {config.n_queries} outcomes match for {dataset}")

    @pytest.mark.parametrize("dataset", ["countries_s3", "family"])
    def test_steps_to_completion_parity(self, dataset: str, base_config):
        """
        Test that number of steps to completion matches exactly.
        """
        config = clone_config(base_config)
        config.dataset = dataset
        config.n_queries = 100
        
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        queries = prepare_queries(config)
        
        sb3_results, tensor_results = run_both_engines(queries, config)
        
        sb3_traces = sb3_results.get('traces', [])
        tensor_traces = tensor_results.get('traces', [])
        
        steps_mismatches = []
        for i, (sb3_t, tensor_t, query) in enumerate(zip(sb3_traces, tensor_traces, queries)):
            sb3_steps = sb3_t.get('steps', 0)
            tensor_steps = tensor_t.get('steps', 0)
            
            if sb3_steps != tensor_steps:
                split, (pred, head, tail) = query
                steps_mismatches.append((i, f"{pred}({head},{tail})", sb3_steps, tensor_steps))
        
        if steps_mismatches:
            error_msg = f"Found {len(steps_mismatches)} steps mismatches:\n"
            for idx, q, sb3_s, tensor_s in steps_mismatches[:10]:
                error_msg += f"  Query {idx} {q}: SB3={sb3_s}, Tensor={tensor_s}\n"
            pytest.fail(error_msg)
        
        print(f"\n✓ All {config.n_queries} step counts match for {dataset}")


# ============================================================================
# CLI Runner (for manual testing)
# ============================================================================

def run_parity_tests(
    dataset: str = "countries_s3",
    n_queries: int = 200,
    seed: int = 42,
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """
    Run parity tests programmatically.
    
    Returns:
        (all_passed, results_dict)
    """
    config = create_default_config()
    config.dataset = dataset
    config.n_queries = n_queries
    config.seed = seed
    config.verbose = verbose
    
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    print(f"\n{'='*80}")
    print(f"UNIFICATION ENGINE PARITY TEST")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Queries: {n_queries}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    queries = prepare_queries(config)
    
    print(f"Prepared {len(queries)} queries\n")
    
    # Run both engines
    print("Running SB3 engine...")
    sb3_results = run_engine(
        name="sb3_engine",
        setup_func=setup_sb3_engine,
        test_func=run_sb3_engine,
        queries=queries,
        config=config,
    )
    print(f"  ✓ {sb3_results['successful']}/{sb3_results['total_queries']} successful")
    
    print("Running tensor engine...")
    tensor_results = run_engine(
        name="tensor_engine",
        setup_func=setup_tensor_engine,
        test_func=run_tensor_engine,
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
        print(f"\n✓ ALL TRACES MATCH - Engines are EXACTLY equivalent")
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
    
    parser = argparse.ArgumentParser(description='Test unification engine parity')
    parser.add_argument('--dataset', type=str, default='family',
                        help='Dataset name (default: countries_s3)')
    parser.add_argument('--n-queries', type=int, default=200,
                        help='Number of queries to test (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    passed, results = run_parity_tests(
        dataset=args.dataset,
        n_queries=args.n_queries,
        seed=args.seed,
        verbose=args.verbose
    )
    
    sys.exit(0 if passed else 1)

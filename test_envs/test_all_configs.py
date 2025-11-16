"""
Main test orchestrator for all engine and environment configurations.

Tests all 8 configurations with deterministic or random policy:
1. SB3 engine (string-based unification engine)
2. Tensor engine (tensor unification engine)
3. Batched tensor engine (batched tensor unification engine)
4. SB3 env (string-based environment)
5. Tensor env (tensor environment)
6. Batched tensor env (batched tensor environment)
7. Eval env (environment tested via evaluate_policy)
8. Rollout env (environment tested via rollout collector with random agent)

Usage:
  python test_envs/test_env_engines.py --dataset family --num-queries 20 --deterministic
  python test_envs/test_env_engines.py --dataset countries_s3 --num-queries 100 --random
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import random
import torch
import numpy as np
from typing import Tuple, Dict, List

# Import test modules
from test_envs.test_engine_sb3 import setup_sb3_engine, test_sb3_engine_batch
from test_envs.test_engine_tensor import setup_tensor_engine, test_tensor_engine_batch
from test_envs.test_env_sb3 import setup_sb3_env, test_sb3_env_batch
from test_envs.test_env_tensor import setup_tensor_env, test_tensor_env_batched, test_tensor_env_single_query
from test_envs.test_env_eval import test_eval_env
from test_envs.test_env_rollout import test_rollout_env


def prepare_queries(dataset: str = "countries_s3", base_path: str = "./data/", 
                    n_queries: int = None, seed: int = 42) -> List[Tuple[str, Tuple[str, str, str]]]:
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
    from str_based.str_dataset import DataHandler
    
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
    
    # Shuffle and take first n (to match test_envs.py behavior)
    rng = random.Random(seed)
    rng.shuffle(all_queries)
    
    if n_queries is not None:
        all_queries = all_queries[:n_queries]
    
    return all_queries


def compare_traces_deterministic(traces_dict: Dict[str, List[Dict]], verbose: bool = False) -> bool:
    """
    Compare traces from all configurations for deterministic policy.
    All should produce identical results at every step.
    
    Args:
        traces_dict: Dict mapping config_name -> list of trace dicts
        verbose: Print detailed comparison
        
    Returns:
        True if all traces match
    """
    config_names = list(traces_dict.keys())
    
    if not config_names:
        return True
    
    # Get reference traces (first config)
    ref_name = config_names[0]
    ref_traces = traces_dict[ref_name]
    n_queries = len(ref_traces)
    
    # Compare each query
    mismatches = []
    
    for q_idx in range(n_queries):
        ref_trace = ref_traces[q_idx]['trace']
        ref_success = ref_traces[q_idx]['success']
        
        # Compare with all other configs
        for config_name in config_names[1:]:
            other_trace = traces_dict[config_name][q_idx]['trace']
            other_success = traces_dict[config_name][q_idx]['success']
            
            # Check success flag
            if ref_success != other_success:
                mismatches.append({
                    'query_idx': q_idx,
                    'configs': (ref_name, config_name),
                    'type': 'success',
                    'ref': ref_success,
                    'other': other_success
                })
            
            # Check trace length
            if len(ref_trace) != len(other_trace):
                mismatches.append({
                    'query_idx': q_idx,
                    'configs': (ref_name, config_name),
                    'type': 'trace_length',
                    'ref_len': len(ref_trace),
                    'other_len': len(other_trace)
                })
                continue
            
            # Compare each step
            for step_idx, (ref_step, other_step) in enumerate(zip(ref_trace, other_trace)):
                # Compare states
                if ref_step['state'] != other_step['state']:
                    mismatches.append({
                        'query_idx': q_idx,
                        'step': step_idx,
                        'configs': (ref_name, config_name),
                        'type': 'state',
                        'ref': ref_step['state'],
                        'other': other_step['state']
                    })
                
                # Compare number of actions
                if ref_step['num_actions'] != other_step['num_actions']:
                    mismatches.append({
                        'query_idx': q_idx,
                        'step': step_idx,
                        'configs': (ref_name, config_name),
                        'type': 'num_actions',
                        'ref': ref_step['num_actions'],
                        'other': other_step['num_actions']
                    })
                
                # Compare derived states (if not terminal)
                if not ref_step['done'] and not other_step['done']:
                    ref_derived = sorted(ref_step['derived_states'])
                    other_derived = sorted(other_step['derived_states'])
                    if ref_derived != other_derived:
                        mismatches.append({
                            'query_idx': q_idx,
                            'step': step_idx,
                            'configs': (ref_name, config_name),
                            'type': 'derived_states',
                            'ref_count': len(ref_derived),
                            'other_count': len(other_derived)
                        })
    
    if mismatches:
        print(f"\n{'='*60}")
        print(f"DETERMINISTIC COMPARISON: {len(mismatches)} MISMATCHES FOUND")
        print(f"{'='*60}")
        
        if verbose:
            for i, m in enumerate(mismatches[:10]):  # Show first 10
                print(f"\nMismatch {i+1}:")
                print(f"  Query: {m['query_idx']}")
                if 'step' in m:
                    print(f"  Step: {m['step']}")
                print(f"  Configs: {m['configs'][0]} vs {m['configs'][1]}")
                print(f"  Type: {m['type']}")
                if 'ref' in m and 'other' in m:
                    print(f"  {m['configs'][0]}: {m['ref']}")
                    print(f"  {m['configs'][1]}: {m['other']}")
                elif 'ref_len' in m:
                    print(f"  {m['configs'][0]} length: {m['ref_len']}")
                    print(f"  {m['configs'][1]} length: {m['other_len']}")
                elif 'ref_count' in m:
                    print(f"  {m['configs'][0]} derived: {m['ref_count']}")
                    print(f"  {m['configs'][1]} derived: {m['other_count']}")
            
            if len(mismatches) > 10:
                print(f"\n... and {len(mismatches) - 10} more mismatches")
        
        return False
    else:
        print(f"\n{'='*60}")
        print(f"✓ DETERMINISTIC COMPARISON: ALL CONFIGURATIONS MATCH!")
        print(f"{'='*60}")
        return True


def debug_compare_step_by_step(traces_dict: Dict[str, List[Dict]], queries: List[Tuple]) -> None:
    """
    DEBUG MODE: Compare traces step-by-step and raise error on first mismatch.
    
    This function compares all configurations query-by-query, step-by-step,
    and raises a detailed error as soon as any mismatch is detected.
    
    Args:
        traces_dict: Dict mapping config_name -> list of trace dicts
        queries: List of (split, (predicate, head, tail)) tuples
        
    Raises:
        AssertionError: On first mismatch with detailed information
    """
    config_names = list(traces_dict.keys())
    
    if len(config_names) < 2:
        print("DEBUG: Only one configuration, nothing to compare")
        return
    
    print(f"\n{'='*80}")
    print(f"DEBUG MODE: Step-by-step comparison of {len(config_names)} configurations")
    print(f"Configurations: {', '.join(config_names)}")
    print(f"{'='*80}\n")
    
    # Get reference traces (first config)
    ref_name = config_names[0]
    ref_traces = traces_dict[ref_name]
    n_queries = len(ref_traces)
    
    # Compare each query
    for q_idx in range(n_queries):
        split, (pred, head, tail) = queries[q_idx]
        query_str = f"{pred}({head}, {tail})"
        
        print(f"Query {q_idx}/{n_queries}: {query_str} [{split}]")
        
        ref_trace = ref_traces[q_idx]['trace']
        ref_success = ref_traces[q_idx]['success']
        ref_reward = ref_traces[q_idx]['reward']
        
        # Compare with all other configs
        for config_name in config_names[1:]:
            other_trace = traces_dict[config_name][q_idx]['trace']
            other_success = traces_dict[config_name][q_idx]['success']
            other_reward = traces_dict[config_name][q_idx]['reward']
            
            # Check success
            if ref_success != other_success:
                error_msg = f"\n{'='*80}\n"
                error_msg += f"MISMATCH DETECTED - Query {q_idx}: {query_str}\n"
                error_msg += f"{'='*80}\n"
                error_msg += f"Configuration comparison: {ref_name} vs {config_name}\n"
                error_msg += f"TYPE: Success flag mismatch\n\n"
                error_msg += f"{ref_name} success: {ref_success}\n"
                error_msg += f"{config_name} success: {other_success}\n"
                error_msg += f"\n{ref_name} reward: {ref_reward}\n"
                error_msg += f"{config_name} reward: {other_reward}\n"
                error_msg += f"\nTrace lengths: {ref_name}={len(ref_trace)}, {config_name}={len(other_trace)}\n"
                error_msg += f"{'='*80}\n"
                raise AssertionError(error_msg)
            
            # Check trace length
            if len(ref_trace) != len(other_trace):
                error_msg = f"\n{'='*80}\n"
                error_msg += f"MISMATCH DETECTED - Query {q_idx}: {query_str}\n"
                error_msg += f"{'='*80}\n"
                error_msg += f"Configuration comparison: {ref_name} vs {config_name}\n"
                error_msg += f"TYPE: Trace length mismatch\n\n"
                error_msg += f"{ref_name} trace length: {len(ref_trace)}\n"
                error_msg += f"{config_name} trace length: {len(other_trace)}\n"
                error_msg += f"\n{ref_name} final success: {ref_success}\n"
                error_msg += f"{config_name} final success: {other_success}\n"
                error_msg += f"\nLast steps:\n"
                if ref_trace:
                    error_msg += f"{ref_name} last step {len(ref_trace)-1}:\n"
                    error_msg += f"  State: {ref_trace[-1]['state']}\n"
                    error_msg += f"  Num actions: {ref_trace[-1]['num_actions']}\n"
                    error_msg += f"  Done: {ref_trace[-1]['done']}\n"
                if other_trace:
                    error_msg += f"{config_name} last step {len(other_trace)-1}:\n"
                    error_msg += f"  State: {other_trace[-1]['state']}\n"
                    error_msg += f"  Num actions: {other_trace[-1]['num_actions']}\n"
                    error_msg += f"  Done: {other_trace[-1]['done']}\n"
                error_msg += f"{'='*80}\n"
                raise AssertionError(error_msg)
            
            # Compare each step
            for step_idx, (ref_step, other_step) in enumerate(zip(ref_trace, other_trace)):
                # Compare states
                if ref_step['state'] != other_step['state']:
                    error_msg = f"\n{'='*80}\n"
                    error_msg += f"MISMATCH DETECTED - Query {q_idx}: {query_str}\n"
                    error_msg += f"{'='*80}\n"
                    error_msg += f"Configuration comparison: {ref_name} vs {config_name}\n"
                    error_msg += f"TYPE: State mismatch at step {step_idx}\n\n"
                    error_msg += f"{ref_name} state:\n  {ref_step['state']}\n"
                    error_msg += f"{config_name} state:\n  {other_step['state']}\n"
                    error_msg += f"\nStep info:\n"
                    error_msg += f"  Step: {step_idx}\n"
                    error_msg += f"  {ref_name} num_actions: {ref_step['num_actions']}\n"
                    error_msg += f"  {config_name} num_actions: {other_step['num_actions']}\n"
                    error_msg += f"  {ref_name} done: {ref_step['done']}\n"
                    error_msg += f"  {config_name} done: {other_step['done']}\n"
                    if step_idx > 0:
                        error_msg += f"\nPrevious step {step_idx-1}:\n"
                        error_msg += f"  {ref_name} state: {ref_trace[step_idx-1]['state']}\n"
                        error_msg += f"  {config_name} state: {other_trace[step_idx-1]['state']}\n"
                        error_msg += f"  {ref_name} action: {ref_trace[step_idx-1].get('action', 'N/A')}\n"
                        error_msg += f"  {config_name} action: {other_trace[step_idx-1].get('action', 'N/A')}\n"
                    error_msg += f"{'='*80}\n"
                    raise AssertionError(error_msg)
                
                # Compare number of actions
                if ref_step['num_actions'] != other_step['num_actions']:
                    error_msg = f"\n{'='*80}\n"
                    error_msg += f"MISMATCH DETECTED - Query {q_idx}: {query_str}\n"
                    error_msg += f"{'='*80}\n"
                    error_msg += f"Configuration comparison: {ref_name} vs {config_name}\n"
                    error_msg += f"TYPE: Number of actions mismatch at step {step_idx}\n\n"
                    error_msg += f"Current state: {ref_step['state']}\n"
                    error_msg += f"{ref_name} num_actions: {ref_step['num_actions']}\n"
                    error_msg += f"{config_name} num_actions: {other_step['num_actions']}\n"
                    error_msg += f"\n{ref_name} derived states ({len(ref_step.get('derived_states', []))}):\n"
                    for i, ds in enumerate(ref_step.get('derived_states', [])[:5]):
                        error_msg += f"  {i}: {ds}\n"
                    if len(ref_step.get('derived_states', [])) > 5:
                        error_msg += f"  ... and {len(ref_step['derived_states']) - 5} more\n"
                    error_msg += f"\n{config_name} derived states ({len(other_step.get('derived_states', []))}):\n"
                    for i, ds in enumerate(other_step.get('derived_states', [])[:5]):
                        error_msg += f"  {i}: {ds}\n"
                    if len(other_step.get('derived_states', [])) > 5:
                        error_msg += f"  ... and {len(other_step['derived_states']) - 5} more\n"
                    error_msg += f"{'='*80}\n"
                    raise AssertionError(error_msg)
                
                # Compare derived states (if not terminal)
                if not ref_step['done'] and not other_step['done']:
                    ref_derived = sorted(ref_step.get('derived_states', []))
                    other_derived = sorted(other_step.get('derived_states', []))
                    if ref_derived != other_derived:
                        error_msg = f"\n{'='*80}\n"
                        error_msg += f"MISMATCH DETECTED - Query {q_idx}: {query_str}\n"
                        error_msg += f"{'='*80}\n"
                        error_msg += f"Configuration comparison: {ref_name} vs {config_name}\n"
                        error_msg += f"TYPE: Derived states mismatch at step {step_idx}\n\n"
                        error_msg += f"Current state: {ref_step['state']}\n"
                        error_msg += f"Both have {ref_step['num_actions']} actions\n"
                        error_msg += f"\n{ref_name} derived states (sorted):\n"
                        for i, ds in enumerate(ref_derived[:10]):
                            error_msg += f"  {i}: {ds}\n"
                        if len(ref_derived) > 10:
                            error_msg += f"  ... and {len(ref_derived) - 10} more\n"
                        error_msg += f"\n{config_name} derived states (sorted):\n"
                        for i, ds in enumerate(other_derived[:10]):
                            error_msg += f"  {i}: {ds}\n"
                        if len(other_derived) > 10:
                            error_msg += f"  ... and {len(other_derived) - 10} more\n"
                        error_msg += f"\nDifferences:\n"
                        # Find first difference
                        for i, (r, o) in enumerate(zip(ref_derived, other_derived)):
                            if r != o:
                                error_msg += f"  First diff at index {i}:\n"
                                error_msg += f"    {ref_name}: {r}\n"
                                error_msg += f"    {config_name}: {o}\n"
                                break
                        if len(ref_derived) != len(other_derived):
                            error_msg += f"  Length difference: {len(ref_derived)} vs {len(other_derived)}\n"
                        error_msg += f"{'='*80}\n"
                        raise AssertionError(error_msg)
        
        print(f"  ✓ All {len(config_names)} configurations match for this query\n")
    
    print(f"\n{'='*80}")
    print(f"✓ DEBUG MODE: ALL QUERIES MATCH ACROSS ALL CONFIGURATIONS!")
    print(f"{'='*80}\n")


def print_results_table(results_dict: Dict[str, Dict], deterministic: bool):
    """
    Print a formatted table of results from all configurations.
    
    Args:
        results_dict: Dict mapping config_name -> results dict
        deterministic: Whether deterministic policy was used
    """
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY ({'Deterministic' if deterministic else 'Random'} Policy)")
    print(f"{'='*80}")
    
    # Table header
    print(f"{'Configuration':<40} {'Queries':<10} {'Success':<10} {'Success %':<12} {'Avg Steps':<10}")
    print(f"{'-'*80}")
    
    # Table rows
    for config_name, results in results_dict.items():
        total = results['total_queries']
        success = results['successful']
        success_pct = (success / total * 100) if total > 0 else 0.0
        avg_steps = results['avg_steps']
        
        print(f"{config_name:<40} {total:<10} {success:<10} {success_pct:<12.2f} {avg_steps:<10.2f}")
    
    print(f"{'='*80}")
    
    # Summary statistics
    all_success_pcts = [
        (r['successful'] / r['total_queries'] * 100) if r['total_queries'] > 0 else 0.0
        for r in results_dict.values()
    ]
    
    if all_success_pcts:
        avg_success = np.mean(all_success_pcts)
        std_success = np.std(all_success_pcts)
        min_success = np.min(all_success_pcts)
        max_success = np.max(all_success_pcts)
        
        print(f"\nOverall Statistics:")
        print(f"  Average success rate: {avg_success:.2f}% ± {std_success:.2f}%")
        print(f"  Min/Max success rate: {min_success:.2f}% / {max_success:.2f}%")
        
        # Check if rollout_env is in the mix (uses random policy)
        has_rollout = 'rollout_env' in results_dict
        
        if deterministic:
            # For deterministic mode, most configs should be identical except rollout_env
            if has_rollout:
                # Expect variation due to rollout_env using random policy
                det_configs = {k: v for k, v in results_dict.items() if k != 'rollout_env'}
                if det_configs:
                    det_pcts = [(r['successful'] / r['total_queries'] * 100) for r in det_configs.values()]
                    det_std = np.std(det_pcts)
                    if det_std < 0.01:
                        print(f"  ✓ Deterministic configurations have identical success rates")
                        print(f"    Note: rollout_env uses random policy, expected to differ")
                    else:
                        print(f"  ✗ WARNING: Deterministic configurations differ!")
            else:
                # All should be identical
                if std_success < 0.01:
                    print(f"  ✓ All configurations have identical success rates (as expected)")
                else:
                    print(f"  ✗ WARNING: Success rates differ between configurations!")
        else:
            # Random should be around 42% with some variation
            if 35 <= avg_success <= 50:
                print(f"  ✓ Average success rate is in expected range (35-50%)")
                if avg_success < 40:
                    print(f"    Note: {avg_success:.2f}% is slightly lower than expected ~42%")
                elif avg_success > 45:
                    print(f"    Note: {avg_success:.2f}% is slightly higher than expected ~42%")
            else:
                print(f"  ✗ WARNING: Average success rate {avg_success:.2f}% is outside expected range!")
    
    print(f"{'='*80}\n")


def run_all_tests(
    dataset: str = "countries_s3",
    n_queries: int = 100,
    deterministic: bool = True,
    max_depth: int = 20,
    seed: int = 42,
    verbose: bool = False,
    debug: bool = False,
    configs: List[str] = None
):
    """
    Run tests on all requested configurations.
    
    Args:
        dataset: Dataset name
        n_queries: Number of queries to test
        deterministic: If True, use canonical ordering; if False, random actions
        max_depth: Maximum proof depth
        seed: Random seed
        verbose: Print detailed information
        debug: If True, compare step-by-step and raise error on first mismatch
        configs: List of config names to test. If None, test all.
                 Options: 'sb3_engine', 'tensor_engine', 'batched_tensor_engine',
                         'sb3_env', 'tensor_env', 'batched_tensor_env', 'eval_env', 'rollout_env'
    """
    # Set all random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"TESTING ALL CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Queries: {n_queries}")
    print(f"Policy: {'Deterministic (canonical)' if deterministic else 'Random'}")
    print(f"Max depth: {max_depth}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Prepare queries
    print("Preparing queries...")
    queries = prepare_queries(dataset=dataset, n_queries=n_queries, seed=seed)
    print(f"  Prepared {len(queries)} queries\n")
    
    # Default to all configs if none specified
    # Order matters: engines first, then environments for proper deterministic comparison
    if configs is None:
        configs = [
            # Engines first (SKIP)
            # 'sb3_engine',
            # 'tensor_engine', 
            # 'batched_tensor_engine',
            # Environments second (with environment wrapper)
            'sb3_env',
            'tensor_env',
            'batched_tensor_env',
            'eval_env',
            'rollout_env'
        ]
    
    results_dict = {}
    traces_dict = {}
    
    # Test each configuration
    for config_name in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print(f"{'='*60}")
        
        try:
            if config_name == 'sb3_engine':
                print("Setting up SB3 engine...")
                engine_data = setup_sb3_engine(dataset=dataset)
                print("Running tests...")
                results = test_sb3_engine_batch(
                    queries, engine_data,
                    deterministic=deterministic,
                    max_depth=max_depth,
                    seed=seed,
                    verbose=verbose
                )
                results_dict[config_name] = results
                traces_dict[config_name] = results['traces']
                
            elif config_name == 'tensor_engine':
                print("Setting up tensor engine...")
                engine_data = setup_tensor_engine(dataset=dataset, batched=False)
                print("Running tests...")
                results = test_tensor_engine_batch(
                    queries, engine_data,
                    deterministic=deterministic,
                    max_depth=max_depth,
                    seed=seed,
                    verbose=verbose
                )
                results_dict[config_name] = results
                traces_dict[config_name] = results['traces']
                
            elif config_name == 'batched_tensor_engine':
                # For batched engine, we still test one query at a time for now
                # (true batching would require different implementation)
                print("Setting up batched tensor engine...")
                engine_data = setup_tensor_engine(dataset=dataset, batched=True)
                print("Running tests...")
                results = test_tensor_engine_batch(
                    queries, engine_data,
                    deterministic=deterministic,
                    max_depth=max_depth,
                    seed=seed,
                    verbose=verbose
                )
                results_dict[config_name] = results
                traces_dict[config_name] = results['traces']
                
            elif config_name == 'sb3_env':
                print("Setting up SB3 environment...")
                env_data = setup_sb3_env(dataset=dataset, seed=seed)
                print("Running tests...")
                results = test_sb3_env_batch(
                    queries, env_data,
                    deterministic=deterministic,
                    max_depth=max_depth,
                    seed=seed,
                    verbose=verbose
                )
                results_dict[config_name] = results
                traces_dict[config_name] = results['traces']
                
            elif config_name == 'tensor_env':
                print(f"Setting up tensor environment (batch_size=1 for sequential testing)...")
                env_data = setup_tensor_env(dataset=dataset, seed=seed, batch_size=1)
                print("Running tests (looping through all queries)...")
                from test_envs.test_env_tensor import test_tensor_env
                results = test_tensor_env(
                    queries, env_data,
                    deterministic=deterministic,
                    max_depth=max_depth,
                    seed=seed,
                    verbose=verbose
                )
                results_dict[config_name] = results
                traces_dict[config_name] = results['traces']
                
            elif config_name == 'batched_tensor_env':
                print("Setting up batched tensor environment (eval mode)...")
                env_data = setup_tensor_env(dataset=dataset, seed=seed, batch_size=len(queries))
                print("Running tests (TRUE BATCH MODE with eval dataset)...")
                results = test_tensor_env_batched(
                    queries, env_data,
                    deterministic=deterministic,
                    max_depth=max_depth,
                    seed=seed,
                    verbose=verbose
                )
                results_dict[config_name] = results
                traces_dict[config_name] = results['traces']
            
            elif config_name == 'eval_env':
                print("Testing with evaluate_policy...")
                results = test_eval_env(
                    queries=queries,
                    dataset=dataset,
                    deterministic=deterministic,
                    max_depth=max_depth,
                    seed=seed,
                    verbose=verbose
                )
                results_dict[config_name] = results
                traces_dict[config_name] = results['traces']
            
            elif config_name == 'rollout_env':
                print("Testing with rollout collector...")
                results = test_rollout_env(
                    queries=queries,
                    dataset=dataset,
                    deterministic=deterministic,
                    max_depth=max_depth,
                    seed=seed,
                    verbose=verbose
                )
                results_dict[config_name] = results
                traces_dict[config_name] = results['traces']
            
            print(f"\n✓ {config_name}: {results['successful']}/{results['total_queries']} successful ({results['avg_reward']*100:.2f}%)")
            
        except Exception as e:
            print(f"\n✗ {config_name}: ERROR - {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print results table
    print_results_table(results_dict, deterministic)
    
    # Compare traces if deterministic
    if deterministic and len(traces_dict) > 1:
        # Separate engine and environment configs
        # Note: eval_env and rollout_env don't track detailed traces, so exclude them from comparison
        engine_configs = {k: v for k, v in traces_dict.items() if 'engine' in k}
        env_configs = {k: v for k, v in traces_dict.items() if 'env' in k and k not in ['eval_env', 'rollout_env']}
        
        if debug:
            # DEBUG MODE: Compare step-by-step and raise error on first mismatch
            # Compare engines first
            if len(engine_configs) > 1:
                print(f"\n{'='*80}")
                print(f"DEBUG MODE: Comparing engines")
                print(f"{'='*80}")
                debug_compare_step_by_step(engine_configs, queries)
            
            # Then compare environments (excluding eval_env which has no traces)
            if len(env_configs) > 1:
                print(f"\n{'='*80}")
                print(f"DEBUG MODE: Comparing environments")
                print(f"{'='*80}")
                debug_compare_step_by_step(env_configs, queries)
        else:
            # Normal mode: Show summary of mismatches
            # Compare engines first
            if len(engine_configs) > 1:
                print(f"\n{'='*80}")
                print(f"COMPARING ENGINE TRACES")
                print(f"{'='*80}")
                compare_traces_deterministic(engine_configs, verbose=verbose)
            
            # Then compare environments (excluding eval_env which has no traces)
            if len(env_configs) > 1:
                print(f"\n{'='*80}")
                print(f"COMPARING ENVIRONMENT TRACES")
                print(f"{'='*80}")
                compare_traces_deterministic(env_configs, verbose=verbose)
    
    return results_dict, traces_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test all engine and environment configurations')
    parser.add_argument('--dataset', type=str, default='family', help='Dataset name')
    parser.add_argument('--n_queries', type=int, default=200, help='Number of queries to test')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic (canonical) policy')
    parser.add_argument('--random', action='store_true', help='Use random policy')
    parser.add_argument('--max-depth', type=int, default=20, help='Maximum proof depth')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    parser.add_argument('--debug', action='store_true', help='Debug mode: compare step-by-step and raise error on first mismatch')
    parser.add_argument('--configs', type=str, nargs='+', help='Specific configs to test')
    
    args = parser.parse_args()
    
    # Default to deterministic if neither specified
    if not args.deterministic and not args.random:
        deterministic = True
    else:
        deterministic = args.deterministic
    
    run_all_tests(
        dataset=args.dataset,
        n_queries=args.n_queries,
        deterministic=deterministic,
        max_depth=args.max_depth,
        seed=args.seed,
        verbose=args.verbose,
        debug=args.debug,
        configs=args.configs
    )

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
from typing import Tuple, Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from types import SimpleNamespace

# Import test modules
from test_envs.test_engine_sb3 import setup_sb3_engine, test_sb3_engine_batch
from test_envs.test_engine_tensor import setup_tensor_engine, test_tensor_engine_batch
from test_envs.test_env_sb3 import setup_sb3_env, test_sb3_env_batch
from test_envs.test_env_tensor import setup_tensor_env, test_tensor_env_batched, test_tensor_env
from test_envs.test_env_eval import test_eval_env
from test_envs.test_env_rollout import test_rolloutsb3_env #,test_rollout_env


def create_default_test_config() -> SimpleNamespace:
    """
    Create the default configuration namespace for all CLI/test arguments.

    Returns:
        SimpleNamespace with default configuration parameters.
    """
    args = SimpleNamespace(
        dataset="countries_s3",
        n_queries=800,
        deterministic=True,
        max_depth=20,
        seed=42,
        verbose=False,
        debug=False,
        configs=None,
        collect_action_stats=True,
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
        # PPO-specific parameters (for rolloutsb3_env)
        n_steps=128,
        n_epochs=1,
        batch_size=1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        learning_rate=1e-4,
    )
    if args.end_proof_action:
        args.deterministic = False
        args.random = True
        print("Note: end_proof_action is enabled, setting deterministic=False")        
        print("random results will be very low if end_proof_action is enabled"
              "and random policy is used (with deterministic it is never chosen")
    if not args.use_exact_memory:
        print("Using BloomFilter for memory instead of ExactMemory"
              "will give different results in deterministic settings")
    return args

def clone_config(config: SimpleNamespace) -> SimpleNamespace:
    """Create a shallow clone of a SimpleNamespace to avoid shared state."""
    return SimpleNamespace(**vars(config))


def add_bool_override_argument(parser, name: str, default: bool, help_text: str) -> None:
    """
    Register paired CLI arguments (--name/--no-name) to override boolean configs.
    """
    cli_name = name.replace("_", "-")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{cli_name}",
        dest=name,
        action="store_true",
        help=f"Enable {help_text} (default: {default})",
    )
    group.add_argument(
        f"--no-{cli_name}",
        dest=name,
        action="store_false",
        help=f"Disable {help_text} (default: {default})",
    )
    parser.set_defaults(**{name: default})


@dataclass
class Config:
    """Configuration for a test setup."""
    name: str
    description: str
    setup_func: Optional[Callable[..., Any]] = None
    test_func: Callable[..., Dict] = None
    setup_kwargs: Optional[Dict[str, Any]] = None
    test_kwargs: Optional[Dict[str, Any]] = None
    has_traces: bool = True
    category: str = "environment"


class Configs:
    """Centralized configuration namespace for all test configurations."""

    sb3_engine = Config(
        name="sb3_engine",
        description="SB3 engine (string-based unification engine)",
        setup_func=setup_sb3_engine,
        test_func=test_sb3_engine_batch,
        setup_kwargs={},
        has_traces=True,
        category="engine"
    )

    tensor_engine = Config(
        name="tensor_engine",
        description="Tensor engine (tensor unification engine)",
        setup_func=setup_tensor_engine,
        test_func=test_tensor_engine_batch,
        setup_kwargs={"batched": False},
        has_traces=True,
        category="engine"
    )

    batched_tensor_engine = Config(
        name="batched_tensor_engine",
        description="Batched tensor engine (batched tensor unification engine)",
        setup_func=setup_tensor_engine,
        test_func=test_tensor_engine_batch,
        setup_kwargs={"batched": True},
        has_traces=True,
        category="engine"
    )

    sb3_env = Config(
        name="sb3_env",
        description="SB3 env (string-based environment)",
        setup_func=setup_sb3_env,
        test_func=test_sb3_env_batch,
        setup_kwargs={},
        has_traces=True,
        category="environment"
    )

    tensor_env = Config(
        name="tensor_env",
        description="Tensor env (tensor environment)",
        setup_func=setup_tensor_env,
        test_func=test_tensor_env,
        setup_kwargs={"batch_size": 1},
        has_traces=True,
        category="environment"
    )

    batched_tensor_env = Config(
        name="batched_tensor_env",
        description="Batched tensor env (batched tensor environment)",
        setup_func=setup_tensor_env,
        test_func=test_tensor_env_batched,
        setup_kwargs={},  # batch_size will be passed dynamically as len(queries)
        has_traces=True,
        category="environment"
    )

    eval_env = Config(
        name="eval_env",
        description="Eval env (environment tested via evaluate_policy)",
        setup_func=None,
        test_func=test_eval_env,
        has_traces=False,
        category="environment"
    )

    # rollout_env = Config(
    #     name="rollout_env",
    #     description="Rollout env (environment tested via rollout collector with random agent)",
    #     setup_func=None,
    #     test_func=test_rollout_env,
    #     has_traces=False,
    #     category="environment"
    # )

    rolloutsb3_env = Config(
        name="rolloutsb3_env",
        description="RolloutSB3 env (environment tested via PPO SB3-style rollout)",
        setup_func=None,
        test_func=test_rolloutsb3_env,
        has_traces=True,
        category="environment"
    )

    @classmethod
    def get_all_configs(cls) -> List[Config]:
        return [
            cls.sb3_engine,
            cls.tensor_engine,
            cls.batched_tensor_engine,
            cls.sb3_env,
            cls.tensor_env,
            cls.batched_tensor_env,
            cls.eval_env,
            # cls.rollout_env,
            cls.rolloutsb3_env,
        ]

    @classmethod
    def get_config_by_name(cls, name: str) -> Config:
        for config in cls.get_all_configs():
            if config.name == name:
                return config
        raise ValueError(f"Unknown configuration: {name}")

    @classmethod
    def get_default_env_configs(cls) -> List[str]:
        return [
            'sb3_env',
            # 'tensor_env',
            'batched_tensor_env',
            'eval_env',
            # 'rollout_env',
            'rolloutsb3_env'
        ]


def run_config(config_meta: Config, queries: List[Tuple[str, Tuple[str, str, str]]], cfg: SimpleNamespace) -> Dict:
    """
    Run a test configuration by calling setup_func (if exists) and test_func.
    
    All test functions now receive:
    - queries: List of query tuples
    - config: SimpleNamespace with all configuration parameters
    
    Test functions with setup_func also receive:
    - setup_result: Result from setup function
    """
    if config_meta.setup_func:
        # Prepare setup kwargs
        setup_kwargs = {
            "dataset": cfg.dataset,
            "seed": cfg.seed,
            "config": cfg,
        }
        
        # Add specific setup kwargs (e.g., batched=True/False)
        if config_meta.setup_kwargs:
            setup_kwargs.update(config_meta.setup_kwargs)
        
        # Special handling for batch_size - compute dynamically
        if config_meta.name == "batched_tensor_env":
            setup_kwargs["batch_size"] = len(queries)
        
        setup_result = config_meta.setup_func(**setup_kwargs)
        
        # Test functions with setup: (queries, setup_result, config)
        return config_meta.test_func(queries, setup_result, cfg)
    else:
        # Test functions without setup: (queries, config)
        return config_meta.test_func(queries, cfg)


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
    print(f"{'-'*84}")
    
    # Table header
    print(f"{'Configuration':<40} {'Queries':<10} {'Success':<10} {'Success %':<12} {'Avg Reward':<12} {'Avg Steps':<10} {'Avg Actions':<12}")
    print(f"{'-'*96}")
    
    # Table rows
    for config_name, results in results_dict.items():
        total = results['total_queries']
        success = results['successful']
        success_pct = (success / total * 100) if total > 0 else 0.0
        avg_reward = results.get('avg_reward', 0.0)
        avg_steps = results['avg_steps']
        avg_actions = results.get('avg_actions', 0.0)
        
        print(f"{config_name:<40} {total:<10} {success:<10} {success_pct:<12.2f} {avg_reward:<12.2f} {avg_steps:<10.2f} {avg_actions:<12.2f}")
    
    print(f"{'='*96}")
    
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
    
    print(f"{'='*80}\n")


def run_all_tests(
    dataset: str = "countries_s3",
    n_queries: int = 100,
    deterministic: bool = True,
    max_depth: int = 20,
    seed: int = 42,
    verbose: bool = False,
    debug: bool = False,
    configs: List[str] = None,
    collect_action_stats: bool = False,
    config: SimpleNamespace = None
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
        configs: List of config names to test. If None, use defaults.
        collect_action_stats: If True, collect action statistics for eval/rollout configs
        config: SimpleNamespace with shared configuration overrides for all setups
    """
    if config is not None:
        cfg = clone_config(config)
    else:
        cfg = create_default_test_config()
        overrides = {
            "dataset": dataset,
            "n_queries": n_queries,
            "deterministic": deterministic,
            "max_depth": max_depth,
            "seed": seed,
            "verbose": verbose,
            "debug": debug,
            "collect_action_stats": collect_action_stats,
        }
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
    if configs is not None:
        cfg.configs = configs
    
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    print(f"\n{'='*80}")
    print(f"TESTING ALL CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"Dataset: {cfg.dataset}")
    print(f"Queries: {cfg.n_queries}")
    print(f"Policy: {'Deterministic (canonical)' if cfg.deterministic else 'Random'}")
    print(f"Max depth: {cfg.max_depth}")
    print(f"Seed: {cfg.seed}")
    print(f"{'='*80}\n")
    
    print("Preparing queries...")
    queries = prepare_queries(dataset=cfg.dataset, n_queries=cfg.n_queries, seed=cfg.seed)
    print(f"  Prepared {len(queries)} queries\n")
    
    selected_configs = cfg.configs or Configs.get_default_env_configs()
    
    results_dict: Dict[str, Dict] = {}
    traces_dict: Dict[str, List[Dict]] = {}
    
    for config_name in selected_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print(f"{'='*60}")
        config_meta = Configs.get_config_by_name(config_name)
        
        try:
            results = run_config(config_meta, queries, cfg)
            results_dict[config_name] = results
            if config_meta.has_traces:
                traces_dict[config_name] = results['traces']
            
            print(f"\n✓ {config_name}: {results['successful']}/{results['total_queries']} successful ({results['avg_reward']*100:.2f}%)")
        except Exception as e:
            print(f"\n✗ {config_name}: ERROR - {str(e)}")
            import traceback
            traceback.print_exc()
    
    print_results_table(results_dict, cfg.deterministic)
    
    if cfg.deterministic and len(traces_dict) > 1:
        engine_configs = {
            name: traces
            for name, traces in traces_dict.items()
            if Configs.get_config_by_name(name).category == 'engine'
        }
        env_configs = {
            name: traces
            for name, traces in traces_dict.items()
            if Configs.get_config_by_name(name).category == 'environment'
        }
        
        # Filter out configs without detailed traces (e.g., eval_env)
        env_configs_with_traces = {
            name: traces
            for name, traces in env_configs.items()
            if traces and len(traces) > 0 and traces[0].get('trace') and len(traces[0]['trace']) > 0 and 'state' in traces[0]['trace'][0]
        }
        
        if cfg.debug:
            # Re-run tests for debug
            debug_traces = {}
            
            # Iterate over configs that were already run and have traces
            for config_name in env_configs_with_traces.keys():
                print(f"\nTesting: {config_name}")
                
                # Setup environment and run test
                if config_name == 'sb3_env':
                    env_data = setup_sb3_env(dataset=cfg.dataset, config=cfg)
                    results = test_sb3_env_batch(queries, env_data, cfg)
                elif config_name == 'batched_tensor_env':
                    env_data = setup_tensor_env(dataset=cfg.dataset, config=cfg)
                    results = test_tensor_env_batched(queries, env_data, cfg)
                else:
                    continue
                
                traces = results['traces']
                debug_traces[config_name] = traces
                
            # Compare traces
            debug_compare_step_by_step(debug_traces, queries)
        else:
            if len(engine_configs) > 1:
                print(f"\n{'='*80}")
                print(f"COMPARING ENGINE TRACES")
                print(f"{'='*80}")
                compare_traces_deterministic(engine_configs, verbose=cfg.verbose)
            if len(env_configs_with_traces) > 1:
                print(f"\n{'='*80}")
                print(f"COMPARING ENVIRONMENT TRACES")
                print(f"{'='*80}")
                compare_traces_deterministic(env_configs_with_traces, verbose=cfg.verbose)
    
    return results_dict, traces_dict


if __name__ == "__main__":
    import argparse
    
    base_cli_config = create_default_test_config()
    parser = argparse.ArgumentParser(description='Test all engine and environment configurations')
    parser.add_argument('--dataset', type=str, default=base_cli_config.dataset, help=f'Dataset name (default: {base_cli_config.dataset})')
    parser.add_argument('--n_queries', type=int, default=base_cli_config.n_queries, help=f'Number of queries to test (default: {base_cli_config.n_queries})')
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument('--deterministic', dest='deterministic', action='store_true',
                              help='Use deterministic (canonical) policy')
    policy_group.add_argument('--random', dest='deterministic', action='store_false',
                              help='Use random policy')
    parser.set_defaults(deterministic=base_cli_config.deterministic)
    parser.add_argument('--max-depth', type=int, default=base_cli_config.max_depth, help=f'Maximum proof depth (default: {base_cli_config.max_depth})')
    parser.add_argument('--seed', type=int, default=base_cli_config.seed, help=f'Random seed (default: {base_cli_config.seed})')
    add_bool_override_argument(parser, 'verbose', base_cli_config.verbose, 'verbose logging')
    add_bool_override_argument(parser, 'debug', base_cli_config.debug, 'debug comparisons')
    add_bool_override_argument(parser, 'collect_action_stats', base_cli_config.collect_action_stats,
                               'collect action statistics for eval/rollout configs')
    parser.add_argument('--configs', type=str, nargs='+', default=None, help='Specific configs to test')
    parser.add_argument('--padding-atoms', dest='padding_atoms', type=int, default=base_cli_config.padding_atoms,
                        help=f'Override padding_atoms (default: {base_cli_config.padding_atoms})')
    parser.add_argument('--padding-states', dest='padding_states', type=int, default=base_cli_config.padding_states,
                        help=f'Override padding_states (default: {base_cli_config.padding_states})')
    parser.add_argument('--reward-type', dest='reward_type', type=int, default=base_cli_config.reward_type,
                        help=f'Set reward_type (default: {base_cli_config.reward_type})')
    parser.add_argument('--prover-verbose', dest='prover_verbose', type=int, default=base_cli_config.prover_verbose,
                        help=f'Set prover verbosity (default: {base_cli_config.prover_verbose})')
    parser.add_argument('--max-total-runtime-vars', dest='max_total_runtime_vars', type=int,
                        default=base_cli_config.max_total_runtime_vars,
                        help=f'Set max_total_runtime_vars (default: {base_cli_config.max_total_runtime_vars})')
    parser.add_argument('--max-derived-per-state', dest='max_derived_per_state', type=int,
                        default=base_cli_config.max_derived_per_state,
                        help=f'Set max derived states per state (default: {base_cli_config.max_derived_per_state})')
    parser.add_argument('--device', dest='device', type=str, default=base_cli_config.device,
                        help=f'Set device for tensor components (default: {base_cli_config.device})')
    add_bool_override_argument(parser, 'skip_unary_actions', base_cli_config.skip_unary_actions,
                               'skip unary actions in env/engine steps')
    add_bool_override_argument(parser, 'end_proof_action', base_cli_config.end_proof_action,
                               'enable synthetic end-of-proof action')
    add_bool_override_argument(parser, 'use_exact_memory', base_cli_config.use_exact_memory,
                               'use exact memory mode for batched env')
    add_bool_override_argument(parser, 'memory_pruning', base_cli_config.memory_pruning,
                               'enable memory pruning')
    
    args = parser.parse_args()
    
    cli_config = clone_config(base_cli_config)
    for key, value in vars(args).items():
        setattr(cli_config, key, value)
    
    run_all_tests(config=cli_config)

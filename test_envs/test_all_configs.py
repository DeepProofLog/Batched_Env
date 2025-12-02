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
# add also the path in sb3 folder
sys.path.insert(0, os.path.join(root_path, 'sb3'))

import random
import torch
import numpy as np
from typing import Tuple, Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from types import SimpleNamespace

# Import test modules
from test_envs.test_engine_sb3 import setup_sb3_engine, test_sb3_engine
from test_envs.test_engine_tensor import setup_tensor_engine, test_tensor_engine
from test_envs.test_env_sb3 import setup_sb3_env, test_sb3_env_batch
from test_envs.test_env_tensor import setup_tensor_env, test_tensor_env_batched, test_tensor_env
from test_envs.test_env_eval import test_tensor_env
from test_envs.test_env_rollout import test_tensor_rollout

from compare_traces_metrics import compare_traces_metrics

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
        # PPO-specific parameters (for tensor_rollout)
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
    """
    Configuration for a test setup.
    setup_kwargs: Optional keyword arguments for setup_func
    test_kwargs: Optional keyword arguments for test_func
    """
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

    # ------------------ ENGINE CONFIGURATIONS ------------------
    sb3_engine = Config(
        name="sb3_engine",
        description="SB3 engine (string-based unification engine)",
        setup_func=setup_sb3_engine,
        test_func=test_sb3_engine,
        setup_kwargs={},
        has_traces=True,
        category="engine"
    )

    tensor_engine = Config(
        name="tensor_engine",
        description="Tensor engine (tensor unification engine)",
        setup_func=setup_tensor_engine,
        test_func=test_tensor_engine,
        setup_kwargs={},
        has_traces=True,
        category="engine"
    )

    # ------------------ ENVIRONMENT CONFIGURATIONS ------------------

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
        description="Eval env (environment tested via evaluate_policy)",
        setup_func=None,
        test_func=test_tensor_env,
        has_traces=False,
        category="environment"
    )

    # ------------------ ROLLOUT CONFIGURATIONS ------------------
    tensor_rollout = Config(
        name="tensor_rollout",
        description="RolloutSB3 env (environment tested via PPO SB3-style rollout)",
        setup_func=None,
        test_func=test_tensor_rollout,
        has_traces=True,
        category="rollout"
    )

    @classmethod
    def get_all_configs(cls) -> List[Config]:
        return [
            cls.sb3_engine,
            cls.tensor_engine,
            # cls.sb3_env,
            # cls.tensor_env,
            # cls.tensor_rollout,
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
            'sb3_engine',
            'tensor_engine',
            # 'sb3_env',
            # 'tensor_env',
            # 'tensor_rollout',
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
        
        # Check if tensor_rollout is in the mix (uses random policy)
        has_rollout = 'tensor_rollout' in results_dict
        
        if deterministic:
            # All should be identical
            if std_success < 0.01:
                print(f"  ✓ All configurations have identical success rates (as expected)")
            else:
                print(f"  ✗ WARNING: Success rates differ between configurations!")
    
    print(f"{'='*80}\n")


def run_all_tests(config: SimpleNamespace):
    """
    Run tests on all requested configurations.
    Args:
        config: SimpleNamespace with shared configuration overrides for all setups
    """
    cfg = clone_config(config)

    
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    print(f"\n{'='*80}")
    print(f"TESTING ALL CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"Dataset: {cfg.dataset}")
    print(f"Queries: {cfg.n_queries}")
    print(f"Policy: {'Deterministic (canonical)' if cfg.deterministic else 'Random'}")
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

        rollout_configs = {
            name: traces
            for name, traces in traces_dict.items()
            if Configs.get_config_by_name(name).category == 'rollout'
        }

        if len(engine_configs) > 1:
            print(f"\n{'='*80}")
            print(f"COMPARING ENGINE TRACES")
            print(f"{'='*80}")
            compare_traces_metrics(engine_configs, verbose=cfg.verbose)
        if len(env_configs) > 1:
            print(f"\n{'='*80}")
            print(f"COMPARING ENVIRONMENT TRACES")
            print(f"{'='*80}")
            compare_traces_metrics(env_configs, verbose=cfg.verbose)
        if len(rollout_configs) > 1:
            print(f"\n{'='*80}")
            print(f"COMPARING ROLLOUT TRACES")
            print(f"{'='*80}")
            compare_traces_metrics(rollout_configs, verbose=cfg.verbose)
    
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
    
    run_all_tests(cli_config)

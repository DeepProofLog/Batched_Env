"""
/home/castellanoontiv/miniconda3/envs/rl/bin/python /home/castellanoontiv/Batched_env/run_comparison.py

Simplified sb3 runner for comparison with batched implementation.

This uses the same config as runner_new.py but runs the sb3 version.
"""

import argparse
import copy
import datetime
import os
import sys
import random
import numpy as np
import torch
import gc

# Add both paths to ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sb3.sb3_utils import FileLogger
from sb3.sb3_train import main as train_main


def _shared_default_config():
    """
    Build the config we want both SB3 and batched paths to use.

    The defaults mirror runner_new.py and force CPU/deterministic settings
    so we can compare step-by-step traces.
    """
    return {
        # Dataset params
        'dataset_name': 'countries_s3',
        'eval_neg_samples': None,
        'test_neg_samples': None,
        'train_depth': None,
        'valid_depth': None,
        'test_depth': None,
        'n_train_queries': None,
        'n_eval_queries': None, # 10 for testing
        'n_test_queries': None, # 10 for testing
        'prob_facts': False,
        'topk_facts': None,
        'topk_facts_threshold': 0.33,

        # Model params
        'model_name': 'PPO',
        'ent_coef': 0.5,
        'clip_range': 0.2,
        'n_epochs': 10,
        'lr': 3e-4,
        'gamma': 0.99,

        # Training params
        'seed': 0,
        'seed_run_i': 0,
        'timesteps_train': 20000, #128 for testing
        'restore_best_val_model': False,
        'load_model': False,
        'save_model': False,
        'n_envs': 128, #1 for testing
        'n_steps': 128,
        'n_eval_envs': 128, #1 for testing
        'batch_size': 4096, #128 for testing
        'use_compile': False,

        # Env params
        'reward_type': 4,
        'train_neg_ratio': 1,
        'engine': 'python',
        'engine_strategy': 'cmp',
        'endf_action': True,  # sb3 uses endf_action instead of end_proof_action
        'endt_action': False,
        'skip_unary_actions': True,
        'canonical_action_order': True,
        'max_depth': 20,
        'memory_pruning': True,
        'corruption_mode': 'dynamic',
        'corruption_scheme': ['head', 'tail'],
        'false_rules': False,

        # KGE params (disabled for fair comparison)
        'kge_action': False,
        'logit_fusion': False,
        'inference_fusion': False,
        'inference_success_only': False,
        'pbrs': False,
        'enable_top_k': False,
        'kge_engine': 'tf',
        'kge_checkpoint_dir': './../../checkpoints/',
        'kge_run_signature': None,
        'kge_scores_file': None,
        'top_k_init_value': 10,
        'top_k_final_value': 7,
        'top_k_start': 0.3,
        'top_k_end': 1,
        'top_k_transform': 'linear',
        'pbrs_beta': 0.0,
        'pbrs_gamma': None,
        'kge_logit_init_value': 1.0,
        'kge_logit_transform': 'log',
        'kge_logit_eps': 1e-6,
        'eval_hybrid_kge_weight': 1.0,
        'eval_hybrid_rl_weight': 1.0,
        'eval_hybrid_success_only': True,

        # Embedding params
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        'atom_embedding_size': 256,
        'constant_embedding_size': 256,
        'predicate_embedding_size': 256,
        'learn_embeddings': True,
        'padding_atoms': 6,
        'padding_states': 20,
        # Keep variable preallocation modest to prevent OOM in the index manager.
        'max_total_vars': 50000,

        # Other params
        'device': 'cuda',
        'min_gpu_memory_gb': 2.0,
        'extended_eval_info': True,
        'eval_best_metric': 'mrr',
        'plot_trajectories': False,
        'plot': False,
        'depth_info': True,
        'verbose_cb': False,
        'verbose_env': 0,
        'verbose_prover': 0,
        'data_path': './data/',
        'models_path': './models/',
        'rules_file': 'rules.txt',
        'facts_file': 'train.txt',
        'use_logger': True,
        'logger_path': './runs/',
        'use_wb': False,
        'wb_path': './../wandb/',
        'debug_ppo': False,
        'janus_file': None,
        'train_file': 'train.txt',
        'valid_file': 'valid.txt',
        'test_file': 'test.txt',
        'annealing_specs': {},
        'allow_small_eval': False,
    }


def _format_stat(mean, std, count):
    """Format metric stats as mean +/- std with optional count."""
    if mean is None:
        return "N/A"
    if std is None and count is None:
        return f"{mean:.3f}"
    if std is None:
        return f"{mean:.3f} (n={count})" if count is not None else f"{mean:.3f}"
    if count is None:
        return f"{mean:.3f} +/- {std:.2f}"
    return f"{mean:.3f} +/- {std:.2f} ({count})"


def _print_test_metrics(metrics: dict) -> None:
    """
    Display test metrics with grouped mean/std/count formatting.

    This mirrors the display used during evaluation for consistency.
    """
    print(f"\nTest metrics:")
    grouped = {}
    grouped_suffixes = ("_mean", "_std", "_count")
    handled_keys = set()

    for key, value in metrics.items():
        if not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        for suffix in grouped_suffixes:
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                grouped.setdefault(base, {"mean": None, "std": None, "count": None})
                grouped[base][suffix.lstrip("_")] = float(value)
                handled_keys.add(key)
                break

    final_output = {}

    for base in grouped:
        stats = grouped[base]
        count_val = stats.get("count")
        display = _format_stat(
            stats.get("mean"),
            stats.get("std"),
            int(count_val) if count_val is not None else None,
        )
        final_output[base] = display

    for key in metrics.keys():
        if key in handled_keys:
            continue
        value = metrics[key]
        if isinstance(value, (float, np.floating)):
            final_output[key] = f"{value:.3f}"
        elif isinstance(value, (int, np.integer)):
            final_output[key] = f"{value}"
        else:
            final_output[key] = f"{value}"

    for key in sorted(final_output.keys()):
        print(f"  {key}: {final_output[key]}")


def main():
    # Keep CPU usage predictable so batched+sb3 back-to-back don't OOM.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    # Match the config from runner_new.py exactly
    DEFAULT_CONFIG = _shared_default_config()
    DEFAULT_CONFIG['eval_freq'] = DEFAULT_CONFIG['n_steps']*DEFAULT_CONFIG['n_eval_envs']

    parser = argparse.ArgumentParser(description='Simplified SB3 Comparison Runner')
    parser.add_argument("--n_envs", type=int, default=DEFAULT_CONFIG['n_envs'], help="Number of parallel environments")
    parser.add_argument("--n_steps", type=int, default=DEFAULT_CONFIG['n_steps'], help="Number of rollout steps")
    parser.add_argument("--timesteps_train", type=int, default=DEFAULT_CONFIG['timesteps_train'], help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG['seed'], help="Random seed")
    parser.add_argument("--smoke", action="store_true", help="Use a very small rollout for quick parity checks")
    parser.add_argument("--verbose_env", type=int, default=DEFAULT_CONFIG['verbose_env'], help="Environment verbosity level")
    # Alias to accept torchrl-style flag when sharing CLI with runner_new
    parser.add_argument("--batch_size_env", type=int, default=None, help="Alias for n_envs to ease parity scripts")
    parser.add_argument("--canonical_action_order", type=int, default=int(DEFAULT_CONFIG['canonical_action_order']), help="Deterministically order derived states")
    parser.add_argument("--n_queries", type=int, default=None, help="Limit train/valid/test queries to this number")
    parser.add_argument("--trace_dir", type=str, default=None, help="Optional directory to dump rollout traces for parity debugging")
    
    args = parser.parse_args()

    # Override with command line args
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.batch_size_env is not None:
        args.n_envs = args.batch_size_env
    config['n_envs'] = args.n_envs
    config['n_eval_envs'] = args.n_envs
    config['n_steps'] = args.n_steps
    config['timesteps_train'] = args.timesteps_train
    config['seed'] = args.seed
    config['seed_run_i'] = args.seed
    config['verbose_env'] = args.verbose_env
    config['verbose_prover'] = DEFAULT_CONFIG['verbose_prover']
    config['canonical_action_order'] = bool(args.canonical_action_order)
    if args.n_queries is not None:
        config['n_train_queries'] = args.n_queries
        config['n_eval_queries'] = args.n_queries
        config['n_test_queries'] = args.n_queries
    if args.trace_dir:
        config['trace_dir'] = args.trace_dir
    if args.smoke:
        print("Using smoke test settings")
        config['n_steps'] = min(16, config['n_steps'])
        config['timesteps_train'] = config['n_steps'] * config['n_envs']
        n_smoke = args.n_queries if args.n_queries is not None else 1
        config['n_train_queries'] = n_smoke
        config['n_eval_queries'] = n_smoke
        config['n_test_queries'] = n_smoke
        config['atom_embedding_size'] = 32
        config['constant_embedding_size'] = 32
        config['predicate_embedding_size'] = 32
        config['max_total_vars'] = 5000
        config['allow_small_eval'] = True
        config['train_neg_ratio'] = 0.0
        config['corruption_mode'] = False

    if 'countries' in config['dataset_name']:
        config['corruption_scheme'] = ['tail']

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    # Eager seeds to align with torchrl runner
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Avoid nonsensical settings where timesteps < rollout_size
    rollout_size = config['n_steps'] * config['n_envs']
    # if config['timesteps_train'] > 0 and rollout_size > config['timesteps_train']:
    #     new_n_steps = max(1, config['timesteps_train'] // config['n_envs'])
    #     if new_n_steps < 1:
    #         new_n_steps = 1
    #     print(f"Adjusting n_steps from {config['n_steps']} to {new_n_steps} "
    #           f"to respect timesteps_train={config['timesteps_train']} with n_envs={config['n_envs']}")
    #     config['n_steps'] = new_n_steps
    #     rollout_size = config['n_steps'] * config['n_envs']
    # Keep eval_freq aligned with rollout size
    config['eval_freq'] = rollout_size
    print(f"Using device: {device}")
    print(f"\nConfig:")
    print(f"  n_envs: {config['n_envs']}")
    print(f"  n_steps: {config['n_steps']}")
    print(f"  timesteps_train: {config['timesteps_train']}")
    print(f"  seed: {config['seed']}")
    print(f"  verbose_env: {config['verbose_env']}")
    print(f"  smoke: {args.smoke}")
    if args.n_queries is not None:
        print(f"  n_queries override: {args.n_queries}")
    if args.trace_dir:
        print(f"  trace_dir: {args.trace_dir}")

    # Build argparse.Namespace from config
    from argparse import Namespace
    cfg = Namespace(**config)

    # Create run signature for logging
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    cfg.run_signature = f"comparison-sb3-{cfg.dataset_name}-n_envs{cfg.n_envs}-{timestamp}"
    
    # Logging setup
    use_logger = config['use_logger']
    log_filename = None
    if use_logger:
        log_dir = config['logger_path']
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(
            log_dir,
            f"_tmp_log-{cfg.run_signature}-seed_{cfg.seed}.csv"
        )
        print(f"\nLog file: {log_filename}")

    # Run main training
    print("\n" + "="*80)
    print("STARTING SB3 COMPARISON RUN")
    print("="*80 + "\n")
    
    metrics_train, metrics_valid, metrics_test = train_main(
        cfg,
        log_filename,
        use_logger,
        config['use_wb'],
        config['wb_path'],
        timestamp
    )

    print("\n" + "="*80)
    print("SB3 COMPARISON RUN COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

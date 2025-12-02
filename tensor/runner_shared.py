"""
Simplified runner for comparing batched vs sb3 implementation.

This runner is designed to match sb3's behavior as closely as possible
to verify that rollout collection, validation, and evaluation produce
identical or very similar results.
"""

import argparse
import copy
import datetime
import os
import random
import numpy as np
import torch
import gc

from utils.utils import FileLogger
from train import main as train_main


def _shared_default_config():
    """
    Build the config we want both SB3 and batched paths to use.

    The defaults mirror sb3_runner_new.py and force CPU/deterministic settings
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
        'load_depth_info': True,
        'n_train_queries': None,
        'n_eval_queries': None,
        'n_test_queries': None,
        'filter_queries_by_rules': True,

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
        'timesteps_train': 2000,
        'restore_best_val_model': False,
        'load_model': False,
        'save_model': False,
        'use_amp': False,  # Disable for debugging
        'use_compile': False,  # Disable for debugging
        'n_steps': 128,
        'batch_size_env': 128,  # Start with 1 to match sb3 n_envs=1
        'batch_size_env_eval': 128,
        'batch_size': 4096,
        'allow_small_eval': False,

        # Env params
        'reward_type': 4,
        'train_neg_ratio': 1,
        'end_proof_action': True,
        'skip_unary_actions': True,
        'max_depth': 20,
        'memory_pruning': True,
        'eval_pruning': True,
        'corruption_mode': 'dynamic',
        'corruption_scheme': ['head', 'tail'],
        'use_exact_memory': True,
        'kge_action': False,

        # Embedding params
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        'atom_embedding_size': 256,
        'constant_embedding_size': 256,
        'predicate_embedding_size': 256,
        'learn_embeddings': True,
        'padding_atoms': 6,
        'padding_states': 20,
        # Large values explode memory in the index manager; keep this reasonable.
        'max_total_vars': 50000,
        'dataset_name_embedding': None,
        'constant_emb_file': None,
        'predicate_emb_file': None,
        'rule_depend_var': False,

        # Other params
        'device': 'cuda',
        'rollout_device': 'cpu',
        'min_gpu_memory_gb': 2.0,
        'extended_eval_info': True,
        'eval_best_metric': 'mrr',
        'plot_trajectories': False,
        'plot': False,
        'depth_info': True,
        'verbose_cb': False,
        'verbose_env': 0,  # Enable for debugging
        'verbose_prover': 0,
        'data_path': './data/',
        'models_path': 'models/',
        'rules_file': 'rules.txt',
        'facts_file': 'train.txt',
        'use_logger': True,
        'logger_path': './runs/',
        'use_wb': False,
        'wb_path': './../wandb/',
        'debug_ppo': False,
        
        # Additional sb3-compatible fields
        'janus_file': None,
        'train_file': 'train.txt',
        'valid_file': 'valid.txt',
        'test_file': 'test.txt',
        'prob_facts': False,
        'topk_facts': None,
        'topk_facts_threshold': 0.33,
    }

def main():
    # Keep CPU memory/thread usage low to avoid OOM when running both paths back-to-back.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    # Use simple, fixed config to start
    DEFAULT_CONFIG = _shared_default_config()
    DEFAULT_CONFIG['eval_freq'] = DEFAULT_CONFIG['n_steps']*DEFAULT_CONFIG['batch_size_env']

    parser = argparse.ArgumentParser(description='Simplified TorchRL vs SB3 Comparison Runner')
    parser.add_argument("--batch_size_env", type=int, default=DEFAULT_CONFIG['batch_size_env'], help="Batch size for environment")
    parser.add_argument("--n_steps", type=int, default=DEFAULT_CONFIG['n_steps'], help="Number of rollout steps")
    parser.add_argument("--timesteps_train", type=int, default=DEFAULT_CONFIG['timesteps_train'], help="Total training timesteps")
    parser.add_argument("--verbose_env", type=int, default=DEFAULT_CONFIG['verbose_env'], help="Environment verbosity level")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG['seed'], help="Random seed")
    parser.add_argument("--smoke", action="store_true", help="Use a very small rollout for quick parity checks")
    parser.add_argument("--n_queries", type=int, default=None, help="Limit train/valid/test queries to this number")
    parser.add_argument("--n_train_queries", type=int, default=None, help="Limit training queries to this number")
    parser.add_argument("--n_eval_queries", type=int, default=None, help="Limit eval queries to this number")
    parser.add_argument("--n_test_queries", type=int, default=None, help="Limit test queries to this number")
    parser.add_argument("--trace_dir", type=str, default=None, help="Optional directory to dump rollout traces for parity debugging")
    # Alias to accept sb3-style flag when using the shared comparison runner
    parser.add_argument("--n_envs", type=int, default=None, help="Alias for batch_size_env to ease parity scripts")
    parser.add_argument("--train_neg_ratio", type=int, default=DEFAULT_CONFIG['train_neg_ratio'], help="Training negative ratio")
    parser.add_argument("--eval_neg_samples", type=int, default=DEFAULT_CONFIG['eval_neg_samples'], help="Evaluation negative samples")
    parser.add_argument("--allow_small_eval", action="store_true", help="Permit eval with <=1 query (parity/debugging)")
    
    args = parser.parse_args()

    # Override with command line args
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.n_envs is not None:
        args.batch_size_env = args.n_envs
    config['batch_size_env'] = args.batch_size_env
    config['batch_size_env_eval'] = args.batch_size_env
    config['n_steps'] = args.n_steps
    config['timesteps_train'] = args.timesteps_train
    config['verbose_env'] = args.verbose_env
    config['seed'] = args.seed
    config['seed_run_i'] = args.seed
    if args.n_queries is not None:
        config['n_train_queries'] = args.n_queries
        config['n_eval_queries'] = args.n_queries
        config['n_test_queries'] = args.n_queries
    if args.n_train_queries is not None:
        config['n_train_queries'] = args.n_train_queries
    if args.n_eval_queries is not None:
        config['n_eval_queries'] = args.n_eval_queries
    if args.n_test_queries is not None:
        config['n_test_queries'] = args.n_test_queries
    if args.trace_dir:
        config['trace_dir'] = args.trace_dir
    config['train_neg_ratio'] = args.train_neg_ratio
    if args.eval_neg_samples is not None:
        config['eval_neg_samples'] = args.eval_neg_samples
    if 'countries' in config['dataset_name']:
        config['corruption_scheme'] = ['tail']
    if args.smoke:
        # Minimal settings for quick step-by-step comparison
        config['n_steps'] = min(16, config['n_steps'])
        config['timesteps_train'] = config['n_steps'] * config['batch_size_env']
        n_smoke = args.n_queries if args.n_queries is not None else 1
        config['n_train_queries'] = n_smoke
        config['n_eval_queries'] = n_smoke
        config['n_test_queries'] = n_smoke
        config['atom_embedding_size'] = 32
        config['constant_embedding_size'] = 32
        config['predicate_embedding_size'] = 32
        config['max_total_vars'] = 5000
        config['train_neg_ratio'] = 0.0
        config['corruption_mode'] = False
        config['allow_small_eval'] = True

    if args.allow_small_eval or (config.get('n_eval_queries') is not None and config['n_eval_queries'] <= 1):
        config['allow_small_eval'] = True
    

    # Eager seeds to align with sb3 runner before train seeds
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    print(f"Using device: {config['device']}")
    print(f"\nConfig:")
    print(f"  batch_size_env: {config['batch_size_env']}")
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
    cfg.run_signature = f"comparison-{cfg.dataset_name}-batch{cfg.batch_size_env}-{timestamp}"
    
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
    print("STARTING COMPARISON RUN")
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
    print("COMPARISON RUN COMPLETE")
    print("="*80)

    
    return True

if __name__ == "__main__":
    main()

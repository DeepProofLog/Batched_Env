"""
Simplified sb3 runner for comparison with batched implementation.

This uses the same config as runner_new.py but runs the sb3 version.
"""

import argparse
import copy
import datetime
import os
import sys
import numpy as np
import torch

# Add both paths to ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sb3_utils import FileLogger
from sb3_train import main

if __name__ == "__main__":
    # Match the config from runner_new.py exactly
    DEFAULT_CONFIG = {
        # Dataset params
        'dataset_name': 'countries_s3',
        'eval_neg_samples': 3,
        'test_neg_samples': None,
        'train_depth': None,
        'valid_depth': None,
        'test_depth': None,
        'n_train_queries': None,
        'n_eval_queries': 500,
        'n_test_queries': 500,
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
        'timesteps_train': 128,  # Small for quick testing
        'restore_best_val_model': False,
        'load_model': False,
        'save_model': False,
        'n_envs': 1,  # sb3 uses n_envs instead of batch_size_env
        'n_steps': 128,
        'n_eval_envs': 1,
        'batch_size': 128,

        # Env params
        'reward_type': 4,
        'train_neg_ratio': 1,
        'engine': 'python',
        'engine_strategy': 'cmp',
        'endf_action': True,  # sb3 uses endf_action instead of end_proof_action
        'endt_action': False,
        'skip_unary_actions': True,
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
        'max_total_vars': 1000000,

        # Other params
        'device': 'cuda',  # Changed from cuda:1 to cuda
        'min_gpu_memory_gb': 2.0,
        'extended_eval_info': True,
        'eval_best_metric': 'mrr',
        'plot_trajectories': False,
        'plot': False,
        'depth_info': True,
        'verbose_cb': False,
        'data_path': '../data/',
        'models_path': 'models/',
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
        'eval_freq': 1000,
        'annealing_specs': {},
    }

    parser = argparse.ArgumentParser(description='Simplified SB3 Comparison Runner')
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--n_steps", type=int, default=128, help="Number of rollout steps")
    parser.add_argument("--timesteps_train", type=int, default=128, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()

    # Override with command line args
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['n_envs'] = args.n_envs
    config['n_eval_envs'] = args.n_envs
    config['n_steps'] = args.n_steps
    config['timesteps_train'] = args.timesteps_train
    config['seed'] = args.seed
    config['seed_run_i'] = args.seed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"\nConfig:")
    print(f"  n_envs: {config['n_envs']}")
    print(f"  n_steps: {config['n_steps']}")
    print(f"  timesteps_train: {config['timesteps_train']}")
    print(f"  seed: {config['seed']}")

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
    
    metrics_train, metrics_valid, metrics_test = main(
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
    print(f"\nTest metrics:")
    for key, value in metrics_test.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

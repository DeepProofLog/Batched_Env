"""
TorchRL-based experiment runner for Neural-guided Grounding.

This module provides the command-line interface and experiment management,
migrated from the original Stable-Baselines3 version to use TorchRL.
"""

import os
import numpy as np
import torch

torch.set_float32_matmul_precision('high')

import argparse
import copy
import datetime
from itertools import product
from typing import List, Optional

from train import main
from utils import (
    FileLogger,
    parse_scalar,
    coerce_config_value,
    update_config_value,
    parse_assignment,
)

if __name__ == "__main__":

    DEFAULT_CONFIG = {
        # Dataset params
        'dataset_name': 'countries_s3',

        'eval_neg_samples': 4,
        'test_neg_samples': None,

        'train_depth': {1,2,3,4,5,6},
        'valid_depth': None,
        'test_depth': None,

        'load_depth_info': True,

        'n_train_queries': None,
        'n_eval_queries': None,
        'n_test_queries': None,

        # Model params
        'use_l2_norm': True,
        'sqrt_scale': False,
        'temperature': 0.1,
        'model_name': 'PPO',
        'clip_range': 0.2,
        'n_epochs': 5,  
        'gamma': 0.99,
        'gae_lambda': 0.95,  
        'clip_range_vf': None,# 0.5,  
        'vf_coef': 2.0,  
        'max_grad_norm': 0.5,  
        'target_kl': 0.07,  
        'hidden_dim': 256,
        'num_layers': 8,
        'dropout_prob': 0.1,

        # Training params
        'seed': [0],
        'timesteps_train': 700000,
        'restore_best_val_model': True,
        'load_best_metric': 'eval', # 'eval' (best MRR) or 'train' (best reward)
        'load_model': False,
        'save_model': True,
        'use_amp': True,  
        'use_compile': True,  
        'batch_size_env': 128,  
        'batch_size_env_eval': 128,
        'n_steps': 128,  
        'batch_size': 512,  
        'eval_freq': 1,  

        # Env params
        'reward_type': 4,  
        'train_neg_ratio': 1,
        'end_proof_action': True,
        'skip_unary_actions': False,
        'max_depth': 20,
        'memory_pruning': True,
        'corruption_mode': 'dynamic',  
        'use_exact_memory': False,


        # Entropy coefficient decay params
        'ent_coef': 0.1,  # Overwritten if using decay

        'ent_coef_decay': True,  # Enable entropy coefficient decay
        'ent_coef_init_value': 0.01,  # Start with moderate exploration
        'ent_coef_final_value': 0.01,  # End with low exploration
        'ent_coef_start': 0.0,  # Fraction of training when decay starts
        'ent_coef_end': 1.0,  # Fraction of training when decay ends
        'ent_coef_transform': 'linear',  # Decay schedule: 'linear', 'exp', 'cos', 'log'
        
        # Learning rate decay params
        'lr': 3e-5, # Overwritten if using decay

        'lr_decay': True,  # Enable learning rate decay
        'lr_init_value': 3e-5,  # Reduced from 5e-5 for more stable updates
        'lr_final_value': 1e-6,  # Final value
        'lr_start': 0.0,  # Fraction of training when decay starts
        'lr_end': 1.0,  # Fraction of training when decay ends
        'lr_transform': 'linear',  # Decay schedule: 'linear', 'exp', 'cos', 'log'
        

        # Embedding params
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        'atom_embedding_size': 250,  
        'learn_embeddings': True,
        'padding_atoms': 6,
        'padding_states': -1,  # auto-computed from dataset
        'max_total_vars': 100, # max vars in the embedder

        # Other params
        'device': 'cuda',
        'rollout_device': None,  # Device for rollout collection, None means same as device
        'min_gpu_memory_gb': 2.0,
        'extended_eval_info': True,
        'eval_best_metric': 'mrr',
        'plot_trajectories': False,
        'plot': False,
        'depth_info': True,  # Enable depth info by default to see metrics by depth
        'verbose': False,
        'verbose_cb': False,  # Verbose callback debugging
        'verbose_env': 0,  # Environment verbosity level (0=quiet, 1=verbose)
        'verbose_prover': 0,  # Prover verbosity level (0=quiet, 1=verbose)
        'prover_verbose': False,
        'data_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
        'models_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
        'rules_file': 'rules.txt',
        'facts_file': 'train.txt',
        'use_logger': True,
        'logger_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
        'wb_path': './../wandb/',
        'debug_ppo': False,

        
        # Determinism settings
        'deterministic': False,  # Enable strict reproducibility (slower, set False for production)
        'sample_deterministic_per_env': False,  # Sample deterministic per environment (slower, set False for production)
        'canonical_action_order': False,


    }

    KNOWN_CONFIG_KEYS = set(DEFAULT_CONFIG.keys())

    parser = argparse.ArgumentParser(description='TorchRL Experiment Runner')

    parser.add_argument("--set", action='append', default=[], metavar="KEY=VALUE",
        help="Override config value, e.g. --set reward_type=3 --set seed='[0,1]'.")
    parser.add_argument("--grid", action='append', default=[], metavar="KEY=V1,V2",
        help="Grid search values, e.g. --grid reward_type=2,3.")
    parser.add_argument("--eval", action='store_true',
        help="Shortcut: load model and skip training (timesteps=0).")
    parser.add_argument("--profile", action='store_true',
        help="Enable cProfile profiling for the training loop.")

    args = parser.parse_args()

    base_config = copy.deepcopy(DEFAULT_CONFIG)

    # Apply command-line overrides
    for assignment in args.set:
        key, raw_value = parse_assignment(assignment)
        parsed_value = parse_scalar(raw_value)
        update_config_value(base_config, key, parsed_value, DEFAULT_CONFIG)

    if args.eval:
        base_config['load_model'] = True
        base_config['timesteps_train'] = 0

    base_config['profile'] = args.profile

    # Prepare grid search
    grid_spec = {}
    if args.grid:
        for entry in args.grid:
            key, raw_values = parse_assignment(entry)
            value_candidates = [v.strip() for v in raw_values.split(',') if v.strip()]
            if not value_candidates:
                raise ValueError(f"No values supplied for grid entry '{entry}'.")
            parsed_values = [
                coerce_config_value(key, parse_scalar(candidate), DEFAULT_CONFIG)
                for candidate in value_candidates
            ]
            grid_spec[key] = parsed_values

    # Generate list of experiment configurations
    run_configs = []
    if grid_spec:
        grid_keys = sorted(grid_spec.keys())
        for combo in product(*(grid_spec[key] for key in grid_keys)):
            config_copy = copy.deepcopy(base_config)
            for key, value in zip(grid_keys, combo):
                update_config_value(config_copy, key, value, DEFAULT_CONFIG, prevalidated=True)
            run_configs.append(config_copy)
        print(
            f"Prepared grid search over {len(grid_spec)} parameter(s), "
            f"yielding {len(run_configs)} experiment(s)."
        )
    else:
        run_configs = [base_config]


    def build_namespace(config):
        """Build argparse.Namespace from config dict."""
        cfg = copy.deepcopy(config)

        namespace = argparse.Namespace(**cfg)

        # Auto-configure padding_states based on dataset
        if namespace.padding_states == -1:
            padding_states = {
                "countries_s3": 20, "countries_s2": 20, "countries_s1": 20,
                "family": 130, "wn18rr": 262, "fb15k237": 358,
            }
            namespace.padding_states = padding_states[namespace.dataset_name]

        # Corruption scheme - unconditional override based on dataset
        namespace.corruption_scheme = ['head', 'tail']
        if 'countries' in namespace.dataset_name or 'ablation' in namespace.dataset_name:
            namespace.corruption_scheme = ['tail']
        namespace.corruption_scheme = list(namespace.corruption_scheme)

        # File names
        namespace.train_file = "train_depths.txt" if namespace.train_depth else "train.txt"
        namespace.valid_file = "valid_depths.txt" if namespace.valid_depth else "valid.txt"
        namespace.test_file = "test_depths.txt" if namespace.test_depth else "test.txt"

        # Embedding sizes
        namespace.state_embedding_size = (
            namespace.atom_embedding_size
            if namespace.state_embedder != "concat"
            else namespace.atom_embedding_size * namespace.padding_atoms
        )

        embedding_multiplier = 1
        if namespace.atom_embedder in ["complex", "rotate"]:
            embedding_multiplier = 2
        
        namespace.constant_embedding_size = namespace.atom_embedding_size * embedding_multiplier
        namespace.predicate_embedding_size = namespace.atom_embedding_size * (
            2 if namespace.atom_embedder == "complex" else 1
        )

        return namespace



    all_args = []
    for config in run_configs:
        args_namespace = build_namespace(config)
        
        if not args_namespace.save_model and args_namespace.restore_best_val_model:
            print(
                "\nWARNING: restore_best_val_model requested but save_model is False. "
                "Disabling best-model restoration for this run.\n"
            )
            args_namespace.restore_best_val_model = False

        if args_namespace.restore_best_val_model and args_namespace.load_model == 'last_epoch':
            print(
                "\nWARNING: restore_best_val_model is True while load_model='last_epoch'. "
                "You may not reproduce evaluation results.\n"
            )

        # Build run signature
        run_vars = (
            args_namespace.dataset_name,
            args_namespace.atom_embedding_size,
            args_namespace.end_proof_action,
            args_namespace.ent_coef,
            args_namespace.clip_range,
            args_namespace.train_neg_ratio,
            args_namespace.reward_type,
            args_namespace.n_epochs,
            args_namespace.lr,
            args_namespace.batch_size_env,
            'torchrl',  # Mark as TorchRL version
        )
        args_namespace.run_signature = '-'.join(str(v) for v in run_vars)
        all_args.append(args_namespace)


    def main_wrapper(args):
        """Wrapper to run main training function."""
        logger = FileLogger(base_folder=args.logger_path) if args.use_logger else None

        for seed in args.seed:
            date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            args.seed_run_i = seed
            print(f"\n{'='*60}\nSeed {seed} in {args.seed}\n{'='*60}")
            print("\nRun vars:", args.run_signature, '\n', {k: args.__dict__[k] for k in sorted(args.__dict__.keys())}, '\n')

            log_filename_tmp = logger.get_tmp_log_filename(args.run_signature, date, seed) if logger else None

            use_wb = getattr(args, 'use_wb', False)
            train_metrics, valid_metrics, test_metrics = main(
                args, log_filename_tmp, args.use_logger, use_wb, args.wb_path, date, profile_run=args.profile,
            )

            if logger:
                logger.log_run(args, train_metrics, valid_metrics, test_metrics, log_filename_tmp, date, seed)

        if logger:
            logger.log_avg_results(args.__dict__, args.run_signature, args.seed)

    # Run all experiments
    total_experiments = len(all_args)
    for idx, experiment_args in enumerate(all_args, start=1):
        print(f"\n{'='*60}")
        print(f"Experiment {idx}/{total_experiments}")
        print(f"{'='*60}\n")
        main_wrapper(experiment_args)

    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"{'='*60}\n")

"""
TorchRL-based experiment runner for Neural-guided Grounding.

This module provides the command-line interface and experiment management,
migrated from the original Stable-Baselines3 version to use TorchRL.
"""

import os
import numpy as np
import torch
from utils.seeding import seed_all

torch.set_float32_matmul_precision('high')

import argparse
import copy
import datetime
from itertools import product
from typing import List, Optional

from utils.utils import FileLogger
from train_new import main
from utils.utils_config import (
    load_experiment_configs,
    parse_scalar,
    coerce_config_value,
    update_config_value,
    parse_assignment,
)

if __name__ == "__main__":

    DEFAULT_CONFIG = {
        # General experiment configuration
        # NOTE: These defaults are aligned with sb3_runner.py for parity testing

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


        # Model params
        'model_name': 'PPO',
        'ent_coef': 0.2,
        'clip_range': 0.2,
        'n_epochs': 5, 
        'lr': 5e-5,
        'gamma': 0.99,
        'target_kl': 0.03,  # KL divergence limit for early stopping (aligned with SB3)

        # Training params
        'seed': [0],
        'timesteps_train': 700000,
        'restore_best_val_model': True,
        'load_model': False,
        'save_model': True,
        'use_amp': True,
        'use_compile': True,
        'n_steps': 128,
        'eval_freq': 1,  # In multiples of n_steps (matches SB3)
        'batch_size_env': 128,
        'batch_size_env_eval': 128,
        'batch_size': 4096,  # Aligned with SB3 (was 1024)

        # Env params
        'reward_type': 4,  # Aligned with SB3 (was 4)
        'train_neg_ratio': 4,
        'end_proof_action': True,
        'skip_unary_actions': True,
        'max_depth': 20,
        'memory_pruning': True,
        'corruption_mode': 'dynamic',  # Aligned with SB3 (was True)
        'corruption_scheme': ['head', 'tail'],
        'canonical_action_order': False,
        'use_exact_memory': False,

        # Embedding params
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        'atom_embedding_size': 250,  # Aligned with SB3 (was 256)
        'learn_embeddings': True,
        'padding_atoms': 6,
        'padding_states': -1,  # Aligned with SB3 (was 20), auto-computed from dataset
        'max_total_vars': 100,  # Aligned with SB3 (was 1000000)

        # Other params
        'device': 'auto',  # Aligned with SB3 (was 'cuda:1')
        'rollout_device': None,  # Device for rollout collection, None means same as device
        'min_gpu_memory_gb': 2.0,
        'extended_eval_info': True,
        'eval_best_metric': 'mrr',
        'plot_trajectories': False,
        'plot': False,
        'depth_info': True,  # Enable depth info by default to see metrics by depth
        'verbose': False,  # Added for SB3 parity
        'verbose_cb': False,  # Verbose callback debugging
        'verbose_env': 0,  # Environment verbosity level (0=quiet, 1=verbose)
        'verbose_prover': 0,  # Prover verbosity level (0=quiet, 1=verbose)
        'prover_verbose': False,  # Added for SB3 parity
        'data_path': './data/',
        'models_path': 'models/',
        'rules_file': 'rules.txt',
        'facts_file': 'train.txt',
        'use_logger': True,
        'logger_path': './runs/',
        'use_wb': False,
        'wb_path': './../wandb/',
        'debug_ppo': False,
        
        # Determinism settings
        'deterministic': False,  # Enable strict reproducibility (slower, set False for production)
        'sample_deterministic_per_env': False,  # Sample deterministic per environment (slower, set False for production)
    }

    KNOWN_CONFIG_KEYS = set(DEFAULT_CONFIG.keys())

    parser = argparse.ArgumentParser(description='TorchRL Experiment Runner')
    parser.add_argument("--config", type=str,
        help="Path to YAML file describing experiments.")
    parser.add_argument("--set", action='append', default=[], metavar="KEY=VALUE",
        help="Override config value, e.g. --set reward_type=3 --set seed='[0,1]'.")
    parser.add_argument("--grid", action='append', default=[], metavar="KEY=V1,V2",
        help="Grid search values, e.g. --grid reward_type=2,3.")
    parser.add_argument("--eval", action='store_true',
        help="Shortcut: load model and skip training (timesteps=0).")

    args = parser.parse_args()

    if args.config and args.grid:
        raise ValueError("Use either --config or --grid, not both.")

    base_config = copy.deepcopy(DEFAULT_CONFIG)

    # Apply command-line overrides
    for assignment in args.set:
        key, raw_value = parse_assignment(assignment)
        parsed_value = parse_scalar(raw_value)
        update_config_value(base_config, key, parsed_value, DEFAULT_CONFIG)

    if args.eval:
        base_config['load_model'] = True
        base_config['timesteps_train'] = 0

    # Determine device
    requested_device = base_config.get('device', 'auto')
    if requested_device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device

    print(f"Using device: {device}\n")

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

    # Load experiments from config file
    experiments_from_file = []
    if args.config:
        overrides_list = load_experiment_configs(args.config)
        for idx, overrides in enumerate(overrides_list):
            unknown_keys = set(overrides) - KNOWN_CONFIG_KEYS
            if unknown_keys:
                unknown_list = ", ".join(sorted(unknown_keys))
                raise ValueError(
                    f"Unknown parameter(s) in experiment {idx}: {unknown_list}"
                )
            experiments_from_file.append(overrides)
        print(f"\n\nLoaded {len(experiments_from_file)} experiment(s) from {args.config}")

    # Generate list of experiment configurations
    run_configs = []
    if experiments_from_file:
        for overrides in experiments_from_file:
            config_copy = copy.deepcopy(base_config)
            for key, value in overrides.items():
                update_config_value(config_copy, key, value, DEFAULT_CONFIG)
            run_configs.append(config_copy)
    elif grid_spec:
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

        # Best metric for model selection
        best_metric = cfg.get('eval_best_metric', 'mrr')
        if not isinstance(best_metric, str):
            raise ValueError("eval_best_metric must be a string.")
        metric_normalized = best_metric.strip().lower()
        allowed_best_metrics = {'auc_pr', 'mrr'}
        if metric_normalized not in allowed_best_metrics:
            allowed = ", ".join(sorted(allowed_best_metrics))
            raise ValueError(
                f"Unsupported eval_best_metric '{best_metric}'. Allowed: {allowed}."
            )
        # Map to actual metric names returned by evaluation
        metric_name_map = {
            'mrr': 'mrr',  # Match SB3 naming
            'auc_pr': 'auc_pr',
        }
        cfg['eval_best_metric'] = metric_name_map.get(metric_normalized, metric_normalized)

        namespace = argparse.Namespace(**cfg)

        # Keep corruption_mode as-is (matches SB3 behavior)
        # Can be 'dynamic', 'static', or boolean values

        # Auto-configure padding_states based on dataset
        if namespace.padding_states == -1:
            if namespace.dataset_name in {"countries_s3", "countries_s2", "countries_s1"}:
                if namespace.test_neg_samples is not None:
                    print("Overriding test_neg_samples.")
                namespace.test_neg_samples = None
                namespace.padding_states = 20
                # Note: atom_embedding_size is kept as 250 (from DEFAULT_CONFIG) to match SB3
            elif namespace.dataset_name == "family":
                namespace.padding_states = 130
            elif namespace.dataset_name == "wn18rr":
                namespace.padding_states = 262
            elif namespace.dataset_name == "fb15k237":
                namespace.padding_states = 358
            else:
                raise ValueError("Unknown dataset name for automatic padding configuration.")



        # Corruption scheme - matches SB3's behavior (unconditional override based on dataset)
        namespace.corruption_scheme = ['head', 'tail']
        if 'countries' in namespace.dataset_name or 'ablation' in namespace.dataset_name:
            namespace.corruption_scheme = ['tail']
        namespace.corruption_scheme = list(namespace.corruption_scheme)

        # File names - match SB3 behavior for parity
        train_file = "train.txt"
        valid_file = "valid.txt"
        test_file = "test.txt"

        # Match SB3: only use depth files when depth filter is explicitly set
        if namespace.train_depth is not None:
            train_file = train_file.replace('.txt', '_depths.txt')
        if namespace.valid_depth is not None:
            valid_file = valid_file.replace('.txt', '_depths.txt')
        if namespace.test_depth is not None:
            test_file = test_file.replace('.txt', '_depths.txt')

        namespace.train_file = train_file
        namespace.valid_file = valid_file
        namespace.test_file = test_file

        # Embedding sizes
        namespace.state_embedding_size = (
            namespace.atom_embedding_size
            if namespace.state_embedder != "concat"
            else namespace.atom_embedding_size * namespace.padding_atoms
        )
        namespace.constant_embedding_size = namespace.atom_embedding_size
        namespace.predicate_embedding_size = namespace.atom_embedding_size
        
        if namespace.atom_embedder == "complex":
            namespace.constant_embedding_size = 2 * namespace.atom_embedding_size
            namespace.predicate_embedding_size = 2 * namespace.atom_embedding_size
        if namespace.atom_embedder == "rotate":
            namespace.constant_embedding_size = 2 * namespace.atom_embedding_size

        namespace.device = device
        namespace.eval_freq = int(namespace.n_steps * namespace.eval_freq)

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
            print(f"\n{'='*60}")
            print(f"Seed {seed} in {args.seed}")
            print(f"{'='*60}")
            dict_ordered = {k: args.__dict__[k] for k in sorted(args.__dict__.keys())}
            print("\nRun vars:", args.run_signature, '\n', dict_ordered, '\n')

            if args.use_logger:
                log_filename_tmp = os.path.join(
                    args.logger_path,
                    f"_tmp_log-{args.run_signature}-{date}-seed_{seed}.csv",
                )
            else:
                log_filename_tmp = None

            train_metrics, valid_metrics, test_metrics = main(
                args,
                log_filename_tmp,
                args.use_logger,
                args.use_wb,
                args.wb_path,
                date,
            )

            if args.use_logger and logger is not None:
                logged_data = copy.deepcopy(args)
                dicts_to_log = {
                    'train': train_metrics,
                    'valid': valid_metrics,
                    'test': test_metrics,
                }
                logger.log(log_filename_tmp, logged_data.__dict__, dicts_to_log)

                # Extract scalar values from metrics (handle both float and list cases)
                rewards_pos_mean_val = test_metrics.get('rewards_pos_mean', 0)
                if isinstance(rewards_pos_mean_val, (list, np.ndarray)):
                    rewards_pos_mean_val = np.mean(rewards_pos_mean_val)
                rewards_pos_mean = np.round(float(rewards_pos_mean_val), 3)
                
                mrr_val = test_metrics.get('mrr_mean', 0)
                if isinstance(mrr_val, (list, np.ndarray)):
                    mrr_val = np.mean(mrr_val)
                mrr = np.round(float(mrr_val), 3)
                
                metrics = f"{rewards_pos_mean:.3f}_{mrr:.3f}"
                log_filename_run_name = os.path.join(
                    args.logger_path,
                    'indiv_runs',
                    f"_ind_log-{args.run_signature}-{date}-{metrics}-seed_{seed}.csv",
                )
                logger.finalize_log_file(log_filename_tmp, log_filename_run_name)

        if args.use_logger and logger is not None:
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

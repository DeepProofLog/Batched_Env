"""
TorchRL-based experiment runner for Neural-guided Grounding.

This module provides the command-line interface and experiment management,
migrated from the original Stable-Baselines3 version to use TorchRL.
"""

import argparse
import copy
import datetime
import os
from itertools import product
from typing import List, Optional

import numpy as np
import torch

from utils import FileLogger
from train import main
from utils_config import (
    load_experiment_configs,
    parse_scalar,
    coerce_config_value,
    update_config_value,
    parse_assignment,
    get_available_gpus,
    select_best_gpu,
)

# Use new API for TF32 settings to avoid deprecation warnings
# Valid precision values: 'ieee', 'tf32', None (for default)
_cuda_matmul_backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
if _cuda_matmul_backend is not None and hasattr(_cuda_matmul_backend, "fp32_precision"):
    _cuda_matmul_backend.fp32_precision = "tf32"

_cudnn_conv_backend = getattr(getattr(torch.backends, "cudnn", None), "conv", None)
if _cudnn_conv_backend is not None and hasattr(_cudnn_conv_backend, "fp32_precision"):
    _cudnn_conv_backend.fp32_precision = "tf32"


if __name__ == "__main__":

    DEFAULT_CONFIG = {
        # General experiment configuration

        # Dataset params
        'dataset_name': 'countries_s3',

        'eval_neg_samples': 4,
        'test_neg_samples': 100,

        'train_depth': None,
        'valid_depth': None,
        'test_depth': None,

        'n_train_queries': None,
        'n_eval_queries': 500,
        'n_test_queries': 500,


        # Model params
        'model_name': 'PPO',
        'ent_coef': 0.5,
        'clip_range': 0.2,
        'n_epochs': 10,
        'lr': 3e-4,
        'gamma': 0.99,

        # Training params
        'seed': [0],
        'timesteps_train': 200000,
        'restore_best_val_model': True,
        'load_model': False,
        'save_model': True,
        'n_envs': 4,  # Now used as batch_size for BatchedVecEnv
        'n_steps': 20, #8192, 16384
        'n_eval_envs': 4,  # Now used as batch_size for eval BatchedVecEnv
        'batch_size': 128,

        # Env params
        'reward_type': 1,
        'train_neg_ratio': 1,
        'engine': 'python',
        'endf_action': False,
        'skip_unary_actions': False,
        'max_depth': 20,
        'memory_pruning': False,
        'corruption_mode': True,
        'min_multiaction_ratio': 0.05,

        # Embedding params
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        'atom_embedding_size': 100,
        'learn_embeddings': True,
        'padding_atoms': 4,
        'padding_states': 20,  # if -1, auto-computed from dataset
        'max_total_vars': 2000000,

        # Other params
        'device': 'cuda:1',  # 'cpu', 'cuda:1' (auto-select), or 'cuda:all'
        'min_gpu_memory_gb': 2.0,
        'extended_eval_info': True,
        'eval_best_metric': 'mrr',
        'plot_trajectories': False,
        'plot': False,
        'depth_info': False,
        'verbose_cb': False,  # Verbose callback debugging
        'verbose_env': 1,  # Environment verbosity level (0=quiet, 1=verbose)
        'verbose_prover': 0,  # Prover verbosity level (0=quiet, 1=verbose)
        'data_path': './data/',
        'models_path': 'models/',
        'rules_file': 'rules.txt',
        'facts_file': 'train.txt',
        'use_logger': True,
        'logger_path': './runs/',
        'use_wb': False,
        'wb_path': './../wandb/',
        'debug_ppo': False,
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

    # Handle device selection
    device_choice = base_config.get('device', 'cuda:1')
    min_memory_gb = base_config.get('min_gpu_memory_gb', 2.0)
    
    if device_choice == "cpu":
        print("\n=== Using CPU ===")
        print("Training will run on CPU (slower but always available)\n")
        device = "cpu"
    
    elif device_choice == "cuda:1":
        print("\n=== Auto-selecting best GPU ===")
        best_gpu = select_best_gpu(min_free_gb=min_memory_gb)
        if best_gpu is not None:
            device = f"cuda:{best_gpu}"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            print(f"Using device: cuda:{best_gpu}\n")
        else:
            device = "cpu"
            print(f"No GPU with at least {min_memory_gb} GB free memory found.")
            print("Falling back to CPU\n")
    
    elif device_choice == "cuda:all":
        print("\n=== Using all available GPUs ===")
        available_gpus = get_available_gpus(min_free_gb=min_memory_gb)
        
        if len(available_gpus) == 0:
            device = "cpu"
            print(f"No GPUs with at least {min_memory_gb} GB free memory found.")
            print("Falling back to CPU\n")
        elif len(available_gpus) == 1:
            device = f"cuda:{available_gpus[0]}"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
            print(f"Only 1 GPU available: GPU {available_gpus[0]}")
            print(f"Using device: cuda:{available_gpus[0]}\n")
        else:
            print(f"Found {len(available_gpus)} GPUs: {available_gpus}")
            print("Note: Multi-GPU training requires additional setup. Using first available GPU.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
            device = f"cuda:0"
            print(f"Using device: cuda:0 (mapped from GPU {available_gpus[0]})\n")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
            'mrr': 'mrr_mean',
            'auc_pr': 'auc_pr',
        }
        cfg['eval_best_metric'] = metric_name_map.get(metric_normalized, metric_normalized)

        namespace = argparse.Namespace(**cfg)

        raw_corruption_mode = getattr(namespace, "corruption_mode", True)
        namespace.corruption_mode = bool(raw_corruption_mode)

        # Auto-configure padding_states based on dataset
        if namespace.padding_states == -1:
            if namespace.dataset_name in {"countries_s3", "countries_s2", "countries_s1"}:
                if namespace.test_neg_samples is not None:
                    print("Overriding test_neg_samples.")
                namespace.test_neg_samples = None
                namespace.padding_states = 20
                namespace.atom_embedding_size = 256
            elif namespace.dataset_name == "family":
                namespace.padding_states = 130
            elif namespace.dataset_name == "wn18rr":
                namespace.padding_states = 262
            elif namespace.dataset_name == "fb15k237":
                namespace.padding_states = 358
            else:
                raise ValueError("Unknown dataset name for automatic padding configuration.")



        # Corruption scheme
        namespace.corruption_scheme = ['head', 'tail']
        if 'countries' in namespace.dataset_name or 'ablation' in namespace.dataset_name:
            namespace.corruption_scheme = ['tail']
        namespace.corruption_scheme = list(namespace.corruption_scheme)

        # File names
        train_file = "train.txt"
        valid_file = "valid.txt"
        test_file = "test.txt"

        # Always use depth files if they exist (depth filter is applied in DataHandler)
        # Check if depth files exist and use them
        import os
        from os.path import join
        dataset_path = join(namespace.data_path, namespace.dataset_name)
        
        if os.path.exists(join(dataset_path, train_file.replace('.txt', '_depths.txt'))):
            train_file = train_file.replace('.txt', '_depths.txt')
        if os.path.exists(join(dataset_path, valid_file.replace('.txt', '_depths.txt'))):
            valid_file = valid_file.replace('.txt', '_depths.txt')
        if os.path.exists(join(dataset_path, test_file.replace('.txt', '_depths.txt'))):
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
        namespace.eval_freq = namespace.n_steps * namespace.n_envs

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
            args_namespace.endf_action,
            args_namespace.ent_coef,
            args_namespace.clip_range,
            args_namespace.train_neg_ratio,
            args_namespace.reward_type,
            args_namespace.n_epochs,
            args_namespace.lr,
            args_namespace.n_envs,
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

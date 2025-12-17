"""
TorchRL-based experiment runner for Neural-guided Grounding.

This module provides the command-line interface and experiment management.
Uses run_experiment from train.py with TrainConfig from config.py.
"""

import os
import copy
import datetime
import argparse
from itertools import product
from typing import List, Optional

import numpy as np
import torch
torch.set_float32_matmul_precision('high')

from train import run_experiment
from config import TrainConfig
from utils import FileLogger, parse_scalar, coerce_config_value, update_config_value, parse_assignment


if __name__ == "__main__":

    DEFAULT_CONFIG = {
        # Dataset
        'dataset': 'countries_s3',
        'data_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
        
        # Training
        'seed': [0],
        'total_timesteps': 700000,
        'n_envs': 128,
        'n_steps': 128,
        'batch_size': 512,
        'n_epochs': 5,
        
        # PPO hyperparams  
        'learning_rate': 3e-5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': None,
        'ent_coef': 0.1,
        'vf_coef': 2.0,
        'max_grad_norm': 0.5,
        'target_kl': 0.07,
        
        # Model architecture
        'atom_embedding_size': 250,
        'hidden_dim': 256,
        'num_layers': 8,
        'dropout_prob': 0.1,
        'use_l2_norm': True,
        'sqrt_scale': False,
        'temperature': 0.1,
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        
        # Environment
        'padding_atoms': 6,
        'padding_states': -1,  # Auto from dataset
        'max_steps': 20,
        'reward_type': 4,
        'negative_ratio': 1,
        'end_proof_action': True,
        'skip_unary_actions': False,
        'memory_pruning': True,
        'use_exact_memory': False,
        'max_total_vars': 100,
        
        # Depths
        'train_depth': {1,2,3,4,5,6},
        'valid_depth': None,
        'test_depth': None,
        
        # Evaluation
        'eval_freq': 1,
        'eval_neg_samples': 4,
        'n_corruptions': None,  # test_neg_samples alias
        'eval_best_metric': 'mrr',
        
        # LR decay
        'lr_decay': True,
        'lr_init_value': 3e-5,
        'lr_final_value': 1e-6,
        'lr_start': 0.0,
        'lr_end': 1.0,
        'lr_transform': 'linear',
        
        # Entropy decay  
        'ent_coef_decay': True,
        'ent_coef_init_value': 0.01,
        'ent_coef_final_value': 0.01,
        'ent_coef_start': 0.0,
        'ent_coef_end': 1.0,
        'ent_coef_transform': 'linear',
        
        # Model saving/loading
        'save_model': True,
        'load_model': False,
        'restore_best': True,
        'load_best_metric': 'eval',
        'models_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
        
        # Logging
        'use_logger': True,
        'logger_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
        'verbose': False,
        
        # Device
        'device': 'cuda',
        
        # Misc
        'parity': False,
        'profile': False,
        'use_callbacks': True,
        'sample_deterministic_per_env': False,
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
        help="Enable profiling.")

    args = parser.parse_args()
    base_config = copy.deepcopy(DEFAULT_CONFIG)

    # Apply command-line overrides
    for assignment in args.set:
        key, raw_value = parse_assignment(assignment)
        parsed_value = parse_scalar(raw_value)
        update_config_value(base_config, key, parsed_value, DEFAULT_CONFIG)

    if args.eval:
        base_config['load_model'] = True
        base_config['total_timesteps'] = 0

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

    # Generate experiment configs
    run_configs = []
    if grid_spec:
        grid_keys = sorted(grid_spec.keys())
        for combo in product(*(grid_spec[key] for key in grid_keys)):
            config_copy = copy.deepcopy(base_config)
            for key, value in zip(grid_keys, combo):
                update_config_value(config_copy, key, value, DEFAULT_CONFIG, prevalidated=True)
            run_configs.append(config_copy)
        print(f"Grid search: {len(grid_spec)} params, {len(run_configs)} experiments.")
    else:
        run_configs = [base_config]


    def config_from_dict(cfg_dict: dict) -> TrainConfig:
        """Convert config dict to TrainConfig dataclass."""
        # Auto-configure padding_states
        if cfg_dict.get('padding_states', -1) == -1:
            dataset = cfg_dict.get('dataset', 'countries_s3')
            padding_map = {
                "countries_s3": 20, "countries_s2": 20, "countries_s1": 20,
                "family": 130, "wn18rr": 262, "fb15k237": 358,
            }
            cfg_dict['padding_states'] = padding_map.get(dataset, 64)
        
        # Build run signature
        dataset = cfg_dict.get('dataset', 'run')
        atom_size = cfg_dict.get('atom_embedding_size', 64)
        n_envs = cfg_dict.get('n_envs', 3)
        cfg_dict['run_signature'] = f"{dataset}-{atom_size}-{n_envs}-torchrl"
        
        # Corruption scheme
        if 'countries' in dataset or 'ablation' in dataset:
            cfg_dict['corruption_scheme'] = ['tail']
        else:
            cfg_dict['corruption_scheme'] = ['head', 'tail']
        
        # File names based on depth
        cfg_dict['train_file'] = "train_depths.txt" if cfg_dict.get('train_depth') else "train.txt"
        cfg_dict['valid_file'] = "valid_depths.txt" if cfg_dict.get('valid_depth') else "valid.txt"
        cfg_dict['test_file'] = "test_depths.txt" if cfg_dict.get('test_depth') else "test.txt"
        cfg_dict['rules_file'] = 'rules.txt'
        cfg_dict['facts_file'] = 'train.txt'
        
        # Create TrainConfig with matching fields
        config_fields = {f.name for f in TrainConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg_dict.items() if k in config_fields}
        
        return TrainConfig(**filtered)


    def run_wrapper(cfg_dict: dict):
        """Wrapper to run experiment with logging."""
        logger = FileLogger(base_folder=cfg_dict.get('logger_path', './runs')) if cfg_dict.get('use_logger') else None
        seeds = cfg_dict.get('seed', [42])
        if not isinstance(seeds, (list, tuple)):
            seeds = [seeds]
        
        for seed in seeds:
            date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            cfg_dict['seed'] = seed
            cfg_dict['seed_run_i'] = seed
            
            print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
            
            # Convert to TrainConfig
            config = config_from_dict(cfg_dict)
            print(f"Run signature: {config.run_signature}")
            
            # Run experiment
            results = run_experiment(config)
            
            # Log results
            if logger:
                # Create metrics dicts for logger compatibility
                test_metrics = {
                    'mrr_mean': results.get('MRR', 0),
                    'hits1_mean': results.get('Hits@1', 0),
                    'hits3_mean': results.get('Hits@3', 0),
                    'hits10_mean': results.get('Hits@10', 0),
                }
                log_filename = logger.get_tmp_log_filename(config.run_signature, date, seed)
                logger.log_run(cfg_dict, {}, {}, test_metrics, log_filename, date, seed)
        
        if logger:
            logger.log_avg_results(cfg_dict, config.run_signature, seeds)


    # Run all experiments
    total = len(run_configs)
    for idx, cfg in enumerate(run_configs, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {idx}/{total}")
        print(f"{'='*60}\n")
        run_wrapper(cfg)

    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"{'='*60}\n")

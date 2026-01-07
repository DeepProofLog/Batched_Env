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
import warnings

if torch.cuda.is_available():
    # Recommended TF32 settings for newer Pytorch versions (matches 'high' precision)
    # This satisfies both core Pytorch and the Inductor compiler
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*use the new API settings to control TF32 behavior.*")
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            # Fallback for older Pytorch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
from train import run_experiment
from config import TrainConfig
from kge_inference import normalize_backend, default_checkpoint_dir
from utils import FileLogger, parse_scalar, coerce_config_value, update_config_value, parse_assignment


if __name__ == "__main__":

    DEFAULT_CONFIG = {
        # Dataset
        'dataset': ['wn18rr'],
        'data_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
        
        # Training
        'seed': [0],
        'total_timesteps': 0,
        'n_envs': 128,
        'n_steps': 256,
        'batch_size': 512,
        'n_epochs': 5,
        'augment_train': True, # for countries dataset
        
        # PPO hyperparams
        'learning_rate': 1e-4,  # Increased from 3e-5 for faster value function learning
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': None,
        'ent_coef': 0.1,
        'vf_coef': 0.5,  # Reduced from 2.0 to SB3 default for stable training
        'max_grad_norm': 0.5,
        'target_kl': None,  # Disabled (was 0.07) - allows all epochs to complete for faster value learning
        
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
        'padding_states': 120,  # Auto from dataset
        'max_steps': 20,
        'reward_type': 4,
        'negative_ratio': 1,
        'end_proof_action': True,
        'skip_unary_actions': False,  # AAAI26 parity: auto-advance when only 1 action
        'memory_pruning': True,
        'use_exact_memory': False,
        'max_total_vars': 100,
        'max_fact_pairs_cap': None,  # Auto-set per dataset in config_from_dict
        
        # Depths
        'train_depth': {1,2,3,4,5,6},
        'valid_depth': None,
        'test_depth': None,
        
        # Evaluation
        'eval_freq': 4,
        'n_eval_queries': 100,
        'n_test_queries': None,
        'eval_neg_samples': 10,
        'test_neg_samples': 100,
        'eval_best_metric': 'mrr',
        'ranking_tie_seed': 0,

        # KGE inference (eval-time fusion)
        'kge_inference': True,
        'kge_inference_success': True,
        'kge_engine': 'pytorch',
        'kge_checkpoint_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kge_pytorch', 'models'),

        'kge_scores_file': None,
        'kge_eval_kge_weight': 2.0,  # Hybrid weight for KGE scores
        'kge_eval_rl_weight': 1.0,   # Binary bonus for proven queries
        'kge_fail_penalty': 0.5,     # Penalty for failed proofs
        'kge_only_eval': False,      # False enables hybrid KGE+RL scoring
        # Hybrid: success = kge_weight * kge_logp + rl_weight, fail = kge_weight * kge_logp - penalty

        # KGE Integration: Probabilistic Facts
        'prob_facts': False,
        'prob_facts_topk': None,
        'prob_facts_threshold': None,

        # KGE Integration: PBRS (Potential-Based Reward Shaping)
        'pbrs_beta': 0.0,
        'pbrs_gamma': 0.99,
        'pbrs_precompute': True,

        # KGE Integration: Neural Bridge
        'neural_bridge': False,
        'neural_bridge_init_alpha': 0.5,
        'neural_bridge_train_epochs': 100,
        'neural_bridge_lr': 0.01,

        # KGE Integration: Unification Scoring
        'unification_scoring': False,
        'unification_scoring_mode': 'offline',
        'unification_top_k': None,

        # LR decay
        'lr_decay': True,
        'lr_init_value': 1e-4,  # Match learning_rate
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
        'load_model': True,
        'restore_best': True,
        'load_best_metric': 'eval',
        'models_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
        
        # Logging
        'use_logger': True,
        'logger_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
        'verbose': True,
        
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

    # Automatically add list-valued parameters to grid search (excluding seed)
    # This allows setting e.g. 'dataset': ['wn18rr', 'family'] in DEFAULT_CONFIG
    for key, value in base_config.items():
        if isinstance(value, list) and key != 'seed' and key not in grid_spec:
            grid_spec[key] = value

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

        # Auto-configure max_fact_pairs_cap for large datasets
        # wn18rr has 35k facts for hypernym predicate, cap to 8000 for 7x speedup
        if cfg_dict.get('max_fact_pairs_cap') is None:
            dataset = cfg_dict.get('dataset', 'countries_s3')
            cap_map = {
                "wn18rr": 1000,     # hypernym has 35k facts, cap for 7x speedup
                "fb15k237": 1000,   # similar issue expected
                # family, countries: no cap needed (max ~2.5k facts per predicate)
            }
            cfg_dict['max_fact_pairs_cap'] = cap_map.get(dataset, None)
        
        # Build run signature
        dataset = cfg_dict.get('dataset', 'run')
        atom_size = cfg_dict.get('atom_embedding_size', 64)
        n_envs = cfg_dict.get('n_envs', 3)
        cfg_dict['run_signature'] = f"{dataset}-{atom_size}-{n_envs}-torchrl"

        # KGE inference defaults
        if cfg_dict.get('kge_inference', False):
            if cfg_dict.get('kge_run_signature') is None and cfg_dict.get('dataset') == 'wn18rr':
                cfg_dict['kge_run_signature'] = 'torch_wn18rr_RotatE_1024_20260107_125531_s42'
            elif cfg_dict.get('kge_run_signature') is None and cfg_dict.get('dataset') == 'family':
                cfg_dict['kge_run_signature'] = 'torch_family_RotatE_1024_20260107_124531_s42'
            else:
                raise ValueError("kge_run_signature must be specified for dataset: {}".format(cfg_dict.get('dataset')))
            if cfg_dict.get('kge_inference'):
                engine = normalize_backend(cfg_dict.get('kge_engine', 'pytorch'))
                cfg_dict['kge_engine'] = engine
                if not cfg_dict.get('kge_checkpoint_dir'):
                    cfg_dict['kge_checkpoint_dir'] = default_checkpoint_dir(engine)


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

        if 'countries' in dataset and cfg_dict.get('augment_train'):
            cfg_dict['train_file'] = "combined_train_oversampled.txt"
        
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
                test_metrics = {}
                for k, v in results.items():
                    # Pass through all scalar metrics (int/float)
                    if isinstance(v, (int, float)):
                        test_metrics[k] = v
                    # Try to parse string numbers if any
                    elif isinstance(v, str):
                        try:
                            test_metrics[k] = float(v)
                        except ValueError:
                            pass
                log_filename = logger.get_tmp_log_filename(config.run_signature, date, seed)
                logger.log_run(config, {}, {}, test_metrics, log_filename, date, seed)

        
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

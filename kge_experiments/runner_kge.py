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
from kge_module import normalize_backend, default_checkpoint_dir
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
        'filter_queries_by_rules': True,
        
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
        'kge_checkpoint_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kge_module', 'kge_trainer', 'models'),

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
        'neural_bridge_type': 'linear',  # 'linear', 'gated', or 'mlp'
        'neural_bridge_init_alpha': 0.5,
        'neural_bridge_init_alpha_success': 0.7,  # For gated bridge
        'neural_bridge_init_alpha_fail': 0.2,  # For gated bridge
        'neural_bridge_train_epochs': 100,
        'neural_bridge_lr': 0.01,
        'neural_bridge_hidden_dim': 32,  # For MLP bridge

        # KGE Integration: Predicate-Aware Scoring
        'predicate_aware_scoring': False,
        'predicate_aware_symmetric_weight': 0.7,  # RL weight for symmetric predicates
        'predicate_aware_chain_weight': 0.0,  # RL weight for chain-only predicates (0 = pure KGE)

        # KGE Integration: KGE-Filtered Candidates
        'kge_filter_candidates': False,
        'kge_filter_top_k': 100,

        # KGE Integration: KGE-Initialized Embeddings
        'kge_init_embeddings': False,

        # KGE Integration: Ensemble KGE Models
        'kge_ensemble': False,
        'kge_ensemble_signatures': None,
        'kge_ensemble_method': 'mean',

        # KGE Integration: Joint KGE-RL Training
        'kge_joint_training': False,
        'kge_joint_lambda': 0.1,
        'kge_joint_margin': 1.0,

        # KGE Integration: Unification Scoring
        'unification_scoring': False,
        'unification_scoring_mode': 'offline',
        'unification_top_k': None,

        # KGE Integration: Rule Attention
        'kge_rule_attention': False,
        'kge_rule_attention_weight': 0.5,
        'kge_rule_attention_temperature': 1.0,

        # KGE Benchmarking
        'kge_benchmark': False,

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
        # Only load RL model if NOT in KGE-only evaluation mode
        if not base_config.get('kge_only_eval', False):
            base_config['load_model'] = True
        else:
            base_config['load_model'] = False
            print("Evaluate with KGE-only mode: RL model loading disabled.")
            
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
                "nations": 64, "umls": 64, "pharmkg_full": 358,
            }
            cfg_dict['padding_states'] = padding_map.get(dataset, 64)

        # Auto-configure max_fact_pairs_cap for large datasets
        # wn18rr has 35k facts for hypernym predicate, cap to 8000 for 7x speedup
        if cfg_dict.get('max_fact_pairs_cap') is None:
            dataset = cfg_dict.get('dataset', 'countries_s3')
            cap_map = {
                "wn18rr": 1000,     # hypernym has 35k facts, cap for 7x speedup
                "fb15k237": 1000,   # similar issue expected
                "pharmkg_full": 1000,
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
            elif cfg_dict.get('kge_run_signature') is None and cfg_dict.get('dataset') == 'fb15k237':
                cfg_dict['kge_run_signature'] = 'torch_fb15k237_TuckER_512_20260111_002222_s42'
            elif cfg_dict.get('kge_run_signature') is None and cfg_dict.get('dataset') == 'pharmkg_full':
                cfg_dict['kge_run_signature'] = 'torch_pharmkg_full_ComplEx_1024_20260111_054518_s42'
            elif cfg_dict.get('kge_run_signature') is None and cfg_dict.get('dataset') == 'umls':
                cfg_dict['kge_run_signature'] = 'torch_umls_ComplEx_1024_20260110_223751_s42'
            elif cfg_dict.get('kge_run_signature') is None and cfg_dict.get('dataset') == 'nations':
                cfg_dict['kge_run_signature'] = 'torch_nations_TuckER_512_20260110_224506_s42'
            elif cfg_dict.get('kge_run_signature') is None:
                # Attempt to find the latest model for this dataset
                if not cfg_dict.get('kge_checkpoint_dir'):
                     # We need to resolve the default dir first if not set
                     engine = normalize_backend(cfg_dict.get('kge_engine', 'pytorch'))
                     cfg_dict['kge_checkpoint_dir'] = default_checkpoint_dir(engine)
                
                ckpt_dir = cfg_dict['kge_checkpoint_dir']
                dataset_name = cfg_dict.get('dataset')
                # Pattern: torch_{dataset}_*
                prefix = f"torch_{dataset_name}_"
                
                if os.path.isdir(ckpt_dir):
                    candidates = [d for d in os.listdir(ckpt_dir) 
                                  if d.startswith(prefix) and os.path.isdir(os.path.join(ckpt_dir, d))]
                    if candidates:
                        # Sort by name (which includes timestamp and hence is chronological)
                        # Format: torch_{dataset}_{model}_{dim}_{timestamp}_s{seed}
                        # Timestamps are YYYYMMDD_HHMMSS, so ASCII sort works for finding latest.
                        candidates.sort(reverse=True)
                        best_candidate = candidates[0]
                        print(f"Auto-discovered latest KGE model for {dataset_name}: {best_candidate}")
                        cfg_dict['kge_run_signature'] = best_candidate
                    else:
                        # Fallback to raising error if no model found
                         raise ValueError(f"kge_run_signature not specified and no auto-discovered model found for dataset: {dataset_name} in {ckpt_dir}")
                else:
                     raise ValueError(f"kge_run_signature not specified and checkpoint dir does not exist: {ckpt_dir}")

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
        
        # File names based on depth - only use if they exist
        data_path = cfg_dict.get('data_path', base_config['data_path'])
        dataset_dir = os.path.join(data_path, dataset)
        
        def sanitize_depth_config(base_name, depth_key):
            if cfg_dict.get(depth_key):
                 depth_file = f"{base_name}_depths.txt"
                 if not os.path.exists(os.path.join(dataset_dir, depth_file)):
                      cfg_dict[depth_key] = None

        sanitize_depth_config('train', 'train_depth')
        sanitize_depth_config('valid', 'valid_depth')
        sanitize_depth_config('test', 'test_depth')

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

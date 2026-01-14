"""
TorchRL-based experiment runner for Neural-guided Grounding.

This module provides the command-line interface and experiment management.
Uses run_experiment from train.py with TrainConfig from config.py.
"""

import os
import copy
import datetime
import argparse
from dataclasses import fields, MISSING
from itertools import product

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


def get_default_config() -> dict:
    """Get default config dict from TrainConfig dataclass."""
    config_dict = {}
    for f in fields(TrainConfig):
        if f.default is not MISSING:
            config_dict[f.name] = f.default
        elif f.default_factory is not MISSING:
            config_dict[f.name] = f.default_factory()
        # Skip fields with no default (required fields)

    # Runner-specific: wrap seed in list for grid search compatibility
    config_dict['seed'] = [config_dict.get('seed', 0)]

    return config_dict


if __name__ == "__main__":

    DEFAULT_CONFIG = get_default_config()
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
        dataset = cfg_dict.get('dataset', 'wn18rr')

        # Build run signature
        atom_size = cfg_dict.get('atom_embedding_size', 250)
        n_envs = cfg_dict.get('n_envs', 128)
        cfg_dict['run_signature'] = f"{dataset}-{atom_size}-{n_envs}-torchrl"

        # KGE inference: normalize engine and set checkpoint_dir
        if cfg_dict.get('kge_inference', False):
            engine = normalize_backend(cfg_dict.get('kge_engine', 'pytorch'))
            cfg_dict['kge_engine'] = engine
            if not cfg_dict.get('kge_checkpoint_dir'):
                cfg_dict['kge_checkpoint_dir'] = default_checkpoint_dir(engine)
        
        # File names based on depth - only use if they exist
        data_path = cfg_dict.get('data_path', DEFAULT_CONFIG['data_path'])
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

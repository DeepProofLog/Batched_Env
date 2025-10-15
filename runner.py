import torch
import argparse
import datetime
import copy
import os
import sys
from itertools import product

import numpy as np
from utils import FileLogger
from train import main
from utils_config import (
    load_experiment_configs,
    parse_scalar,
    coerce_config_value,
    update_config_value,
    parse_assignment,
)
torch.set_float32_matmul_precision('high')
# import gc
# gc.disable()  
# torch.cuda.set_allocator_config(garbage_collection_threshold=0.9)


if __name__ == "__main__":

    DEFAULT_CONFIG = {
        # General experiment configuration

        # Dataset params
        'dataset_name': 'family',
        'eval_neg_samples': 3,
        'test_neg_samples': 100,
        'train_depth': None, # {-1,3}
        'valid_depth': None,
        'test_depth': None,
        'n_train_queries': None,
        'n_eval_queries': 500,
        'n_test_queries': None,

        # Model params
        'model_name': 'PPO',
        'ent_coef': 0.2,
        'clip_range': 0.2,
        'n_epochs': 10,
        'lr': 3e-4,
        'gamma': 0.99,

        # Training params
        'seed': [0],
        'timesteps_train': 700000,
        'restore_best_val_model': True,
        'load_model': True,
        'save_model': True,
        'n_envs': 128,
        'n_steps': 128,
        'n_eval_envs': 100,
        'batch_size': 128,

        # Env params
        'reward_type': 3,
        'train_neg_ratio': 1,
        'engine_strategy': 'rft',
        'endf_action': True,
        'endt_action': False,
        'skip_unary_actions': True,
        'engine': 'python',
        'max_depth': 20,
        'memory_pruning': True,
        'corruption_mode': 'dynamic',
        'false_rules': False,

        # Top-K actions filtering: Curriculum learning params
        'top_k_curriculum': None,
        'top_k_initial': 10,
        'top_k_final': 7,
        'top_k_start_step': 200000,

        # KGE integration params
        'kge_integration_strategy': None, # None, 'train', 'train_bias', 'logit_shaping', 'sum_eval'
        'kge_checkpoint_dir': './../../checkpoints/',
        'kge_run_signature': None,
        'kge_scores_file': None,
        'use_kge_action': None,  # Auto-computed from kge_integration_strategy unless overridden

        # KGE logit shaping params
        'kge_logit_gain_init': 1.0,
        'kge_logit_gain_final': 0.2,
        'kge_logit_gain_anneal_steps': 300000,
        'kge_logit_gain_warmup_steps': 0,
        'kge_logit_transform': 'log',
        'kge_logit_eps': 1e-6,

        # Potential-based shaping params
        'pbrs_beta': 0.0, # Set it as the same value of PPO's gamma
        'pbrs_gamma': None,

        # Evaluation hybrid fusion
        'eval_hybrid_kge_weight': 2.0,
        'eval_hybrid_rl_weight': 1.0,
        'eval_hybrid_success_only': True,

        # Embedding params
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        'atom_embedding_size': 256,
        'learn_embeddings': True,
        'padding_atoms': 6,
        'padding_states': -1, # Auto-computed from dataset unless overridden
        'max_total_vars': 100,

        # Other params
        'extended_eval_info': True,
        'eval_best_metric': 'mrr',
        'plot': False,
        'data_path': './data/',
        'models_path': 'models/',
        'rules_file': 'rules.txt',
        'facts_file': 'train.txt',
        'use_logger': True,
        'logger_path': './runs/',
        'use_wb': False,
        'wb_path': './../wandb/',
    }

    KNOWN_CONFIG_KEYS = set(DEFAULT_CONFIG.keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description='Experiment Runner')
    parser.add_argument("--config",type=str,
        help="Path to YAML file describing a list of experiments (each as key/value overrides).",)
    parser.add_argument("--set",action='append',default=[],metavar="KEY=VALUE",
        help="Override a configuration value, e.g. --set reward_type=3 --set seed='[0,1]'.",)
    parser.add_argument("--grid",action='append',default=[],metavar="KEY=V1,V2",
        help="Grid search values for a parameter, e.g. --grid reward_type=2,3 --grid ent_coef=0.2,0.5.",)
    parser.add_argument("--eval",action='store_true',
        help="Shortcut: load model and skip training (timesteps=0).",)

    args = parser.parse_args()

    if args.config and args.grid:
        raise ValueError("Use either --config or --grid, not both at the same time.")

    base_config = copy.deepcopy(DEFAULT_CONFIG)

    # Apply command-line overrides
    for assignment in args.set:
        key, raw_value = parse_assignment(assignment)
        parsed_value = parse_scalar(raw_value)
        update_config_value(base_config, key, parsed_value, DEFAULT_CONFIG)

    if args.eval:
        base_config['load_model'] = True
        base_config['timesteps_train'] = 0

    # Prepare grid search specification (if any)
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

    # Load experiments from config file (if any)
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

    # Generate the list of experiment configurations to run
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
                update_config_value(
                    config_copy,
                    key,
                    value,
                    DEFAULT_CONFIG,
                    prevalidated=True,
                )
            run_configs.append(config_copy)
        print(
            f"Prepared grid search over {len(grid_spec)} parameter(s), "
            f"yielding {len(run_configs)} experiment(s)."
        )
    else:
        run_configs = [base_config]


    def build_namespace(config):
        """Given a config dict, build the argparse.Namespace by applying defaults and derived values."""
        cfg = copy.deepcopy(config)

        best_metric = cfg.get('eval_best_metric', 'auc_pr')
        if not isinstance(best_metric, str):
            raise ValueError("eval_best_metric must be provided as a string.")
        metric_normalized = best_metric.strip().lower()
        allowed_best_metrics = {'auc_pr', 'mrr'}
        if metric_normalized not in allowed_best_metrics:
            allowed = ", ".join(sorted(allowed_best_metrics))
            raise ValueError(
                f"Unsupported eval_best_metric '{best_metric}'. Allowed values: {allowed}."
            )
        cfg['eval_best_metric'] = metric_normalized

        if cfg.get('use_kge_action') is None:
            cfg['use_kge_action'] = cfg.get('kge_integration_strategy') in {"train", "train_bias"}

        curriculum = cfg.get('top_k_curriculum')
        if isinstance(curriculum, str):
            normalized = curriculum.strip()
            if not normalized or normalized.lower() == 'none':
                cfg['top_k_curriculum'] = None
            else:
                cfg['top_k_curriculum'] = normalized.lower()
        elif isinstance(curriculum, bool):
            cfg['top_k_curriculum'] = 'cte' if curriculum else None
        elif curriculum is not None:
            raise ValueError(
                "top_k_curriculum must be a string or None. "
                f"Received type {type(curriculum).__name__}."
            )

        allowed_curricula = {None, 'linear', 'exponential', 'step', 'cte'}
        if cfg['top_k_curriculum'] not in allowed_curricula:
            raise ValueError(
                f"Unsupported top_k_curriculum '{cfg['top_k_curriculum']}'. "
                "Allowed values: None, linear, exponential, step, cte."
            )

        if cfg.get('kge_run_signature') is None:
            dataset = cfg['dataset_name']
            if dataset in {"countries_s3", "countries_s2", "countries_s1"}:
                cfg['kge_run_signature'] = f"{dataset}-backward_0_1-no_reasoner-rotate-True-256-256-128-rules.txt"
            elif dataset == "family":
                cfg['kge_run_signature'] = "kinship_family-backward_0_1-no_reasoner-rotate-True-256-256-4-rules.txt"
            elif dataset == "wn18rr":
                cfg['kge_run_signature'] = "wn18rr-backward_0_1-no_reasoner-rotate-True-256-256-1-rules.txt"
            else:
                raise ValueError(f"No default KGE run signature defined for dataset '{dataset}'. "
                                 "Set 'kge_run_signature' manually or extend the defaults.")

        namespace = argparse.Namespace(**cfg)

        if namespace.padding_states == -1:
            if namespace.dataset_name in {"countries_s3", "countries_s2", "countries_s1"}:
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

        if namespace.dataset_name == "mnist_addition":
            namespace.corruption_mode = None

        namespace.corruption_scheme = ['head', 'tail']
        if 'countries' in namespace.dataset_name or 'ablation' in namespace.dataset_name:
            namespace.corruption_scheme = ['tail']

        if namespace.false_rules:
            if namespace.engine == 'prolog':
                namespace.janus_file = "countries_false_rules.pl"
            else:
                raise ValueError("False rules are not implemented for the python engine.")
        elif namespace.engine == 'python':
            namespace.janus_file = None
        else:
            namespace.janus_file = f"{namespace.dataset_name}.pl"
            print("Using prolog file:", namespace.janus_file)

        train_file = "train.txt"
        valid_file = "valid.txt"
        test_file = "test.txt"

        if namespace.corruption_mode == "static":
            train_file = "train_label_corruptions.json"
            valid_file = "valid_label_corruptions.json"
            test_file = "test_label_corruptions.json"

        if namespace.train_depth is not None:
            train_file = train_file.replace('.txt', '_depths.txt')
        if namespace.valid_depth is not None:
            valid_file = valid_file.replace('.txt', '_depths.txt')
        if namespace.test_depth is not None:
            test_file = test_file.replace('.txt', '_depths.txt')

        namespace.train_file = train_file
        namespace.valid_file = valid_file
        namespace.test_file = test_file

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
            raise ValueError(
                "restore_best_val_model=True but save_model=False. "
                "Enable model saving or disable best-model restoration."
            )

        if args_namespace.restore_best_val_model and args_namespace.load_model == 'last_epoch':
            print(
                "\nWARNING: restore_best_val_model is True while load_model='last_epoch'. "
                "You may not reproduce evaluation results.\n"
            )

        run_vars = (
            args_namespace.dataset_name,
            args_namespace.atom_embedder,
            args_namespace.state_embedder,
            args_namespace.atom_embedding_size,
            args_namespace.padding_atoms,
            args_namespace.padding_states,
            args_namespace.false_rules,
            args_namespace.endt_action,
            args_namespace.endf_action,
            args_namespace.skip_unary_actions,
            args_namespace.memory_pruning,
            args_namespace.max_depth,
            args_namespace.ent_coef,
            args_namespace.clip_range,
            args_namespace.engine,
            args_namespace.engine_strategy,
            args_namespace.train_neg_ratio,
            args_namespace.use_kge_action,
            args_namespace.reward_type,
            args_namespace.kge_integration_strategy,
            args_namespace.top_k_curriculum,
            args_namespace.top_k_initial,
            args_namespace.top_k_final,
            args_namespace.top_k_start_step,
            args_namespace.n_epochs,
            args_namespace.lr,
            args_namespace.n_envs,
        )
        args_namespace.run_signature = '-'.join(str(v) for v in run_vars)
        all_args.append(args_namespace)

        

    def main_wrapper(args):
        logger = FileLogger(base_folder=args.logger_path) if args.use_logger else None

        for seed in args.seed:
            date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            args.seed_run_i = seed
            print(f"Seed {seed} in {args.seed}")
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

                rewards_pos_mean = np.round(np.mean(test_metrics['rewards_pos_mean']), 3)
                mrr = np.round(np.mean(test_metrics['mrr_mean']), 3)
                metrics = f"{rewards_pos_mean:.3f}_{mrr:.3f}"
                log_filename_run_name = os.path.join(
                    args.logger_path,
                    'indiv_runs',
                    f"_ind_log-{args.run_signature}-{date}-{metrics}-seed_{seed}.csv",
                )
                logger.finalize_log_file(log_filename_tmp, log_filename_run_name)

        if args.use_logger and logger is not None:
            logger.log_avg_results(args.__dict__, args.run_signature, args.seed)

    total_experiments = len(all_args)
    for idx, experiment_args in enumerate(all_args, start=1):
        print(f"Experiment {idx}/{total_experiments}")
        main_wrapper(experiment_args)

# ==============================================================================
# CRITICAL: Early seeding for deterministic initialization
# Must happen BEFORE importing sb3_train (which triggers many nested imports)
# ==============================================================================
import os
import sys
import warnings

# Set environment variables for determinism before any CUDA operations
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '0')


# Add parent to path for utils import
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Early imports for seeding
import numpy as np
import torch
from utils.seeding import seed_all

# Default seed for module initialization - will be overridden by config
_INIT_SEED = 0
seed_all(_INIT_SEED, deterministic=True, warn=False)

# Set float32 matmul precision
torch.set_float32_matmul_precision('high')

# ==============================================================================
# Now import remaining modules (order matches runner.py)
# ==============================================================================
import argparse
import copy
import datetime
from itertools import product
from typing import Optional, List

try:
    # Try relative import first (when sb3/ is in sys.path)
    from sb3_utils import FileLogger
    from sb3_train import main
    from sb3_utils_config import (
        load_experiment_configs,
        parse_scalar,
        coerce_config_value,
        update_config_value,
        parse_assignment,
        get_available_gpus,
        select_best_gpu,
    )
except ImportError:
    # Fallback to package import (when imported as sb3.sb3_runner)
    from sb3.sb3_utils import FileLogger
    from sb3.sb3_train import main
    from sb3.sb3_utils_config import (
        load_experiment_configs,
        parse_scalar,
        coerce_config_value,
        update_config_value,
        parse_assignment,
        get_available_gpus,
        select_best_gpu,
    )

if __name__ == "__main__":

    DEFAULT_CONFIG = {
        # General experiment configuration
        # Countries: (lr, entropy) decay true, else 5e-5 and 0.2, train_neg_ratio 4, envsxsteps=16x128. 
        # family: (lr, entropy) decay false, 5e-5 and 0.05, train_neg_ratio 0 (nothing to learn from neg), envsxsteps=128x128. 
        # countries: 111(train queries)x5(pos+4 neg)x4(avg_len)=2,220 steps. set 16(envs)x128(steps)=2048. Recomended to cover 10%
        # family: 20k(train queries)x5(pos+4 neg)x3(avg_len)=300k-->cover 30k steps= 128x12
        # Dataset params
        'dataset_name': 'countries_s3',

        'eval_neg_samples': None,
        'test_neg_samples': None, # 5

        'train_depth': None, # {-1,3}
        'valid_depth': None,
        'test_depth': None,

        'n_train_queries': None,
        'n_eval_queries': None,
        'n_test_queries': None,

        'prob_facts': False,
        'topk_facts': None,
        'topk_facts_threshold': 0.33,

        # Model params
        'model_name': 'PPO',
        'ent_coef': 0.2,
        'clip_range': 0.2,
        'n_epochs': 20,
        'lr': 5e-5,
        'gamma': 0.99,
        'clip_range_vf': None,
        'target_kl': 0.03,

        # Training params
        'seed': [0],
        'timesteps_train': 2000,
        'restore_best_val_model': True,
        'load_model': False,
        'save_model': True,
        'n_envs': 20,
        'n_steps': 40,
        'n_eval_envs': 20,
        'batch_size': 4096,
        'eval_freq': 1, # in multiples of (n_steps * n_envs) -> how many rollouts between evaluations

        # Env params
        'reward_type': 0,
        'train_neg_ratio': 4,
        'engine': 'python',
        'engine_strategy': 'cmp', # 'cmp', 'rft'
        'endf_action': True,
        'endt_action': False,
        'skip_unary_actions': True,
        'max_depth': 20,
        'memory_pruning': True,
        'corruption_mode': 'dynamic',
        'false_rules': False,

        # Entropy coefficient decay params
        'ent_coef_decay': False,  # Enable entropy coefficient decay
        'ent_coef_init_value': 0.5,  # Initial value (defaults to ent_coef if None)
        'ent_coef_final_value': 0.01,  # Final value
        'ent_coef_start': 0.0,  # Fraction of training when decay starts
        'ent_coef_end': 1.0,  # Fraction of training when decay ends
        'ent_coef_transform': 'linear',  # Decay schedule: 'linear', 'exp', 'cos', 'log'

        # Learning rate decay params
        'lr_decay': False,  # Enable learning rate decay
        'lr_init_value': 3e-4,  # Initial value (defaults to lr if None)
        'lr_final_value': 1e-6,  # Final value
        'lr_start': 0.0,  # Fraction of training when decay starts
        'lr_end': 1.0,  # Fraction of training when decay ends
        'lr_transform': 'linear',  # Decay schedule: 'linear', 'exp', 'cos', 'log'

        # # KGE integration params
        # 'kge_action': False,        # Add an action which is a new predicate with '_kge', whose logits are from KGE
        # 'logit_fusion': False,      # Combine KGE logits with RL logits during training and evaluation
        # 'inference_fusion': False,  # Enable KGE inference fusion at evaluation time (True/False)
        # 'inference_success_only': False,  # When inference_fusion=True, only use KGE on RL success
        # 'pbrs': False,              # Enable potential-based reward shaping
        # 'enable_top_k': False,      # Enable Top-K actions filtering with the value function
        # 'kge_engine': 'tf',         # KGE backend: 'tf', 'pytorch', or 'pykeen'
        # 'kge_checkpoint_dir': './../../checkpoints/',
        # 'kge_run_signature': None,
        # 'kge_scores_file': None,

        # # Top-K actions filtering: Curriculum learning params
        # 'top_k_init_value': 10,
        # 'top_k_final_value': 7,
        # 'top_k_start': 0.3,
        # 'top_k_end': 1,
        # 'top_k_transform': 'linear',

        # # KGE logit shaping params for logit_fusion
        # 'kge_logit_init_value': 1.0,
        # 'kge_logit_final_value': 0.2,
        # 'kge_logit_start': 0.2, # at what fraction of training timesteps the annealing starts
        # 'kge_logit_end': 1, # at what fraction of training timesteps the annealing ends
        # 'kge_logit_transform': 'log',
        # 'kge_logit_eps': 1e-6,

        # # Potential-based shaping params
        # 'pbrs_beta': 0.5, 
        # 'pbrs_gamma': 0.99, # Set it as the same value of PPO's gamma

        # Evaluation hybrid fusion
        'eval_hybrid_success_only': True,
        'eval_hybrid_kge_weight': 2.0,
        'eval_hybrid_rl_weight': 1.0,

        # Embedding params
        'atom_embedder': 'transe',
        'state_embedder': 'mean',
        'atom_embedding_size': 250,
        'learn_embeddings': True,
        'padding_atoms': 6,
        'padding_states': -1, # Auto-computed from dataset unless overridden
        'max_total_vars': 100,

        # Other params
        'verbose': False,
        'prover_verbose': False,
        'device': 'auto',  # Device: 'cpu', 'cuda:1' (auto-select best GPU), or 'cuda:all' (use all available GPUs)
        'min_gpu_memory_gb': 2.0,  # Minimum free GPU memory in GB to consider a GPU available
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
        
        # Determinism settings
        'deterministic': True,  # Enable strict reproducibility (slower, set False for production)
        'eval_deterministic': True,  # Use argmax in evaluation (vs sampling)
    }

    KNOWN_CONFIG_KEYS = set(DEFAULT_CONFIG.keys())

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

    # Handle device selection based on user choice
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
            # Set CUDA_VISIBLE_DEVICES to use only the selected GPU
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
            print(f"Only 1 GPU available with sufficient memory: GPU {available_gpus[0]}")
            print(f"Using device: cuda:{available_gpus[0]}\n")
        else:
            # Multiple GPUs available
            print(f"Found {len(available_gpus)} GPUs with sufficient memory: {available_gpus}")
            print("Note: Multi-GPU training requires additional setup. Using first available GPU.")
            # Set CUDA_VISIBLE_DEVICES to only the available GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
            device = f"cuda:0"  # First GPU in the visible list
            print(f"Using device: cuda:0 (mapped from GPU {available_gpus[0]})\n")
    else:
        # Fallback to default behavior
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}\n")

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

        # Best metric for model selection
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

        cfg['kge_action'] = bool(cfg.get('kge_action', False))
        cfg['logit_fusion'] = bool(cfg.get('logit_fusion', False))
        cfg['inference_fusion'] = bool(cfg.get('inference_fusion', False))
        cfg['inference_success_only'] = bool(cfg.get('inference_success_only', True))
        cfg['pbrs'] = bool(cfg.get('pbrs', False))
        cfg['enable_top_k'] = bool(cfg.get('enable_top_k', False))
        cfg['prob_facts'] = bool(cfg.get('prob_facts', False))

        topk_facts = cfg.get('topk_facts')
        if topk_facts is None:
            cfg['topk_facts'] = None
        elif isinstance(topk_facts, str) and topk_facts.strip().lower() in {'none', ''}:
            cfg['topk_facts'] = None
        elif isinstance(topk_facts, bool):
            cfg['topk_facts'] = None if not topk_facts else int(topk_facts)
        else:
            cfg['topk_facts'] = int(topk_facts)

        topk_threshold = cfg.get('topk_facts_threshold')
        if topk_threshold is None:
            cfg['topk_facts_threshold'] = None
        elif isinstance(topk_threshold, str) and topk_threshold.strip().lower() in {'none', ''}:
            cfg['topk_facts_threshold'] = None
        else:
            cfg['topk_facts_threshold'] = float(topk_threshold)

        # Set eval_hybrid_success_only based on inference_success_only
        if cfg['inference_fusion']:
            cfg['eval_hybrid_success_only'] = cfg['inference_success_only']
        else:
            cfg['eval_hybrid_success_only'] = True  # Default when not using inference_fusion

        # Align potential-based shaping parameters with toggle
        if not cfg['pbrs']:
            cfg['pbrs_beta'] = 0.0
            cfg['pbrs_gamma'] = None
        else:
            if float(cfg.get('pbrs_beta', 0.0)) == 0.0:
                raise ValueError("pbrs is True but pbrs_beta is 0. Set a positive value for pbrs_beta.")
            if cfg.get('pbrs_gamma') is None:
                raise ValueError("pbrs is True but pbrs_gamma is None. Set a value for pbrs_gamma.")

        annealing_specs = {}

        # --- Top-K schedule ---
        if cfg['enable_top_k']:
            top_k_init = cfg.get('top_k_init_value')
            top_k_final = cfg.get('top_k_final_value', top_k_init)
            if top_k_init is not None:
                top_k_init = int(top_k_init)
                top_k_final = int(top_k_final)
                top_k_start = float(cfg.get('top_k_start', 0.0))
                top_k_end = float(cfg.get('top_k_end', 1.0))
                annealing_specs['top_k_value'] = {
                    'initial': top_k_init,
                    'final': top_k_final,
                    'start_point': max(0.0, min(1.0, top_k_start)),
                    'end_point': max(0.0, min(1.0, top_k_end)),
                    'transform': cfg.get('top_k_transform', 'linear'),
                    'value_type': 'int',
                }
                cfg['top_k_value'] = top_k_init
            else:
                cfg['top_k_value'] = 10  # Default value
        else:
            cfg['top_k_value'] = None

        # --- KGE logit gain schedule ---
        kge_logit_init = cfg.get('kge_logit_init_value')
        kge_logit_final = cfg.get('kge_logit_final_value', kge_logit_init)
        if kge_logit_init is not None:
            kge_logit_init = float(kge_logit_init)
            kge_logit_final = float(kge_logit_final)
            kge_logit_start = float(cfg.get('kge_logit_start', 0.0))
            kge_logit_end = float(cfg.get('kge_logit_end', 1.0))
            annealing_specs['kge_logit_gain'] = {
                'initial': kge_logit_init,
                'final': kge_logit_final,
                'start_point': max(0.0, min(1.0, kge_logit_start)),
                'end_point': max(0.0, min(1.0, kge_logit_end)),
                'transform': cfg.get('kge_logit_transform', 'linear'),
                'value_type': 'float',
            }

        # --- Entropy coefficient decay schedule ---
        if cfg.get('ent_coef_decay', False):
            ent_coef_init = cfg.get('ent_coef_init_value')
            if ent_coef_init is None:
                ent_coef_init = cfg.get('ent_coef', 0.2)
            ent_coef_final = cfg.get('ent_coef_final_value', 0.01)
            ent_coef_start = float(cfg.get('ent_coef_start', 0.0))
            ent_coef_end = float(cfg.get('ent_coef_end', 1.0))
            annealing_specs['ent_coef'] = {
                'initial': float(ent_coef_init),
                'final': float(ent_coef_final),
                'start_point': max(0.0, min(1.0, ent_coef_start)),
                'end_point': max(0.0, min(1.0, ent_coef_end)),
                'transform': cfg.get('ent_coef_transform', 'linear'),
                'value_type': 'float',
            }
            print(
                f"Entropy coefficient decay configured: {ent_coef_init} -> {ent_coef_final} "
                f"({cfg.get('ent_coef_transform', 'linear')})"
            )

        # --- Learning rate decay schedule ---
        if cfg.get('lr_decay', False):
            lr_init = cfg.get('lr_init_value')
            if lr_init is None:
                lr_init = cfg.get('lr', 5e-5)
            lr_final = cfg.get('lr_final_value', 1e-6)
            lr_start = float(cfg.get('lr_start', 0.0))
            lr_end = float(cfg.get('lr_end', 1.0))
            annealing_specs['lr'] = {
                'initial': float(lr_init),
                'final': float(lr_final),
                'start_point': max(0.0, min(1.0, lr_start)),
                'end_point': max(0.0, min(1.0, lr_end)),
                'transform': cfg.get('lr_transform', 'linear'),
                'value_type': 'float',
            }
            print(
                f"Learning rate decay configured: {lr_init} -> {lr_final} "
                f"({cfg.get('lr_transform', 'linear')})"
            )

        cfg['annealing_specs'] = annealing_specs
        cfg['annealing'] = annealing_specs

        # Validate KGE engine parameter
        kge_engine = cfg.get('kge_engine', 'tf').strip().lower()
        if kge_engine not in {'tf', 'tensorflow', 'pytorch', 'torch', 'pykeen'}:
            raise ValueError(f"Invalid kge_engine '{kge_engine}'. Must be one of: 'tf', 'pytorch', or 'pykeen'.")
        
        # Normalize engine name
        if kge_engine in {'tensorflow'}:
            kge_engine = 'tf'
        elif kge_engine in {'torch'}:
            kge_engine = 'pytorch'
        cfg['kge_engine'] = kge_engine

        # Default KGE run signatures per dataset and backend
        if cfg.get('kge_run_signature') is None:
            dataset = cfg['dataset_name']
            backend = cfg['kge_engine']
            
            # TensorFlow backend uses the original signatures
            if backend == 'tf':
                if dataset in {"countries_s3", "countries_s2", "countries_s1"}:
                    cfg['kge_run_signature'] = f"{dataset}-backward_0_1-no_reasoner-rotate-True-256-256-128-rules.txt"
                elif dataset == "family":
                    cfg['kge_run_signature'] = "kinship_family-backward_0_1-no_reasoner-rotate-True-256-256-4-rules.txt"
                elif dataset == "wn18rr":
                    cfg['kge_run_signature'] = "wn18rr-backward_0_1-no_reasoner-rotate-True-256-256-1-rules.txt"
                else:
                    raise ValueError(f"No default KGE run signature defined for dataset '{dataset}' with backend 'tf'. "
                                     "Set 'kge_run_signature' manually or extend the defaults.")
            
            # PyTorch backend - adapt signatures as needed
            elif backend == 'pytorch':
                if dataset in {"countries_s3", "countries_s2", "countries_s1"}:
                    cfg['kge_run_signature'] = f"{dataset}_pytorch_rotate_256"
                elif dataset == "family":
                    cfg['kge_run_signature'] = "family_pytorch_rotate_256"
                elif dataset == "wn18rr":
                    cfg['kge_run_signature'] = "wn18rr_pytorch_rotate_256"
                else:
                    raise ValueError(f"No default KGE run signature defined for dataset '{dataset}' with backend 'pytorch'. "
                                     "Set 'kge_run_signature' manually or extend the defaults.")
            
            # PyKEEN backend - adapt signatures as needed
            elif backend == 'pykeen':
                if dataset in {"countries_s3", "countries_s2", "countries_s1"}:
                    cfg['kge_run_signature'] = f"{dataset}_pykeen_rotate"
                elif dataset == "family":
                    cfg['kge_run_signature'] = "family_pykeen_rotate"
                elif dataset == "wn18rr":
                    cfg['kge_run_signature'] = "wn18rr_pykeen_rotate"
                else:
                    raise ValueError(f"No default KGE run signature defined for dataset '{dataset}' with backend 'pykeen'. "
                                     "Set 'kge_run_signature' manually or extend the defaults.")
            else:
                raise ValueError(f"Unsupported KGE backend: {backend}")


        namespace = argparse.Namespace(**cfg)

        if namespace.padding_states == -1:
            if namespace.dataset_name in {"countries_s3", "countries_s2", "countries_s1"}:
                if namespace.test_neg_samples != None: print("Overriding test_neg_samples.")
                namespace.test_neg_samples = None
                namespace.padding_states = 20
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
        namespace.eval_freq = int(namespace.n_steps * namespace.eval_freq)

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
            # args_namespace.atom_embedder,
            # args_namespace.state_embedder,
            args_namespace.atom_embedding_size,
            # args_namespace.padding_atoms,
            # args_namespace.padding_states,
            # args_namespace.false_rules,
            # args_namespace.endt_action,
            args_namespace.endf_action,
            # args_namespace.skip_unary_actions,
            # args_namespace.memory_pruning,
            # args_namespace.max_depth,
            args_namespace.ent_coef,
            args_namespace.clip_range,
            # args_namespace.gamma,
            # args_namespace.engine,
            args_namespace.engine_strategy,
            args_namespace.train_neg_ratio,
            args_namespace.reward_type,
            # args_namespace.kge_action,
            # args_namespace.logit_fusion,
            # args_namespace.inference_fusion,
            # args_namespace.pbrs,
            # args_namespace.enable_top_k,
            # args_namespace.prob_facts,
            # args_namespace.topk_facts,
            # args_namespace.topk_facts_threshold,
            # args_namespace.top_k_initial,
            # args_namespace.top_k_final,
            # args_namespace.top_k_start_step,
            args_namespace.n_epochs,
            args_namespace.lr,
            args_namespace.n_envs,
            # args_namespace.pbrs_beta,
            # args_namespace.pbrs_gamma,
            args_namespace.eval_hybrid_kge_weight,
            args_namespace.eval_hybrid_rl_weight,
            args_namespace.eval_hybrid_success_only,
            # args_namespace.kge_logit_transform,
            # args_namespace.kge_logit_gain_init,
            # args_namespace.kge_logit_gain_final,
            # args_namespace.kge_logit_gain_anneal_steps,
            # args_namespace.kge_logit_gain_warmup_steps, 
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

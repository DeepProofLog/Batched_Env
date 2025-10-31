"""
RL Runner - Inherits from BaseKGERunner for unified experiment management.
Orchestrates reinforcement learning experiments for neural-guided grounding.
"""
import torch
import argparse
import datetime
import copy
import os
import sys
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from kge_base_runner import BaseKGERunner, BaseKGEConfig
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

torch.set_float32_matmul_precision('high')


@dataclass
class RLConfig(BaseKGEConfig):
    """Configuration for RL experiments extending BaseKGEConfig."""
    
    # Override base defaults
    dataset: str = 'countries_s3'
    epochs: int = None  # Not used in RL, use timesteps_train instead
    batch_size: int = 128
    lr: float = 3e-4
    seed: List[int] = None  # Will be set in __post_init__
    
    # Dataset params
    eval_neg_samples: int = 3
    test_neg_samples: int = 100
    train_depth: Optional[int] = None
    valid_depth: Optional[int] = None
    test_depth: Optional[int] = None
    n_train_queries: Optional[int] = None
    n_eval_queries: int = 500
    n_test_queries: Optional[int] = None
    prob_facts: bool = False
    topk_facts: Optional[int] = None
    topk_facts_threshold: float = 0.33
    
    # Model params
    model_name: str = 'PPO'
    ent_coef: float = 0.5
    clip_range: float = 0.2
    n_epochs: int = 10
    gamma: float = 0.99
    
    # Training params
    timesteps_train: int = 2000000
    restore_best_val_model: bool = True
    load_model: bool = False
    save_model: bool = True
    n_envs: int = 128
    n_steps: int = 128
    n_eval_envs: int = 100
    
    # Env params
    reward_type: int = 4
    train_neg_ratio: int = 1
    engine: str = 'python'
    engine_strategy: str = 'cmp'
    endf_action: bool = True
    endt_action: bool = False
    skip_unary_actions: bool = True
    max_depth: int = 20
    memory_pruning: bool = True
    corruption_mode: str = 'dynamic'
    false_rules: bool = False
    
    # KGE integration params
    kge_action: bool = False
    logit_fusion: bool = False
    inference_fusion: bool = False
    inference_success_only: bool = False
    pbrs: bool = False
    enable_top_k: bool = False
    kge_engine: str = 'tf'
    kge_checkpoint_dir: str = './../../checkpoints/'
    kge_run_signature: Optional[str] = None
    kge_scores_file: Optional[str] = None
    
    # Top-K actions filtering
    top_k_init_value: int = 10
    top_k_final_value: int = 7
    top_k_start: float = 0.3
    top_k_end: float = 1.0
    top_k_transform: str = 'linear'
    
    # KGE logit shaping params
    kge_logit_init_value: float = 1.0
    kge_logit_final_value: float = 0.2
    kge_logit_start: float = 0.2
    kge_logit_end: float = 1.0
    kge_logit_transform: str = 'log'
    kge_logit_eps: float = 1e-6
    
    # Potential-based shaping params
    pbrs_beta: float = 0.5
    pbrs_gamma: float = 0.99
    
    # Evaluation hybrid fusion
    eval_hybrid_success_only: bool = True
    eval_hybrid_kge_weight: float = 2.0
    eval_hybrid_rl_weight: float = 1.0
    
    # Embedding params
    atom_embedder: str = 'transe'
    state_embedder: str = 'mean'
    atom_embedding_size: int = 256
    learn_embeddings: bool = True
    padding_atoms: int = 6
    padding_states: int = -1  # Auto-computed
    max_total_vars: int = 100
    
    # Other params
    min_gpu_memory_gb: float = 2.0
    extended_eval_info: bool = True
    eval_best_metric: str = 'mrr'
    plot: bool = False
    data_path: str = './data/'
    models_path: str = 'models/'
    rules_file: str = 'rules.txt'
    facts_file: str = 'train.txt'
    use_logger: bool = True
    logger_path: str = './runs/'
    use_wb: bool = False
    wb_path: str = './../wandb/'
    
    def __post_init__(self):
        """Set defaults that can't be set in field definitions."""
        if self.seed is None:
            self.seed = [0]
        # Note: dataset is inherited from BaseKGEConfig but we use dataset_name in RL
        # Store it under both names for compatibility
        self.dataset_name = self.dataset


class RLRunner(BaseKGERunner):
    """
    Runner for RL experiments.
    
    Inherits from BaseKGERunner to provide unified experiment management
    for reinforcement learning experiments in neural-guided grounding.
    """
    
    # RL doesn't use MODEL_CONFIGS like KGE, but we define it for compatibility
    MODEL_CONFIGS = {
        'PPO': {
            'description': 'Proximal Policy Optimization for neural-guided grounding',
        }
    }
    
    # Available datasets (same as KGE)
    AVAILABLE_DATASETS = [
        'family', 'countries_s1', 'countries_s2', 'countries_s3',
        'wn18rr', 'fb15k237', 'mnist_addition'
    ]
    
    def __init__(self, config: Optional[RLConfig] = None):
        """Initialize RL runner."""
        super().__init__(config or RLConfig())
        self.known_config_keys = set(asdict(RLConfig()).keys())
        
    def get_default_config(self) -> RLConfig:
        """Get default configuration for RL."""
        return RLConfig()
    
    def build_parser(self) -> argparse.ArgumentParser:
        """Build argument parser with RL-specific arguments."""
        parser = argparse.ArgumentParser(
            description='Reinforcement Learning Experiment Runner',
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        
        # Core arguments from original runner.py
        parser.add_argument(
            "--config",
            type=str,
            help="Path to YAML file describing a list of experiments (each as key/value overrides).",
        )
        parser.add_argument(
            "--set",
            action='append',
            default=[],
            metavar="KEY=VALUE",
            help="Override a configuration value, e.g. --set reward_type=3 --set seed='[0,1]'.",
        )
        parser.add_argument(
            "--grid",
            action='append',
            default=[],
            metavar="KEY=V1,V2",
            help="Grid search values for a parameter, e.g. --grid reward_type=2,3 --grid ent_coef=0.2,0.5.",
        )
        parser.add_argument(
            "--eval",
            action='store_true',
            help="Shortcut: load model and skip training (timesteps=0).",
        )
        
        # Hyperparameter search (inherited from base but we use config/grid instead)
        parser.add_argument(
            '--hparam_search',
            action='store_true',
            help='Enable hyperparameter search (use --config for experiments list)',
        )
        
        # Common overrides
        parser.add_argument('--dataset', type=str, help='Dataset name')
        parser.add_argument('--timesteps', type=int, help='Training timesteps')
        parser.add_argument('--seed', type=str, help='Seed(s) as list, e.g., "[0,1,2]"')
        parser.add_argument('--device', type=str, default='cuda:1', 
                          help="Device: 'cpu', 'cuda:1' (auto-select), 'cuda:all'")
        parser.add_argument('--save_models', action='store_true', help='Save trained models')
        parser.add_argument('--results', type=str, help='Path to save results JSON')
        
        # Add examples
        parser.epilog = """
Examples:
  # Single experiment with default config
  python runner_rl.py
  
  # Override parameters
  python runner_rl.py --set dataset_name=family --set timesteps_train=1000000
  
  # Load experiments from YAML
  python runner_rl.py --config experiments.yaml
  
  # Grid search
  python runner_rl.py --grid reward_type=3,4 --grid ent_coef=0.2,0.5
  
  # Evaluation only
  python runner_rl.py --eval --set dataset_name=family
  
  # Hyperparameter search with config file
  python runner_rl.py --hparam_search --config experiments.yaml --results results.json
        """
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = self.build_parser()
        return parser.parse_args(args)
    
    def args_to_config(self, args: argparse.Namespace) -> RLConfig:
        """
        Convert parsed arguments to configuration.
        
        This handles the RL-specific argument processing.
        """
        # Start with default config
        config_dict = asdict(self.get_default_config())
        base_config = copy.deepcopy(config_dict)
        
        # Apply command-line overrides from --set
        for assignment in getattr(args, 'set', []):
            key, raw_value = parse_assignment(assignment)
            parsed_value = parse_scalar(raw_value)
            update_config_value(base_config, key, parsed_value, config_dict)
        
        # Handle --eval shortcut
        if getattr(args, 'eval', False):
            base_config['load_model'] = True
            base_config['timesteps_train'] = 0
        
        # Apply direct argument overrides
        if hasattr(args, 'dataset') and args.dataset:
            base_config['dataset_name'] = args.dataset
            base_config['dataset'] = args.dataset
        if hasattr(args, 'timesteps') and args.timesteps:
            base_config['timesteps_train'] = args.timesteps
        if hasattr(args, 'seed') and args.seed:
            seed_val = parse_scalar(args.seed)
            base_config['seed'] = seed_val if isinstance(seed_val, list) else [seed_val]
        if hasattr(args, 'save_models') and args.save_models:
            base_config['save_model'] = True
        if hasattr(args, 'results') and args.results:
            base_config['results_file'] = args.results
        
        # Device is handled separately in setup_device_and_experiments
        if hasattr(args, 'device') and args.device:
            base_config['device'] = args.device
        
        # Convert back to RLConfig
        config = RLConfig(**base_config)
        return config
    
    def setup_device_and_experiments(
        self,
        args: argparse.Namespace,
        base_config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Setup device and generate experiment configurations.
        
        Args:
            args: Parsed command-line arguments
            base_config: Base configuration dictionary
            
        Returns:
            Tuple of (device_string, list of experiment configs)
        """
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
        
        # Prepare grid search or config-based experiments
        run_configs = []
        
        # Check for --grid
        grid_spec = {}
        if hasattr(args, 'grid') and args.grid:
            for entry in args.grid:
                key, raw_values = parse_assignment(entry)
                value_candidates = [v.strip() for v in raw_values.split(',') if v.strip()]
                if not value_candidates:
                    raise ValueError(f"No values supplied for grid entry '{entry}'.")
                parsed_values = [
                    coerce_config_value(key, parse_scalar(candidate), base_config)
                    for candidate in value_candidates
                ]
                grid_spec[key] = parsed_values
        
        # Load experiments from config file
        experiments_from_file = []
        if hasattr(args, 'config') and args.config:
            overrides_list = load_experiment_configs(args.config)
            for idx, overrides in enumerate(overrides_list):
                unknown_keys = set(overrides) - self.known_config_keys
                if unknown_keys:
                    unknown_list = ", ".join(sorted(unknown_keys))
                    raise ValueError(
                        f"Unknown parameter(s) in experiment {idx}: {unknown_list}"
                    )
                experiments_from_file.append(overrides)
            print(f"\n\nLoaded {len(experiments_from_file)} experiment(s) from {args.config}")
        
        # Generate run configurations
        if experiments_from_file:
            for overrides in experiments_from_file:
                config_copy = copy.deepcopy(base_config)
                for key, value in overrides.items():
                    update_config_value(config_copy, key, value, base_config)
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
                        base_config,
                        prevalidated=True,
                    )
                run_configs.append(config_copy)
            print(
                f"Prepared grid search over {len(grid_spec)} parameter(s), "
                f"yielding {len(run_configs)} experiment(s)."
            )
        else:
            run_configs = [base_config]
        
        return device, run_configs
    
    def build_namespace(self, config: Dict[str, Any], device: str) -> argparse.Namespace:
        """
        Build argparse.Namespace from config dict with RL-specific processing.
        
        This replicates the build_namespace logic from original runner.py.
        """
        cfg = copy.deepcopy(config)
        
        # Best metric for model selection
        best_metric = cfg.get('eval_best_metric', 'mrr')
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
        
        # Boolean conversions
        cfg['kge_action'] = bool(cfg.get('kge_action', False))
        cfg['logit_fusion'] = bool(cfg.get('logit_fusion', False))
        cfg['inference_fusion'] = bool(cfg.get('inference_fusion', False))
        cfg['inference_success_only'] = bool(cfg.get('inference_success_only', True))
        cfg['pbrs'] = bool(cfg.get('pbrs', False))
        cfg['enable_top_k'] = bool(cfg.get('enable_top_k', False))
        cfg['prob_facts'] = bool(cfg.get('prob_facts', False))
        
        # Top-k facts handling
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
            cfg['eval_hybrid_success_only'] = True
        
        # Align PBRS parameters
        if not cfg['pbrs']:
            cfg['pbrs_beta'] = 0.0
            cfg['pbrs_gamma'] = None
        else:
            if float(cfg.get('pbrs_beta', 0.0)) == 0.0:
                raise ValueError("pbrs is True but pbrs_beta is 0.")
            if cfg.get('pbrs_gamma') is None:
                raise ValueError("pbrs is True but pbrs_gamma is None.")
        
        # Annealing specifications
        annealing_specs = {}
        
        # Top-K schedule
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
                cfg['top_k_value'] = 10
        else:
            cfg['top_k_value'] = None
        
        # KGE logit gain schedule
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
        
        cfg['annealing_specs'] = annealing_specs
        cfg['annealing'] = annealing_specs
        
        # Validate KGE engine
        kge_engine = cfg.get('kge_engine', 'tf').strip().lower()
        if kge_engine not in {'tf', 'tensorflow', 'pytorch', 'torch', 'pykeen'}:
            raise ValueError(f"Invalid kge_engine '{kge_engine}'.")
        
        # Normalize engine name
        if kge_engine in {'tensorflow'}:
            kge_engine = 'tf'
        elif kge_engine in {'torch'}:
            kge_engine = 'pytorch'
        cfg['kge_engine'] = kge_engine
        
        # Set default KGE run signatures
        if cfg.get('kge_run_signature') is None:
            dataset = cfg['dataset_name']
            backend = cfg['kge_engine']
            
            if backend == 'tf':
                if dataset in {"countries_s3", "countries_s2", "countries_s1"}:
                    cfg['kge_run_signature'] = f"{dataset}-backward_0_1-no_reasoner-rotate-True-256-256-128-rules.txt"
                elif dataset == "family":
                    cfg['kge_run_signature'] = "kinship_family-backward_0_1-no_reasoner-rotate-True-256-256-4-rules.txt"
                elif dataset == "wn18rr":
                    cfg['kge_run_signature'] = "wn18rr-backward_0_1-no_reasoner-rotate-True-256-256-1-rules.txt"
            elif backend == 'pytorch':
                if dataset in {"countries_s3", "countries_s2", "countries_s1"}:
                    cfg['kge_run_signature'] = f"{dataset}_pytorch_rotate_256"
                elif dataset == "family":
                    cfg['kge_run_signature'] = "family_pytorch_rotate_256"
                elif dataset == "wn18rr":
                    cfg['kge_run_signature'] = "wn18rr_pytorch_rotate_256"
            elif backend == 'pykeen':
                if dataset in {"countries_s3", "countries_s2", "countries_s1"}:
                    cfg['kge_run_signature'] = f"{dataset}_pykeen_rotate"
                elif dataset == "family":
                    cfg['kge_run_signature'] = "family_pykeen_rotate"
                elif dataset == "wn18rr":
                    cfg['kge_run_signature'] = "wn18rr_pykeen_rotate"
        
        # Create namespace
        namespace = argparse.Namespace(**cfg)
        
        # Dataset-specific padding
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
                raise ValueError("Unknown dataset for automatic padding configuration.")
        
        if namespace.dataset_name == "mnist_addition":
            namespace.corruption_mode = None
        
        # Corruption scheme
        namespace.corruption_scheme = ['head', 'tail']
        if 'countries' in namespace.dataset_name or 'ablation' in namespace.dataset_name:
            namespace.corruption_scheme = ['tail']
        
        # Prolog file setup
        if namespace.false_rules:
            if namespace.engine == 'prolog':
                namespace.janus_file = "countries_false_rules.pl"
            else:
                raise ValueError("False rules not implemented for python engine.")
        elif namespace.engine == 'python':
            namespace.janus_file = None
        else:
            namespace.janus_file = f"{namespace.dataset_name}.pl"
            print("Using prolog file:", namespace.janus_file)
        
        # Data files
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
    
    def build_run_signature(self, namespace: argparse.Namespace) -> str:
        """Build run signature for experiment tracking."""
        run_vars = (
            namespace.dataset_name,
            namespace.atom_embedding_size,
            namespace.endf_action,
            namespace.ent_coef,
            namespace.clip_range,
            namespace.engine_strategy,
            namespace.train_neg_ratio,
            namespace.reward_type,
            namespace.kge_action,
            namespace.logit_fusion,
            namespace.inference_fusion,
            namespace.pbrs,
            namespace.enable_top_k,
            namespace.prob_facts,
            namespace.topk_facts,
            namespace.topk_facts_threshold,
            namespace.n_epochs,
            namespace.lr,
            namespace.n_envs,
            namespace.pbrs_beta,
            namespace.pbrs_gamma,
            namespace.eval_hybrid_kge_weight,
            namespace.eval_hybrid_rl_weight,
            namespace.eval_hybrid_success_only,
            namespace.kge_logit_transform,
        )
        return '-'.join(str(v) for v in run_vars)
    
    def train_single_model(
        self,
        model_name: str,
        config: RLConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a single RL model.
        
        This is required by BaseKGERunner but for RL we don't use it directly.
        Instead, experiments are run through run_experiments.
        """
        raise NotImplementedError(
            "RL experiments should be run through run() or run_experiments() "
            "which handles the full RL training pipeline."
        )
    
    def load_model(self, model_path: str, **kwargs) -> Any:
        """Load a trained RL model."""
        # RL models are loaded through the standard RL pipeline
        # This could be extended to load SB3 models
        raise NotImplementedError("Model loading for RL should use the standard RL pipeline.")
    
    def save_model(self, model: Any, save_path: str, **kwargs) -> None:
        """Save a trained RL model."""
        # RL models are saved through the standard RL pipeline
        raise NotImplementedError("Model saving for RL should use the standard RL pipeline.")
    
    def run_single_experiment(
        self,
        namespace: argparse.Namespace,
        logger: Optional[FileLogger] = None
    ) -> Dict[str, Any]:
        """
        Run a single RL experiment for all seeds.
        
        Args:
            namespace: Experiment configuration
            logger: Optional file logger
            
        Returns:
            Dictionary containing experiment results
        """
        all_results = []
        
        for seed in namespace.seed:
            date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            namespace.seed_run_i = seed
            
            print(f"Seed {seed} in {namespace.seed}")
            dict_ordered = {k: namespace.__dict__[k] for k in sorted(namespace.__dict__.keys())}
            print("\nRun vars:", namespace.run_signature, '\n', dict_ordered, '\n')
            
            if namespace.use_logger:
                log_filename_tmp = os.path.join(
                    namespace.logger_path,
                    f"_tmp_log-{namespace.run_signature}-{date}-seed_{seed}.csv",
                )
            else:
                log_filename_tmp = None
            
            # Run training
            train_metrics, valid_metrics, test_metrics = main(
                namespace,
                log_filename_tmp,
                namespace.use_logger,
                namespace.use_wb,
                namespace.wb_path,
                date,
            )
            
            # Log results
            if namespace.use_logger and logger is not None:
                logged_data = copy.deepcopy(namespace)
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
                    namespace.logger_path,
                    'indiv_runs',
                    f"_ind_log-{namespace.run_signature}-{date}-{metrics}-seed_{seed}.csv",
                )
                logger.finalize_log_file(log_filename_tmp, log_filename_run_name)
            
            all_results.append({
                'seed': seed,
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics,
                'test_metrics': test_metrics,
            })
        
        # Log average results
        if namespace.use_logger and logger is not None:
            logger.log_avg_results(namespace.__dict__, namespace.run_signature, namespace.seed)
        
        # Aggregate metrics
        aggregated = {
            'dataset': namespace.dataset_name,
            'run_signature': namespace.run_signature,
            'seeds': namespace.seed,
            'results': all_results,
        }
        
        return aggregated
    
    def run_experiments(
        self,
        configs: List[Dict[str, Any]],
        device: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run multiple RL experiments.
        
        Args:
            configs: List of configuration dictionaries
            device: Device to use for training
            **kwargs: Additional parameters
            
        Returns:
            List of experiment results
        """
        all_results = []
        
        total_experiments = len(configs)
        for idx, config in enumerate(configs, start=1):
            print(f"\n{'='*80}")
            print(f"Experiment {idx}/{total_experiments}")
            print(f"{'='*80}\n")
            
            try:
                # Build namespace
                namespace = self.build_namespace(config, device)
                
                # Validate configuration
                if not namespace.save_model and namespace.restore_best_val_model:
                    raise ValueError(
                        "restore_best_val_model=True but save_model=False. "
                        "Enable model saving or disable best-model restoration."
                    )
                
                if namespace.restore_best_val_model and namespace.load_model == 'last_epoch':
                    print(
                        "\nWARNING: restore_best_val_model is True while load_model='last_epoch'. "
                        "You may not reproduce evaluation results.\n"
                    )
                
                # Build run signature
                namespace.run_signature = self.build_run_signature(namespace)
                
                # Create logger
                logger = FileLogger(base_folder=namespace.logger_path) if namespace.use_logger else None
                
                # Run experiment
                result = self.run_single_experiment(namespace, logger)
                all_results.append(result)
                
                print(f"\n✓ Experiment {idx} completed successfully!")
                
            except Exception as e:
                print(f"\n✗ Error in experiment {idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                all_results.append({
                    'error': str(e),
                    'config': config,
                    'timestamp': datetime.datetime.now().isoformat(),
                })
        
        return all_results
    
    def run(self, args: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Main entry point for running RL experiments.
        
        Args:
            args: Command-line arguments (None = use sys.argv)
            
        Returns:
            List of experiment results
        """
        # Parse arguments
        parsed_args = self.parse_args(args)
        
        # Check for conflicts
        if hasattr(parsed_args, 'config') and hasattr(parsed_args, 'grid'):
            if parsed_args.config and parsed_args.grid:
                raise ValueError("Use either --config or --grid, not both.")
        
        # Convert to config
        base_config = self.args_to_config(parsed_args)
        base_config_dict = asdict(base_config)
        
        # Setup device and generate experiment configs
        device, run_configs = self.setup_device_and_experiments(
            parsed_args,
            base_config_dict
        )
        
        # Run experiments
        results = self.run_experiments(run_configs, device)
        
        # Save results if requested
        if base_config.results_file:
            self.save_results(results, base_config.results_file)
        
        # Check for failures
        failed_count = sum(1 for r in results if 'error' in r)
        if failed_count > 0:
            print(f"\n⚠ Warning: {failed_count}/{len(results)} experiments failed")
            sys.exit(1)
        else:
            print(f"\n✓ All {len(results)} experiments completed successfully!")
        
        return results


def main():
    """Main entry point."""
    runner = RLRunner()
    runner.run()


if __name__ == '__main__':
    main()

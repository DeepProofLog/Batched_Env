"""
PyTorch KGE Runner - Single unified file.
Orchestrates training of PyTorch KGE models with hyperparameter search support.
"""
import os
import sys
import argparse
import json
import itertools
import time
import datetime
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path


@dataclass
class BaseKGEConfig:
    """Base configuration for KGE experiments."""
    
    # Dataset
    dataset: str = "nations,umls,fb15k237,pharmkg_full,family,wn18rr"
    data_root: str = "/home/castellanoontiv/Batched_Env/kge_experiments/data"
    train_split: str = "train.txt"
    valid_split: str = "valid.txt"
    test_split: str = "test.txt"
    use_local: bool = True
    
    # Training
    epochs: int = 100
    batch_size: int = 512
    seed: int = 42
    lr: Optional[float] = None
    embedding_dim: Optional[int] = None
    weight_decay: Optional[float] = None
    
    # Regularization
    use_early_stopping: bool = True
    patience: int = 10
    
    # Device
    device: str = "cpu"
    min_gpu_memory_gb: float = 2.0
    
    # Saving/Loading
    save_dir: str = "models"
    save_models: bool = True
    load_checkpoint: Optional[str] = None
    results_file: Optional[str] = None
    run_signature: Optional[str] = None
    
    # Misc
    verbose: bool = True


class BaseKGERunner:
    """Base runner class for KGE experiments."""
    
    BACKEND_NAME = "base"
    
    def __init__(self, config: Optional[BaseKGEConfig] = None):
        """Initialize base runner."""
        self.config = config or BaseKGEConfig()
        
    def build_parser(self) -> argparse.ArgumentParser:
        """Build argument parser with common KGE arguments."""
        parser = argparse.ArgumentParser(description="KGE Experiment Runner")
        
        # Dataset args
        parser.add_argument("--dataset", type=str, default="nations,umls,fb15k237,pharmkg_full", help="Dataset name")
        parser.add_argument("--data_root", type=str, default="/home/castellanoontiv/Batched_Env/kge_experiments/data", help="Root directory for data")
        parser.add_argument("--train_split", type=str, default="train.txt", help="Training split name")
        parser.add_argument("--valid_split", type=str, default="valid.txt", help="Validation split name")
        parser.add_argument("--test_split", type=str, default="test.txt", help="Test split name")
        parser.add_argument("--use_builtin", action="store_true", help="Use builtin datasets instead of local files")
        
        # Models to run
        parser.add_argument("--model", type=str, default="TransE", help="Single model to run")
        parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models to run")
        
        # Training args
        parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides model default)")
        parser.add_argument("--embedding_dim", type=int, default=None, help="Embedding dimension (overrides model default)")
        parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
        
        # Regularization
        parser.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping")
        parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
        
        # Device
        parser.add_argument("--device", type=str, default="cpu", help="Device to use")
        
        # Saving/Loading
        parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
        parser.add_argument("--save_models", action="store_true", help="Save trained model weights")
        parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
        parser.add_argument("--results", type=str, default=None, help="Path to save results JSON")
        
        # Hyperparameter search
        parser.add_argument("--hparam_search", action="store_true", help="Run hyperparameter search")
        parser.add_argument("--hparam_config", type=str, default=None, help="Path to hyperparameter search config JSON")
        
        # System
        parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
        parser.add_argument("--quiet", action="store_true", help="Suppress output")
        
        return parser

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = self.build_parser()
        return parser.parse_args(args)
    
    def parse_model_list(self, models_str: str) -> List[str]:
        """Parse comma-separated list of models."""
        if not models_str:
            return []
        if models_str.lower() == "all" and hasattr(self, "MODEL_CONFIGS"):
            return list(self.MODEL_CONFIGS.keys())
        return [m.strip() for m in models_str.split(",") if m.strip()]
    
    def parse_dataset_list(self, datasets_str: str) -> List[str]:
        """Parse comma-separated list of datasets."""
        if not datasets_str:
            return ["countries_s3"]
        return [d.strip() for d in datasets_str.split(",") if d.strip()]
    
    def load_hparam_config(self, path: str) -> Dict[str, Any]:
        """Load hyperparameter search configuration."""
        with open(path, 'r') as f:
            return json.load(f)
            
    def train_single_model(self, model_name: str, config: BaseKGEConfig, **kwargs) -> Dict[str, Any]:
        """
        Train a single model. Must be implemented by subclasses.
        
        args:
            model_name: Name of model to train
            config: Configuration object
            
        Returns:
            Dictionary with results/metrics
        """
        raise NotImplementedError("Subclasses must implement train_single_model")
        
    def run_experiments(self, models: List[str], config: BaseKGEConfig) -> List[Dict[str, Any]]:
        """Run experiments for multiple models."""
        results = []
        for model_name in models:
            if config.verbose:
                print(f"\n{'='*50}")
                print(f"Training {model_name} on {config.dataset}")
                print(f"{'='*50}")
            
            try:
                start_time = time.time()
                result = self.train_single_model(model_name, config)
                duration = time.time() - start_time
                
                result['model'] = model_name
                result['dataset'] = config.dataset
                result['duration'] = duration
                result['timestamp'] = datetime.datetime.now().isoformat()
                
                if config.verbose:
                    print(f"Finished {model_name} in {duration:.2f}s")
                    if 'metrics' in result:
                         print("Metrics:", result['metrics'])
                         
                results.append(result)
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'model': model_name,
                    'dataset': config.dataset,
                    'error': str(e),
                    'status': 'failed'
                })
                
        if config.results_file:
            self._save_results(results, config.results_file)
            
        return results

    def run_hparam_search(self, hparam_config: Dict[str, Any], base_config: BaseKGEConfig) -> List[Dict[str, Any]]:
        """
        Run hyperparameter search.
        
        Args:
            hparam_config: Dictionary defining search space per model
            base_config: Base configuration to start from
            
        Returns:
            List of results for all trials
        """
        # Determine models to search
        models_to_search = hparam_config.keys()
        
        all_results = []
        
        for model_name in models_to_search:
            search_space = hparam_config[model_name]
            
            # Generate grid
            keys = sorted(search_space.keys())
            combinations = list(itertools.product(*(search_space[k] for k in keys)))
            
            print(f"\nHyperparameter search for {model_name}: {len(combinations)} trials")
            
            for i, params in enumerate(combinations):
                # Apply params to config copy
                # Using type(base_config)() to create new instance of same class
                # But simplified: just copy needed
                import copy
                trial_config = copy.deepcopy(base_config)
                
                trial_params = dict(zip(keys, params))
                print(f"Trial {i+1}/{len(combinations)}: {trial_params}")
                
                # Update config attributes
                for k, v in trial_params.items():
                    if hasattr(trial_config, k):
                        setattr(trial_config, k, v)
                    else:
                        print(f"Warning: Config has no attribute '{k}', ignoring")
                        
                # Run trial
                try:
                    result = self.train_single_model(model_name, trial_config)
                    result['hparams'] = trial_params
                    result['trial'] = i
                    all_results.append(result)
                except Exception as e:
                    print(f"Trial failed: {e}")
                    
        return all_results

    def _save_results(self, results: List[Dict[str, Any]], filepath: str) -> None:
        """Save results list to JSON file."""
        if os.path.dirname(filepath):
             os.makedirs(os.path.dirname(filepath), exist_ok=True)
             
        # Helper to serialize non-serializable objects
        def json_serial(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            return str(obj)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=json_serial)
        print(f"Results saved to {filepath}")


class PyTorchConfig(BaseKGEConfig):
    """Extended configuration for PyTorch KGE models."""
    
    # PyTorch-specific parameters
    gamma: float = 12.0  # Margin for distance-based models
    p: int = 1  # Norm (1 for L1, 2 for L2)
    relation_dim: Optional[int] = None  # For TuckER
    dropout: float = 0.0
    
    # ConvE specific parameters
    input_dropout: float = 0.2
    feature_map_dropout: float = 0.2
    hidden_dropout: float = 0.3
    embedding_height: int = 10
    embedding_width: int = 20
    
    # Training enhancements
    neg_ratio: int = 1
    use_reciprocal: bool = False
    adv_temp: float = 0.0  # Self-adversarial temperature
    grad_clip: float = 0.0
    warmup_ratio: float = 0.0
    scheduler: str = 'none'  # 'none', 'cosine'
    
    # Performance
    amp: bool = True
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = True
    num_workers: int = 2
    eval_chunk_size: int = 2048
    eval_rank_mode: str = "realistic"
    report_train_mrr: bool = False
    
    # GPU settings
    min_gpu_memory_gb: float = 2.0
    multi_gpu: bool = False


class PyTorchRunner(BaseKGERunner):
    """Runner for PyTorch KGE models."""
    BACKEND_NAME = "torch"
    
    # Model-specific configurations (presets)
    MODEL_CONFIGS = {
        'RotatE': {
            'lr': 1e-3,
            'embedding_dim': 1024,
            'gamma': 12.0,
            'p': 1,
            'weight_decay': 0.0,
            'use_reciprocal': False,
            'adv_temp': 0.0,
            'grad_clip': 0.0,
            'warmup_ratio': 0.0,
            'scheduler': 'none',
            'description': 'Distance-based rotation model',
        },
        'ComplEx': {
            'lr': 5e-4,
            'embedding_dim': 1024,
            'weight_decay': 1e-6,
            'use_reciprocal': False,
            'adv_temp': 0.0,
            'grad_clip': 0.0,
            'warmup_ratio': 0.0,
            'scheduler': 'none',
            'description': 'Complex bilinear model',
        },
        'DistMult': {
            'lr': 5e-4,
            'embedding_dim': 512,
            'weight_decay': 1e-6,
            'use_reciprocal': False,
            'adv_temp': 0.0,
            'grad_clip': 0.0,
            'warmup_ratio': 0.0,
            'scheduler': 'none',
            'description': 'Bilinear diagonal model',
        },
        'TuckER': {
            'lr': 5e-4,
            'embedding_dim': 512,
            'relation_dim': 256,
            'dropout': 0.3,
            'weight_decay': 1e-6,
            'use_reciprocal': False,
            'adv_temp': 0.0,
            'grad_clip': 1.0,
            'warmup_ratio': 0.0,
            'scheduler': 'none',
            'description': 'Tucker decomposition model',
        },
        'TransE': {
            'lr': 1e-3,
            'embedding_dim': 512,
            'p': 1,
            'weight_decay': 0.0,
            'use_reciprocal': False,
            'adv_temp': 0.0,
            'grad_clip': 0.0,
            'warmup_ratio': 0.0,
            'scheduler': 'none',
            'description': 'Translational embedding model',
        },
        'ConvE': {
            'lr': 1e-3,
            'embedding_dim': 200,
            'embedding_height': 10,
            'embedding_width': 20,
            'input_dropout': 0.2,
            'feature_map_dropout': 0.2,
            'hidden_dropout': 0.3,
            'weight_decay': 1e-6,
            'use_reciprocal': False,
            'adv_temp': 0.0,
            'grad_clip': 1.0,
            'warmup_ratio': 0.0,
            'scheduler': 'none',
            'description': 'Convolutional model',
        },
        'mrr_boost': {
            'lr': 1e-3,
            'embedding_dim': 1024,
            'gamma': 12.0,
            'p': 1,
            'weight_decay': 1e-6,
            'use_reciprocal': True,
            'adv_temp': 0.0,
            'grad_clip': 2.0,
            'warmup_ratio': 0.1,
            'scheduler': 'cosine',
            'description': 'RotatE with MRR boost (reciprocal + warmup + scheduler)',
        },
    }
    
    def __init__(self, config: Optional[PyTorchConfig] = None):
        """Initialize PyTorch runner."""
        super().__init__(config or PyTorchConfig())
        self.use_cpu = False
        self.use_multi_gpu = False
    
    def get_default_config(self) -> PyTorchConfig:
        """Get default configuration for PyTorch."""
        config = PyTorchConfig()
        base_dir = Path(__file__).resolve().parent
        config.data_root = "/home/castellanoontiv/Batched_Env/kge_experiments/data"
        config.save_dir = str(base_dir / "models")
        return config
    
    def build_parser(self) -> argparse.ArgumentParser:
        """Build argument parser with PyTorch-specific arguments."""
        parser = super().build_parser()
        base_dir = Path(__file__).resolve().parent
        
        # Override device choices for PyTorch
        for action in parser._actions:
            if action.dest == 'device':
                action.choices = ['cpu', 'cuda:1', 'cuda:all']
                action.help = "Device: 'cpu' (use CPU), 'cuda:1' (auto-select best GPU), 'cuda:all' (use all available GPUs)"
                action.default = 'cuda:all'
                break
        
        # Add PyTorch-specific arguments
        parser.add_argument(
            '--min_gpu_memory_gb',
            type=float,
            default=2.0,
            help='Minimum free GPU memory in GB'
        )
        parser.add_argument(
            '--gamma',
            type=float,
            default=None,
            help='Margin for distance-based models'
        )
        parser.add_argument(
            '--p',
            type=int,
            default=None,
            choices=[1, 2],
            help='Norm (1 for L1, 2 for L2)'
        )
        parser.add_argument(
            '--relation_dim',
            type=int,
            default=None,
            help='Relation embedding dimension (for TuckER)'
        )
        parser.add_argument(
            '--dropout',
            type=float,
            default=None,
            help='Dropout rate'
        )
        parser.add_argument(
            '--neg_ratio',
            type=int,
            default=1,
            help='Negative sampling ratio'
        )
        parser.add_argument(
            '--use_reciprocal',
            action='store_true',
            help='Add reciprocal/inverse relations'
        )
        parser.add_argument(
            '--adv_temp',
            type=float,
            default=0.0,
            help='Self-adversarial temperature (0 = disabled)'
        )
        parser.add_argument(
            '--grad_clip',
            type=float,
            default=0.0,
            help='Gradient clipping threshold (0 = disabled)'
        )
        parser.add_argument(
            '--warmup_ratio',
            type=float,
            default=0.0,
            help='Warmup ratio (fraction of total steps)'
        )
        parser.add_argument(
            '--scheduler',
            type=str,
            default='none',
            choices=['none', 'cosine'],
            help='Learning rate scheduler'
        )
        parser.add_argument(
            '--amp',
            action='store_true',
            default=True,
            help='Use automatic mixed precision'
        )
        parser.add_argument(
            '--no_amp',
            action='store_true',
            help='Disable automatic mixed precision'
        )
        parser.add_argument(
            '--compile',
            action='store_true',
            default=True,
            help='Use torch.compile'
        )
        parser.add_argument(
            '--no_compile',
            action='store_true',
            help='Disable torch.compile'
        )
        parser.add_argument(
            '--compile_mode',
            type=str,
            default='reduce-overhead',
            choices=['default', 'reduce-overhead', 'max-autotune'],
            help='torch.compile mode'
        )
        parser.add_argument(
            '--compile_fullgraph',
            action='store_true',
            default=True,
            help='Request full-graph torch.compile'
        )
        parser.add_argument(
            '--no_compile_fullgraph',
            dest='compile_fullgraph',
            action='store_false',
            help='Disable full-graph torch.compile'
        )
        parser.add_argument(
            '--sampled_eval',
            action='store_true',
            default=True,
            help='Use sampled evaluation (100 negatives)'
        )
        parser.add_argument(
            '--no_sampled_eval',
            dest='sampled_eval',
            action='store_false',
            help='Use exhaustive evaluation (all entities)'
        )
        parser.add_argument(
            '--sampled_negatives',
            type=int,
            default=100,
            help='Number of negatives for sampled evaluation'
        )
        
        # Update description
        parser.description = 'Train PyTorch KGE models with hyperparameter search support'
        parser.set_defaults(
            data_root="/home/castellanoontiv/Batched_Env/kge_experiments/data",
            save_dir=str(base_dir / "models"),
            save_models=True,
        )
        
        # Add examples
        parser.epilog = """
Examples:
  # Train a single model
  python runner_torch.py --model RotatE --dataset family
  
  # Train multiple models
  python runner_torch.py --models TransE,ComplEx,RotatE --dataset family
  
  # Train all models
  python runner_torch.py --models all --dataset wn18rr
  
  # Train with MRR boost preset
  python runner_torch.py --model mrr_boost --dataset family
  
  # Custom hyperparameters
  python runner_torch.py --model RotatE --dataset family --epochs 700 --lr 0.001 --use_reciprocal
  
  # Hyperparameter search
  python runner_torch.py --hparam_search --hparam_config hparam_config.json
  
  # Multi-GPU training
  python runner_torch.py --model RotatE --dataset wn18rr --device cuda:all
  
  # Save models and results
  python runner_torch.py --models all --dataset family --save_models --results results.json
        """
        
        return parser
    
    def args_to_config(self, args: argparse.Namespace) -> PyTorchConfig:
        """Convert parsed arguments to PyTorch configuration."""
        # Start with base config
        config = PyTorchConfig()
        
        # Update with base arguments
        config.dataset = args.dataset
        config.data_root = args.data_root
        config.train_split = args.train_split
        config.valid_split = args.valid_split
        config.test_split = args.test_split
        config.use_local = not args.use_builtin
        
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.seed = args.seed
        
        # PyTorch-specific arguments
        if hasattr(args, 'gamma') and args.gamma is not None:
            config.gamma = args.gamma
        if hasattr(args, 'p') and args.p is not None:
            config.p = args.p
        if hasattr(args, 'relation_dim') and args.relation_dim is not None:
            config.relation_dim = args.relation_dim
        if hasattr(args, 'dropout') and args.dropout is not None:
            config.dropout = args.dropout
        if hasattr(args, 'neg_ratio'):
            config.neg_ratio = args.neg_ratio
        if hasattr(args, 'use_reciprocal'):
            config.use_reciprocal = args.use_reciprocal
        if hasattr(args, 'adv_temp'):
            config.adv_temp = args.adv_temp
        if hasattr(args, 'grad_clip'):
            config.grad_clip = args.grad_clip
        if hasattr(args, 'warmup_ratio'):
            config.warmup_ratio = args.warmup_ratio
        if hasattr(args, 'scheduler'):
            config.scheduler = args.scheduler
        
        # Performance settings
        if hasattr(args, 'amp') and hasattr(args, 'no_amp'):
            config.amp = args.amp and not args.no_amp
        if hasattr(args, 'compile') and hasattr(args, 'no_compile'):
            config.compile = args.compile and not args.no_compile
        if hasattr(args, 'compile_mode'):
            config.compile_mode = args.compile_mode
        if hasattr(args, 'compile_fullgraph'):
            config.compile_fullgraph = args.compile_fullgraph
        if hasattr(args, 'sampled_eval'):
            config.sampled_eval = args.sampled_eval
        if hasattr(args, 'sampled_negatives'):
            config.sampled_negatives = args.sampled_negatives
        
        # Only update if explicitly provided
        if args.lr is not None:
            config.lr = args.lr
        if args.embedding_dim is not None:
            config.embedding_dim = args.embedding_dim
        if args.weight_decay is not None:
            config.weight_decay = args.weight_decay
        
        config.use_early_stopping = not args.no_early_stopping
        config.patience = args.patience
        
        config.device = args.device
        if hasattr(args, 'min_gpu_memory_gb'):
            config.min_gpu_memory_gb = args.min_gpu_memory_gb
        
        config.save_dir = args.save_dir
        config.save_models = args.save_models
        config.load_checkpoint = args.load_checkpoint
        config.results_file = args.results
        
        config.verbose = args.verbose and not args.quiet
        
        
        return config
    
    def setup_device(self, device_choice: str, min_memory_gb: float) -> None:
        """
        Setup device selection before importing torch modules.
        
        Args:
            device_choice: One of 'cpu', 'cuda:1', 'cuda:all'
            min_memory_gb: Minimum GPU memory required
        """
        if device_choice == "cpu":
            print("\n=== Using CPU ===")
            print("Training will run on CPU (slower but always available)\n")
            self.use_cpu = True
            self.use_multi_gpu = False
            return
        
        elif device_choice == "cuda:1":
            print("\n=== Auto-selecting best GPU ===")
            best_gpu = self.select_best_gpu_early(min_free_gb=min_memory_gb)
            if best_gpu is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
                print(f"Set CUDA_VISIBLE_DEVICES={best_gpu}\n")
                self.use_cpu = False
                self.use_multi_gpu = False
            else:
                print(f"No GPU with at least {min_memory_gb} GB free memory found.")
                print("Falling back to CPU\n")
                self.use_cpu = True
                self.use_multi_gpu = False
        
        elif device_choice == "cuda:all":
            print("\n=== Using all available GPUs ===")
            available_gpus = self.get_available_gpus_early(min_free_gb=min_memory_gb)
            
            if len(available_gpus) == 0:
                print(f"No GPUs with at least {min_memory_gb} GB free memory found.")
                print("Falling back to CPU\n")
                self.use_cpu = True
                self.use_multi_gpu = False
            elif len(available_gpus) == 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
                print(f"Only 1 GPU available: GPU {available_gpus[0]}")
                print(f"Set CUDA_VISIBLE_DEVICES={available_gpus[0]}\n")
                self.use_cpu = False
                self.use_multi_gpu = False
            else:
                print(f"Found {len(available_gpus)} GPUs: {available_gpus}")
                print(f"Multi-GPU training will use DataParallel.")
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
                print(f"Set CUDA_VISIBLE_DEVICES={','.join(map(str, available_gpus))}\n")
                self.use_cpu = False
                self.use_multi_gpu = True
    
    def select_best_gpu_early(self, min_free_gb: float = 1.0) -> Optional[int]:
        """Select GPU with most free memory using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) == 2:
                        gpu_id = int(parts[0].strip())
                        free_mb = float(parts[1].strip())
                        free_gb = free_mb / 1024.0
                        print(f"GPU {gpu_id}: {free_gb:.2f} GB free")
                        if free_gb >= min_free_gb:
                            gpus.append((gpu_id, free_gb))
            
            if gpus:
                gpus.sort(key=lambda x: x[1], reverse=True)
                best_gpu = gpus[0][0]
                print(f"Selected GPU {best_gpu} with {gpus[0][1]:.2f} GB free")
                return best_gpu
            else:
                return None
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not query GPUs: {e}")
            return None
    
    def get_available_gpus_early(self, min_free_gb: float = 1.0) -> List[int]:
        """Get all GPUs with sufficient memory using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            
            available_gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) == 2:
                        gpu_id = int(parts[0].strip())
                        free_mb = float(parts[1].strip())
                        free_gb = free_mb / 1024.0
                        print(f"GPU {gpu_id}: {free_gb:.2f} GB free")
                        if free_gb >= min_free_gb:
                            available_gpus.append(gpu_id)
            
            return available_gpus
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not query GPUs: {e}")
            return []
    
    def train_single_model(
        self,
        model_name: str,
        config: PyTorchConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a single PyTorch KGE model.
        
        Args:
            model_name: Name of the model to train
            config: Configuration for training
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Import here after device setup
        from train_torch import TrainConfig, train_model
        
        # Get model-specific config
        base_config = self.MODEL_CONFIGS.get(model_name, {})
        
        # Generate timestamp and run signature
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dim = config.embedding_dim or base_config.get('embedding_dim', 1024)
        run_signature = f"torch_{config.dataset}_{model_name}_{dim}_{timestamp}_s{config.seed}"
        
        # Update save_dir to include run_signature
        save_dir = os.path.join(config.save_dir, run_signature)
        
        # Build training config from our config
        train_cfg = TrainConfig(
            save_dir=save_dir,
            run_signature=run_signature,
            dataset=config.dataset,
            data_root=config.data_root,
            train_split=config.train_split,
            valid_split=config.valid_split,
            test_split=config.test_split,
            model=model_name if model_name != 'mrr_boost' else 'RotatE',
            dim=config.embedding_dim or base_config.get('embedding_dim', 1024),
            gamma=config.gamma if hasattr(config, 'gamma') else base_config.get('gamma', 12.0),
            p=config.p if hasattr(config, 'p') else base_config.get('p', 1),
            relation_dim=config.relation_dim if hasattr(config, 'relation_dim') else base_config.get('relation_dim'),
            dropout=config.dropout if hasattr(config, 'dropout') else base_config.get('dropout', 0.0),
            input_dropout=config.input_dropout if hasattr(config, 'input_dropout') else base_config.get('input_dropout', 0.2),
            feature_map_dropout=config.feature_map_dropout if hasattr(config, 'feature_map_dropout') else base_config.get('feature_map_dropout', 0.2),
            hidden_dropout=config.hidden_dropout if hasattr(config, 'hidden_dropout') else base_config.get('hidden_dropout', 0.3),
            embedding_height=config.embedding_height if hasattr(config, 'embedding_height') else base_config.get('embedding_height', 10),
            embedding_width=config.embedding_width if hasattr(config, 'embedding_width') else base_config.get('embedding_width', 20),
            lr=config.lr or base_config.get('lr', 1e-3),
            batch_size=config.batch_size,
            neg_ratio=config.neg_ratio if hasattr(config, 'neg_ratio') else base_config.get('neg_ratio', 1),
            epochs=config.epochs,
            num_workers=config.num_workers if hasattr(config, 'num_workers') else 2,
            amp=config.amp if hasattr(config, 'amp') else True,
            compile=config.compile if hasattr(config, 'compile') else True,
            compile_mode=config.compile_mode if hasattr(config, 'compile_mode') else "reduce-overhead",
            compile_fullgraph=config.compile_fullgraph if hasattr(config, 'compile_fullgraph') else True,
            cpu=self.use_cpu,
            multi_gpu=self.use_multi_gpu,
            seed=config.seed,
            eval_chunk_size=config.eval_chunk_size if hasattr(config, 'eval_chunk_size') else 2048,
            eval_rank_mode=config.eval_rank_mode if hasattr(config, 'eval_rank_mode') else "realistic",
            report_train_mrr=config.report_train_mrr if hasattr(config, 'report_train_mrr') else False,
            use_reciprocal=config.use_reciprocal if hasattr(config, 'use_reciprocal') else base_config.get('use_reciprocal', False),
            adv_temp=config.adv_temp if hasattr(config, 'adv_temp') else base_config.get('adv_temp', 0.0),
            weight_decay=config.weight_decay or base_config.get('weight_decay', 0.0),
            grad_clip=config.grad_clip if hasattr(config, 'grad_clip') else base_config.get('grad_clip', 0.0),
            warmup_ratio=config.warmup_ratio if hasattr(config, 'warmup_ratio') else base_config.get('warmup_ratio', 0.0),
            scheduler=config.scheduler if hasattr(config, 'scheduler') else base_config.get('scheduler', 'none'),
            sampled_eval=config.sampled_eval,
            sampled_negatives=config.sampled_negatives,
        )
        
        if config.verbose:
            print("Training config:")
            print(train_cfg)
            print()
        
        # Train the model
        artifacts = train_model(train_cfg)
        
        # Build result dictionary
        result = {
            'metrics': artifacts.metrics or {},
            'model_path': artifacts.weights_path if config.save_models else None,
            'config_path': artifacts.config_path,
        }
        
        return result
    
    def load_model(self, model_path: str, **kwargs) -> Any:
        """
        Load a trained PyTorch model.
        
        Args:
            model_path: Path to saved model weights
            **kwargs: Additional parameters (should include model config)
            
        Returns:
            Loaded PyTorch model
        """
        import torch
        from model_torch import build_model
        import json
        
        # Load config
        model_dir = Path(model_path).parent
        config_path = model_dir / 'config.json'
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Build model
        model = build_model(
            model_name=config_dict['model'],
            num_entities=config_dict['num_entities'],
            num_relations=config_dict['num_relations'],
            dim=config_dict['dim'],
            gamma=config_dict.get('gamma', 12.0),
            p=config_dict.get('p', 1),
            relation_dim=config_dict.get('relation_dim'),
            dropout=config_dict.get('dropout', 0.0),
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location=kwargs.get('device', 'cpu'))
        model.load_state_dict(state_dict)
        
        return model
    
    def save_model(self, model: Any, save_path: str, **kwargs) -> None:
        """
        Save a trained PyTorch model.
        
        Args:
            model: PyTorch model to save
            save_path: Path where model should be saved
            **kwargs: Additional parameters
        """
        import torch
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
    
    def run(self, args: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Main entry point for running experiments.
        
        Args:
            args: Command-line arguments (None = use sys.argv)
            
        Returns:
            List of experiment results
        """
        # Parse arguments
        parsed_args = self.parse_args(args)
        
        # Setup device BEFORE importing torch training modules
        config = self.args_to_config(parsed_args)
        self.setup_device(
            parsed_args.device,
            parsed_args.min_gpu_memory_gb if hasattr(parsed_args, 'min_gpu_memory_gb') else 2.0
        )
        
        # Now run experiments (imports happen inside train_single_model)
        if parsed_args.hparam_search:
            if parsed_args.hparam_config is None:
                raise ValueError("--hparam_config must be provided when --hparam_search is enabled")
            
            hparam_config = self.load_hparam_config(parsed_args.hparam_config)
            results = self.run_hparam_search(hparam_config, config)
        else:
            # Determine which models and datasets to run
            if parsed_args.models:
                models = self.parse_model_list(parsed_args.models)
            else:
                models = self.parse_model_list(parsed_args.model)
            
            datasets = self.parse_dataset_list(parsed_args.dataset)
            
            # Run experiments for each dataset
            all_results = []
            for dataset in datasets:
                config.dataset = dataset
                results = self.run_experiments(models, config)
                all_results.extend(results)
            results = all_results
        
        # Check for failures
        failed_count = sum(1 for r in results if 'error' in r)
        if failed_count > 0:
            sys.exit(1)
        
        return results


def main():
    """Main entry point."""
    runner = PyTorchRunner()
    
    # Print available models
    print("\nAvailable PyTorch models:")
    for model_name, config in runner.MODEL_CONFIGS.items():
        print(f"  - {model_name}: {config['description']}")
    print()
    
    # Run experiments
    runner.run()


if __name__ == '__main__':
    main()

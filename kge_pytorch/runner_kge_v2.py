"""
PyTorch KGE Runner - Inherits from BaseKGERunner.
Orchestrates training of PyTorch KGE models with hyperparameter search support.
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
import argparse
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kge_base_runner import BaseKGERunner, BaseKGEConfig


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
    num_workers: int = 2
    eval_chunk_size: int = 2048
    report_train_mrr: bool = False
    
    # GPU settings
    min_gpu_memory_gb: float = 2.0
    multi_gpu: bool = False


class PyTorchRunner(BaseKGERunner):
    """Runner for PyTorch KGE models."""
    
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
        return PyTorchConfig()
    
    def build_parser(self) -> argparse.ArgumentParser:
        """Build argument parser with PyTorch-specific arguments."""
        parser = super().build_parser()
        
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
        
        # Update description
        parser.description = 'Train PyTorch KGE models with hyperparameter search support'
        
        # Add examples
        parser.epilog = """
Examples:
  # Train a single model
  python runner_kge_v2.py --model RotatE --dataset family
  
  # Train multiple models
  python runner_kge_v2.py --models TransE,ComplEx,RotatE --dataset family
  
  # Train all models
  python runner_kge_v2.py --models all --dataset wn18rr
  
  # Train with MRR boost preset
  python runner_kge_v2.py --model mrr_boost --dataset family
  
  # Custom hyperparameters
  python runner_kge_v2.py --model RotatE --dataset family --epochs 700 --lr 0.001 --use_reciprocal
  
  # Hyperparameter search
  python runner_kge_v2.py --hparam_search --hparam_config hparam_config.json
  
  # Multi-GPU training
  python runner_kge_v2.py --model RotatE --dataset wn18rr --device cuda:all
  
  # Save models and results
  python runner_kge_v2.py --models all --dataset family --save_models --results results.json
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
        
        # Build training config from our config
        train_cfg = TrainConfig(
            save_dir=config.save_dir,
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
            cpu=self.use_cpu,
            multi_gpu=self.use_multi_gpu,
            seed=config.seed,
            eval_chunk_size=config.eval_chunk_size if hasattr(config, 'eval_chunk_size') else 2048,
            report_train_mrr=config.report_train_mrr if hasattr(config, 'report_train_mrr') else False,
            use_reciprocal=config.use_reciprocal if hasattr(config, 'use_reciprocal') else base_config.get('use_reciprocal', False),
            adv_temp=config.adv_temp if hasattr(config, 'adv_temp') else base_config.get('adv_temp', 0.0),
            weight_decay=config.weight_decay or base_config.get('weight_decay', 0.0),
            grad_clip=config.grad_clip if hasattr(config, 'grad_clip') else base_config.get('grad_clip', 0.0),
            warmup_ratio=config.warmup_ratio if hasattr(config, 'warmup_ratio') else base_config.get('warmup_ratio', 0.0),
            scheduler=config.scheduler if hasattr(config, 'scheduler') else base_config.get('scheduler', 'none'),
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
            # Determine which models to run
            if parsed_args.models:
                models = self.parse_model_list(parsed_args.models)
            else:
                models = [parsed_args.model]
            
            # Run experiments
            results = self.run_experiments(models, config)
        
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

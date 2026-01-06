"""
Base runner class for KGE model training.
Provides common functionality for PyKeen and PyTorch KGE runners.
"""
import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import product
import warnings


@dataclass
class BaseKGEConfig:
    """Base configuration for KGE training."""
    
    # Dataset parameters
    dataset: str = 'family'
    data_root: str = './data'
    train_split: str = 'train.txt'
    valid_split: str = 'valid.txt'
    test_split: str = 'test.txt'
    use_local: bool = True
    
    # Model parameters
    model: str = 'TransE'
    embedding_dim: Optional[int] = None
    
    # Training parameters
    epochs: int = 1500
    batch_size: int = 4096
    lr: float = 0.001
    weight_decay: float = 0.0
    seed: int = 42
    
    # Early stopping
    use_early_stopping: bool = True
    patience: int = 100
    
    # Device configuration
    device: str = 'cuda'
    
    # I/O parameters
    save_dir: str = './models'
    save_models: bool = False
    load_checkpoint: Optional[str] = None
    results_file: Optional[str] = None
    
    # Logging
    verbose: bool = True
    log_interval: int = 50

    # Run metadata
    run_signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseKGEConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown config parameter: {key}")


class BaseKGERunner(ABC):
    """
    Base class for KGE runners.
    
    Provides:
    - Default configuration management
    - Command-line argument parsing
    - Hyperparameter search
    - Model loading/saving utilities
    - Results tracking
    """
    
    # Model-specific configurations (to be overridden by subclasses)
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {}
    BACKEND_NAME: str = "kge"
    
    # Available datasets
    AVAILABLE_DATASETS = [
        'family', 'countries_s1', 'countries_s2', 'countries_s3',
        'wn18rr', 'fb15k237'
    ]
    
    def __init__(self, config: Optional[BaseKGEConfig] = None):
        """
        Initialize runner with configuration.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or BaseKGEConfig()
        self.results: List[Dict[str, Any]] = []

    def build_run_signature(
        self,
        backend: str,
        dataset: str,
        model_name: str,
        embedding_dim: int,
        seed: int,
    ) -> str:
        """Build a run signature with backend prefix and timestamp."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{backend}_{dataset}_{model_name}_{embedding_dim}_{stamp}_s{seed}"

    def _prepare_run_config(self, config: BaseKGEConfig, model_name: str) -> BaseKGEConfig:
        """Fill in run_signature and save_dir for a specific model run."""
        if config.run_signature is None:
            embedding_dim = int(getattr(config, "embedding_dim", 0) or 0)
            config.run_signature = self.build_run_signature(
                backend=self.BACKEND_NAME,
                dataset=config.dataset,
                model_name=model_name,
                embedding_dim=embedding_dim,
                seed=config.seed,
            )

        if config.save_models:
            save_root = Path(config.save_dir)
            config.save_dir = str(save_root / config.run_signature)
        return config
        
    @abstractmethod
    def train_single_model(
        self,
        model_name: str,
        config: BaseKGEConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a single model. Must be implemented by subclasses.
        
        Args:
            model_name: Name of the model to train
            config: Configuration for training
            **kwargs: Additional framework-specific parameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> Any:
        """
        Load a trained model. Must be implemented by subclasses.
        
        Args:
            model_path: Path to saved model
            **kwargs: Additional framework-specific parameters
            
        Returns:
            Loaded model object
        """
        pass
    
    @abstractmethod
    def save_model(self, model: Any, save_path: str, **kwargs) -> None:
        """
        Save a trained model. Must be implemented by subclasses.
        
        Args:
            model: Model object to save
            save_path: Path where model should be saved
            **kwargs: Additional framework-specific parameters
        """
        pass
    
    def get_default_config(self) -> BaseKGEConfig:
        """Get default configuration."""
        return BaseKGEConfig()
    
    def build_parser(self) -> argparse.ArgumentParser:
        """
        Build argument parser with common arguments.
        Subclasses can extend this to add framework-specific arguments.
        """
        parser = argparse.ArgumentParser(
            description='Train KGE models',
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        
        # Model selection
        parser.add_argument(
            '--models',
            type=str,
            default=None,
            help='Comma-separated list of models to train (e.g., "TransE,ComplEx") or "all"'
        )
        parser.add_argument(
            '--model',
            type=str,
            default='TransE',
            help='Single model to train (ignored if --models is provided)'
        )
        
        # Dataset
        parser.add_argument(
            '--dataset',
            type=str,
            default='family',
            choices=self.AVAILABLE_DATASETS,
            help='Dataset to use'
        )
        parser.add_argument(
            '--data_root',
            type=str,
            default='./data',
            help='Root directory for datasets'
        )
        parser.add_argument(
            '--train_split',
            type=str,
            default='train.txt',
            help='Training split filename'
        )
        parser.add_argument(
            '--valid_split',
            type=str,
            default='valid.txt',
            help='Validation split filename'
        )
        parser.add_argument(
            '--test_split',
            type=str,
            default='test.txt',
            help='Test split filename'
        )
        parser.add_argument(
            '--use_builtin',
            action='store_true',
            help='Use built-in dataset instead of local files'
        )
        
        # Training hyperparameters
        parser.add_argument(
            '--epochs',
            type=int,
            default=1500,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--batch_size',
            type=int,
            default=4096,
            help='Training batch size'
        )
        parser.add_argument(
            '--lr',
            type=float,
            default=None,
            help='Learning rate (default: model-specific)'
        )
        parser.add_argument(
            '--embedding_dim',
            type=int,
            default=None,
            help='Embedding dimension (default: model-specific)'
        )
        parser.add_argument(
            '--weight_decay',
            type=float,
            default=None,
            help='Weight decay (default: model-specific)'
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=42,
            help='Random seed'
        )
        
        # Early stopping
        parser.add_argument(
            '--patience',
            type=int,
            default=100,
            help='Early stopping patience (epochs)'
        )
        parser.add_argument(
            '--no_early_stopping',
            action='store_true',
            help='Disable early stopping'
        )
        
        # Device
        parser.add_argument(
            '--device',
            type=str,
            default='cuda',
            choices=['cuda', 'cpu'],
            help='Device to use for training'
        )
        
        # I/O
        parser.add_argument(
            '--save_dir',
            type=str,
            default='./models',
            help='Directory to save models'
        )
        parser.add_argument(
            '--save_models',
            action='store_true',
            help='Save trained models to disk'
        )
        parser.add_argument(
            '--load_checkpoint',
            type=str,
            default=None,
            help='Path to checkpoint to load'
        )
        parser.add_argument(
            '--results',
            type=str,
            default=None,
            help='Path to save results JSON file'
        )
        
        # Logging
        parser.add_argument(
            '--verbose',
            action='store_true',
            default=True,
            help='Enable verbose logging'
        )
        parser.add_argument(
            '--quiet',
            action='store_true',
            help='Disable verbose logging'
        )
        
        # Hyperparameter search
        parser.add_argument(
            '--hparam_search',
            action='store_true',
            help='Enable hyperparameter search'
        )
        parser.add_argument(
            '--hparam_config',
            type=str,
            default=None,
            help='Path to hyperparameter search configuration JSON file'
        )
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = self.build_parser()
        return parser.parse_args(args)
    
    def args_to_config(self, args: argparse.Namespace) -> BaseKGEConfig:
        """
        Convert parsed arguments to configuration object.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Configuration object
        """
        config = self.get_default_config()
        
        # Update config from args
        config.dataset = args.dataset
        config.data_root = args.data_root
        config.train_split = args.train_split
        config.valid_split = args.valid_split
        config.test_split = args.test_split
        config.use_local = not args.use_builtin
        
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.seed = args.seed
        
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
        config.save_dir = args.save_dir
        config.save_models = args.save_models
        config.load_checkpoint = args.load_checkpoint
        config.results_file = args.results
        
        config.verbose = args.verbose and not args.quiet
        
        return config
    
    def parse_model_list(self, model_arg: str) -> List[str]:
        """
        Parse model argument.
        
        Args:
            model_arg: Model string (single model, comma-separated, or "all")
            
        Returns:
            List of model names
        """
        if model_arg.lower() == 'all':
            return list(self.MODEL_CONFIGS.keys())
        
        models = [m.strip() for m in model_arg.split(',')]
        
        # Validate models
        invalid_models = [m for m in models if m not in self.MODEL_CONFIGS]
        if invalid_models:
            raise ValueError(
                f"Invalid model(s): {invalid_models}. "
                f"Available models: {list(self.MODEL_CONFIGS.keys())}"
            )
        
        return models
    
    def run_experiments(
        self,
        models: List[str],
        base_config: BaseKGEConfig,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run training experiments for multiple models.
        
        Args:
            models: List of model names to train
            base_config: Base configuration for training
            **kwargs: Additional framework-specific parameters
            
        Returns:
            List of result dictionaries
        """
        all_results = []
        
        print("\n" + "="*80)
        print(f"RUNNING EXPERIMENTS")
        print("="*80)
        print(f"Models: {', '.join(models)}")
        print(f"Dataset: {base_config.dataset}")
        print(f"Max Epochs: {base_config.epochs}, Batch Size: {base_config.batch_size}")
        if base_config.use_early_stopping:
            print(f"Early Stopping: Enabled (patience={base_config.patience} epochs)")
        else:
            print(f"Early Stopping: Disabled")
        print(f"Device: {base_config.device}")
        if base_config.save_models:
            print(f"Saving models to: {base_config.save_dir}")
        print("="*80 + "\n")
        
        for i, model_name in enumerate(models, 1):
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {i}/{len(models)}: {model_name}")
            print(f"{'='*80}\n")
            
            # Create config for this model
            model_config = BaseKGEConfig.from_dict(base_config.to_dict())
            model_config.model = model_name
            
            # Update with model-specific defaults if not overridden
            if model_name in self.MODEL_CONFIGS:
                model_defaults = self.MODEL_CONFIGS[model_name]
                if base_config.lr is None and 'lr' in model_defaults:
                    model_config.lr = model_defaults['lr']
                if base_config.embedding_dim is None and 'embedding_dim' in model_defaults:
                    model_config.embedding_dim = model_defaults['embedding_dim']
                if base_config.weight_decay is None and 'weight_decay' in model_defaults:
                    model_config.weight_decay = model_defaults['weight_decay']

            model_config = self._prepare_run_config(model_config, model_name)
            
            try:
                # Train the model
                result = self.train_single_model(
                    model_name=model_name,
                    config=model_config,
                    **kwargs
                )
                
                # Store results
                result_summary = {
                    'model': model_name,
                    'dataset': model_config.dataset,
                    'config': model_config.to_dict(),
                    'metrics': result.get('metrics', {}),
                    'timestamp': datetime.now().isoformat(),
                    'run_signature': model_config.run_signature,
                }
                
                if result.get('model_path'):
                    result_summary['model_path'] = str(result['model_path'])
                
                all_results.append(result_summary)
                
                print(f"\n✓ {model_name} training completed successfully!")
                
            except Exception as e:
                print(f"\n✗ Error training {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Store error in results
                all_results.append({
                    'model': model_name,
                    'dataset': model_config.dataset,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                })
        
        # Print summary
        self.print_summary(all_results)
        
        # Save results
        if base_config.results_file:
            self.save_results(all_results, base_config.results_file)
        
        self.results = all_results
        return all_results
    
    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print experiment summary."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        for result in results:
            model = result['model']
            if 'error' in result:
                print(f"\n{model}: FAILED")
                print(f"  Error: {result['error']}")
            else:
                print(f"\n{model}:")
                metrics = result.get('metrics', {})
                if 'mrr' in metrics:
                    print(f"  MRR: {metrics['mrr']:.4f}")
                if 'hits_at_1' in metrics:
                    print(f"  Hits@1: {metrics['hits_at_1']:.4f}")
                if 'hits_at_3' in metrics:
                    print(f"  Hits@3: {metrics['hits_at_3']:.4f}")
                if 'hits_at_10' in metrics:
                    print(f"  Hits@10: {metrics['hits_at_10']:.4f}")
                if 'final_loss' in metrics:
                    print(f"  Final Loss: {metrics['final_loss']:.4f}")
                if 'actual_epochs' in metrics:
                    print(f"  Epochs: {metrics['actual_epochs']}", end='')
                    if metrics.get('early_stopped', False):
                        print(f" (early stopped)")
                    else:
                        print()
        
        print("\n" + "="*80)
        failed_count = sum(1 for r in results if 'error' in r)
        if failed_count > 0:
            print(f"COMPLETED WITH {failed_count}/{len(results)} FAILURES")
        else:
            print(f"ALL {len(results)} EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
    
    def save_results(self, results: List[Dict[str, Any]], filepath: str) -> None:
        """Save results to JSON file."""
        results_path = Path(filepath)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
    
    def load_hparam_config(self, config_path: str) -> Dict[str, List[Any]]:
        """
        Load hyperparameter search configuration from JSON file.
        
        Expected format:
        {
            "models": ["TransE", "ComplEx"],
            "datasets": ["family", "countries_s1"],
            "lr": [0.001, 0.0005],
            "embedding_dim": [256, 512],
            ...
        }
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Dictionary mapping parameter names to lists of values
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def generate_hparam_combinations(
        self,
        hparam_config: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate all combinations of hyperparameters.
        
        Args:
            hparam_config: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of parameter combination dictionaries
        """
        # Separate models and datasets from other params
        models = hparam_config.pop('models', ['TransE'])
        datasets = hparam_config.pop('datasets', ['family'])
        
        # Generate combinations of remaining parameters
        param_names = list(hparam_config.keys())
        param_values = [hparam_config[name] for name in param_names]
        
        combinations = []
        for model in models:
            for dataset in datasets:
                if param_values:
                    for values in product(*param_values):
                        combo = {
                            'model': model,
                            'dataset': dataset,
                        }
                        combo.update(dict(zip(param_names, values)))
                        combinations.append(combo)
                else:
                    combinations.append({
                        'model': model,
                        'dataset': dataset,
                    })
        
        return combinations
    
    def run_hparam_search(
        self,
        hparam_config: Dict[str, List[Any]],
        base_config: BaseKGEConfig,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run hyperparameter search.
        
        Args:
            hparam_config: Hyperparameter search configuration
            base_config: Base configuration
            **kwargs: Additional framework-specific parameters
            
        Returns:
            List of results for all combinations
        """
        combinations = self.generate_hparam_combinations(hparam_config)
        
        print("\n" + "="*80)
        print(f"HYPERPARAMETER SEARCH")
        print("="*80)
        print(f"Total combinations: {len(combinations)}")
        print("="*80 + "\n")
        
        all_results = []
        
        for i, combo in enumerate(combinations, 1):
            print(f"\n{'='*80}")
            print(f"COMBINATION {i}/{len(combinations)}")
            print(f"{'='*80}")
            print(f"Parameters: {combo}")
            print(f"{'='*80}\n")
            
            # Create config for this combination
            trial_config = BaseKGEConfig.from_dict(base_config.to_dict())
            trial_config.update(**combo)

            model_name = combo['model']
            trial_config = self._prepare_run_config(trial_config, model_name)
            
            try:
                result = self.train_single_model(
                    model_name=model_name,
                    config=trial_config,
                    **kwargs
                )
                
                result_summary = {
                    'combination': combo,
                    'metrics': result.get('metrics', {}),
                    'timestamp': datetime.now().isoformat(),
                    'run_signature': trial_config.run_signature,
                }
                
                if result.get('model_path'):
                    result_summary['model_path'] = str(result['model_path'])
                
                all_results.append(result_summary)
                
                print(f"\n✓ Combination {i} completed successfully!")
                
            except Exception as e:
                print(f"\n✗ Error in combination {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                all_results.append({
                    'combination': combo,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                })
        
        # Print hyperparameter search summary
        self.print_hparam_summary(all_results)
        
        return all_results
    
    def print_hparam_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print hyperparameter search summary."""
        print("\n" + "="*80)
        print("HYPERPARAMETER SEARCH SUMMARY")
        print("="*80)
        
        # Sort by MRR if available
        valid_results = [r for r in results if 'error' not in r and 'metrics' in r]
        if valid_results and 'mrr' in valid_results[0].get('metrics', {}):
            valid_results.sort(key=lambda x: x['metrics']['mrr'], reverse=True)
            
            print("\nTop 5 configurations by MRR:")
            for i, result in enumerate(valid_results[:5], 1):
                combo = result['combination']
                metrics = result['metrics']
                print(f"\n{i}. MRR: {metrics['mrr']:.4f}")
                print(f"   Config: {combo}")
                if 'hits_at_10' in metrics:
                    print(f"   Hits@10: {metrics['hits_at_10']:.4f}")
        
        failed_count = sum(1 for r in results if 'error' in r)
        print(f"\n{len(valid_results)}/{len(results)} successful, {failed_count} failed")
        print("="*80 + "\n")
    
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
        
        # Convert to config
        base_config = self.args_to_config(parsed_args)
        
        # Check if hyperparameter search is requested
        if parsed_args.hparam_search:
            if parsed_args.hparam_config is None:
                raise ValueError("--hparam_config must be provided when --hparam_search is enabled")
            
            hparam_config = self.load_hparam_config(parsed_args.hparam_config)
            results = self.run_hparam_search(hparam_config, base_config)
        else:
            # Determine which models to run
            if parsed_args.models:
                models = self.parse_model_list(parsed_args.models)
            else:
                models = [parsed_args.model]
            
            # Run experiments
            results = self.run_experiments(models, base_config)
        
        # Check for failures
        failed_count = sum(1 for r in results if 'error' in r)
        if failed_count > 0:
            sys.exit(1)
        
        return results

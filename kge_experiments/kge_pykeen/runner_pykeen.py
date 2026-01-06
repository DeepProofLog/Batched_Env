"""
PyKeen KGE Runner - Inherits from BaseKGERunner.
Orchestrates training of PyKeen models with hyperparameter search support.
"""
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kge_base_runner import BaseKGERunner, BaseKGEConfig
from pykeen.losses import SoftplusLoss, BCEAfterSigmoidLoss, MarginRankingLoss
from train_pykeen import train_model
import torch


class PyKeenConfig(BaseKGEConfig):
    """Extended configuration for PyKeen models."""
    
    # PyKeen-specific parameters
    loss: str = 'MarginRanking'  # 'Softplus', 'BCE', 'MarginRanking'
    
    # Model-specific parameters
    relation_dim: Optional[int] = None  # For TuckER
    dropout_0: float = 0.3  # For TuckER - input dropout
    dropout_1: float = 0.4  # For TuckER - hidden dropout
    dropout_2: float = 0.5  # For TuckER - output dropout
    scoring_fct_norm: int = 1  # For TransE - L1 or L2 norm


class PyKeenRunner(BaseKGERunner):
    """Runner for PyKeen KGE models."""
    BACKEND_NAME = "pykeen"
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        'ComplEx': {
            'loss': 'BCE',
            'lr': 0.0005,
            'weight_decay': 1e-6,
            'embedding_dim': 512,  # PyKeen doubles this internally (512 -> 1024 total)
            'description': 'Bilinear model - requires BCE/Softplus loss',
            'extra_kwargs': {}
        },
        'RotatE': {
            'loss': 'MarginRanking',
            'lr': 0.001,
            'weight_decay': 0.0,
            'embedding_dim': 500,
            'description': 'Distance-based model - works with MarginRankingLoss',
            'extra_kwargs': {}
        },
        'TransE': {
            'loss': 'MarginRanking',
            'lr': 0.001,
            'weight_decay': 0.0,
            'embedding_dim': 500,
            'description': 'Distance-based translational model - works with MarginRankingLoss',
            'extra_kwargs': {'scoring_fct_norm': 1}  # L1 norm by default
        },
        'TuckER': {
            'loss': 'BCE',
            'lr': 0.0005,
            'weight_decay': 1e-6,
            'embedding_dim': 512,  # Entity embedding dimension
            'description': 'Tucker decomposition model - works with BCE',
            'extra_kwargs': {
                'relation_dim': 256,  # Relation embedding dimension
                'dropout_0': 0.3,  # Input dropout
                'dropout_1': 0.4,  # Hidden dropout
                'dropout_2': 0.5,  # Output dropout
            }
        },
        'DistMult': {
            'loss': 'Softplus',
            'lr': 0.0005,
            'weight_decay': 1e-6,
            'embedding_dim': 1024,
            'description': 'Bilinear model - requires BCE/Softplus loss',
            'extra_kwargs': {}
        },
        'ConvE': {
            'loss': 'BCE',
            'lr': 0.001,
            'weight_decay': 0.0,
            'embedding_dim': 200,
            'description': 'Convolutional model - works with BCE',
            'extra_kwargs': {}
        },
    }
    
    # Loss function mapping
    LOSS_MAP = {
        'Softplus': SoftplusLoss,
        'BCE': BCEAfterSigmoidLoss,
        'MarginRanking': MarginRankingLoss,
    }
    
    def __init__(self, config: Optional[PyKeenConfig] = None):
        """Initialize PyKeen runner."""
        super().__init__(config or PyKeenConfig())
    
    def get_default_config(self) -> PyKeenConfig:
        """Get default configuration for PyKeen."""
        config = PyKeenConfig()
        base_dir = Path(__file__).resolve().parent.parent
        config.data_root = str(base_dir / "data")
        config.save_dir = str(base_dir / "kge_pykeen" / "models")
        return config
    
    def build_parser(self) -> argparse.ArgumentParser:
        """Build argument parser with PyKeen-specific arguments."""
        parser = super().build_parser()
        base_dir = Path(__file__).resolve().parent.parent
        
        # Add PyKeen-specific arguments
        parser.add_argument(
            '--loss',
            type=str,
            default=None,
            choices=['Softplus', 'BCE', 'MarginRanking'],
            help='Loss function (default: model-specific)'
        )
        parser.add_argument(
            '--relation_dim',
            type=int,
            default=None,
            help='Relation embedding dimension (for TuckER)'
        )
        
        # Update description with PyKeen models
        parser.description = 'Train PyKeen KGE models with hyperparameter search support'
        parser.set_defaults(
            data_root=str(base_dir / "data"),
            save_dir=str(base_dir / "kge_pykeen" / "models"),
        )
        
        # Add examples
        parser.epilog = """
Examples:
  # Train a single model
  python runner_pykeen.py --model ComplEx --dataset family
  
  # Train multiple models
  python runner_pykeen.py --models ComplEx,TransE,RotatE --dataset family
  
  # Train all models
  python runner_pykeen.py --models all --dataset wn18rr
  
  # Custom hyperparameters
  python runner_pykeen.py --model ComplEx --dataset family --epochs 2000 --lr 0.001
  
  # Hyperparameter search
  python runner_pykeen.py --hparam_search --hparam_config hparam_config.json
  
  # Save models and results
  python runner_pykeen.py --models all --dataset family --save_models --results results.json
        """
        
        return parser
    
    def args_to_config(self, args: argparse.Namespace) -> PyKeenConfig:
        """Convert parsed arguments to PyKeen configuration."""
        # Start with base config
        config = PyKeenConfig()
        
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
        
        # PyKeen-specific arguments
        if hasattr(args, 'loss') and args.loss is not None:
            config.loss = args.loss
        if hasattr(args, 'relation_dim') and args.relation_dim is not None:
            config.relation_dim = args.relation_dim
        
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
    
    def train_single_model(
        self,
        model_name: str,
        config: PyKeenConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a single PyKeen model.
        
        Args:
            model_name: Name of the model to train
            config: Configuration for training
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Get model-specific config
        base_config = self.MODEL_CONFIGS[model_name].copy()
        
        # Determine loss function
        loss_name = config.loss if hasattr(config, 'loss') else base_config['loss']
        loss_fn = self.LOSS_MAP[loss_name]
        
        # Build model config
        model_config = {
            'loss': loss_fn,
            'lr': config.lr if config.lr is not None else base_config['lr'],
            'weight_decay': config.weight_decay if config.weight_decay is not None else base_config['weight_decay'],
            'embedding_dim': config.embedding_dim if config.embedding_dim is not None else base_config['embedding_dim'],
            'extra_kwargs': base_config.get('extra_kwargs', {}).copy(),
        }
        
        # Add model-specific extra kwargs
        if model_name == 'TuckER' and hasattr(config, 'relation_dim') and config.relation_dim is not None:
            model_config['extra_kwargs']['relation_dim'] = config.relation_dim
        
        # Build training config
        training_config = {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'patience': config.patience,
            'use_early_stopping': config.use_early_stopping,
            'seed': config.seed,
        }
        
        # Setup paths
        data_dir = Path(config.data_root) / config.dataset if config.use_local else None
        save_dir = Path(config.save_dir) if config.save_models else None
        
        if config.verbose:
            print(f"Model Config:")
            print(f"  Description: {base_config['description']}")
            print(f"  Loss: {loss_fn.__name__}")
            print(f"  Learning rate: {model_config['lr']}")
            print(f"  Weight decay: {model_config['weight_decay']}")
            print(f"  Embedding dim: {model_config['embedding_dim']}")
            if model_config['extra_kwargs']:
                print(f"  Extra kwargs: {model_config['extra_kwargs']}")
            print()
        
        # Train the model
        result = train_model(
            model_name=model_name,
            dataset_name=config.dataset,
            model_config=model_config,
            training_config=training_config,
            data_dir=data_dir,
            use_local=config.use_local,
            save_dir=save_dir,
            run_signature=config.run_signature,
            device=config.device,
            verbose=config.verbose
        )
        
        return result
    
    def load_model(self, model_path: str, **kwargs) -> Any:
        """
        Load a trained PyKeen model.
        
        Args:
            model_path: Path to saved model
            **kwargs: Additional parameters
            
        Returns:
            Loaded PyKeen model
        """
        from pykeen.models import Model
        
        model = torch.load(model_path, map_location=kwargs.get('device', 'cpu'))
        return model
    
    def save_model(self, model: Any, save_path: str, **kwargs) -> None:
        """
        Save a trained PyKeen model.
        
        Args:
            model: PyKeen model to save
            save_path: Path where model should be saved
            **kwargs: Additional parameters
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(model, save_path)
        print(f"Model saved to: {save_path}")


def main():
    """Main entry point."""
    runner = PyKeenRunner()
    
    # Print available models
    print("\nAvailable PyKeen models:")
    for model_name, config in runner.MODEL_CONFIGS.items():
        print(f"  - {model_name}: {config['description']}")
    print()
    
    # Run experiments
    runner.run()


if __name__ == '__main__':
    main()

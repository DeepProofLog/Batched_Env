"""
Example: How to extend BaseKGERunner for custom implementations.

This demonstrates creating a custom KGE runner for a hypothetical framework.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import argparse

from kge_base_runner import BaseKGERunner, BaseKGEConfig


class CustomConfig(BaseKGEConfig):
    """
    Extended configuration for custom KGE framework.
    Add framework-specific parameters here.
    """
    
    # Custom parameters
    custom_param_1: float = 0.5
    custom_param_2: str = 'default_value'
    custom_param_3: bool = False
    
    # Override defaults if needed
    lr: float = 0.0005  # Different default LR
    embedding_dim: int = 256  # Different default embedding dim


class CustomKGERunner(BaseKGERunner):
    """
    Custom KGE runner implementation.
    
    This is a template showing how to extend BaseKGERunner
    for your own KGE framework.
    """
    
    # Define model-specific configurations
    MODEL_CONFIGS = {
        'CustomModel1': {
            'lr': 0.001,
            'embedding_dim': 512,
            'custom_param_1': 0.8,
            'description': 'First custom model',
        },
        'CustomModel2': {
            'lr': 0.0005,
            'embedding_dim': 256,
            'custom_param_1': 0.5,
            'custom_param_2': 'special_mode',
            'description': 'Second custom model',
        },
        'CustomModel3': {
            'lr': 0.001,
            'embedding_dim': 1024,
            'custom_param_3': True,
            'description': 'Third custom model with special features',
        },
    }
    
    def __init__(self, config: Optional[CustomConfig] = None):
        """Initialize custom runner."""
        super().__init__(config or CustomConfig())
    
    def get_default_config(self) -> CustomConfig:
        """Get default configuration."""
        return CustomConfig()
    
    def build_parser(self) -> argparse.ArgumentParser:
        """
        Build argument parser with custom-specific arguments.
        
        Override this to add framework-specific CLI arguments.
        """
        # Start with base parser
        parser = super().build_parser()
        
        # Add custom arguments
        parser.add_argument(
            '--custom_param_1',
            type=float,
            default=None,
            help='Custom parameter 1 (default: model-specific)'
        )
        parser.add_argument(
            '--custom_param_2',
            type=str,
            default=None,
            help='Custom parameter 2 (default: model-specific)'
        )
        parser.add_argument(
            '--custom_param_3',
            action='store_true',
            help='Enable custom parameter 3'
        )
        
        # Update description
        parser.description = 'Train Custom KGE models with hyperparameter search'
        
        # Add usage examples
        parser.epilog = """
        Examples:
        # Train single model
        python custom_runner_example.py --model CustomModel1 --dataset family
        
        # Train multiple models
        python custom_runner_example.py --models CustomModel1,CustomModel2 --dataset wn18rr
        
        # Train all models
        python custom_runner_example.py --models all --dataset family
        
        # Hyperparameter search
        python custom_runner_example.py --hparam_search --hparam_config hparam.json
        """
        
        return parser
    
    def args_to_config(self, args: argparse.Namespace) -> CustomConfig:
        """
        Convert parsed arguments to custom configuration.
        
        Override this to handle custom parameters.
        """
        # Start with base conversion
        config = CustomConfig()
        
        # Base parameters
        config.dataset = args.dataset
        config.data_root = args.data_root
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.seed = args.seed
        
        # Custom parameters (only if explicitly provided)
        if hasattr(args, 'custom_param_1') and args.custom_param_1 is not None:
            config.custom_param_1 = args.custom_param_1
        if hasattr(args, 'custom_param_2') and args.custom_param_2 is not None:
            config.custom_param_2 = args.custom_param_2
        if hasattr(args, 'custom_param_3'):
            config.custom_param_3 = args.custom_param_3
        
        # Standard parameters
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
        config.results_file = args.results
        config.verbose = args.verbose and not args.quiet
        
        return config
    
    def train_single_model(
        self,
        model_name: str,
        config: CustomConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a single model using your custom framework.
        
        This is where you implement your actual training logic.
        
        Args:
            model_name: Name of the model to train
            config: Configuration for training
            **kwargs: Additional framework-specific parameters
            
        Returns:
            Dictionary containing:
                - metrics: Dict of evaluation metrics
                - model_path: Path to saved model (optional)
                - any other relevant information
        """
        # Get model-specific defaults
        base_config = self.MODEL_CONFIGS.get(model_name, {})
        
        # Apply model-specific defaults if not overridden
        custom_param_1 = config.custom_param_1
        if 'custom_param_1' in base_config:
            custom_param_1 = base_config['custom_param_1']
        
        if config.verbose:
            print(f"Training {model_name} on {config.dataset}")
            print(f"  LR: {config.lr}")
            print(f"  Embedding dim: {config.embedding_dim}")
            print(f"  Custom param 1: {custom_param_1}")
            print(f"  Custom param 2: {config.custom_param_2}")
            print(f"  Custom param 3: {config.custom_param_3}")
        
        # ============================================
        # YOUR TRAINING CODE GOES HERE
        # ============================================
        
        # Example pseudocode:
        # 1. Load dataset
        # data = load_dataset(config.dataset, config.data_root)
        
        # 2. Create model
        # model = create_model(
        #     model_name,
        #     embedding_dim=config.embedding_dim,
        #     custom_param_1=custom_param_1,
        # )
        
        # 3. Train model
        # trainer = Trainer(
        #     model=model,
        #     data=data,
        #     lr=config.lr,
        #     epochs=config.epochs,
        #     batch_size=config.batch_size,
        # )
        # result = trainer.train()
        
        # 4. Evaluate
        # metrics = evaluate(model, data.test)
        
        # For this example, return dummy metrics
        metrics = {
            'mrr': 0.75,
            'hits_at_1': 0.65,
            'hits_at_3': 0.80,
            'hits_at_10': 0.90,
            'final_loss': 0.25,
            'actual_epochs': config.epochs,
            'early_stopped': False,
        }
        
        # Save model if requested
        model_path = None
        if config.save_models:
            model_path = Path(config.save_dir) / f"{model_name}_{config.dataset}.pt"
            # Save your model here
            # save_model(model, model_path)
            if config.verbose:
                print(f"Model saved to: {model_path}")
        
        # Return results
        return {
            'metrics': metrics,
            'model_path': str(model_path) if model_path else None,
        }
    
    def load_model(self, model_path: str, **kwargs) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to saved model file
            **kwargs: Additional parameters (e.g., device)
            
        Returns:
            Loaded model object
        """
        # ============================================
        # YOUR MODEL LOADING CODE GOES HERE
        # ============================================
        
        # Example pseudocode:
        # model = load_custom_model(model_path, device=kwargs.get('device', 'cpu'))
        # return model
        
        print(f"Loading model from: {model_path}")
        return f"DummyModel(path={model_path})"  # Replace with actual loading
    
    def save_model(self, model: Any, save_path: str, **kwargs) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Model object to save
            save_path: Path where model should be saved
            **kwargs: Additional parameters
        """
        # ============================================
        # YOUR MODEL SAVING CODE GOES HERE
        # ============================================
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Example pseudocode:
        # save_custom_model(model, save_path)
        
        print(f"Model saved to: {save_path}")


def main():
    """Main entry point."""
    runner = CustomKGERunner()
    
    # Print available models
    print("\nAvailable Custom models:")
    for model_name, config in runner.MODEL_CONFIGS.items():
        print(f"  - {model_name}: {config['description']}")
    print()
    
    # Run experiments
    runner.run()


if __name__ == '__main__':
    # Example: Run programmatically instead of CLI
    from kge_base_runner import BaseKGEConfig
    
    
    # Create custom configuration
    config = CustomConfig(
        dataset='family',
        epochs=100,
        batch_size=2048,
        lr=0.001,
        embedding_dim=512,
        custom_param_1=0.75,
        custom_param_2='test_mode',
        custom_param_3=True,
        save_models=True,
        verbose=True,
    )
    
    # Run experiments
    print("\n" + "="*80)
    print("Running Custom KGE Runner Example")
    print("="*80 + "\n")
    
    runner = CustomKGERunner()
    results = runner.run_experiments(['CustomModel1', 'CustomModel2'], config)
    
    # Print results
    print("\nResults:")
    for result in results:
        print(f"\n{result['model']}:")
        if 'error' not in result:
            for key, value in result['metrics'].items():
                print(f"  {key}: {value}")

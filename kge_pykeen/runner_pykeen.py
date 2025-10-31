"""
Runner for PyKeen KGE models.
Orchestrates training of multiple models sequentially with configurable parameters.
"""
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
from pykeen.losses import SoftplusLoss, BCEAfterSigmoidLoss, MarginRankingLoss

from train_pykeen import train_model


# Model-specific configurations
MODEL_CONFIGS = {
    'ComplEx': {
        'loss': BCEAfterSigmoidLoss,
        'lr': 0.0005,
        'weight_decay': 1e-6,
        'embedding_dim': 512,  # PyKeen doubles this internally (512 -> 1024 total)
        'description': 'Bilinear model - requires BCE/Softplus loss',
        'extra_kwargs': {}
    },
    'RotatE': {
        'loss': MarginRankingLoss,
        'lr': 0.001,
        'weight_decay': 0.0,
        'embedding_dim': 500,
        'description': 'Distance-based model - works with MarginRankingLoss',
        'extra_kwargs': {}
    },
    'TransE': {
        'loss': MarginRankingLoss,
        'lr': 0.001,
        'weight_decay': 0.0,
        'embedding_dim': 500,
        'description': 'Distance-based translational model - works with MarginRankingLoss',
        'extra_kwargs': {'scoring_fct_norm': 1}  # L1 norm by default
    },
    'TuckER': {
        'loss': BCEAfterSigmoidLoss,
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
        'loss': SoftplusLoss,
        'lr': 0.0005,
        'weight_decay': 1e-6,
        'embedding_dim': 1024,
        'description': 'Bilinear model - requires BCE/Softplus loss',
        'extra_kwargs': {}
    },
    'ConvE': {
        'loss': BCEAfterSigmoidLoss,
        'lr': 0.001,
        'weight_decay': 0.0,
        'embedding_dim': 200,
        'description': 'Convolutional model - works with BCE',
        'extra_kwargs': {}
    },
}


def parse_model_list(model_arg: str) -> List[str]:
    """
    Parse model argument which can be:
    - Single model: 'ComplEx'
    - Multiple models: 'ComplEx,TransE,RotatE'
    - All models: 'all'
    """
    if model_arg.lower() == 'all':
        return list(MODEL_CONFIGS.keys())
    
    models = [m.strip() for m in model_arg.split(',')]
    
    # Validate models
    invalid_models = [m for m in models if m not in MODEL_CONFIGS]
    if invalid_models:
        raise ValueError(
            f"Invalid model(s): {invalid_models}. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )
    
    return models


def run_experiments(
    models: List[str],
    dataset: str,
    epochs: int,
    batch_size: int,
    embedding_dim: int = None,
    lr: float = None,
    weight_decay: float = None,
    loss: str = None,
    local: bool = True,
    save_models: bool = False,
    results_file: str = None,
    device: str = 'cuda',
    patience: int = 100,
    use_early_stopping: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run experiments for multiple models sequentially.
    
    Args:
        models: List of model names to train
        dataset: Dataset name
        epochs: Maximum number of training epochs
        batch_size: Batch size
        embedding_dim: Override embedding dimension (None = use model default)
        lr: Override learning rate (None = use model default)
        weight_decay: Override weight decay (None = use model default)
        loss: Override loss function (None = use model default)
        local: Use local dataset
        save_models: Whether to save trained models
        results_file: Path to save results JSON
        device: Device to train on
        patience: Early stopping patience in epochs
        use_early_stopping: Whether to use early stopping
    
    Returns:
        List of result dictionaries for each model
    """
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / dataset if local else None
    save_dir = base_dir / 'models' / 'pykeen' if save_models else None
    
    # Loss function mapping
    loss_map = {
        'Softplus': SoftplusLoss,
        'BCE': BCEAfterSigmoidLoss,
        'MarginRanking': MarginRankingLoss,
    }
    
    all_results = []
    
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENTS")
    print("="*80)
    print(f"Models: {', '.join(models)}")
    print(f"Dataset: {dataset}")
    print(f"Max Epochs: {epochs}, Batch Size: {batch_size}")
    if use_early_stopping:
        print(f"Early Stopping: Enabled (patience={patience} epochs)")
    else:
        print(f"Early Stopping: Disabled")
    print(f"Device: {device}")
    if save_models:
        print(f"Saving models to: {save_dir}")
    print("="*80 + "\n")
    
    for i, model_name in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(models)}: {model_name}")
        print(f"{'='*80}\n")
        
        # Get model-specific config
        base_config = MODEL_CONFIGS[model_name].copy()
        
        # Override with command-line arguments if provided
        model_config = {
            'loss': loss_map[loss] if loss else base_config['loss'],
            'lr': lr if lr is not None else base_config['lr'],
            'weight_decay': weight_decay if weight_decay is not None else base_config['weight_decay'],
            'embedding_dim': embedding_dim if embedding_dim is not None else base_config['embedding_dim'],
            'extra_kwargs': base_config.get('extra_kwargs', {}),
        }
        
        training_config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience,
            'use_early_stopping': use_early_stopping,
        }
        
        print(f"Model Config:")
        print(f"  Description: {base_config['description']}")
        print(f"  Loss: {model_config['loss'].__name__}")
        print(f"  Learning rate: {model_config['lr']}")
        print(f"  Weight decay: {model_config['weight_decay']}")
        print(f"  Embedding dim: {model_config['embedding_dim']}")
        if model_config['extra_kwargs']:
            print(f"  Extra kwargs: {model_config['extra_kwargs']}")
        print()
        
        try:
            # Train the model
            result = train_model(
                model_name=model_name,
                dataset_name=dataset,
                model_config=model_config,
                training_config=training_config,
                data_dir=data_dir,
                use_local=local,
                save_dir=save_dir,
                device=device,
                verbose=True
            )
            
            # Store results
            result_summary = {
                'model': model_name,
                'dataset': dataset,
                'config': {
                    'loss': model_config['loss'].__name__,
                    'lr': model_config['lr'],
                    'weight_decay': model_config['weight_decay'],
                    'embedding_dim': model_config['embedding_dim'],
                    'epochs': epochs,
                    'batch_size': batch_size,
                },
                'metrics': result['metrics'],
                'timestamp': datetime.now().isoformat(),
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
                'dataset': dataset,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            })
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for result in all_results:
        model = result['model']
        if 'error' in result:
            print(f"\n{model}: FAILED")
            print(f"  Error: {result['error']}")
        else:
            print(f"\n{model}:")
            metrics = result['metrics']
            print(f"  MRR: {metrics['mrr']:.4f}")
            print(f"  Hits@1: {metrics['hits_at_1']:.4f}")
            print(f"  Hits@3: {metrics['hits_at_3']:.4f}")
            print(f"  Hits@10: {metrics['hits_at_10']:.4f}")
            print(f"  Final Loss: {metrics['final_loss']:.4f}")
            print(f"  Epochs: {metrics['actual_epochs']}", end='')
            if metrics.get('early_stopped', False):
                print(f" (early stopped)")
            else:
                print()
    
    # Save results to file
    if results_file:
        results_path = Path(results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80 + "\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Train multiple KGE models sequentially on local datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a single model
  python runner_pykeen.py --models ComplEx --dataset family
  
  # Train multiple models
  python runner_pykeen.py --models ComplEx,TransE,RotatE --dataset family
  
  # Train all models
  python runner_pykeen.py --models all --dataset wn18rr
  
  # Train with custom parameters
  python runner_pykeen.py --models ComplEx --dataset family --epochs 2000 --lr 0.001
  
  # Disable early stopping
  python runner_pykeen.py --models ComplEx --dataset family --no_early_stopping
  
  # Custom early stopping patience
  python runner_pykeen.py --models all --dataset family --patience 50
  
  # Save models and results
  python runner_pykeen.py --models all --dataset family --save_models --results results.json
        """
    )
    
    # Model selection
    parser.add_argument(
        '--models', 
        type=str, 
        default='RotatE',
        help='Models to train (comma-separated or "all"). Available: %(choices)s',
        metavar='MODELS'
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='family', 
        choices=['wn18rr', 'family', 'countries_s1', 'countries_s2', 'countries_s3', 'fb15k237'],
        help='Dataset to use from data/ folder'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=1500,
        help='Maximum number of training epochs (default: 1500)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=4092,
        help='Training batch size (default: 4092)'
    )
    parser.add_argument(
        '--patience', 
        type=int, 
        default=100,
        help='Early stopping patience in epochs (default: 100)'
    )
    parser.add_argument(
        '--no_early_stopping', 
        action='store_true',
        help='Disable early stopping'
    )
    
    # Model hyperparameters (optional overrides)
    parser.add_argument(
        '--embedding_dim', 
        type=int, 
        default=None,
        help='Override embedding dimension (default: model-specific)'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=None,
        help='Override learning rate (default: model-specific)'
    )
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=None,
        help='Override weight decay (default: model-specific)'
    )
    parser.add_argument(
        '--loss', 
        type=str, 
        default=None,
        choices=['Softplus', 'BCE', 'MarginRanking'],
        help='Override loss function (default: model-specific)'
    )
    
    # Data source
    parser.add_argument(
        '--local', 
        action='store_true',
        default=True,
        help='Use local dataset from data/ folder (default: True)'
    )
    parser.add_argument(
        '--builtin', 
        action='store_true',
        help='Use PyKeen built-in dataset instead of local'
    )
    
    # Output options
    parser.add_argument(
        '--save_models', 
        action='store_true',
        help='Save trained models to disk'
    )
    parser.add_argument(
        '--results', 
        type=str, 
        default=None,
        help='Path to save results JSON file'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to train on (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Parse model list
    try:
        models = parse_model_list(args.models)
    except ValueError as e:
        parser.error(str(e))
    
    # Print available models for reference
    print("\nAvailable models:")
    for model_name, config in MODEL_CONFIGS.items():
        print(f"  - {model_name}: {config['description']}")
    print()
    
    # Run experiments
    results = run_experiments(
        models=models,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        local=not args.builtin,  # Use local unless --builtin is specified
        save_models=args.save_models,
        results_file=args.results,
        device=args.device,
        patience=args.patience,
        use_early_stopping=not args.no_early_stopping,
    )
    
    # Exit with error code if any model failed
    failed_count = sum(1 for r in results if 'error' in r)
    if failed_count > 0:
        print(f"\n⚠ Warning: {failed_count}/{len(results)} models failed")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(results)} models trained successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()


"""
Training module for PyKeen KGE models.
Handles single model training, evaluation, and saving.
"""
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.losses import SoftplusLoss, BCEAfterSigmoidLoss, MarginRankingLoss
from pykeen.stoppers import EarlyStopper


def parse_prolog_file(filepath):
    """Parse Prolog format: relation(entity1,entity2). to numpy array."""
    triples = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Parse: relation(entity1,entity2).
            match = re.match(r'(\w+)\(([^,]+),([^)]+)\)\.?', line)
            if match:
                relation, head, tail = match.groups()
                triples.append([head, relation, tail])
    return np.array(triples, dtype=str)


def train_model(
    model_name: str,
    dataset_name: str,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    data_dir: Optional[Path] = None,
    use_local: bool = True,
    save_dir: Optional[Path] = None,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a single KGE model.
    
    Args:
        model_name: Name of the model (e.g., 'ComplEx', 'TransE')
        dataset_name: Name of the dataset (e.g., 'family', 'wn18rr')
        model_config: Model-specific configuration dict with keys:
            - loss: Loss function class
            - lr: Learning rate
            - weight_decay: Weight decay
            - embedding_dim: Embedding dimension
            - extra_kwargs: Additional model-specific parameters
        training_config: Training configuration dict with keys:
            - epochs: Number of training epochs
            - batch_size: Batch size
            - patience: Early stopping patience (default: 100)
            - use_early_stopping: Whether to use early stopping (default: True)
        data_dir: Path to local dataset directory (if use_local=True)
        use_local: Whether to use local dataset or PyKeen built-in
        save_dir: Directory to save trained model (optional)
        device: Device to train on ('cuda' or 'cpu')
        verbose: Whether to print training progress
    
    Returns:
        Dictionary containing:
            - result: PyKeen pipeline result object
            - metrics: Dictionary of evaluation metrics
            - model_path: Path where model was saved (if save_dir provided)
    """
    
    if verbose:
        print("="*60)
        print(f"Training {model_name} on {dataset_name}")
        print("="*60)
        print(f"Model Config: {model_config}")
        print(f"Training Config: {training_config}")
        print(f"Device: {device}")
        print("="*60)
    
    # Extract configurations
    loss_fn = model_config['loss']
    lr = model_config['lr']
    weight_decay = model_config['weight_decay']
    embedding_dim = model_config['embedding_dim']
    extra_model_kwargs = model_config.get('extra_kwargs', {})
    
    epochs = training_config['epochs']
    batch_size = training_config['batch_size']
    patience = training_config.get('patience', 100)
    use_early_stopping = training_config.get('use_early_stopping', True)
    
    # Prepare model kwargs
    model_kwargs = dict(
        embedding_dim=embedding_dim,
        entity_initializer='xavier_uniform_',
        relation_initializer='xavier_uniform_',
    )
    model_kwargs.update(extra_model_kwargs)
    
    # Setup early stopping
    stopper = None
    if use_early_stopping:
        stopper = EarlyStopper(
            model=model_name,
            patience=patience,
            frequency=1,  # Check every epoch
            metric='loss',  # Monitor training loss
            relative_delta=0.0,  # Any improvement counts
        )
        if verbose:
            print(f"Early stopping enabled: patience={patience} epochs")
            print()
    
    # Train the model
    if use_local:
        if data_dir is None:
            raise ValueError("data_dir must be provided when use_local=True")
        
        # Load local dataset (Prolog format)
        train_path = data_dir / 'train.txt'
        valid_path = data_dir / 'valid.txt'
        test_path = data_dir / 'test.txt'
        
        train_triples = parse_prolog_file(train_path)
        valid_triples = parse_prolog_file(valid_path)
        test_triples = parse_prolog_file(test_path)
        
        training = TriplesFactory.from_labeled_triples(train_triples)
        validation = TriplesFactory.from_labeled_triples(
            valid_triples, 
            entity_to_id=training.entity_to_id, 
            relation_to_id=training.relation_to_id
        )
        testing = TriplesFactory.from_labeled_triples(
            test_triples, 
            entity_to_id=training.entity_to_id, 
            relation_to_id=training.relation_to_id
        )
        
        if verbose:
            print(f"\nDataset Stats:")
            print(f"  Entities: {training.num_entities}")
            print(f"  Relations: {training.num_relations}")
            print(f"  Train triples: {training.num_triples}")
            print(f"  Valid triples: {validation.num_triples}")
            print(f"  Test triples: {testing.num_triples}")
            print()
        
        result = pipeline(
            training=training,
            validation=validation,
            testing=testing,
            model=model_name,
            loss=loss_fn,
            model_kwargs=model_kwargs,
            training_kwargs=dict(
                num_epochs=epochs,
                batch_size=batch_size,
            ),
            optimizer='Adam',
            optimizer_kwargs=dict(
                lr=lr,
                weight_decay=weight_decay,
            ),
            regularizer=None,
            stopper=stopper,  # Add early stopping
            random_seed=42,
            device=device,
        )
    else:
        # Use PyKeen's built-in dataset
        dataset_map = {
            'wn18rr': 'WN18RR',
            'fb15k237': 'FB15k237',
            'kinships': 'Kinships',
        }
        
        pykeen_dataset = dataset_map.get(dataset_name)
        if pykeen_dataset is None:
            raise ValueError(
                f"PyKeen built-in dataset not available for '{dataset_name}'. "
                f"Available: {list(dataset_map.keys())}. Use use_local=True instead."
            )
        
        if verbose:
            print(f"\nLoading PyKeen built-in dataset: {pykeen_dataset}")
        
        result = pipeline(
            dataset=pykeen_dataset,
            model=model_name,
            loss=loss_fn,
            model_kwargs=model_kwargs,
            training_kwargs=dict(
                num_epochs=epochs,
                batch_size=batch_size,
            ),
            optimizer='Adam',
            optimizer_kwargs=dict(
                lr=lr,
                weight_decay=weight_decay,
            ),
            regularizer=None,
            stopper=stopper,  # Add early stopping
            random_seed=42,
            device=device,
        )
    
    # Extract metrics
    metrics = result.metric_results.to_dict()
    mrr = metrics.get("both", {}).get("realistic", {}).get("inverse_harmonic_mean_rank", 0)
    hits_at_10 = metrics.get("both", {}).get("realistic", {}).get("hits_at_10", 0)
    hits_at_1 = metrics.get("both", {}).get("realistic", {}).get("hits_at_1", 0)
    hits_at_3 = metrics.get("both", {}).get("realistic", {}).get("hits_at_3", 0)
    
    if verbose:
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # Show early stopping info if used
        actual_epochs = len(result.losses)
        if use_early_stopping and actual_epochs < epochs:
            print(f"\nEarly stopping triggered after {actual_epochs} epochs (max: {epochs})")
        else:
            print(f"\nCompleted {actual_epochs} epochs")
        
        print(f"\nTest Results:")
        print(f"  MRR: {mrr:.4f}")
        print(f"  Hits@1: {hits_at_1:.4f}")
        print(f"  Hits@3: {hits_at_3:.4f}")
        print(f"  Hits@10: {hits_at_10:.4f}")
        print(f"\nFinal training loss: {result.losses[-1]:.4f}")
    
    # Save model if save_dir is provided
    model_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model-specific subdirectory
        model_subdir = save_dir / f"{dataset_name}_{model_name}"
        model_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_path = model_subdir / "trained_model.pkl"
        result.save_to_directory(str(model_subdir))
        
        if verbose:
            print(f"\nModel saved to: {model_subdir}")
    
    return {
        'result': result,
        'metrics': {
            'mrr': mrr,
            'hits_at_1': hits_at_1,
            'hits_at_3': hits_at_3,
            'hits_at_10': hits_at_10,
            'final_loss': result.losses[-1],
            'actual_epochs': len(result.losses),
            'early_stopped': use_early_stopping and len(result.losses) < epochs,
        },
        'model_path': model_path,
        'model_name': model_name,
        'dataset_name': dataset_name,
    }


if __name__ == '__main__':
    # Example usage
    from pykeen.losses import BCEAfterSigmoidLoss
    
    # Define model config
    model_config = {
        'loss': BCEAfterSigmoidLoss,
        'lr': 0.0005,
        'weight_decay': 1e-6,
        'embedding_dim': 512,
        'extra_kwargs': {}
    }
    
    # Define training config
    training_config = {
        'epochs': 1500,
        'batch_size': 4092,
        'patience': 100,
        'use_early_stopping': True,
    }
    
    # Train model
    data_dir = Path(__file__).parent.parent / 'data' / 'family'
    result_dict = train_model(
        model_name='ComplEx',
        dataset_name='family',
        model_config=model_config,
        training_config=training_config,
        data_dir=data_dir,
        use_local=True,
        device='cuda',
        verbose=True
    )
    
    print("\nTraining completed successfully!")
    print(f"Metrics: {result_dict['metrics']}")

"""
Bayesian Hyperparameter Optimization using Optuna.

Searches over key hyperparameters to maximize test MRR.
Uses the Phase 1 fixes (n_eval_queries=200, randomized sampling, dropout, weight_decay).

Usage:
    python hyperparam_search.py --n_trials 25 --timeout 36000
    python hyperparam_search.py --n_trials 10 --dataset family
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch

# Add parent directory to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    sys.exit(1)

from config import TrainConfig
from train import run_experiment


def create_config_from_trial(trial: Trial, base_config: Dict[str, Any]) -> TrainConfig:
    """Create a TrainConfig from Optuna trial suggestions.

    Args:
        trial: Optuna trial object.
        base_config: Base configuration to modify.

    Returns:
        TrainConfig with suggested hyperparameters.
    """
    # Hyperparameter search space (expanded)
    config_dict = dict(base_config)

    # Architecture
    config_dict['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    config_dict['num_layers'] = trial.suggest_categorical('num_layers', [4, 6, 8])
    config_dict['atom_embedding_size'] = trial.suggest_categorical('atom_embedding_size', [128, 250, 512])

    # Regularization (NOW EFFECTIVE with dropout fix!)
    config_dict['dropout_prob'] = trial.suggest_categorical('dropout_prob', [0.0, 0.1, 0.2, 0.3])

    # Learning dynamics
    config_dict['learning_rate'] = trial.suggest_categorical('learning_rate', [5e-5, 1e-4, 2e-4])
    config_dict['batch_size'] = trial.suggest_categorical('batch_size', [256, 512, 1024])
    config_dict['n_epochs'] = trial.suggest_categorical('n_epochs', [3, 5, 10])

    # PPO-specific
    config_dict['target_kl'] = trial.suggest_categorical('target_kl', [0.05, 0.1, 0.15, 0.2])
    config_dict['clip_range'] = trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3])
    config_dict['gae_lambda'] = trial.suggest_categorical('gae_lambda', [0.9, 0.95, 0.99])
    config_dict['vf_coef'] = trial.suggest_categorical('vf_coef', [0.5, 1.0, 2.0])

    # Exploration
    config_dict['ent_coef_init_value'] = trial.suggest_categorical('ent_coef_init_value', [0.1, 0.15, 0.2, 0.3])
    config_dict['ent_coef_final_value'] = trial.suggest_categorical('ent_coef_final_value', [0.01, 0.02, 0.05])
    config_dict['temperature'] = trial.suggest_categorical('temperature', [0.05, 0.1, 0.2])

    # LR scheduling
    config_dict['lr_warmup'] = True
    config_dict['lr_warmup_steps'] = trial.suggest_float('lr_warmup_steps', 0.05, 0.15)
    config_dict['lr_final_value'] = trial.suggest_categorical('lr_final_value', [1e-7, 1e-6, 1e-5])

    # Fixed settings
    config_dict['n_eval_queries'] = 200
    config_dict['lr_init_value'] = config_dict['learning_rate']
    config_dict['lr_decay'] = True

    # Generate unique run signature
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dict['run_signature'] = f"optuna_trial_{trial.number}_{timestamp}"

    return TrainConfig(**config_dict)


def objective(trial: Trial, base_config: Dict[str, Any], verbose: bool = False) -> float:
    """Optuna objective function to maximize test MRR.

    Args:
        trial: Optuna trial object.
        base_config: Base configuration dictionary.
        verbose: Whether to print progress.

    Returns:
        Test MRR (negative for minimization, but we use direction='maximize').
    """
    try:
        config = create_config_from_trial(trial, base_config)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Trial {trial.number}: lr={config.learning_rate}, hidden={config.hidden_dim}, "
                  f"layers={config.num_layers}, ent_coef={config.ent_coef_init_value}")
            print(f"{'='*60}")

        # Run experiment
        results = run_experiment(config)

        # Extract test MRR
        test_mrr = results.get('MRR', 0.0)
        val_mrr = results.get('mrr_mean', 0.0)

        # Log intermediate values for pruning
        trial.report(test_mrr, step=0)

        # Check for pruning (early stopping for poor trials)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if verbose:
            print(f"\nTrial {trial.number} completed: Test MRR = {test_mrr:.4f}, Val MRR = {val_mrr:.4f}")

        return test_mrr

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return 0.0  # Return low score for failed trials


def run_optimization(
    n_trials: int = 25,
    timeout: Optional[int] = None,
    dataset: str = 'family',
    total_timesteps: int = 5000000,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run Bayesian hyperparameter optimization.

    Args:
        n_trials: Number of trials to run.
        timeout: Maximum time in seconds (None for unlimited).
        dataset: Dataset to use.
        total_timesteps: Training timesteps per trial.
        study_name: Name for the Optuna study.
        storage: Database URL for distributed optimization.
        verbose: Whether to print progress.

    Returns:
        Dictionary with best parameters and results.
    """
    # Base configuration (with Phase 1 fixes)
    base_config = {
        'dataset': dataset,
        'total_timesteps': total_timesteps,
        'n_eval_queries': 200,
        'dropout_prob': 0.1,
        'seed': 0,
        'verbose': verbose,
        'save_model': True,
        'eval_freq': 4,
    }

    # Create or load study
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"hyperparam_search_{dataset}_{timestamp}"

    # Use TPE sampler (default, good for small budgets)
    sampler = optuna.samplers.TPESampler(seed=42)

    # Use median pruner for early stopping
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Don't prune first 5 trials
        n_warmup_steps=0,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',  # Maximize test MRR
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, verbose),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Get best results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    print(f"\n{'='*60}")
    print(f"Optimization complete!")
    print(f"{'='*60}")
    print(f"Best Test MRR: {best_value:.4f}")
    print(f"Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    # Save results
    results_dir = ROOT / 'logs' / 'hyperparam_search'
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'study_name': study_name,
        'best_trial': best_trial.number,
        'best_value': best_value,
        'best_params': best_params,
        'n_trials': len(study.trials),
        'base_config': base_config,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
            }
            for t in study.trials
        ]
    }

    results_path = results_dir / f"{study_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Bayesian Hyperparameter Optimization')
    parser.add_argument('--n_trials', type=int, default=25, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=None, help='Max time in seconds')
    parser.add_argument('--dataset', type=str, default='family', help='Dataset name')
    parser.add_argument('--total_timesteps', type=int, default=5000000, help='Training timesteps')
    parser.add_argument('--study_name', type=str, default=None, help='Study name')
    parser.add_argument('--storage', type=str, default=None, help='Database URL')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')

    args = parser.parse_args()

    results = run_optimization(
        n_trials=args.n_trials,
        timeout=args.timeout,
        dataset=args.dataset,
        total_timesteps=args.total_timesteps,
        study_name=args.study_name,
        storage=args.storage,
        verbose=not args.quiet,
    )

    return results


if __name__ == '__main__':
    main()

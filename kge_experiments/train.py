"""
Training script for Neural-Guided Logical Reasoning (Production).

This module provides the run_experiment function for training using optimized components:
- EnvVec for vectorized environments
- PPO for PPO training
- UnificationEngineVectorized for compiled unification

Usage:
    from train import run_experiment
    from config import TrainConfig
    
    config = TrainConfig(dataset="countries_s3", total_timesteps=1000)
    results = run_experiment(config)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

# Add root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngineVectorized
from env import EnvVec
from nn.embeddings import EmbedderLearnable
from policy import ActorCriticPolicy
from ppo import PPO
from nn.sampler import Sampler
from config import TrainConfig

from callbacks import (
    TorchRLCallbackManager, 
    MetricsCallback, 
    RankingCallback,
    CheckpointCallback,
    ScalarAnnealingCallback, 
    AnnealingTarget
)

from utils import seed_all
from kge_inference import build_kge_inference


def build_callbacks(config, ppo, policy, sampler, dh, eval_env=None, date: str = None):
    """Build callbacks for training.
    
    Callbacks included:
    - MetricsCallback: Always included for logging
    - RankingCallback: If eval_freq > 0 and eval_env provided
    - CheckpointCallback: If save_model is True
    - ScalarAnnealingCallback: If lr_decay or ent_coef_decay enabled
    """
    callbacks = [MetricsCallback(
        log_interval=1, 
        verbose=getattr(config, 'verbose', True), 
        collect_detailed=True
    )]
    best_model_path_train, best_model_path_eval = None, None
    
    save_model = getattr(config, 'save_model', False)
    eval_freq = getattr(config, 'eval_freq', 0)
    
    if save_model or eval_freq > 0:
        models_path = getattr(config, 'models_path', getattr(config, 'model_path', './models/'))
        run_sig = getattr(config, 'run_signature', getattr(config, 'dataset', 'run'))
        save_path = Path(models_path) / run_sig
        best_model_path_train = save_path / "best_model_train.pt"
        best_model_path_eval = save_path / "best_model_eval.pt"
        
    # 1. CheckpointCallback for saving/loading
    if save_model:
        best_metric = getattr(config, 'eval_best_metric', 'mrr_mean')
        if best_metric == 'mrr':
            best_metric = 'mrr_mean'
        callbacks.append(CheckpointCallback(
            save_path=save_path, policy=policy,
            train_metric="ep_rew_mean", eval_metric=best_metric,
            verbose=True, date=date,
            restore_best=getattr(config, 'restore_best', False),
            load_best_metric=getattr(config, 'load_best_metric', 'eval'),
            load_model=getattr(config, 'load_model', False)
        ))

    # 2. RankingCallback for evaluation
    if eval_freq > 0 and eval_env is not None:
        valid_split = dh.get_materialized_split('valid')
        valid_queries = valid_split.queries.squeeze(1)
        valid_depths = valid_split.depths
        n_eval = getattr(config, 'n_eval_queries', None)
        if n_eval:
            valid_queries, valid_depths = valid_queries[:n_eval], valid_depths[:n_eval]
        
        n_corruptions = getattr(config, 'eval_neg_samples')
        scheme = getattr(config, 'corruption_scheme', ('head', 'tail'))
        
        callbacks.append(RankingCallback(
            eval_env=eval_env, policy=policy, sampler=sampler,
            eval_data=valid_queries, eval_data_depths=valid_depths,
            eval_freq=int(eval_freq), n_corruptions=n_corruptions,
            corruption_scheme=tuple(scheme), ppo_agent=ppo
        ))
    
    # 3. ScalarAnnealingCallback for lr/entropy decay
    annealing_targets = []
    total_timesteps = getattr(config, 'timesteps_train', getattr(config, 'total_timesteps', 0))
    
    if getattr(config, 'lr_decay', False):
        lr_init = getattr(config, 'lr_init_value', getattr(config, 'lr', getattr(config, 'learning_rate', 3e-4)))
        lr_final = getattr(config, 'lr_final_value', 1e-6)
        def _set_lr(v): 
            for pg in ppo.optimizer.param_groups:
                pg['lr'] = float(v)
            ppo.learning_rate = float(v)
        annealing_targets.append(AnnealingTarget(
            name='lr', setter=_set_lr, initial=float(lr_init), final=float(lr_final),
            start_point=float(getattr(config, 'lr_start', 0.0)),
            end_point=float(getattr(config, 'lr_end', 1.0)),
            transform=getattr(config, 'lr_transform', 'linear'), value_type='float',
        ))
    
    if getattr(config, 'ent_coef_decay', False):
        ent_init = getattr(config, 'ent_coef_init_value', getattr(config, 'ent_coef', 0.01))
        ent_final = getattr(config, 'ent_coef_final_value', 0.01)
        def _set_ent(v):
            ppo.ent_coef = float(v)
        annealing_targets.append(AnnealingTarget(
            name='ent_coef', setter=_set_ent, initial=float(ent_init), final=float(ent_final),
            start_point=float(getattr(config, 'ent_coef_start', 0.0)),
            end_point=float(getattr(config, 'ent_coef_end', 1.0)),
            transform=getattr(config, 'ent_coef_transform', 'linear'), value_type='float',
        ))
    
    if annealing_targets:
        callbacks.append(ScalarAnnealingCallback(
            total_timesteps=total_timesteps, targets=annealing_targets, verbose=1
        ))
    
    callback_manager = TorchRLCallbackManager(callbacks=callbacks) if callbacks else None
    return callback_manager, best_model_path_train, best_model_path_eval


def create_components(config: TrainConfig) -> Dict[str, Any]:
    """Create training components (data handler, index manager, env, model)."""
    device = torch.device(config.device)
    seed_all(config.seed)
    
    # Data handler
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file, 
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
        corruption_mode="dynamic",
    )
    
    # Index manager
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    
    # Materialize indices
    dh.materialize_indices(im=im, device=device)
    
    # Sampler
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode=config.sampler_default_mode,
        seed=config.seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    # Convert queries
    train_queries_tensor = im.queries_to_tensor(dh.train_queries, device)
    valid_queries_tensor = im.queries_to_tensor(dh.valid_queries, device)

    
    # Create vectorized engine
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=config.padding_atoms,
        parity_mode=False,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
        max_fact_pairs_cap=getattr(config, 'max_fact_pairs_cap', None),
    )
    
    # Create environment
    env = EnvVec(
        vec_engine=vec_engine,
        batch_size=config.n_envs,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_steps,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
        train_queries=train_queries_tensor,
        valid_queries=valid_queries_tensor,
        sample_deterministic_per_env=config.sample_deterministic_per_env,
        sampler=sampler,
        order=False,  # Random query selection (production)
        negative_ratio=config.negative_ratio,
        reward_type=config.reward_type,
        compile=True,
        compile_mode='reduce-overhead',
        compile_fullgraph=True,
        skip_unary_actions=config.skip_unary_actions,  # AAAI26 parity: auto-advance when only 1 action
    )
    
    # Create embedder
    embedder = EmbedderLearnable(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=1000,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=str(device),
    )
    embedder.embed_dim = config.atom_embedding_size
    
    # Create policy
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=config.padding_states,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        parity=False,
    ).to(device)
    
    return {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'embedder': embedder,
        'vec_engine': vec_engine,
        'env': env,
        'policy': policy,
        'device': device,
    }


def run_experiment(config: TrainConfig, return_traces: bool = False) -> Dict[str, Any]:
    """Run full training experiment and return evaluation metrics.
    
    Args:
        config: Training configuration (TrainConfig dataclass).
        return_traces: If True, return detailed traces for debugging.
        
    Returns:
        Dict containing evaluation metrics and optionally traces.
    """
    print("=" * 70)
    print(f"Training: {config.dataset}")
    print(f"Envs: {config.n_envs}, Steps: {config.n_steps}, Timesteps: {config.total_timesteps}")
    print(f"Seed: {config.seed}")
    print("=" * 70)
    
    # Create components
    print("\n[1/3] Creating components...")
    comp = create_components(config)
    
    im = comp['im']
    policy = comp['policy']
    env = comp['env']
    device = comp['device']
    
    # Create PPO
    print("\n[2/3] Running training...")
    
    # Extract metadata for metrics
    query_labels, query_depths = None, None
    dh = comp.get('dh')
    if dh and hasattr(dh, 'get_materialized_split'):
        try:
            train_split = dh.get_materialized_split('train')
            query_depths = train_split.depths
            if hasattr(train_split, 'labels') and train_split.labels is not None:
                query_labels = train_split.labels
        except Exception as e:
            print(f"Warning: Could not extract training metadata: {e}")

    kge_engine = build_kge_inference(config, index_manager=im)
    ppo = PPO(
        policy,
        env,
        config,
        query_labels=query_labels,
        query_depths=query_depths,
        kge_inference_engine=kge_engine,
        kge_index_manager=im,
    )
    
    # Build callbacks
    date = None
    if config.use_callbacks:
        import datetime
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        # Create eval env if needed (reuse training env or create new one)
        # For simplicity, we use the same env instance but could create a dedicated one
        eval_env = env 
        
        callback_manager, _, _ = build_callbacks(
            config, ppo, policy, comp['sampler'], comp['dh'], 
            eval_env=eval_env, date=date
        )
        ppo.callback = callback_manager
     
    # Train
    learn_result = ppo.learn(
        total_timesteps=config.total_timesteps,
        return_traces=return_traces,
    )
    
    # Evaluation
    print("\n[3/3] Running evaluation...")
    policy.eval()
    
    test_queries = comp['dh'].test_queries[:config.n_envs * 4]
    queries_tensor = im.queries_to_tensor(test_queries, device)

    
    eval_results = ppo.evaluate(
        queries=queries_tensor,
        sampler=comp['sampler'],
        n_corruptions=config.test_neg_samples,
        corruption_modes=tuple(config.corruption_scheme),
        query_depths=torch.as_tensor(
            comp['dh'].test_depths[:config.n_envs * 4], 
            dtype=torch.long, device=device
        ),
        verbose=False,
    )
    
    # Extract results
    mrr = eval_results.get('MRR', 0.0)
    hits1 = eval_results.get('Hits@1', 0.0)
    hits3 = eval_results.get('Hits@3', 0.0)
    hits10 = eval_results.get('Hits@10', 0.0)
    
    proven_pos = eval_results.get('proven_pos', 0.0)
    proven_neg = eval_results.get('proven_neg', 0.0)
    print(f"\nResults: MRR={mrr:.4f}, Hits@1={hits1:.4f}, Hits@3={hits3:.4f}, Hits@10={hits10:.4f}")
    print(f"Proven: pos={proven_pos:.4f}, neg={proven_neg:.4f}")
    
    train_stats = getattr(ppo, 'last_train_metrics', {})
    results = {
        "MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10,
        "policy_loss": train_stats.get('policy_loss', 0.0),
        "value_loss": train_stats.get('value_loss', 0.0),
        "entropy": train_stats.get('entropy', 0.0),
    }
    # Merge all detailed eval metrics (e.g. depth breakdown)
    results.update(eval_results)
    
    if return_traces:
        results['rollout_traces'] = learn_result.get('rollout_traces', [])
        results['train_traces'] = learn_result.get('train_traces', [])
    
    return results

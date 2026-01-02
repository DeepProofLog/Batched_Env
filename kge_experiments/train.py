"""
Training script for Neural-Guided Logical Reasoning (Optimized/Compiled Version).

This module provides the run_experiment function for training using optimized components:
- Env_vec (EnvVec) for vectorized environments  
- PPOOptimized for PPO training
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
from typing import List, Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
import numpy as np

# Add root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngineVectorized
from env import EnvVec
from nn.embeddings import EmbedderLearnable as TensorEmbedder
from policy import ActorCriticPolicy as TensorPolicy
from ppo import PPO as PPOOptimized
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

# Parity mode imports (lazy-loaded to avoid circular imports)
def get_parity_modules():
    """Get tensor modules for parity mode (BatchedEnv, UnificationEngine, TensorPPO, tensor_eval_corruptions)."""
    from tensor.tensor_env import BatchedEnv
    from tensor.tensor_unification import UnificationEngine
    from tensor.tensor_ppo import PPO as TensorPPO
    from tensor.tensor_model_eval import eval_corruptions as tensor_eval_corruptions
    return BatchedEnv, UnificationEngine, TensorPPO, tensor_eval_corruptions


# ==============================================================================
# build_callbacks
# ==============================================================================

def build_callbacks(
    config,  # Config object or namespace with relevant attributes
    ppo,
    policy,
    sampler,
    dh,
    eval_env=None,
    date: str = None,
):
    """
    Build callbacks for training. Works with both TrainConfig and args namespace.
    
    Callbacks included:
    - MetricsCallback: Always included for logging
    - RankingCallback: If eval_freq > 0 and eval_env provided
    - CheckpointCallback: If save_model is True
    - ScalarAnnealingCallback: If lr_decay or ent_coef_decay enabled
    """
    callbacks = [MetricsCallback(log_interval=1, verbose=getattr(config, 'verbose', True), collect_detailed=True)]
    best_model_path_train, best_model_path_eval = None, None
    
    save_model = getattr(config, 'save_model', False)
    eval_freq = getattr(config, 'eval_freq', 0)
    
    if save_model or eval_freq > 0:
        # Get save path
        models_path = getattr(config, 'models_path', getattr(config, 'model_path', './models/'))
        run_sig = getattr(config, 'run_signature', getattr(config, 'dataset', 'run'))
        save_path = Path(models_path) / run_sig
        best_model_path_train = save_path / "best_model_train.pt"
        best_model_path_eval = save_path / "best_model_eval.pt"
        
        # RankingCallback for evaluation
        if eval_freq > 0 and eval_env is not None:
            valid_split = dh.get_materialized_split('valid')
            valid_queries = valid_split.queries.squeeze(1)
            valid_depths = valid_split.depths
            n_eval = getattr(config, 'n_eval_queries', None)
            if n_eval:
                valid_queries, valid_depths = valid_queries[:n_eval], valid_depths[:n_eval]
            
            n_corruptions = getattr(config, 'eval_neg_samples', getattr(config, 'n_corruptions', 50))
            scheme = getattr(config, 'corruption_scheme', ('head', 'tail'))
            
            callbacks.append(RankingCallback(
                eval_env=eval_env, policy=policy, sampler=sampler,
                eval_data=valid_queries, eval_data_depths=valid_depths,
                eval_freq=int(eval_freq), n_corruptions=n_corruptions,
                corruption_scheme=tuple(scheme), ppo_agent=ppo
            ))
        
        # CheckpointCallback for saving
        if save_model:
            best_metric = getattr(config, 'eval_best_metric', 'mrr_mean')
            if best_metric == 'mrr': best_metric = 'mrr_mean'
            callbacks.append(CheckpointCallback(
                save_path=save_path, policy=policy,
                train_metric="ep_rew_mean", eval_metric=best_metric,
                verbose=True, date=date
            ))
    
    # ScalarAnnealingCallback for lr/entropy decay
    annealing_targets = []
    total_timesteps = getattr(config, 'timesteps_train', getattr(config, 'total_timesteps', 0))
    
    if getattr(config, 'lr_decay', False):
        lr_init = getattr(config, 'lr_init_value', getattr(config, 'lr', getattr(config, 'learning_rate', 3e-4)))
        lr_final = getattr(config, 'lr_final_value', 1e-6)
        def _set_lr(v): 
            for pg in ppo.optimizer.param_groups: pg['lr'] = float(v)
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
        def _set_ent(v): ppo.ent_coef = float(v)
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


# ==============================================================================
# create_compiled_components
# ==============================================================================

def create_compiled_components(config: TrainConfig) -> Dict[str, Any]:
    """Create compiled training components (data handler, index manager, env, model)."""
    device = torch.device(config.device)
    
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
    
    # Create queries in tensor format
    def convert_queries_to_tensor(queries):
        return torch.stack([
            im.atom_to_tensor(q.predicate, q.args[0], q.args[1]) for q in queries
        ], dim=0)
    
    def convert_queries_to_padded_tensor(queries):
        """Convert queries to padded format for BatchedEnv."""
        query_tensors = []
        for q in queries:
            query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            query_padded = torch.full((config.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
            query_padded[0] = query_atom
            query_tensors.append(query_padded)
        return torch.stack(query_tensors, dim=0)
    
    train_queries_tensor = convert_queries_to_tensor(dh.train_queries)
    test_queries_tensor = convert_queries_to_tensor(dh.test_queries)
    
    # PARITY MODE: Use BatchedEnv (same as tensor_train_parity.py)
    # NON-PARITY: Use EnvVec (optimized/compiled)
    parity_ppo_class = None
    parity_eval_fn = None
    eval_env = None
    if config.parity:
        # Get tensor modules for parity mode
        BatchedEnv, TensorUnificationEngine, TensorPPO, tensor_eval_corruptions = get_parity_modules()
        parity_ppo_class = TensorPPO
        parity_eval_fn = tensor_eval_corruptions
        
        # Get stringifier params for engine
        stringifier_params = {
            'verbose': 0,
            'idx2predicate': im.idx2predicate,
            'idx2constant': im.idx2constant,
            'idx2template_var': im.idx2template_var,
            'padding_idx': im.padding_idx,
            'n_constants': im.constant_no
        }
        
        # Create tensor unification engine (same as tensor_train_parity.py)
        engine = TensorUnificationEngine.from_index_manager(
            im, take_ownership=False,
            stringifier_params=stringifier_params,
            end_pred_idx=im.end_pred_idx,
            end_proof_action=config.end_proof_action,
            max_derived_per_state=config.padding_states,
        )
        engine.index_manager = im
        
        # Create padded queries for BatchedEnv
        train_queries_padded = convert_queries_to_padded_tensor(dh.train_queries)
        
        # Create BatchedEnv (same as tensor_train_parity.py)
        env = BatchedEnv(
            batch_size=config.n_envs,
            queries=train_queries_padded,
            labels=torch.ones(len(dh.train_queries), dtype=torch.long, device=device),
            query_depths=torch.as_tensor(dh.train_depths, dtype=torch.long, device=device),
            unification_engine=engine,
            mode='train',
            max_depth=config.max_steps,
            memory_pruning=config.memory_pruning,
            use_exact_memory=config.use_exact_memory,
            skip_unary_actions=config.skip_unary_actions,
            end_proof_action=config.end_proof_action,
            reward_type=getattr(config, 'reward_type', 0),
            padding_atoms=config.padding_atoms,
            padding_states=config.padding_states,
            true_pred_idx=im.predicate_str2idx.get('True'),
            false_pred_idx=im.predicate_str2idx.get('False'),
            end_pred_idx=im.predicate_str2idx.get('Endf'),
            verbose=0,
            prover_verbose=0,
            device=device,
            runtime_var_start_index=im.constant_no + 1,
            total_vocab_size=im.constant_no + config.max_total_vars,
            sample_deterministic_per_env=config.sample_deterministic_per_env,
            train_neg_ratio=config.negative_ratio,
            corruption_mode=(config.negative_ratio > 0),
            sampler=sampler,
        )
        
        # Create parity eval environment (same as tensor_train_parity.py)
        test_queries_padded = convert_queries_to_padded_tensor(dh.test_queries)
        eval_env = BatchedEnv(
            batch_size=config.n_envs,
            queries=test_queries_padded,
            labels=torch.ones(len(dh.test_queries), dtype=torch.long, device=device),
            query_depths=torch.as_tensor(dh.test_depths, dtype=torch.long, device=device),
            unification_engine=engine,
            mode='eval',
            max_depth=config.max_steps,
            memory_pruning=config.memory_pruning,
            use_exact_memory=config.use_exact_memory,
            skip_unary_actions=config.skip_unary_actions,
            end_proof_action=config.end_proof_action,
            reward_type=getattr(config, 'reward_type', 0),
            padding_atoms=config.padding_atoms,
            padding_states=config.padding_states,
            true_pred_idx=im.predicate_str2idx.get('True'),
            false_pred_idx=im.predicate_str2idx.get('False'),
            end_pred_idx=im.predicate_str2idx.get('Endf'),
            verbose=0,
            prover_verbose=0,
            device=device,
            runtime_var_start_index=im.constant_no + 1,
            total_vocab_size=im.constant_no + config.max_total_vars,
            sample_deterministic_per_env=config.sample_deterministic_per_env,
        )
        
        vec_engine = None  # Not used in parity mode
    else:
        # Create vectorized engine
        vec_engine = UnificationEngineVectorized.from_index_manager(
            im,
            padding_atoms=config.padding_atoms,
            parity_mode=config.parity,
            max_derived_per_state=config.padding_states,
            end_proof_action=config.end_proof_action,
        )
        
        # Clean up index manager tensors
        im.facts_idx = None
        im.rules_idx = None
        im.rule_lens = None
        im.rules_heads_idx = None
        
        # Create EnvVec environment
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
            valid_queries=test_queries_tensor,
            sample_deterministic_per_env=config.sample_deterministic_per_env,
            sampler=sampler,
            order=True if config.parity else False,
            negative_ratio=config.negative_ratio,
            compile=not config.parity,
            compile_mode='reduce-overhead',
            compile_fullgraph=True,
        )
    
    # Create embedder
    torch.manual_seed(config.seed)
    embedder = TensorEmbedder(
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
    torch.manual_seed(config.seed)
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=config.padding_states,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        parity=True,
    ).to(device)
    
    return {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'embedder': embedder,
        'vec_engine': vec_engine,
        'env': env,
        'eval_env': eval_env,  # Parity eval env when parity=True
        'policy': policy,
        'device': device,
        'train_queries_tensor': train_queries_tensor,
        'parity_ppo_class': parity_ppo_class,  # TensorPPO class when parity=True, else None
        'parity_eval_fn': parity_eval_fn,  # tensor_eval_corruptions when parity=True, else None
    }


# ==============================================================================
# run_experiment
# ==============================================================================

def run_experiment(config: TrainConfig, return_traces: bool = False) -> Dict[str, Any]:
    """Run full training experiment and return evaluation metrics.
    
    Args:
        config: Training configuration (TrainConfig dataclass).
        return_traces: If True, return detailed traces for debugging.
        
    Returns:
        Dict containing evaluation metrics and optionally traces.
    """
    print("=" * 70)
    print("COMPILED TRAINING (using Env_vec / PPOOptimized)")
    print(f"Dataset: {config.dataset}, Envs: {config.n_envs}, Steps: {config.n_steps}")
    print(f"Timesteps: {config.total_timesteps}, Seed: {config.seed}")
    print("=" * 70)
    
    # Create components
    print("\n[1/3] Creating compiled components...")
    seed_all(config.seed, deterministic=config.parity)
    comp = create_compiled_components(config)
    
    im = comp['im']
    embedder = comp['embedder']
    policy = comp['policy']
    env = comp['env']
    device = comp['device']
    
    # Checksums for parity verification
    embedder_checksum = sum(p.sum().item() for p in embedder.parameters())
    policy_checksum_init = sum(p.sum().item() for p in policy.parameters())
    print(f"[PARITY] IndexManager: constants={im.constant_no}, predicates={im.predicate_no}")
    print(f"[PARITY] Embedder checksum: {embedder_checksum:.6f}")
    print(f"[PARITY] Policy checksum: {policy_checksum_init:.6f}")
    
    # Create PPO - use TensorPPO in parity mode for exact match with tensor_train_parity
    print("\n[2/3] Running training...")
    seed_all(config.seed, deterministic=config.parity)
    
    parity_ppo_class = comp.get('parity_ppo_class')
    if parity_ppo_class is not None:
        # PARITY MODE: Use TensorPPO (same as tensor_train_parity.py)
        print("[PARITY] Using TensorPPO for exact parity with tensor_train_parity.py")
        ppo = parity_ppo_class(
            policy=policy,
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            n_steps=config.n_steps,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            target_kl=config.target_kl,
            device=device,
            verbose=True,  # Match tensor_train_parity.py
            parity=config.parity,  # Enable parity mode for consistent RNG behavior
        )
    else:
        # NON-PARITY: Use PPOOptimized
        ppo = PPOOptimized(policy, env, config)
    
    # Load model via CheckpointCallback if requested (Now handled in PPO.learn)
    
    if hasattr(ppo, 'callback') and ppo.callback:
        ppo.callback.on_training_start()
    
    # Train
    learn_result = ppo.learn(
        total_timesteps=config.total_timesteps,
        return_traces=return_traces,
    )
    
    policy_checksum_trained = sum(p.sum().item() for p in policy.parameters())
    print(f"[PARITY] Policy checksum after training: {policy_checksum_trained:.6f}")

    
    # Evaluation
    print("\n[3/3] Running evaluation...")
    seed_all(config.seed + 1000, deterministic=config.parity)
    policy.eval()
    
    test_queries = comp['dh'].test_queries[:config.n_envs * 4]
    queries_tensor = torch.stack([
        im.atom_to_tensor(q.predicate, q.args[0], q.args[1]) for q in test_queries
    ], dim=0)
    
    parity_eval_fn = comp.get('parity_eval_fn')
    eval_env = comp.get('eval_env')
    
    if parity_eval_fn is not None and eval_env is not None:
        # PARITY MODE: Use tensor_eval_corruptions (same as tensor_train_parity.py)
        print("[PARITY] Using tensor_eval_corruptions for exact parity")
        eval_results = parity_eval_fn(
            actor=policy,
            env=eval_env,
            queries=queries_tensor,
            sampler=comp['sampler'],
            n_corruptions=config.n_corruptions,
            corruption_modes=tuple(config.corruption_scheme),
            query_depths=torch.as_tensor(
                comp['dh'].test_depths[:config.n_envs * 4], 
                dtype=torch.long, device=device
            ),
            verbose=False,
        )
    else:
        # NON-PARITY: Use PPOOptimized.evaluate
        eval_results = ppo.evaluate(
            queries=queries_tensor,
            sampler=comp['sampler'],
            n_corruptions=config.n_corruptions,
            corruption_modes=tuple(config.corruption_scheme),
            query_depths=torch.as_tensor(
                comp['dh'].test_depths[:config.n_envs * 4], 
                dtype=torch.long, device=device
            ),
            verbose=False,
            parity_mode=config.parity,
        )
    
    # Extract results
    mrr = eval_results.get('MRR', 0.0)
    hits1 = eval_results.get('Hits@1', 0.0)
    hits3 = eval_results.get('Hits@3', 0.0)
    hits10 = eval_results.get('Hits@10', 0.0)
    
    print(f"\n[PARITY] MRR: {mrr:.4f}, Hits@1: {hits1:.4f}, Hits@3: {hits3:.4f}, Hits@10: {hits10:.4f}")
    
    train_stats = getattr(ppo, 'last_train_metrics', {})
    results = {
        "MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10,
        "index_manager_constants": im.constant_no,
        "index_manager_predicates": im.predicate_no,
        "embedder_checksum": embedder_checksum,
        "policy_checksum_init": policy_checksum_init,
        "policy_checksum_trained": policy_checksum_trained,
        "policy_loss": train_stats.get('policy_loss', 0.0),
        "value_loss": train_stats.get('value_loss', 0.0),
        "entropy": train_stats.get('entropy', 0.0),
        "approx_kl": train_stats.get('approx_kl', 0.0),
        "clip_fraction": train_stats.get('clip_fraction', 0.0),
    }
    
    if return_traces:
        results['rollout_traces'] = learn_result.get('rollout_traces', [])
        results['train_traces'] = learn_result.get('train_traces', [])
    
    return results

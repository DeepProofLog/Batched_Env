"""
Training script for Neural-Guided Logical Reasoning (Optimized/Compiled Version).

This module manages the training loop using optimized/compiled components:
- Env_vec (EvalEnvOptimized) instead of BatchedEnv  
- PPOOptimized instead of TensorPPO
- UnificationEngineVectorized for compiled unification

Key Components:
1. **Data Handler**: Loads and processes the knowledge graph data.
2. **Index Manager**: Manages mapping between symbols and integer indices.
3. **Environment**: Creates vectorized environments using Env_vec.
4. **Policy**: Instantiates the Actor-Critic network with Embedder.
5. **PPO**: Runs the Proximal Policy Optimization algorithm (optimized).
6. **Evaluation**: Uses PPOOptimized.evaluate_with_corruptions for ranking.
"""

import gc
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
import cProfile
import pstats
import io

# Add root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handler import DataHandler
from index_manager import IndexManager

from unification import UnificationEngineVectorized
from env import Env_vec as EvalEnvOptimized, EvalObs, EvalState
from embeddings import EmbedderLearnable as TensorEmbedder
from model import ActorCriticPolicy as TensorPolicy
from ppo import PPOOptimized
from sampler import Sampler

from callbacks import (
    TorchRLCallbackManager, 
    MetricsCallback, 
    RankingCallback,
    CheckpointCallback,
    ScalarAnnealingCallback, 
    AnnealingTarget
)

from utils.utils import save_profile_results


# ==============================================================================
# TrainCompiledConfig - Configuration for test_compiled_script.py
# ==============================================================================

@dataclass
class TrainCompiledConfig:
    """Configuration for compiled training (used by test_compiled_script.py)."""
    # Dataset / data files
    dataset: str = "countries_s3"
    data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"
    rules_file: str = "rules.txt"
    facts_file: str = "train.txt"
    train_depth: Any = None
    
    # Environment / padding
    padding_atoms: int = 6
    padding_states: int = 64
    max_steps: int = 20
    use_exact_memory: bool = True
    memory_pruning: bool = True
    skip_unary_actions: bool = True
    end_proof_action: bool = True
    reward_type: int = 0
    max_total_vars: int = 1000
    sample_deterministic_per_env: bool = True
    
    # PPO / training
    n_envs: int = 3
    n_steps: int = 20
    n_epochs: int = 4
    batch_size: int = 20
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    total_timesteps: int = 120
    n_corruptions: int = 10
    corruption_scheme: List[str] = None
    sampler_default_mode: str = "both"
    
    def __post_init__(self):
        if self.corruption_scheme is None:
            if 'countries' in self.dataset or 'ablation' in self.dataset:
                self.corruption_scheme = ['tail']
            else:
                self.corruption_scheme = ['head', 'tail']
    
    # Embedding / model
    atom_embedding_size: int = 64
    
    # Model saving / evaluation
    eval_freq: int = 0
    save_model: bool = False
    model_path: str = "./models/"  # Default relative to run location, but can be updated if needed
    restore_best: bool = True
    
    # Misc
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True
    parity: bool = False


# ==============================================================================
# Seed utilities
# ==============================================================================

def seed_all(seed: int):
    """Set all random seeds."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# _build_data_and_index 
# ==============================================================================

def _build_data_and_index(args: Any, device: torch.device) -> Tuple[DataHandler, IndexManager, Any, Any]:
    """
    Prepare knowledge graph data components and indices.
    
    Initializes the following components:
    1. **DataHandler**: Loads raw triples, rules, and splits.
    2. **IndexManager**: Builds integer mappings for entities, predicates, and variables.
    3. **Sampler**: Constructs the negative sampler and corruptor (with domain info).
    4. **Embedder**: Initializes learnable embeddings for the policy.
    
    Args:
        args (Any): Configuration namespace containing paths and hyperparameters.
        device (torch.device): Target device for tensors.
        
    Returns:
        Tuple[DataHandler, IndexManager, Sampler, Embedder]: Initialized components.
    """

    # Dataset (matching sb3)
    dh = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        train_depth=args.train_depth,
        valid_depth=args.valid_depth,
        test_depth=args.test_depth,
        corruption_mode=args.corruption_mode,
    )
    
    # Respect caps from args (matching sb3)
    args.n_train_queries = (
        len(dh.train_queries)
        if args.n_train_queries is None
        else min(args.n_train_queries, len(dh.train_queries))
    )
    args.n_eval_queries = (
        len(dh.valid_queries)
        if args.n_eval_queries is None
        else min(args.n_eval_queries, len(dh.valid_queries))
    )
    assert args.n_eval_queries > 1, "Number of evaluation queries must be greater than 1 for callbacks."
    args.n_test_queries = (
        len(dh.test_queries)
        if args.n_test_queries is None
        else min(args.n_test_queries, len(dh.test_queries))
    )
    
    # Index manager
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=args.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=args.padding_atoms,
        device=device,
        rules=dh.rules,
    )

    # Materialize indices (tensor-specific)
    dh.materialize_indices(im=im, device=device)

    # Sampler
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode="both",
        seed=args.seed_run_i,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    # Embedder
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=im.variable_no,
        max_arity=dh.max_arity,
        padding_atoms=args.padding_atoms,
        atom_embedder=getattr(args, 'atom_embedder', 'transe'),
        state_embedder=getattr(args, 'state_embedder', 'sum'),
        constant_embedding_size=args.atom_embedding_size,
        predicate_embedding_size=args.atom_embedding_size,
        atom_embedding_size=args.atom_embedding_size,
        device=str(device),
    )
    
    # Derived dims for concat options (matching sb3)
    args.atom_embedding_size = (
        args.atom_embedding_size
        if getattr(args, 'atom_embedder', 'transe') != "concat"
        else (1 + dh.max_arity) * args.atom_embedding_size
    )
    args.state_embedding_size = (
        args.atom_embedding_size
        if getattr(args, 'state_embedder', 'sum') != "concat"
        else args.atom_embedding_size * args.padding_atoms
    )
    embedder.embed_dim = args.state_embedding_size
    
    return dh, im, sampler, embedder


def create_environments(
    args: Any, 
    dh: DataHandler, 
    im: IndexManager, 
    sampler: Sampler, 
    parity: bool = False,
    **kwargs
):
    """
    Create training and evaluation environments using Env_vec (EvalEnvOptimized).
    
    Args:
        args (Any): Configuration namespace.
        dh (DataHandler): Data handler with query splits.
        im (IndexManager): Index manager for symbol mapping.
        sampler (Sampler): Negative sampler.
        parity (bool): If True, use parity mode for exact matching with tensor version.
        **kwargs: Extensible keyword arguments.
        
    Returns:
        Tuple[EvalEnvOptimized, EvalEnvOptimized, EvalEnvOptimized]: 
            (train_env, eval_env, callback_env). 
    """
    device = torch.device(args.device)
    
    # Create stringifier params
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    # Create vectorized engine directly (no base_engine dependency)
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=args.padding_atoms,
        parity_mode=parity,
        max_derived_per_state=args.padding_states,
        end_proof_action=args.end_proof_action,
    )
    
    # Clean up index manager tensors to save memory (same as take_ownership=True)
    im.facts_idx = None
    im.rules_idx = None
    im.rule_lens = None
    im.rules_heads_idx = None
    
    # Use pre-materialized tensors from DataHandler
    train_split = dh.get_materialized_split('train')
    valid_split = dh.get_materialized_split('valid')
    test_split = dh.get_materialized_split('test')

    # Convert queries to [N, 3] format (squeeze the padding dimension)
    train_queries_tensor = train_split.queries.squeeze(1)  # [N, A, 3] -> [N, 3]
    valid_queries_tensor = valid_split.queries.squeeze(1)
    test_queries_tensor = test_split.queries.squeeze(1)
    
    batch_size = args.batch_size_env

    # Train environment using Env_vec
    train_env = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=batch_size,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        max_depth=args.max_depth,
        end_proof_action=args.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=args.memory_pruning,
        queries=train_queries_tensor,
        sample_deterministic_per_env=args.sample_deterministic_per_env,
    )
    
    # Eval environment using Env_vec (for test queries)
    eval_env = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=batch_size,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        max_depth=args.max_depth,
        end_proof_action=args.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=args.memory_pruning,
        queries=test_queries_tensor,
        sample_deterministic_per_env=args.sample_deterministic_per_env,
    )
    
    # Callback environment using Env_vec (for valid queries)
    callback_env = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=batch_size,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        max_depth=args.max_depth,
        end_proof_action=args.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=args.memory_pruning,
        queries=valid_queries_tensor,
        sample_deterministic_per_env=args.sample_deterministic_per_env,
    )
    
    return train_env, eval_env, callback_env


def _evaluate(
    args: Any, 
    ppo: PPOOptimized,
    sampler, 
    dh: DataHandler, 
    im: IndexManager, 
    device: torch.device
) -> Tuple[dict, dict, dict]:
    """
    Evaluate the policy on the test set corrupted queries using PPOOptimized.
    
    Uses PPOOptimized.evaluate_with_corruptions for MRR and Hits@K evaluation.
    
    Args:
        args (Any): Configuration.
        ppo (PPOOptimized): PPO instance with compiled policy.
        sampler (Sampler): Negative sampler.
        dh (DataHandler): Data handler.
        im (IndexManager): Index manager.
        device (torch.device): Compute device.
        
    Returns:
        Tuple[Dict, Dict, Dict]: (train_metrics, valid_metrics, test_metrics).
    """
    print("\nTest set evaluation...")
    
    ppo.policy.eval()
    
    # Get test queries
    n_test = args.n_test_queries
    test_queries = dh.get_materialized_split('test').queries
    test_queries = test_queries[:n_test].squeeze(1)  # [N, 3]
    
    print(f"Evaluating {len(test_queries)} queries...")
    
    # Use PPOOptimized.evaluate_with_corruptions instead of tensor_eval_corruptions
    eval_results = ppo.evaluate_with_corruptions(
        queries=test_queries,
        sampler=sampler,
        n_corruptions=args.test_neg_samples,
        corruption_modes=tuple(args.corruption_scheme),
        verbose=0,
    )
    
    def _parse_metric(val):
        """Parse metric value (may be string like '0.792 +/- 0.41')."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            if '+/-' in val:
                try:
                    return float(val.split('+/-')[0].strip())
                except (ValueError, IndexError):
                    pass
            try:
                return float(val)
            except ValueError:
                pass
        return 0.0
    
    # Build metrics dict matching expected format
    per_mode = eval_results.get('per_mode', {})
    metrics_test = {
        'mrr_mean': _parse_metric(eval_results.get('MRR', 0.0)),
        'hits1_mean': _parse_metric(eval_results.get('Hits@1', 0.0)),
        'hits3_mean': _parse_metric(eval_results.get('Hits@3', 0.0)),
        'hits10_mean': _parse_metric(eval_results.get('Hits@10', 0.0)),
        'rewards_pos_mean': _parse_metric(eval_results.get('reward_pos_mean', 0.0)),
        'rewards_neg_mean': _parse_metric(eval_results.get('reward_neg_mean', 0.0)),
        'reward_label_pos': _parse_metric(eval_results.get('reward_label_pos', 0.0)),
        'reward_label_neg': _parse_metric(eval_results.get('reward_label_neg', 0.0)),
        'success_rate': _parse_metric(eval_results.get('success_rate', 0.0)),
        'reward_overall': eval_results.get('reward_overall', ''),
        'proven_pos': eval_results.get('proven_pos', ''),
        'proven_neg': eval_results.get('proven_neg', ''),
        # Per-mode metrics
        'mrr_tail_mean': _parse_metric(per_mode.get('tail', {}).get('MRR', 0.0)),
        'mrr_head_mean': _parse_metric(per_mode.get('head', {}).get('MRR', 0.0)),
        'hits1_tail_mean': _parse_metric(per_mode.get('tail', {}).get('Hits@1', 0.0)),
        'hits1_head_mean': _parse_metric(per_mode.get('head', {}).get('Hits@1', 0.0)),
        'hits3_tail_mean': _parse_metric(per_mode.get('tail', {}).get('Hits@3', 0.0)),
        'hits3_head_mean': _parse_metric(per_mode.get('head', {}).get('Hits@3', 0.0)),
        'hits10_tail_mean': _parse_metric(per_mode.get('tail', {}).get('Hits@10', 0.0)),
        'hits10_head_mean': _parse_metric(per_mode.get('head', {}).get('Hits@10', 0.0)),
    }
    
    # Add depth-based metrics if available
    for key in eval_results.keys():
        if key.startswith('len_d_') or key.startswith('proven_d_') or key.startswith('reward_d_'):
            metrics_test[key] = eval_results[key]
    
    print(f"results for: {getattr(args, 'run_signature', 'compiled')}") 
    print("\nTest set metrics:")
    for k in sorted(metrics_test.keys()):
        v = metrics_test[k]
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")
    
    # Placeholder for train/valid
    metrics_train = {k: 0 for k in metrics_test.keys()}
    metrics_valid = {k: 0 for k in metrics_test.keys()}
    
    return metrics_train, metrics_valid, metrics_test


def build_callbacks(
    args, 
    eval_env, 
    policy, 
    sampler, 
    dh, 
    index_manager, 
    ppo,
    date: str
):
    """
    Constructs and returns the callback manager and paths to best models.
    """
    # Build schedule configs for lr and entropy if enabled
    annealing_targets = []
    
    if getattr(args, 'lr_decay', False):
        lr_init = getattr(args, 'lr_init_value', args.lr)
        lr_final = getattr(args, 'lr_final_value', 1e-6)
        
        def _set_lr(value: float) -> None:
            for param_group in ppo.optimizer.param_groups:
                param_group['lr'] = float(value)
            ppo.learning_rate = float(value)

        annealing_targets.append(AnnealingTarget(
            name='lr',
            setter=_set_lr,
            initial=float(lr_init),
            final=float(lr_final),
            start_point=float(getattr(args, 'lr_start', 0.0)),
            end_point=float(getattr(args, 'lr_end', 1.0)),
            transform=getattr(args, 'lr_transform', 'linear'),
            value_type='float',
        ))
        print(f"LR Decay: {lr_init} -> {lr_final}")
    
    if getattr(args, 'ent_coef_decay', False):
        ent_init = getattr(args, 'ent_coef_init_value', args.ent_coef)
        ent_final = getattr(args, 'ent_coef_final_value', 0.01)
        
        def _set_ent_coef(value: float) -> None:
            ppo.ent_coef = float(value)

        annealing_targets.append(AnnealingTarget(
            name='ent_coef',
            setter=_set_ent_coef,
            initial=float(ent_init),
            final=float(ent_final),
            start_point=float(getattr(args, 'ent_coef_start', 0.0)),
            end_point=float(getattr(args, 'ent_coef_end', 1.0)),
            transform=getattr(args, 'ent_coef_transform', 'linear'),
            value_type='float',
        ))
        print(f"Entropy Decay: {ent_init} -> {ent_final}")

    callbacks_list = []
    
    # 1. Training metrics callback
    callbacks_list.append(MetricsCallback(
        log_interval=1,
        verbose=True,
        collect_detailed=True
    ))

    # 2. Evaluation callback (for finding Best Model)
    best_model_path_train = None
    best_model_path_eval = None
    
    if getattr(args, 'save_model', False):
        save_path = Path(args.models_path) / args.run_signature
        
        best_model_path_train = save_path / "best_model_train.pt"
        best_model_path_eval = save_path / "best_model_eval.pt"
        
        # Prepare Eval Data for RankingCallback
        valid_split = dh.get_materialized_split('valid')
        valid_queries_tensor = valid_split.queries.squeeze(1)
        valid_depths_tensor = valid_split.depths
        
        n_eval = getattr(args, 'n_eval_queries', None)
        if n_eval:
            valid_queries_tensor = valid_queries_tensor[:n_eval]
            valid_depths_tensor = valid_depths_tensor[:n_eval]
           
        best_metric_key = getattr(args, 'eval_best_metric', 'mrr_mean')
        if best_metric_key == 'mrr': best_metric_key = 'mrr_mean'
        if best_metric_key == 'auc_pr': best_metric_key = 'auc_pr'

        # Ranking Callback
        ranking_cb = RankingCallback(
            eval_env=eval_env,
            policy=policy,
            sampler=sampler,
            eval_data=valid_queries_tensor,
            eval_data_depths=valid_depths_tensor,
            eval_freq=int(args.eval_freq),
            n_corruptions=args.eval_neg_samples,
            corruption_scheme=tuple(args.corruption_scheme),
            ppo_agent=ppo
        )
        callbacks_list.append(ranking_cb)
        
        # Checkpoint Callback
        ckpt_cb = CheckpointCallback(
            save_path=save_path,
            policy=policy,
            best_model_name_train="best_model_train.pt",
            best_model_name_eval="best_model_eval.pt",
            train_metric="ep_rew_mean",
            eval_metric=best_metric_key,
            verbose=True,
            date=date
        )
        callbacks_list.append(ckpt_cb)

        # Annealing
        if annealing_targets:
            callbacks_list.append(ScalarAnnealingCallback(
                total_timesteps=args.timesteps_train,
                targets=annealing_targets,
                verbose=1
            ))

    callback_manager = None
    if callbacks_list:
        callback_manager = TorchRLCallbackManager(callbacks=callbacks_list)
        
    return callback_manager, best_model_path_train, best_model_path_eval


# ==============================================================================
# create_compiled_components - For run_experiment
# ==============================================================================

def create_compiled_components(config: TrainCompiledConfig) -> Dict[str, Any]:
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
    
    # Create stringifier params
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    # Create vectorized engine directly
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
    
    # Convert queries to tensor format
    train_queries = dh.train_queries
    test_queries = dh.test_queries
    
    def convert_queries_to_tensor(queries):
        query_tensors = []
        for q in queries:
            query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            query_tensors.append(query_atom)
        return torch.stack(query_tensors, dim=0)
    
    train_queries_tensor = convert_queries_to_tensor(train_queries)
    test_queries_tensor = convert_queries_to_tensor(test_queries)
    
    # Create environments using Env_vec
    train_env = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=config.n_envs,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_steps,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
        queries=train_queries_tensor,
        sample_deterministic_per_env=config.sample_deterministic_per_env,
    )
    
    eval_env = EvalEnvOptimized(
        vec_engine=vec_engine,
        batch_size=config.n_envs,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_steps,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
        queries=test_queries_tensor,
        sample_deterministic_per_env=config.sample_deterministic_per_env,
    )
    
    # Create embedder with fixed seed
    n_vars_for_embedder = 1000
    torch.manual_seed(config.seed)
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=n_vars_for_embedder,
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
    
    # Create policy with fixed seed
    action_size = config.padding_states
    torch.manual_seed(config.seed)
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        parity=True,  # Use SB3-identical initialization for parity testing
    ).to(device)
    
    return {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'embedder': embedder,
        'vec_engine': vec_engine,
        'train_env': train_env,
        'eval_env': eval_env,
        'policy': policy,
        'device': device,
        'train_queries_tensor': train_queries_tensor,
    }


# ==============================================================================
# run_experiment - Required by test_compiled_script.py
# ==============================================================================

def run_experiment(config: TrainCompiledConfig) -> Dict[str, float]:
    """Run full training experiment and return evaluation metrics."""
    print("=" * 70)
    print("COMPILED TRAINING (using Env_vec / PPOOptimized)")
    print(f"Dataset: {config.dataset}")
    print(f"N envs: {config.n_envs}, N steps: {config.n_steps}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Seed: {config.seed}")
    print("=" * 70)
    
    # Create compiled components
    print("\n[1/3] Creating compiled components...")
    seed_all(config.seed)
    comp = create_compiled_components(config)
    
    # [PARITY] Output IndexManager info
    im = comp['im']
    print(f"[PARITY] IndexManager: constants={im.constant_no}, predicates={im.predicate_no}")
    
    # [PARITY] Output Embedder checksum
    embedder = comp['embedder']
    embedder_checksum = sum(p.sum().item() for p in embedder.parameters())
    print(f"[PARITY] Embedder checksum: {embedder_checksum:.6f}")
    
    # [PARITY] Output Policy init checksum
    policy = comp['policy']
    policy_checksum_init = sum(p.sum().item() for p in policy.parameters())
    print(f"[PARITY] Policy checksum after creation: {policy_checksum_init:.6f}")
    
    # [PARITY] Output RNG state before sampler
    print(f"[PARITY] RNG state before sampler: {torch.get_rng_state().sum().item():.0f}")
    
    # Compile the policy for the environment
    # When parity=True, use fullgraph=False because parity_mode uses functions with @torch.compiler.disable
    train_env = comp['train_env']
    compile_fullgraph = not config.parity
    train_env.compile(policy, fullgraph=compile_fullgraph, mode='default' if config.parity else 'reduce-overhead')
    
    # Create PPOOptimized
    print("\n[2/3] Running training...")
    seed_all(config.seed)
    ppo = PPOOptimized(
        policy=policy,
        env=train_env,
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        gamma=config.gamma,
        target_kl=config.target_kl,
        device=comp['device'],
        verbose=True,
        parity=config.parity,
        # query_labels=comp['dh'].get_materialized_split('train').labels,
        # query_depths=comp['dh'].get_materialized_split('train').depths,
    )
    ppo.learn(total_timesteps=config.total_timesteps, queries=comp['train_queries_tensor'])
    
    # [PARITY] Output Policy trained checksum
    policy_checksum_trained = sum(p.sum().item() for p in policy.parameters())
    print(f"[PARITY] Policy checksum after training: {policy_checksum_trained:.6f}")
    
    # Evaluation
    print("\n[3/3] Running evaluation...")
    seed_all(config.seed + 1000)
    
    print(f"[PARITY] RNG before eval: {torch.get_rng_state().sum().item():.0f}")
    
    policy.eval()
    
    # Compile eval env (same settings as train_env)
    eval_env = comp['eval_env']
    eval_env.compile(policy, fullgraph=compile_fullgraph, mode='default' if config.parity else 'reduce-overhead')
    
    test_queries = comp['dh'].test_queries[:config.n_envs * 4]
    
    query_atoms = []
    for q in test_queries:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_atoms.append(query_atom)
    queries_tensor = torch.stack(query_atoms, dim=0)
    
    print(f"\n[COMPILED EVAL DEBUG]")
    print(f"  corruption_scheme: {config.corruption_scheme}")
    print(f"  n_corruptions: {config.n_corruptions}")
    print(f"  num test queries: {len(test_queries)}")
    print(f"  sampler default_mode: {config.sampler_default_mode}")
    print(f"  first query: {test_queries[0]}")
    
    # Use PPOOptimized with eval_env for evaluation
    eval_ppo = PPOOptimized(
        policy=policy,
        env=eval_env,
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        device=comp['device'],
        verbose=False,
        eval_only=True,  # Skip buffer allocation for eval
    )
    
    eval_results = eval_ppo.evaluate_with_corruptions(
        queries=queries_tensor,
        sampler=comp['sampler'],
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_scheme),
        query_depths=torch.as_tensor(comp['dh'].test_depths[:config.n_envs * 4], 
                                      dtype=torch.long, device=comp['device']),
        verbose=False,
        parity_mode=config.parity,
    )
    
    # Extract results
    mrr = eval_results.get('MRR', 0.0)
    hits1 = eval_results.get('Hits@1', 0.0)
    hits3 = eval_results.get('Hits@3', 0.0)
    hits10 = eval_results.get('Hits@10', 0.0)
    
    print(f"\n[PARITY] Evaluation Results:")
    print(f"[PARITY] Compiled MRR: {mrr:.4f}")
    print(f"[PARITY] Compiled Hits@1: {hits1:.4f}")
    print(f"[PARITY] Compiled Hits@3: {hits3:.4f}")
    print(f"[PARITY] Compiled Hits@10: {hits10:.4f}")
    
    # Get training stats from PPO
    train_stats = getattr(ppo, 'last_train_metrics', {})
    
    # Comprehensive results dict
    results = {
        # Evaluation metrics
        "MRR": mrr,
        "Hits@1": hits1,
        "Hits@3": hits3,
        "Hits@10": hits10,
        # Checksums
        "index_manager_constants": im.constant_no,
        "index_manager_predicates": im.predicate_no,
        "embedder_checksum": embedder_checksum,
        "policy_checksum_init": policy_checksum_init,
        "policy_checksum_trained": policy_checksum_trained,
        # Training losses (from last epoch)
        "policy_loss": train_stats.get('policy_loss', 0.0),
        "value_loss": train_stats.get('value_loss', 0.0),
        "entropy": train_stats.get('entropy', 0.0),
        "approx_kl": train_stats.get('approx_kl', 0.0),
        "clip_fraction": train_stats.get('clip_fraction', 0.0),
    }
    
    return results


# ==============================================================================
# main - Legacy entry point for runner.py compatibility
# ==============================================================================

def main(args, log_filename, use_logger, use_WB, WB_path, date, external_components=None, profile_run=False):
    """
    Main training entry point.

    Steps:
    1. Check reproducibility settings.
    2. Set random seeds.
    3. Initialize data components (DataHandler, IndexManager, Sampler, Embedder).
    4. Create environments (Train, Eval).
    5. Initialize Policy and PPO algorithm.
    6. Run training loop (`ppo.learn`).
    7. Run evaluation (`_evaluate`).
    
    Args:
        args (Namespace): Parsed command-line arguments.
        log_filename (str): Path to log file.
        use_logger (bool): Whether to enable logging.
        use_WB (bool): Whether to use Weights & Biases.
        WB_path (str): W&B run path.
        date (str): Timestamp string.
        external_components (Optional[Dict]): Pre-initialized components.
    
    Returns:
        Tuple[Dict, Dict, Dict]: Metrics (train, valid, test).
    """
    
    device = torch.device(args.device)
    print(f"Device: {device}. CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")
    
    # Step 1: build important components
    dh, index_manager, sampler, embedder = _build_data_and_index(args, device)

    # Step 2: Create environments using Env_vec
    parity = getattr(args, 'parity', False)
    env, eval_env, callback_env = create_environments(
        args,
        dh,
        index_manager,
        sampler,
        parity=parity,
    )

    # Step 3: Create policy
    action_size = args.padding_states
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=args.state_embedding_size,
        action_dim=action_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_prob=args.dropout_prob,
        device=device,
        temperature=getattr(args, 'temperature', 1.0),
        use_l2_norm=getattr(args, 'use_l2_norm', True),
        sqrt_scale=getattr(args, 'sqrt_scale', False),
    ).to(device)
    
    # Compile policy for the environment
    env.compile(policy)
    
    # Create PPOOptimized
    ppo = PPOOptimized(
        policy=policy,
        env=env,
        n_steps=args.n_steps,
        learning_rate=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        target_kl=args.target_kl,
        device=device,
        verbose=1,
        seed=args.seed_run_i,
        parity=parity,
        query_labels=dh.get_materialized_split('train').labels,
        query_depths=dh.get_materialized_split('train').depths,
    )
    
    # Configure callbacks
    callback_manager, best_model_path_train, best_model_path_eval = build_callbacks(
        args, eval_env, policy, sampler, dh, index_manager, ppo, date
    )
    callbacks_list = callback_manager.callbacks if callback_manager else []


    # Step 4: Train
    if args.timesteps_train > 0 and not getattr(args, 'load_model', False): 
        cb_func = callback_manager if callback_manager else None
        iteration_start_cb = callback_manager.on_iteration_start if callback_manager else None
        step_cb = callback_manager.on_step if callback_manager else None
        
        if callback_manager:
            callback_manager.on_training_start()
            
        if profile_run:
            print(f"\nProfiling PPO.learn() for {args.timesteps_train} timesteps...")
            profiler = cProfile.Profile()
            profiler.enable()

        # Get train queries for PPO.learn
        train_split = dh.get_materialized_split('train')
        train_queries = train_split.queries.squeeze(1)  # [N, 3]

        ppo.learn(
            total_timesteps=args.timesteps_train, 
            queries=train_queries,
            callback=cb_func,
            on_iteration_start_callback=iteration_start_cb,
            on_step_callback=step_cb
        )

        if profile_run:
            profiler.disable()
            print("\nProfiling completed.")
            save_profile_results(profiler, args, device)

    
    # Restore best model if configured
    save_model = getattr(args, 'save_model', False)
    restore_best = getattr(args, 'restore_best_val_model', True)
    load_metric = getattr(args, 'load_best_metric', 'eval')
    training_occurred = args.timesteps_train > 0 and not getattr(args, 'load_model', False)
        

    if training_occurred and save_model and restore_best and callback_manager:
        ckpt_cb = next((cb for cb in callbacks_list if isinstance(cb, CheckpointCallback)), None)
        if ckpt_cb:
            ckpt_cb.load_best_model(load_metric=load_metric, device=device)

    # Step 5: Evaluate
    # Compile eval_env for evaluation
    eval_env.compile(policy)
    metrics_train, metrics_valid, metrics_test = _evaluate(
        args, ppo, sampler, dh, index_manager, device
    )
    
    return metrics_train, metrics_valid, metrics_test


def main_cli():
    """Command-line interface for train.py (standalone use)."""
    import argparse
    parser = argparse.ArgumentParser(description="Train Compiled (Optimized)")
    parser.add_argument("--dataset", type=str, default="countries_s3")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=2000)
    parser.add_argument("--n-corruptions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eval-freq", type=int, default=0,
                        help="Evaluate every N timesteps (0=only at end)")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="Save model checkpoints")
    parser.add_argument("--model-path", type=str, default="./models/",
                        help="Path to save models")
    parser.add_argument("--no-restore-best", action="store_true", default=False,
                        help="Don't restore best model after training")
    
    args = parser.parse_args()
    
    config = TrainCompiledConfig(
        dataset=args.dataset,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        total_timesteps=args.total_timesteps,
        n_corruptions=args.n_corruptions,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        eval_freq=args.eval_freq,
        save_model=args.save_model,
        model_path=args.model_path,
        restore_best=not args.no_restore_best,
    )
    
    run_experiment(config)


if __name__ == "__main__":
    main_cli()

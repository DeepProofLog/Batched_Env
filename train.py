"""
Training script for Neural-Guided Logical Reasoning (Batched Version).

This module manages the training loop for the Agent

Key Components:
1. **Data Handler**: Loads and processes the knowledge graph data.
2. **Index Manager**: Manages mapping between symbols and integer indices.
3. **Environment**: Creates batched logical reasoning environments.
4. **Policy**: Instantiates the Actor-Critic network with Embedder.
5. **PPO**: Runs the Proximal Policy Optimization algorithm.
6. **Evaluation**: Periodically evaluates performance on test queries.
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
from unification import UnificationEngine
from env import BatchedEnv
from embeddings import EmbedderLearnable as TensorEmbedder
from model import ActorCriticPolicy as TensorPolicy
from ppo import PPO as TensorPPO
from model_eval import eval_corruptions as tensor_eval_corruptions
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
        default_mode="both", # this allows the sampler to corrupt both head and tail by default. Can be overridden per-eval.
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


def create_environments(args: Any, dh: DataHandler, im: IndexManager, sampler: Sampler, **kwargs):
    """
    Create training and evaluation environments.
    
    Constructs `BatchedEnv` instances for training (using `train.txt` queries)
    and evaluation (using `test.txt` queries). Configures the `UnificationEngine`
    and other environment parameters (padding, depth, reward type).
    
    Args:
        args (Any): Configuration namespace.
        dh (DataHandler): Data handler with query splits.
        im (IndexManager): Index manager for symbol mapping.
        **kwargs: Extensible keyword arguments.
        
    Returns:
        Tuple[BatchedEnv, BatchedEnv, BatchedEnv]: 
            (train_env, eval_env, callback_env). 
            Note: callback_env is typically aliased to eval_env.
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
    
    # Create unification engine
    engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=args.end_proof_action,
        max_derived_per_state=args.padding_states,
    )
    engine.index_manager = im
    
    # Use pre-materialized tensors from DataHandler
    train_split = dh.get_materialized_split('train')
    valid_split = dh.get_materialized_split('valid')
    test_split = dh.get_materialized_split('test')

    train_queries_tensor = train_split.queries
    valid_queries_tensor = valid_split.queries
    test_queries_tensor = test_split.queries
    
    batch_size = args.batch_size_env
    
    stringifier_params = im.get_stringifier_params()

    # Train environment
    train_env = BatchedEnv(
        batch_size=batch_size,
        queries=train_queries_tensor,
        labels=torch.ones(len(dh.train_queries), dtype=torch.long, device=device),
        query_depths=torch.as_tensor(dh.train_depths, dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',
        max_depth=args.max_depth,
        memory_pruning=args.memory_pruning,
        use_exact_memory=args.use_exact_memory,
        skip_unary_actions=args.skip_unary_actions,
        end_proof_action=args.end_proof_action,
        reward_type=args.reward_type,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=args.verbose_env,
        prover_verbose=args.prover_verbose,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + args.max_total_vars,
        sample_deterministic_per_env=args.sample_deterministic_per_env,
        sampler=sampler,
        train_neg_ratio=args.train_neg_ratio,
        corruption_mode=True,
        stringifier_params=stringifier_params,
    )
    
    # Eval environment
    eval_env = BatchedEnv(
        batch_size=batch_size,
        queries=test_queries_tensor,
        labels=torch.ones(len(dh.test_queries), dtype=torch.long, device=device),
        query_depths=torch.as_tensor(dh.test_depths, dtype=torch.long, device=device),
        unification_engine=engine,
        mode='eval',
        max_depth=args.max_depth,
        memory_pruning=args.memory_pruning,
        use_exact_memory=args.use_exact_memory,
        skip_unary_actions=args.skip_unary_actions,
        end_proof_action=args.end_proof_action,
        reward_type=args.reward_type,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=args.verbose_env,
        prover_verbose=args.prover_verbose,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + args.max_total_vars,
        sample_deterministic_per_env=args.sample_deterministic_per_env,
        stringifier_params=stringifier_params,
    )
    
    callback_env = BatchedEnv(
        batch_size=batch_size,
        queries=valid_split.queries,
        labels=torch.ones(len(dh.valid_queries), dtype=torch.long, device=device),
        query_depths=torch.as_tensor(dh.valid_depths, dtype=torch.long, device=device),
        unification_engine=engine,
        mode='eval',
        max_depth=args.max_depth,
        memory_pruning=args.memory_pruning,
        use_exact_memory=args.use_exact_memory,
        skip_unary_actions=args.skip_unary_actions,
        end_proof_action=args.end_proof_action,
        reward_type=args.reward_type,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=args.verbose_env,
        prover_verbose=args.prover_verbose,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + args.max_total_vars,
        sample_deterministic_per_env=args.sample_deterministic_per_env,
        stringifier_params=stringifier_params,
    )
    return train_env, eval_env, callback_env


def _evaluate(args: Any, policy, eval_env, sampler, dh: DataHandler, im: IndexManager, device: torch.device) -> Tuple[dict, dict, dict]:
    """
    Evaluate the policy on the test set corrupted queries.
    
    Performs MRR (Mean Reciprocal Rank) and Hits@K evaluation.
    1. Reseeds RNG for deterministic evaluation (if configured).
    2. Selects test queries.
    3. Runs `tensor_eval_corruptions` (vectorized ranking).
    4. Formats metrics for logging.
    
    Args:
        args (Any): Configuration.
        policy (ActorCriticPolicy): Trained policy network.
        eval_env (BatchedEnv): Evaluation environment.
        sampler (Sampler): Negative sampler.
        dh (DataHandler): Data handler.
        im (IndexManager): Index manager.
        device (torch.device): Compute device.
        
    Returns:
        Tuple[Dict, Dict, Dict]: (train_metrics, valid_metrics, test_metrics).
            Note: train and valid metrics are currently placeholders (zeros).
    """
    print("\nTest set evaluation...")
    
    policy.eval()
    
    # Get test queries
    n_test = args.n_test_queries
    test_queries = dh.get_materialized_split('test').queries
    test_queries = test_queries[:n_test].squeeze(1)
    
    print(f"Evaluating {len(test_queries)} queries...")
    eval_results = tensor_eval_corruptions(
        actor=policy,
        env=eval_env,
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
    
    # Build metrics dict matching sb3 format
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
        # Per-mode metrics (renamed to mrr_head/mrr_tail convention)
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
    
    print(f"results for: {getattr(args, 'run_signature', 'tensor')}")
    print("\nTest set metrics:")
    # Print metrics in alphabetical order
    for k in sorted(metrics_test.keys()):
        v = metrics_test[k]
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")
    
    # Placeholder for train/valid (matching sb3 - not evaluated by default)
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
            # Update the learning rate for all parameter groups
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
    # Collects aggregated and detailed stats (depth/label) and prints every log_interval
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
        
        # Define paths for potential best models
        best_model_path_train = save_path / "best_model_train.pt"
        best_model_path_eval = save_path / "best_model_eval.pt"
        
        # Prepare Eval Data for RankingCallback
        valid_split = dh.get_materialized_split('valid')
        valid_queries_tensor = valid_split.queries.squeeze(1)
        valid_depths_tensor = valid_split.depths
        
        # Use subset if configured
        n_eval = getattr(args, 'n_eval_queries', None)
        if n_eval:
            valid_queries_tensor = valid_queries_tensor[:n_eval]
            valid_depths_tensor = valid_depths_tensor[:n_eval]
           
        best_metric_key = getattr(args, 'eval_best_metric', 'mrr_mean')
        # Map nice name to callback output key
        if best_metric_key == 'mrr': best_metric_key = 'mrr_mean'
        if best_metric_key == 'auc_pr': best_metric_key = 'auc_pr'

        # Ranking Callback (Evaluates MRR)
        ranking_cb = RankingCallback(
            eval_env=eval_env,
            policy=policy,
            sampler=sampler,
            eval_data=valid_queries_tensor,
            eval_data_depths=valid_depths_tensor,
            eval_freq=int(args.n_steps * args.eval_freq),
            n_corruptions=args.eval_neg_samples,
            corruption_scheme=tuple(args.corruption_scheme)
        )
        callbacks_list.append(ranking_cb)
        
        # Checkpoint Callback (Saves best models)
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
        # Create manager
        callback_manager = TorchRLCallbackManager(callbacks=callbacks_list)
        
    return callback_manager, best_model_path_train, best_model_path_eval

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
        external_components (Optional[Dict]): Pre-initialized components (dh, im, sampler, embedder)
                                              for dependency injection during testing.
    
    Returns:
        Tuple[Dict, Dict, Dict]: Metrics (train, valid, test).
    """
    
    device = torch.device(args.device)
    print(f"Device: {device}. CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")
    
    # Step 1: build important components
    dh, index_manager, sampler, embedder = _build_data_and_index(args, device)

    
    # Step 2: Create environments (matching sb3)
    env, eval_env, callback_env = create_environments(
        args,
        dh,
        index_manager,
        sampler,
    )

    
    # Step 3: Create policy/PPO (matching sb3 flow)
    action_size = args.padding_states
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=args.state_embedding_size,
        action_dim=action_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_prob=args.dropout_prob,
        device=device,
        temperature=getattr(args, 'temperature', 1.0),  # Temperature for entropy control
        use_l2_norm=getattr(args, 'use_l2_norm', True),  # L2 norm for cosine similarity
        sqrt_scale=getattr(args, 'sqrt_scale', False),  # sqrt(E) attention scaling
    ).to(device)
    
    ppo = TensorPPO(
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
        target_kl=args.target_kl,  # Early stopping threshold (aligned with SB3)
        device=device,
        verbose=1,
        seed=args.seed_run_i,  # For RNG synchronization between rollouts
        use_amp=args.use_amp,
        total_timesteps=args.timesteps_train,  # For schedule computation
        use_compile=args.use_compile,
        debug_ppo=getattr(args, 'debug_ppo', False),  # Enable detailed training diagnostics
    )
    
    # Configure callbacks
    callback_manager, best_model_path_train, best_model_path_eval = build_callbacks(
        args, eval_env, policy, sampler, dh, index_manager, ppo, date
    )
    callbacks_list = callback_manager.callbacks if callback_manager else []


    # Step 4: Train (matching sb3 flow)
    if args.timesteps_train > 0 and not getattr(args, 'load_model', False): 
        cb_func = callback_manager if callback_manager else None
        iteration_start_cb = callback_manager.on_iteration_start if callback_manager else None
        step_cb = callback_manager.on_step if callback_manager else None
        
        # Ensure initial values are set before training starts
        if callback_manager:
            callback_manager.on_training_start()
            
        if profile_run:
            print(f"\nProfiling PPO.learn() for {args.timesteps_train} timesteps...")
            profiler = cProfile.Profile()
            profiler.enable()

        ppo.learn(
            total_timesteps=args.timesteps_train, 
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
        # Find ckpt callback
        ckpt_cb = next((cb for cb in callbacks_list if isinstance(cb, CheckpointCallback)), None)
        if ckpt_cb:
            ckpt_cb.load_best_model(load_metric=load_metric, device=device)

    # Step 8: Evaluate (matching sb3)
    metrics_train, metrics_valid, metrics_test = _evaluate(
        args, policy, eval_env, sampler, dh, index_manager, device
    )
    
    return metrics_train, metrics_valid, metrics_test




if __name__ == "__main__":
    main_cli()

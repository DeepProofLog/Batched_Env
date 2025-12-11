"""
Training script for Neural-Guided Logical Reasoning (Batched Version).

This module manages the training loop for the Agent, ensuring functional parity
with the SB3 implementation where applicable.

Key Components:
1. **Data Handler**: Loads and processes the knowledge graph data.
2. **Index Manager**: Manages mapping between symbols and integer indices.
3. **Environment**: Creates batched logical reasoning environments.
4. **Policy**: Instantiates the Actor-Critic network with Embedder.
5. **PPO**: Runs the Proximal Policy Optimization algorithm.
6. **Evaluation**: Periodically evaluates performance on test queries.

Usage:
    Run directly or via `test_runner_simple.py` for parity checks.
    
    ```bash
    python train.py --dataset countries_s3
    ```
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

from utils.seeding import seed_all
from callbacks import (
    TorchRLCallbackManager, 
    MRREvaluationCallback, 
    TrainingMetricsCallback, 
    ScalarAnnealingCallback, 
    AnnealingTarget
)
try:
    from training_stability import MRRTracker
    MRR_TRACKER_AVAILABLE = True
except ImportError:
    MRR_TRACKER_AVAILABLE = False


def _set_seeds(seed: int) -> None:
    """Match sb3_utils._set_seeds exactly."""
    seed_all(seed,
            deterministic=False,
            deterministic_cudnn=False)

def _warn_non_reproducible(args: Any) -> None:
    """Match sb3_utils._warn_non_reproducible."""
    if getattr(args, 'restore_best_val_model', True) is False:
        print(
            "Warning: This setting is not reproducible when creating 2 models from scratch, "
            "but it is when loading pretrained models."
        )


def get_device(device: str = "auto") -> torch.device:
    """Match sb3_utils.get_device."""
    if device == "auto":
        device = "cuda"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


# ==============================================================================
# _build_data_and_index - MATCHING sb3_train._build_data_and_index exactly
# ==============================================================================

def _build_data_and_index(args: Any, device: torch.device) -> Tuple[DataHandler, IndexManager, Any, Any]:
    """
    Prepare knowledge graph data components and indices.
    
    Initializes the following components in a deterministic order for SB3 parity:
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
    
    # Convert queries to tensor format
    def convert_queries(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            padded = torch.full((args.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
            padded[0] = atom
            tensors.append(padded)
        return torch.stack(tensors, dim=0)
    
    train_queries_tensor = convert_queries(dh.train_queries)
    test_queries_tensor = convert_queries(dh.test_queries)
    
    batch_size = args.batch_size_env
    
    # DEBUG: Check predicate mapping
    print(f"[Create Environments] Predicate Mapping: True={im.predicate_str2idx.get('True')}, False={im.predicate_str2idx.get('False')}, Endf={im.predicate_str2idx.get('Endf')}")

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
    
    # Return train_env, eval_env, callback_env (matching sb3 signature)
    callback_env = eval_env  # Use eval_env for callbacks
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
    test_queries = dh.test_queries
    n_test = args.n_test_queries
    test_queries = test_queries[:n_test]
    
    # Convert queries to tensor
    query_atoms = []
    for q in test_queries:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_atoms.append(query_atom)
    queries_tensor = torch.stack(query_atoms, dim=0)
    
    print(f"Evaluating {len(test_queries)} queries...")
    eval_results = tensor_eval_corruptions(
        actor=policy,
        env=eval_env,
        queries=queries_tensor,
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



def main(args, log_filename, use_logger, use_WB, WB_path, date, external_components=None):
    """
    Main training entry point.
    
    Orchestrates the entire training pipeline, designed to match the control flow
    of the reference SB3 implementation `sb3_train.py` exactly for parity verification.
    
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
    
    # Step 3: Get device (matching sb3)
    device = get_device(args.device)
    print(f"Device: {device}. CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")
    
    # Build pieces - use external components if provided (for parity testing)
    if external_components is not None:
        dh = external_components['dh']
        index_manager = external_components['index_manager']
        sampler = external_components['sampler']
        embedder = external_components['embedder']
    else:
        dh, index_manager, sampler, embedder = _build_data_and_index(args, device)

    
    # Step 5: Create environments (matching sb3)
    env, eval_env, callback_env = create_environments(
        args,
        dh,
        index_manager,
        sampler,
    )

    
    # Step 6: Create policy/PPO (matching sb3 flow)
    action_size = args.padding_states
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=args.state_embedding_size,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        temperature=getattr(args, 'temperature', 1.0),  # Temperature for entropy control
        use_l2_norm=getattr(args, 'use_l2_norm', True),  # L2 norm for cosine similarity
        sqrt_scale=getattr(args, 'sqrt_scale', False),  # sqrt(E) attention scaling
        # parity=False is default - uses custom value_head initialization for better training stability
    ).to(device)

    
    
    # Get schedule parameters if available
    gae_lambda = getattr(args, 'gae_lambda', 0.95)
    vf_coef = getattr(args, 'vf_coef', 0.5)
    clip_range_vf = getattr(args, 'clip_range_vf', None)
    
    # Build schedule configs for lr and entropy if enabled
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
    
    ppo = TensorPPO(
        policy=policy,
        env=env,
        n_steps=args.n_steps,
        learning_rate=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=args.ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=getattr(args, 'max_grad_norm', 0.5),
        gamma=args.gamma,
        gae_lambda=gae_lambda,
        target_kl=args.target_kl,  # Early stopping threshold (aligned with SB3)
        device=device,
        verbose=1,
        seed=args.seed_run_i,  # For RNG synchronization between rollouts
        use_amp=getattr(args, 'use_amp', True),
        total_timesteps=args.timesteps_train,  # For schedule computation
        use_compile=getattr(args, 'use_compile', True),
        debug_ppo=getattr(args, 'debug_ppo', False),  # Enable detailed training diagnostics
    )
    
    # NOTE: Initial evaluation commented out for SB3 parity.
    # Running eval here consumes RNG for negative sampling BEFORE training,
    # but SB3's EvalCallback runs AFTER the first rollout. This causes rollout
    # data divergence. For exact parity, skip initial eval.
    # 
    # Step 6.5: Initial evaluation with untrained model (matching sb3 eval callback at step 0)
    print("\n" + "="*60)
    print("Initial evaluation (untrained model)")
    print("="*60)
    policy.eval()
    initial_eval_results = tensor_eval_corruptions(
        actor=policy,
        env=eval_env,
        queries=torch.stack([
            index_manager.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            for q in dh.valid_queries[:getattr(args, 'n_eval_queries', 10) or 10]
        ]),
        sampler=sampler,
        n_corruptions=getattr(args, 'eval_neg_samples', 10) or 10,
        corruption_modes=tuple(getattr(args, 'corruption_scheme', ['tail'])),
        verbose=0,
    )
    print(f"Initial MRR: {initial_eval_results.get('MRR', 0.0):.4f}")
    print(f"Initial Hits@1: {initial_eval_results.get('Hits@1', 0.0):.4f}")
    print(f"Initial success_rate: {initial_eval_results.get('success_rate', 0.0):.4f}")
    print("="*60 + "\n")
    
    # Configure callbacks (PARITY)
    # We manually construct the callback system to match SB3's functionality
    callback_manager = None
    callbacks_list = []
    
    # 1. Training metrics callback
    callbacks_list.append(TrainingMetricsCallback(
        log_interval=1,
        verbose=True,
        collect_detailed=True
    ))

    # 2. Evaluation callback (for finding Best Model)
    best_model_path = None
    if getattr(args, 'save_model', False):
        save_path = Path(args.models_path) / args.run_signature
        best_model_path = save_path / "best_model.pt"
        
        # Create evaluation callback
        # Get validation queries as tensor
        valid_queries_tensor = torch.stack([
            index_manager.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            for q in dh.valid_queries
        ])
        
        # Use full validation set or subset
        n_eval = getattr(args, 'n_eval_queries', None)
        if n_eval:
           valid_queries_tensor = valid_queries_tensor[:n_eval]
           
        best_metric = getattr(args, 'eval_best_metric', 'mrr_mean')
        if best_metric == 'mrr':
            best_metric = 'mrr_mean'
        
        # Get validation depths
        valid_depths = torch.as_tensor(dh.valid_depths, dtype=torch.long, device=device)
        if n_eval:
            valid_depths = valid_depths[:n_eval]
            
        eval_cb = MRREvaluationCallback(
            eval_env=eval_env,
            sampler=sampler,
            eval_data=valid_queries_tensor,
            eval_data_depths=valid_depths,
            n_corruptions=args.eval_neg_samples,
            eval_freq=1,  # Evaluate every iteration for best model tracking
            best_metric=best_metric,
            save_path=save_path,
            model_name="model",
            verbose=True,
            policy=policy,  # Pass policy to enable saving
            corruption_scheme=args.corruption_scheme,
        )
        if annealing_targets:
            callbacks_list.append(ScalarAnnealingCallback(
                total_timesteps=args.timesteps_train,
                targets=annealing_targets,
                verbose=1
            ))
            
        callbacks_list.append(eval_cb)

    if callbacks_list:
        # Create manager

        callback_manager = TorchRLCallbackManager(
            train_callback=next((cb for cb in callbacks_list if isinstance(cb, TrainingMetricsCallback)), None),
            eval_callback=next((cb for cb in callbacks_list if isinstance(cb, MRREvaluationCallback)), None)
        )
        
        # Register other callbacks
        for cb in callbacks_list:
            if isinstance(cb, ScalarAnnealingCallback):
                callback_manager.add_callback(cb)

        # Initialize MRR tracker for monitoring training progress
        mrr_tracker = MRRTracker(patience=20) if MRR_TRACKER_AVAILABLE else None

        # Create wrapper for PPO.learn
        def ppo_callback(locals_, globals_):
            # Extract info needed by manager
            iteration = locals_.get('iteration', 0)
            total_steps = locals_.get('total_steps_done', 0)
            
            # Evaluation callback (saves best model during training)
            if callback_manager.eval_callback:
                if callback_manager.eval_callback.should_evaluate(iteration):
                    callback_manager.on_evaluation_start(iteration, total_steps)
                    mrr_metrics = callback_manager.eval_callback.evaluate_mrr(policy)
                    callback_manager.on_evaluation_end(iteration, total_steps, mrr_metrics)
                    
                    # Track MRR for monitoring
                    if mrr_tracker is not None:
                        current_mrr = mrr_metrics.get('_mrr', 0.0)
                        if isinstance(current_mrr, str):
                            try:
                                current_mrr = float(current_mrr)
                            except ValueError:
                                current_mrr = 0.0
                        track_result = mrr_tracker.update(current_mrr, iteration)
                        
                        # Log MRR tracking info
                        if track_result['is_best']:
                            print(f"[MRR] New best: {track_result['best_mrr']:.4f} at iteration {iteration}")
                        print(f"[MRR] {mrr_tracker.get_summary()}")

            # Training metrics callback
            if callback_manager.train_callback:
                 callback_manager.train_callback.on_iteration_end(iteration, total_steps, n_envs=env.batch_size)

            return True

    # Step 7: Train (matching sb3 flow)
    if args.timesteps_train > 0 and not getattr(args, 'load_model', False): 
        cb_func = ppo_callback if callback_manager else None
        iteration_start_cb = callback_manager.on_iteration_start if callback_manager else None
        
        # Ensure initial values are set before training starts
        if callback_manager:
            callback_manager.on_training_start()
            
        ppo.learn(
            total_timesteps=args.timesteps_train, 
            callback=cb_func,
            on_iteration_start_callback=iteration_start_cb
        )

    
    # Restore best model if configured (PARITY with SB3)
    # Only restore if training actually occurred (timesteps_train > 0)
    save_model = getattr(args, 'save_model', False)
    restore_best = getattr(args, 'restore_best_val_model', True)
    training_occurred = args.timesteps_train > 0 and not getattr(args, 'load_model', False)
        
    if training_occurred and save_model and restore_best and best_model_path and best_model_path.exists():
        print(f"Restored best val model from {best_model_path}")
        policy.load_state_dict(torch.load(best_model_path, map_location=device))

    # Step 8: Evaluate (matching sb3)
    metrics_train, metrics_valid, metrics_test = _evaluate(
        args, policy, eval_env, sampler, dh, index_manager, device
    )
    
    return metrics_train, metrics_valid, metrics_test




if __name__ == "__main__":
    main_cli()

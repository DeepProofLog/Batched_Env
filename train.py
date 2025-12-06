"""
Training script following the EXACT same logical flow as sb3_train.py.
Uses tensor-based components (BatchedEnv, TensorPPO, etc.) but with
matching initialization sequence for parity testing.
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

# Import shared seeding from utils (same as sb3 uses)
try:
    from utils.seeding import seed_all
except ImportError:
    # Fallback implementation matching sb3_utils._set_seeds
    def seed_all(seed: int, deterministic_cudnn: bool = False):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ==============================================================================
# Configuration dataclass (for test_runner_simple compatibility)
# ==============================================================================

@dataclass
class TrainParityConfig:
    """Configuration for train parity tests."""
    # Dataset / data files
    dataset: str = "countries_s3"
    data_path: str = "./data/"
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
    
    # PPO / training
    n_envs: int = 3
    n_steps: int = 20
    n_epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 120
    n_corruptions: int = 10
    
    # Embedding / model
    atom_embedding_size: int = 64
    
    # Misc
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True


# ==============================================================================
# Matching sb3_utils functions exactly
# ==============================================================================

def _set_seeds(seed: int) -> None:
    """Match sb3_utils._set_seeds exactly."""
    seed_all(seed, deterministic_cudnn=False)
    print(f"ensuring determinism in the torch algorithm")


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
    """Prepare DataHandler, IndexManager, sampler and embedder.
    
    MUST MATCH sb3_train._build_data_and_index initialization order exactly:
    1. Create DataHandler
    2. Create IndexManager  
    3. Create sampler
    4. RESEED before embedder creation (torch.manual_seed(args.seed_run_i))
    5. Create embedder
    """
    # PARITY: Reseed at start for deterministic alignment
    deterministic_parity = getattr(args, 'deterministic_parity', False)
    if deterministic_parity:
        _set_seeds(args.seed_run_i)
    
    # Dataset (matching sb3)
    dh = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        train_depth=getattr(args, 'train_depth', None),
        valid_depth=getattr(args, 'valid_depth', None),
        test_depth=getattr(args, 'test_depth', None),
        corruption_mode=getattr(args, 'corruption_mode', 'dynamic'),
    )
    print(f"DEBUG: DataHandler n_train_queries={getattr(args, 'n_train_queries', None)}")
    print(f"DEBUG: DataHandler predicates ({len(dh.predicates)}): {sorted(list(dh.predicates))}")
    
    # Respect caps from args (matching sb3)
    args.n_train_queries = (
        len(dh.train_queries)
        if getattr(args, 'n_train_queries', None) is None
        else min(args.n_train_queries, len(dh.train_queries))
    )
    args.n_eval_queries = (
        len(dh.valid_queries)
        if getattr(args, 'n_eval_queries', None) is None
        else min(args.n_eval_queries, len(dh.valid_queries))
    )
    assert args.n_eval_queries > 1, "Number of evaluation queries must be greater than 1 for callbacks."
    args.n_test_queries = (
        len(dh.test_queries)
        if getattr(args, 'n_test_queries', None) is None
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
    
    # CRITICAL: Reseed before embedder creation (matching sb3_train line 162)
    # "Seed is already set at the beginning of main(), but reseed to align with torchrl stack"
    torch.manual_seed(args.seed_run_i)
    
    # Embedder
    n_vars_for_embedder = 1000
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=n_vars_for_embedder,
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


# ==============================================================================
# create_environments - MATCHING sb3_custom_dummy_env.create_environments
# ==============================================================================

def create_environments(args: Any, dh: DataHandler, im: IndexManager, **kwargs):
    """Create train and eval environments matching sb3 create_environments."""
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
        end_proof_action=getattr(args, 'end_proof_action', True) or getattr(args, 'endf_action', True),
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
    
    batch_size = getattr(args, 'batch_size_env', None) or getattr(args, 'n_envs', 16)
    
    # Train environment
    train_env = BatchedEnv(
        batch_size=batch_size,
        queries=train_queries_tensor,
        labels=torch.ones(len(dh.train_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(dh.train_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',
        max_depth=getattr(args, 'max_depth', 20),
        memory_pruning=args.memory_pruning,
        use_exact_memory=True,
        skip_unary_actions=args.skip_unary_actions,
        end_proof_action=getattr(args, 'end_proof_action', True) or getattr(args, 'endf_action', True),
        reward_type=args.reward_type,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + args.max_total_vars,
        sample_deterministic_per_env=True,
    )
    
    # Eval environment
    eval_env = BatchedEnv(
        batch_size=batch_size,
        queries=test_queries_tensor,
        labels=torch.ones(len(dh.test_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(dh.test_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='eval',
        max_depth=getattr(args, 'max_depth', 20),
        memory_pruning=args.memory_pruning,
        use_exact_memory=True,
        skip_unary_actions=args.skip_unary_actions,
        end_proof_action=getattr(args, 'end_proof_action', True) or getattr(args, 'endf_action', True),
        reward_type=args.reward_type,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + args.max_total_vars,
        sample_deterministic_per_env=True,
    )
    
    # Return train_env, eval_env, callback_env (matching sb3 signature)
    callback_env = eval_env  # Use eval_env for callbacks
    return train_env, eval_env, callback_env


# ==============================================================================
# _evaluate - MATCHING sb3_train._evaluate
# ==============================================================================

def _evaluate(args: Any, policy, eval_env, sampler, dh: DataHandler, im: IndexManager, device: torch.device) -> Tuple[dict, dict, dict]:
    """Evaluate model and return metrics matching sb3_train._evaluate."""
    print("\nTest set evaluation...")
    
    # Reseed before evaluation (matching sb3)
    deterministic_parity = getattr(args, "deterministic_parity", False)
    if deterministic_parity:
        eval_seed = 12345  # Same as sb3_train
        torch.manual_seed(eval_seed)
        np.random.seed(eval_seed)
    
    policy.eval()
    
    # Get test queries
    test_queries = dh.test_queries
    n_test = getattr(args, 'n_test_queries', None) or len(test_queries)
    test_queries = test_queries[:n_test]
    
    # Convert queries to tensor
    query_atoms = []
    for q in test_queries:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_atoms.append(query_atom)
    queries_tensor = torch.stack(query_atoms, dim=0)
    
    n_corruptions = getattr(args, 'test_neg_samples', 10) or 10
    corruption_scheme = getattr(args, 'corruption_scheme', ['tail'])
    
    eval_results = tensor_eval_corruptions(
        actor=policy,
        env=eval_env,
        queries=queries_tensor,
        sampler=sampler,
        n_corruptions=n_corruptions,
        corruption_modes=tuple(corruption_scheme),
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
    metrics_test = {
        'mrr_mean': _parse_metric(eval_results.get('MRR', 0.0)),
        'hits1_mean': _parse_metric(eval_results.get('Hits@1', 0.0)),
        'hits3_mean': _parse_metric(eval_results.get('Hits@3', 0.0)),
        'hits10_mean': _parse_metric(eval_results.get('Hits@10', 0.0)),
        'rewards_pos_mean': _parse_metric(eval_results.get('reward_pos_mean', 0.0)),
        'rewards_neg_mean': _parse_metric(eval_results.get('reward_neg_mean', 0.0)),
        'reward_label_pos': _parse_metric(eval_results.get('reward_pos_mean', 0.0)),
        'reward_label_neg': _parse_metric(eval_results.get('reward_neg_mean', 0.0)),
        'success_rate': _parse_metric(eval_results.get('proven_pos', 0.0)),
        'reward_overall': _parse_metric(eval_results.get('proven_pos', 0.0)),  # For parity comparison
        'proven_pos': _parse_metric(eval_results.get('proven_pos', 0.0)),
        'tail_mrr_mean': _parse_metric(eval_results.get('MRR', 0.0)),  # Tail-only corruption
        'head_mrr_mean': 0.0,  # No head corruption in default config
    }
    
    print(f"results for: {getattr(args, 'run_signature', 'tensor')}")
    print("\nTest set metrics:")
    for k, v in metrics_test.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")
    
    # Placeholder for train/valid (matching sb3 - not evaluated by default)
    metrics_train = {k: 0 for k in metrics_test.keys()}
    metrics_valid = {k: 0 for k in metrics_test.keys()}
    
    return metrics_train, metrics_valid, metrics_test


# ==============================================================================
# main - MATCHING sb3_train.main signature and flow EXACTLY
# ==============================================================================

def main(args, log_filename, use_logger, use_WB, WB_path, date, external_components=None):
    """
    Main training function matching sb3_train.main() signature and flow EXACTLY.
    
    Args:
        args: Configuration namespace
        log_filename: Log file path
        use_logger: Whether to use logging
        use_WB: Whether to use Weights & Biases
        WB_path: W&B path
        date: Date string
        external_components: Optional dict with pre-created components for parity testing:
            {'dh': DataHandler, 'index_manager': IndexManager, 'sampler': Sampler, 'embedder': Embedder}
    
    Flow MUST match sb3_train.main():
    1. _warn_non_reproducible(args)
    2. _set_seeds(args.seed_run_i)
    3. get_device
    4. _build_data_and_index (includes reseed before embedder)
    5. create_environments
    6. Create policy/PPO
    7. Train
    8. Evaluate
    """
    # Step 1: Warn (matching sb3)
    _warn_non_reproducible(args)
    
    # Step 2: Set seeds (matching sb3)
    _set_seeds(args.seed_run_i)
    
    # Deterministic parity mode for exact alignment with SB3 implementation
    deterministic_parity = getattr(args, 'deterministic_parity', False)
    
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
    
    # PARITY: Reseed before environment creation for deterministic alignment
    if deterministic_parity:
        _set_seeds(args.seed_run_i)
    
    # Step 5: Create environments (matching sb3)
    env, eval_env, callback_env = create_environments(
        args,
        dh,
        index_manager,
    )
    
    # PARITY: Reseed before model creation for deterministic alignment
    if deterministic_parity:
        _set_seeds(args.seed_run_i)
    
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
    ).to(device)
    
    ppo = TensorPPO(
        policy=policy,
        env=env,
        n_steps=args.n_steps,
        learning_rate=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        device=device,
        verbose=1,
    )
    
    # Step 7: Train (matching sb3 flow)
    if args.timesteps_train > 0 and not getattr(args, 'load_model', False):
        ppo.learn(total_timesteps=args.timesteps_train)
    
    # Step 8: Evaluate (matching sb3)
    metrics_train, metrics_valid, metrics_test = _evaluate(
        args, policy, eval_env, sampler, dh, index_manager, device
    )
    
    return metrics_train, metrics_valid, metrics_test


# ==============================================================================
# run_experiment (for test_runner_simple compatibility)
# ==============================================================================

def seed_all_compat(seed: int):
    """Set all random seeds."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_tensor_components(config: TrainParityConfig) -> Dict[str, Any]:
    """Create tensor training components (data handler, index manager, env, model)."""
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
        default_mode="both",
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
    
    # Create unification engine
    engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.padding_states,
    )
    engine.index_manager = im
    
    # Convert queries to tensor format
    train_queries = dh.train_queries
    test_queries = dh.test_queries
    
    def convert_queries_to_tensor(queries):
        query_tensors = []
        for q in queries:
            query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            query_padded = torch.full((config.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
            query_padded[0] = query_atom
            query_tensors.append(query_padded)
        return torch.stack(query_tensors, dim=0)
    
    train_queries_tensor = convert_queries_to_tensor(train_queries)
    test_queries_tensor = convert_queries_to_tensor(test_queries)
    
    # Create environments
    train_env = BatchedEnv(
        batch_size=config.n_envs,
        queries=train_queries_tensor,
        labels=torch.ones(len(train_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(train_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',
        max_depth=config.max_steps,
        memory_pruning=config.memory_pruning,
        use_exact_memory=config.use_exact_memory,
        skip_unary_actions=config.skip_unary_actions,
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
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
        sample_deterministic_per_env=True,
    )
    
    eval_env = BatchedEnv(
        batch_size=config.n_envs,
        queries=test_queries_tensor,
        labels=torch.ones(len(test_queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(test_queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='eval',
        max_depth=config.max_steps,
        memory_pruning=config.memory_pruning,
        use_exact_memory=True,
        skip_unary_actions=config.skip_unary_actions,
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
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
        sample_deterministic_per_env=True,
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
    ).to(device)
    
    return {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'embedder': embedder,
        'engine': engine,
        'train_env': train_env,
        'eval_env': eval_env,
        'policy': policy,
        'device': device,
    }


def run_experiment(config: TrainParityConfig) -> Dict[str, float]:
    """Run full training experiment and return evaluation metrics."""
    print("=" * 70)
    print("TENSOR TRAINING")
    print(f"Dataset: {config.dataset}")
    print(f"N envs: {config.n_envs}, N steps: {config.n_steps}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Seed: {config.seed}")
    print(f"N corruptions: {config.n_corruptions}")
    print("=" * 70)
    
    # Create tensor components
    print("\n[1/3] Creating tensor components...")
    seed_all_compat(config.seed)
    tensor_comp = create_tensor_components(config)
    
    # Tensor training
    print("\n[2/3] Running training...")
    seed_all_compat(config.seed)
    tensor_ppo = TensorPPO(
        policy=tensor_comp['policy'],
        env=tensor_comp['train_env'],
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        gamma=config.gamma,
        device=tensor_comp['device'],
        verbose=True,
    )
    tensor_ppo.learn(total_timesteps=config.total_timesteps)
    
    # Tensor evaluation
    print("\n[3/3] Running evaluation...")
    seed_all_compat(config.seed + 1000)
    tensor_comp['policy'].eval()
    
    # Get test queries and convert to tensor
    tensor_im = tensor_comp['im']
    test_query_objs = tensor_comp['dh'].test_queries[:config.n_envs * 4]
    tensor_query_atoms = []
    for q in test_query_objs:
        query_atom = tensor_im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        tensor_query_atoms.append(query_atom)
    tensor_queries = torch.stack(tensor_query_atoms, dim=0)
    
    tensor_eval_results = tensor_eval_corruptions(
        actor=tensor_comp['policy'],
        env=tensor_comp['eval_env'],
        queries=tensor_queries,
        sampler=tensor_comp['sampler'],
        n_corruptions=config.n_corruptions,
        corruption_modes=('tail',),
        verbose=False,
    )
    
    results = {
        "MRR": tensor_eval_results.get('MRR', 0.0),
        "Hits@1": tensor_eval_results.get('Hits@1', 0.0)
    }
    
    print("\nEvaluation Results:")
    print(f"  Tensor MRR: {results['MRR']:.4f}")
    print(f"  Tensor Hits@1: {results['Hits@1']:.4f}")
    
    return results


# ==============================================================================
# CLI entry point
# ==============================================================================

def main_cli():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Train (Tensor)")
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
    
    args = parser.parse_args()
    
    config = TrainParityConfig(
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
    )
    
    run_experiment(config)


if __name__ == "__main__":
    main_cli()

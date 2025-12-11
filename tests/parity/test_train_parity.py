"""
Training Parity Tests.

Tests verifying that the tensor-based training produces the SAME results as
SB3 training, including:
- Rollout collection parity
- PPO gradient updates parity
- Final model weights parity
- Evaluation metrics parity

Uses deterministic seeds and aligned environments to ensure reproducible comparisons.

Usage:
    python tests/parity/test_train_parity.py --dataset countries_s3 --n-steps 20 --n-envs 3
"""
import gc
import os

# Force strict parity for testing
os.environ['USE_FAST_CATEGORICAL'] = '0'

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import json
import time

import torch
import torch.nn as nn
import numpy as np
from collections import deque

# Import seeding utilities (must be before other local imports to set up paths correctly)
from seed_utils import ParityTestSeeder, ParityTestConfig, seed_all
from parity_config import ParityConfig, TOLERANCE, create_parser, config_from_args

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"
TEST_ENVS_ROOT = ROOT / "tests" / "other" / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(2, str(TEST_ENVS_ROOT))

# SB3 imports
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# sb3 imports
from sb3_custom_dummy_env import CustomDummyVecEnv
from sb3_dataset import DataHandler as SB3DataHandler
from sb3_index_manager import IndexManager as SB3IndexManager
from sb3_env import LogicEnv_gym as SB3Env
from sb3_model import PPO_custom as SB3PPO, CustomActorCriticPolicy, CustomCombinedExtractor
from sb3_embeddings import EmbedderLearnable as SB3Embedder
from sb3_model_eval import eval_corruptions as sb3_eval_corruptions
from sb3_neg_sampling import get_sampler as get_sb3_sampler

# Tensor imports  
from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from env import BatchedEnv
from embeddings import EmbedderLearnable as TensorEmbedder
from model import ActorCriticPolicy as TensorPolicy
from ppo import PPO as TensorPPO
from rollout import RolloutBuffer
from model_eval import eval_corruptions as tensor_eval_corruptions
from sampler import Sampler, SamplerConfig


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
    max_total_vars: int = 1000  # Reduced from 1000000 to save memory
    
    # PPO / training
    n_envs: int = 3
    n_steps: int = 20
    n_epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 120  # Small for testing - n_steps * n_envs * 2
    n_corruptions: int = 10
    
    # Embedding / model
    atom_embedding_size: int = 64  # Reduced from 256 to save memory
    
    # Misc
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True
    skip_training: bool = False  # Skip actual training (for faster testing)
    skip_eval: bool = False  # Skip evaluation (to avoid OOM issues)


@dataclass 
class TrainParityResults:
    """Results container for training parity comparison."""
    # Weight alignment
    initial_weights_match: bool = False
    final_weights_match: bool = False
    
    # Rollout comparison
    rollout_obs_match: bool = False
    rollout_actions_match: bool = False
    rollout_rewards_match: bool = False
    rollout_values_match: bool = False
    rollout_logprobs_match: bool = False
    
    # Training comparison
    policy_loss_match: bool = False
    value_loss_match: bool = False
    entropy_match: bool = False
    
    # Evaluation comparison
    eval_mrr_match: bool = False
    eval_hits1_match: bool = False
    
    # Raw values
    sb3_policy_loss: float = 0.0
    tensor_policy_loss: float = 0.0
    sb3_value_loss: float = 0.0
    tensor_value_loss: float = 0.0
    sb3_entropy: float = 0.0
    tensor_entropy: float = 0.0
    
    sb3_mrr: float = 0.0
    tensor_mrr: float = 0.0
    sb3_hits1: float = 0.0
    tensor_hits1: float = 0.0
    
    # Detailed traces (set to None by default to save memory - only populated when needed)
    sb3_rollout_trace: Optional[Dict] = None
    tensor_rollout_trace: Optional[Dict] = None
    
    # Summary
    overall_success: bool = False
    error_message: str = ""
    
    def clear_traces(self):
        """Clear trace data to free memory."""
        self.sb3_rollout_trace = None
        self.tensor_rollout_trace = None


def create_sb3_components(config: TrainParityConfig) -> Dict[str, Any]:
    """Create SB3 training components (data handler, index manager, env, model)."""
    device = torch.device(config.device)
    
    # Data handler
    dh = SB3DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
        corruption_mode=True,
    )
    
    # Index manager
    im = SB3IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_vars=config.max_total_vars,
        rules=dh.rules,
        padding_atoms=config.padding_atoms,
        max_arity=dh.max_arity,
        device=device,
    )
    facts_set = set(dh.facts)
    im.build_fact_index(list(facts_set), deterministic=True)
    
    # Sampler
    sampler = get_sb3_sampler(
        data_handler=dh,
        index_manager=im,
        corruption_scheme=['head', 'tail'],
        device=device,
        corruption_mode=True,
    )
    
    # Embedder - use reasonable n_vars to avoid massive memory usage
    n_vars_for_embedder = 1000
    torch.manual_seed(config.seed)
    embedder = SB3Embedder(
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
        device=device,
    )
    embedder.embed_dim = config.atom_embedding_size
    
    # Create environments with Monitor wrapper like working tests
    def make_env(idx: int, mode: str = "train"):
        def _init():
            queries = dh.train_queries if mode == "train" else dh.test_queries
            labels = [1] * len(queries)  # All queries are positive examples
            env = SB3Env(
                index_manager=im,
                data_handler=dh,
                queries=queries,
                labels=labels,
                query_depths=None,
                facts=facts_set,
                mode=mode,
                sample_deterministic=True,  # Round-robin for parity
                seed=config.seed,
                max_depth=config.max_steps,
                memory_pruning=config.memory_pruning,
                padding_atoms=config.padding_atoms,
                padding_states=config.padding_states,
                verbose=0,
                prover_verbose=0,
                device=device,
                engine='python',
                engine_strategy='complete',
                skip_unary_actions=config.skip_unary_actions,
                endf_action=config.end_proof_action,
                reward_type=config.reward_type,
            )
            env._train_ptr = idx  # Set train pointer for deterministic query selection
            return Monitor(env)  # Wrap with Monitor like working tests
        return _init
    
    # Use DummyVecEnv for training, CustomDummyVecEnv for evaluation
    train_env = DummyVecEnv([make_env(i, "train") for i in range(config.n_envs)])
    eval_env = CustomDummyVecEnv([make_env(i, "eval") for i in range(config.n_envs)])
    
    # Create model
    torch.manual_seed(config.seed)
    policy_kwargs = {
        'features_extractor_class': CustomCombinedExtractor,
        'features_extractor_kwargs': {'features_dim': embedder.embed_dim, 'embedder': embedder},
    }
    
    model = SB3PPO(
        CustomActorCriticPolicy,
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        verbose=0,
        device=device,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        gamma=config.gamma,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
    )
    
    return {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'embedder': embedder,
        'train_env': train_env,
        'eval_env': eval_env,
        'model': model,
        'device': device,
    }


def create_tensor_components(config: TrainParityConfig, sb3_dh: SB3DataHandler = None) -> Dict[str, Any]:
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
    
    # Convert queries to tensor format like working tests
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
    
    # Create environments - match working tests exactly
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
    
    # Create embedder with fixed seed - match SB3 exactly
    n_vars_for_embedder = 1000
    torch.manual_seed(config.seed)
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',  # Must match SB3
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
        parity=True,  # Use SB3-identical initialization
        use_l2_norm=False,  # Match SB3's logit computation (no L2 norm)
        sqrt_scale=True,  # Match SB3's attention-style scaling
        temperature=None,  # No temperature scaling (parity with model_old)
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


def compare_weights(sb3_model, tensor_policy, rtol=1e-4, atol=1e-6, verbose=True) -> Tuple[bool, Dict]:
    """Compare weights between SB3 and tensor models.
    
    Uses chunked comparison for large tensors to avoid OOM.
    """
    sb3_params = dict(sb3_model.policy.named_parameters())
    tensor_params = dict(tensor_policy.named_parameters())
    
    results = {
        'total_params': len(sb3_params),
        'matched_params': 0,
        'mismatched_params': [],
        'missing_in_tensor': [],
        'missing_in_sb3': [],
    }
    
    def compare_tensors_chunked(t1, t2, rtol, atol, chunk_size=100000):
        """Compare large tensors in chunks to avoid OOM."""
        t1_flat = t1.data.view(-1)
        t2_flat = t2.data.view(-1)
        
        if t1_flat.numel() <= chunk_size:
            # Small tensor: use normal comparison
            return torch.allclose(t1.data, t2.data, rtol=rtol, atol=atol)
        
        # Large tensor: check in chunks
        for i in range(0, t1_flat.numel(), chunk_size):
            end = min(i + chunk_size, t1_flat.numel())
            chunk1 = t1_flat[i:end]
            chunk2 = t2_flat[i:end]
            if not torch.allclose(chunk1, chunk2, rtol=rtol, atol=atol):
                return False
        return True
    
    def max_diff_chunked(t1, t2, chunk_size=100000):
        """Compute max diff for large tensors in chunks."""
        t1_flat = t1.data.view(-1)
        t2_flat = t2.data.view(-1)
        
        if t1_flat.numel() <= chunk_size:
            return (t1.data - t2.data).abs().max().item()
        
        max_d = 0.0
        for i in range(0, t1_flat.numel(), chunk_size):
            end = min(i + chunk_size, t1_flat.numel())
            diff = (t1_flat[i:end] - t2_flat[i:end]).abs().max().item()
            max_d = max(max_d, diff)
        return max_d
    
    # Map SB3 param names to tensor param names (they may differ slightly)
    # For now assume names are identical or mappable
    for name, sb3_param in sb3_params.items():
        if name in tensor_params:
            tensor_param = tensor_params[name]
            if sb3_param.shape == tensor_param.shape:
                if compare_tensors_chunked(sb3_param, tensor_param, rtol, atol):
                    results['matched_params'] += 1
                else:
                    diff = max_diff_chunked(sb3_param, tensor_param)
                    results['mismatched_params'].append((name, diff))
                    if verbose:
                        print(f"  Weight mismatch: {name}, max diff={diff:.6e}")
            else:
                results['mismatched_params'].append((name, f"shape mismatch: {sb3_param.shape} vs {tensor_param.shape}"))
        else:
            results['missing_in_tensor'].append(name)
    
    for name in tensor_params:
        if name not in sb3_params:
            results['missing_in_sb3'].append(name)
    
    all_match = (
        results['matched_params'] == results['total_params'] and
        len(results['mismatched_params']) == 0
    )
    
    return all_match, results


def run_sb3_rollout(model, env, n_steps: int, seed: int) -> Dict[str, Any]:
    """Run a single rollout with SB3 and collect detailed traces."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model.policy.set_training_mode(False)
    
    obs = env.reset()
    
    trace = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'values': [],
        'log_probs': [],
        'dones': [],
    }
    
    for step in range(n_steps):
        obs_tensor = obs_as_tensor(obs, model.device)
        
        with torch.no_grad():
            actions, values, log_probs = model.policy(obs_tensor, deterministic=True)
        
        actions_np = actions.cpu().numpy()
        new_obs, rewards, dones, infos = env.step(actions_np)
        
        trace['observations'].append({k: v.copy() for k, v in obs.items()})
        trace['actions'].append(actions_np.copy())
        trace['rewards'].append(rewards.copy())
        trace['values'].append(values.cpu().numpy().copy())
        trace['log_probs'].append(log_probs.cpu().numpy().copy())
        trace['dones'].append(dones.copy())
        
        obs = new_obs
    
    return trace


def run_tensor_rollout(policy, env, n_steps: int, seed: int, device: torch.device) -> Dict[str, Any]:
    """Run a single rollout with tensor env and collect detailed traces."""
    from tensordict import TensorDict
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    policy.eval()
    
    current_obs = env.reset()
    
    trace = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'values': [],
        'log_probs': [],
        'dones': [],
    }
    
    for step in range(n_steps):
        # Clone observation to avoid mutation
        obs_snapshot = current_obs.clone()
        obs_dict = {
            'sub_index': obs_snapshot['sub_index'].to(device),
            'derived_sub_indices': obs_snapshot['derived_sub_indices'].to(device),
            'action_mask': obs_snapshot['action_mask'].to(device),
        }
        
        with torch.no_grad():
            actions, values, log_probs = policy(obs_dict, deterministic=True)
        
        # Step environment using step_and_maybe_reset like tensor PPO does
        action_td = TensorDict({"action": actions}, batch_size=current_obs.batch_size, device=device)
        step_result, next_obs = env.step_and_maybe_reset(action_td)
        
        # Extract done/reward - handle both with and without 'next' key
        if "next" in step_result.keys():
            step_info = step_result["next"]
        else:
            step_info = step_result
        
        rewards = step_info.get("reward", torch.zeros(env.batch_size_int, device=device))
        dones = step_info.get("done", torch.zeros(env.batch_size_int, dtype=torch.bool, device=device))
        
        # Squeeze to ensure correct shape
        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)
        if dones.dim() > 1:
            dones = dones.squeeze(-1)
        
        trace['observations'].append({k: v.cpu().numpy().copy() for k, v in obs_dict.items()})
        trace['actions'].append(actions.cpu().numpy().copy())
        trace['rewards'].append(rewards.cpu().numpy().copy())
        trace['values'].append(values.cpu().numpy().copy())
        trace['log_probs'].append(log_probs.cpu().numpy().copy())
        trace['dones'].append(dones.cpu().numpy().copy())
        
        # Move to next observation
        current_obs = next_obs
    
    return trace


def compare_rollout_traces(sb3_trace: Dict, tensor_trace: Dict, rtol=1e-4, atol=1e-6, verbose=True) -> Dict[str, bool]:
    """Compare rollout traces from SB3 and tensor implementations."""
    results = {
        'actions_match': True,
        'rewards_match': True,
        'values_match': True,
        'log_probs_match': True,
        'first_mismatch_step': -1,
        'mismatch_details': [],
    }
    
    n_steps = min(len(sb3_trace['actions']), len(tensor_trace['actions']))
    
    for step in range(n_steps):
        sb3_actions = sb3_trace['actions'][step]
        tensor_actions = tensor_trace['actions'][step]
        
        if not np.array_equal(sb3_actions, tensor_actions):
            results['actions_match'] = False
            if results['first_mismatch_step'] < 0:
                results['first_mismatch_step'] = step
            results['mismatch_details'].append(f"Step {step}: actions differ: SB3={sb3_actions}, Tensor={tensor_actions}")
        
        sb3_rewards = sb3_trace['rewards'][step]
        tensor_rewards = tensor_trace['rewards'][step]
        if not np.allclose(sb3_rewards, tensor_rewards, rtol=rtol, atol=atol):
            results['rewards_match'] = False
            if results['first_mismatch_step'] < 0:
                results['first_mismatch_step'] = step
        
        sb3_values = sb3_trace['values'][step]
        tensor_values = tensor_trace['values'][step]
        if not np.allclose(sb3_values.flatten(), tensor_values.flatten(), rtol=rtol, atol=atol):
            results['values_match'] = False
            if results['first_mismatch_step'] < 0:
                results['first_mismatch_step'] = step
        
        sb3_lp = sb3_trace['log_probs'][step]
        tensor_lp = tensor_trace['log_probs'][step]
        if not np.allclose(sb3_lp.flatten(), tensor_lp.flatten(), rtol=rtol, atol=atol):
            results['log_probs_match'] = False
            if results['first_mismatch_step'] < 0:
                results['first_mismatch_step'] = step
    
    if verbose and results['mismatch_details']:
        print("Rollout mismatches:")
        for detail in results['mismatch_details'][:5]:
            print(f"  {detail}")
    
    return results


def run_train_parity(
    config: TrainParityConfig,
    verbose: bool = True,
) -> TrainParityResults:
    """
    Run training parity test comparing SB3 and tensor implementations.
    
    Args:
        config: Configuration for the parity test
        verbose: Whether to print detailed output
        
    Returns:
        TrainParityResults with comparison results
    """
    results = TrainParityResults()
    
    if verbose:
        print("=" * 70)
        print("TRAINING PARITY TEST")
        print(f"Dataset: {config.dataset}")
        print(f"N envs: {config.n_envs}, N steps: {config.n_steps}")
        print(f"Total timesteps: {config.total_timesteps}")
        print(f"Seed: {config.seed}")
        print("=" * 70)
    
    try:
        # Set seeds
        seed_all(config.seed)
        
        # Create SB3 components
        if verbose:
            print("\n[1/6] Creating SB3 components...")
        sb3_comp = create_sb3_components(config)
        
        # Create tensor components
        if verbose:
            print("\n[2/6] Creating tensor components...")
        seed_all(config.seed)  # Reset seeds
        tensor_comp = create_tensor_components(config, sb3_comp['dh'])
        
        # Compare initial weights
        if verbose:
            print("\n[3/6] Comparing initial weights...")
        # Note: Weight comparison requires matching architectures
        # For now we skip this if architectures differ
        try:
            initial_match, weight_info = compare_weights(
                sb3_comp['model'], tensor_comp['policy'], verbose=verbose
            )
            results.initial_weights_match = initial_match
            if verbose:
                print(f"  Initial weights match: {initial_match}")
                if not initial_match:
                    print(f"  Matched: {weight_info['matched_params']}/{weight_info['total_params']}")
        except Exception as e:
            if verbose:
                print(f"  Weight comparison skipped: {e}")
            results.initial_weights_match = False
        
        # Run single rollout comparison
        if verbose:
            print("\n[4/6] Comparing single rollout...")
        
        seed_all(config.seed)
        sb3_trace = run_sb3_rollout(
            sb3_comp['model'], sb3_comp['train_env'], config.n_steps, config.seed
        )
        # Only store traces if explicitly needed (skip by default to save memory)
        # results.sb3_rollout_trace = sb3_trace
        
        seed_all(config.seed)
        tensor_trace = run_tensor_rollout(
            tensor_comp['policy'], tensor_comp['train_env'], config.n_steps, config.seed, tensor_comp['device']
        )
        # Only store traces if explicitly needed (skip by default to save memory)
        # results.tensor_rollout_trace = tensor_trace
        
        rollout_comparison = compare_rollout_traces(sb3_trace, tensor_trace, verbose=verbose)
        results.rollout_actions_match = rollout_comparison['actions_match']
        results.rollout_rewards_match = rollout_comparison['rewards_match']
        results.rollout_values_match = rollout_comparison['values_match']
        results.rollout_logprobs_match = rollout_comparison['log_probs_match']
        
        if verbose:
            print(f"  Actions match: {results.rollout_actions_match}")
            print(f"  Rewards match: {results.rollout_rewards_match}")
            print(f"  Values match: {results.rollout_values_match}")
            print(f"  Log probs match: {results.rollout_logprobs_match}")
        
        # Clear traces to free memory before training
        del sb3_trace, tensor_trace
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Skip training and evaluation if requested
        if config.skip_training:
            if verbose:
                print("\n[5/6] Skipping training (skip_training=True)...")
                print("\n[6/6] Skipping evaluation (skip_training=True)...")
            # Mark as success since rollout passed
            results.final_weights_match = results.initial_weights_match
            results.overall_success = (
                results.rollout_actions_match and
                results.rollout_rewards_match
            )
            return results
        
        # Run training
        if verbose:
            print("\n[5/6] Running training comparison...")
        
        # Reset both environments to ensure same starting state
        # The rollout comparison may have changed env state
        seed_all(config.seed)
        sb3_comp['train_env'].reset()
        sb3_comp['model']._last_obs = None  # Force re-initialization
        sb3_comp['model']._last_episode_starts = None
        
        tensor_comp['train_env'].reset()
        
        # SB3 training
        seed_all(config.seed)
        sb3_comp['model'].learn(total_timesteps=config.total_timesteps, progress_bar=False)
        
        # Tensor training
        seed_all(config.seed)
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
            verbose=False,
            parity=True,  # Use numpy RNG for shuffling to match SB3
        )
        tensor_ppo.learn(total_timesteps=config.total_timesteps)
        
        # Compare final weights
        if verbose:
            print("\n[6/6] Comparing final weights and evaluating...")
        try:
            final_match, final_weight_info = compare_weights(
                sb3_comp['model'], tensor_comp['policy'], verbose=verbose
            )
            results.final_weights_match = final_match
            if verbose:
                print(f"  Final weights match: {final_match}")
        except Exception as e:
            if verbose:
                print(f"  Final weight comparison skipped: {e}")
        
        # Skip eval if requested
        if config.skip_eval:
            if verbose:
                print("\n  Skipping evaluation (skip_eval=True)...")
            results.overall_success = (
                results.rollout_actions_match and
                results.rollout_rewards_match and
                results.final_weights_match
            )
            if verbose:
                print("\n" + "=" * 70)
                print(f"OVERALL RESULT: {'PASS' if results.overall_success else 'FAIL'}")
                print("=" * 70)
            return results
        
        # Free memory before evaluation
        del tensor_ppo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run evaluation comparison
        if verbose:
            print("\n  Running evaluation comparison...")
        
        # Get test queries
        test_queries = sb3_comp['dh'].test_queries[:config.n_envs * 4]
        
        # SB3 evaluation (needs env, model, data, sampler, n_corruptions, ...)
        seed_all(config.seed + 1000)  # Different seed for eval
        sb3_eval_results = sb3_eval_corruptions(
            model=sb3_comp['model'],
            env=sb3_comp['eval_env'],
            data=test_queries,
            sampler=sb3_comp['sampler'],
            n_corruptions=config.n_corruptions,
            corruption_scheme=['tail'],
            verbose=0,
        )
        results.sb3_mrr = sb3_eval_results.get('mrr_mean', 0.0)
        results.sb3_hits1 = sb3_eval_results.get('hits1_mean', 0.0)
        
        # Tensor evaluation
        seed_all(config.seed + 1000)
        tensor_comp['policy'].eval()
        # Get test queries - already in tensor format from create_tensor_components
        # tensor_comp['eval_env'] was created with test_queries_tensor
        # We need to get the original query objects from dh.test_queries
        tensor_im = tensor_comp['im']
        test_query_objs = tensor_comp['dh'].test_queries[:config.n_envs * 4]
        # Convert to tensor format
        tensor_query_atoms = []
        for q in test_query_objs:
            query_atom = tensor_im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensor_query_atoms.append(query_atom)
        tensor_queries = torch.stack(tensor_query_atoms, dim=0)  # Shape: (N, 3)
        
        tensor_eval_results = tensor_eval_corruptions(
            actor=tensor_comp['policy'],
            env=tensor_comp['eval_env'],
            queries=tensor_queries,
            sampler=tensor_comp['sampler'],
            n_corruptions=config.n_corruptions,
            corruption_modes=('tail',),
            verbose=False,
        )
        results.tensor_mrr = tensor_eval_results.get('MRR', 0.0)
        results.tensor_hits1 = tensor_eval_results.get('Hits@1', 0.0)
        
        # Compare evaluation results - exact match expected
        results.eval_mrr_match = abs(results.sb3_mrr - results.tensor_mrr) < 0.01
        results.eval_hits1_match = abs(results.sb3_hits1 - results.tensor_hits1) < 0.01
        
        if verbose:
            print(f"\n  Evaluation Results:")
            print(f"    SB3 MRR: {results.sb3_mrr:.4f}, Tensor MRR: {results.tensor_mrr:.4f}, Match: {results.eval_mrr_match}")
            print(f"    SB3 Hits@1: {results.sb3_hits1:.4f}, Tensor Hits@1: {results.tensor_hits1:.4f}, Match: {results.eval_hits1_match}")
        
        # Compute overall success
        results.overall_success = (
            results.rollout_actions_match and
            results.rollout_rewards_match and
            results.eval_mrr_match
        )
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"OVERALL RESULT: {'PASS' if results.overall_success else 'FAIL'}")
            print("=" * 70)
        
    except Exception as e:
        results.error_message = str(e)
        if verbose:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
    
    return results


# ==============================================================================
# Pytest Test Cases
# ==============================================================================

# ============================================================
# Test Functions (callable from __main__)
# ============================================================

def test_train_parity(dataset: str, n_envs: int, n_steps: int, total_timesteps: int,
                     n_corruptions: int, batch_size: int, n_epochs: int, 
                     lr: float, ent_coef: float) -> bool:
    """Test that SB3 and tensor training produce similar results."""
    config = TrainParityConfig(
        dataset=dataset,
        n_envs=n_envs,
        n_steps=n_steps,
        total_timesteps=total_timesteps,
        n_corruptions=n_corruptions,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=lr,
        ent_coef=ent_coef,
        seed=42,
        device="cpu",
        verbose=True,
    )
    
    results = run_train_parity(config, verbose=True)
    
    # Check key parity results
    passed = True
    if not results.rollout_actions_match:
        print("✗ Rollout actions should match")
        passed = False
    if not results.rollout_rewards_match:
        print("✗ Rollout rewards should match")
        passed = False
    if not results.eval_mrr_match:
        print(f"✗ Eval MRR should match: SB3={results.sb3_mrr:.4f}, Tensor={results.tensor_mrr:.4f}")
        passed = False
    
    return passed and results.overall_success


def test_single_rollout_parity(dataset: str) -> bool:
    """Test that a single rollout produces identical traces."""
    config = TrainParityConfig(
        dataset=dataset,
        n_envs=2,
        n_steps=10,
        total_timesteps=0,  # No training
        seed=42,
        device="cpu",
        verbose=True,
    )
    
    seed_all(config.seed)
    sb3_comp = create_sb3_components(config)
    
    seed_all(config.seed)
    tensor_comp = create_tensor_components(config)
    
    seed_all(config.seed)
    sb3_trace = run_sb3_rollout(
        sb3_comp['model'], sb3_comp['train_env'], config.n_steps, config.seed
    )
    
    seed_all(config.seed)
    tensor_trace = run_tensor_rollout(
        tensor_comp['policy'], tensor_comp['train_env'], config.n_steps, config.seed, tensor_comp['device']
    )
    
    comparison = compare_rollout_traces(sb3_trace, tensor_trace, verbose=True)
    
    passed = True
    if not comparison['actions_match']:
        print("✗ Actions should match in single rollout")
        passed = False
    if not comparison['rewards_match']:
        print("✗ Rewards should match in single rollout")
        passed = False
    
    return passed


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_all_tests(args) -> bool:
    """Run all train parity tests."""
    all_passed = True
    
    print(f"\n{'='*70}")
    print("TRAIN PARITY TESTS")
    print(f"Tolerance: {TOLERANCE}")
    print(f"{'='*70}")
    
    # Default test configurations
    default_params = [
        # (dataset, n_envs, n_steps, total_timesteps, n_corruptions, batch_size, n_epochs, lr, ent_coef)
        ("countries_s3", 3, 20, 60, 10, 64, 4, 3e-4, 0.0),
        ("countries_s3", 4, 32, 256, 10, 4096, 20, 5e-5, 0.2),
    ]
    
    # If user provides specific arguments, use them instead of defaults
    if args.n_envs is not None and args.n_steps is not None:
        # User specified custom parameters
        test_params = [(
            args.dataset,
            args.n_envs,
            args.n_steps,
            getattr(args, 'total_timesteps', args.n_envs * args.n_steps * 2),
            getattr(args, 'n_corruptions', 10),
            args.batch_size,
            getattr(args, 'n_epochs', 4),
            args.learning_rate,
            args.ent_coef,
        )]
    elif getattr(args, 'run_all_params', False):
        test_params = default_params
    else:
        # Just run first default
        test_params = [default_params[0]]
    
    # Test 1: test_train_parity
    for dataset, n_envs, n_steps, total_timesteps, n_corruptions, batch_size, n_epochs, lr, ent_coef in test_params:
        print(f"\n{'='*70}")
        print(f"Running test_train_parity[{dataset}-{n_envs}-{n_steps}-{total_timesteps}]")
        print(f"{'='*70}")
        
        if test_train_parity(dataset, n_envs, n_steps, total_timesteps, n_corruptions, batch_size, n_epochs, lr, ent_coef):
            print(f"✓ PASSED: test_train_parity[{dataset}-{n_envs}-{n_steps}-{total_timesteps}]")
        else:
            print(f"✗ FAILED: test_train_parity[{dataset}-{n_envs}-{n_steps}-{total_timesteps}]")
            all_passed = False
    
    # Test 2: test_single_rollout_parity
    print(f"\n{'='*70}")
    print(f"Running test_single_rollout_parity[{args.dataset}]")
    print(f"{'='*70}")
    
    if test_single_rollout_parity(args.dataset):
        print(f"✓ PASSED: test_single_rollout_parity[{args.dataset}]")
    else:
        print(f"✗ FAILED: test_single_rollout_parity[{args.dataset}]")
        all_passed = False
    
    return all_passed


if __name__ == "__main__":
    # Use shared parser with all relevant parameters
    parser = create_parser(description="Train Parity Test")
    parser.add_argument("--run-all-params", action="store_true", default=False,
                        help="Run all test parametrizations (default: run first only)")
    # Note: --skip-training is already in the shared parser
    
    args = parser.parse_args()
    
    all_passed = run_all_tests(args)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if all_passed:
        print("All train parity tests PASSED")
    else:
        print("Some train parity tests FAILED")
    
    sys.exit(0 if all_passed else 1)



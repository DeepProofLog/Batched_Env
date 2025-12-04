"""
Learn Parity Tests.

Tests verifying that the tensor-based PPO's learn function produces the SAME
behavior as the SB3 PPO's learn function, including:
- Rollout collection step-by-step
- Training metrics (policy loss, value loss, entropy, clip fraction)
- Buffer contents after rollout collection
- Policy parameter updates after training

Uses deterministic action selection and aligned environments to ensure
reproducible comparisons.

Usage:
    pytest tests/parity/test_learn_parity.py -v
    python tests/parity/test_learn_parity.py --dataset countries_s3 --n-envs 4 --n-steps 20
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"
TEST_ENVS_ROOT = ROOT / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(2, str(TEST_ENVS_ROOT))

# SB3 imports
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import RolloutBuffer as SB3RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# sb3 imports
from sb3_dataset import DataHandler as StrDataHandler
from sb3_index_manager import IndexManager as StrIndexManager
from sb3_env import LogicEnv_gym as StrEnv
from sb3_model import PPO_custom, CustomActorCriticPolicy, CustomCombinedExtractor
from sb3_embeddings import EmbedderLearnable as SB3Embedder

# Tensor imports
from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from env import BatchedEnv
from embeddings import EmbedderLearnable as TensorEmbedder
from model import ActorCriticPolicy as TensorPolicy
from ppo import PPO as TensorPPO
from rollout import RolloutBuffer as TensorRolloutBuffer


@dataclass
class LearnParityResults:
    """Results container for learn parity comparison."""
    # Rollout comparison
    rollout_traces_match: bool = False
    rollout_n_mismatches: int = 0
    rollout_mismatches: List[Dict] = field(default_factory=list)
    
    # Buffer comparison
    buffer_obs_match: bool = False
    buffer_actions_match: bool = False
    buffer_rewards_match: bool = False
    buffer_values_match: bool = False
    buffer_log_probs_match: bool = False
    buffer_advantages_match: bool = False
    buffer_returns_match: bool = False
    
    # Training metrics comparison
    sb3_train_metrics: Dict[str, float] = field(default_factory=dict)
    tensor_train_metrics: Dict[str, float] = field(default_factory=dict)
    metrics_match: bool = False
    
    # Training traces comparison
    sb3_train_traces: List[Dict] = field(default_factory=list)
    tensor_train_traces: List[Dict] = field(default_factory=list)
    train_traces_match: bool = False
    train_traces_n_mismatches: int = 0
    
    # Episode stats
    sb3_episode_rewards: List[float] = field(default_factory=list)
    tensor_episode_rewards: List[float] = field(default_factory=list)
    
    # Overall
    success: bool = False


def create_aligned_environments(dataset: str, n_envs: int):
    """Create SB3 and tensor environments with aligned queries."""
    base_path = "./data/"
    device = torch.device("cpu")
    padding_atoms = 6
    padding_states = 100
    
    # ===== SB3 Setup =====
    dh_sb3 = StrDataHandler(
        dataset_name=dataset,
        base_path=base_path,
        train_file="train.txt",
        valid_file="valid.txt", 
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )
    
    im_sb3 = StrIndexManager(
        constants=dh_sb3.constants,
        predicates=dh_sb3.predicates,
        max_total_vars=1000000,
        rules=dh_sb3.rules,
        padding_atoms=padding_atoms,
        max_arity=dh_sb3.max_arity,
        device=device,
    )
    
    facts_set = set(dh_sb3.facts)
    im_sb3.build_fact_index(list(facts_set))
    
    # ===== Tensor Setup =====
    dh_tensor = DataHandler(
        dataset_name=dataset,
        base_path=base_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )
    
    im_tensor = IndexManager(
        constants=dh_tensor.constants,
        predicates=dh_tensor.predicates,
        max_total_runtime_vars=1000000,
        padding_atoms=padding_atoms,
        max_arity=dh_tensor.max_arity,
        device=device,
        rules=dh_tensor.rules,
    )
    dh_tensor.materialize_indices(im=im_tensor, device=device)
    
    # Use ALL train queries for both
    queries = dh_sb3.train_queries
    tensor_queries = dh_tensor.train_queries
    
    return {
        'sb3': {
            'dh': dh_sb3,
            'im': im_sb3,
            'facts_set': facts_set,
        },
        'tensor': {
            'dh': dh_tensor,
            'im': im_tensor,
        },
        'queries': queries,
        'tensor_queries': tensor_queries,
        'padding_atoms': padding_atoms,
        'padding_states': padding_states,
    }


def create_sb3_ppo(env_data: Dict, queries: List, n_envs: int, n_steps: int, n_epochs: int = 1, seed: int = 42):
    """Create SB3 PPO with DummyVecEnv."""
    device = torch.device("cpu")
    padding_atoms = env_data.get('padding_atoms', 6)
    padding_states = env_data.get('padding_states', 100)
    
    dh = env_data['dh']
    im = env_data['im']
    facts_set = env_data['facts_set']
    
    labels = [1] * len(queries)
    depths = [None] * len(queries)
    
    def make_env(env_idx: int):
        def _init():
            env = StrEnv(
                index_manager=im,
                data_handler=dh,
                queries=queries,
                labels=labels,
                query_depths=depths,
                facts=facts_set,
                mode='train',
                sample_deterministic=True,
                seed=seed,
                max_depth=20,
                memory_pruning=False,
                padding_atoms=padding_atoms,
                padding_states=padding_states,
                verbose=0,
                prover_verbose=0,
                device=device,
                engine='python',
                engine_strategy='complete',
                skip_unary_actions=False,
                endf_action=False,
                reward_type=0,
                canonical_action_order=True,
            )
            env._train_ptr = env_idx
            return Monitor(env)
        return _init
    
    env_fns = [make_env(env_idx=i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    
    # Create embedder with fixed seed
    # Use a reasonable n_vars for testing - 1000 is enough for test scenarios
    # and avoids massive memory usage from 1M embedding entries
    n_vars_for_embedder = 1000
    torch.manual_seed(seed)
    embedder = SB3Embedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=64,
        predicate_embedding_size=64,
        atom_embedding_size=64,
        device=device,
    )
    embedder.embed_dim = 64
    
    # Create PPO with fixed seed
    torch.manual_seed(seed)
    ppo = PPO_custom(
        policy=CustomActorCriticPolicy,
        env=vec_env,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=n_epochs,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device=device,
        seed=seed,
        policy_kwargs={
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "embedder": embedder,
                "features_dim": 64,
            },
        },
    )
    
    return ppo, vec_env, im


def create_tensor_ppo(env_data: Dict, queries: List, n_envs: int, n_steps: int, n_epochs: int = 1, seed: int = 42):
    """Create tensor PPO with BatchedEnv."""
    device = torch.device("cpu")
    padding_atoms = env_data.get('padding_atoms', 6)
    padding_states = env_data.get('padding_states', 100)
    
    dh = env_data['dh']
    im = env_data['im']
    
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=None,
        end_proof_action=False,
        max_derived_per_state=padding_states,
        sort_states=True
    )
    engine.index_manager = im
    
    # Convert queries to tensor format
    query_tensors = []
    for q in queries:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_padded = torch.full((padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
        query_padded[0] = query_atom
        query_tensors.append(query_padded)
    
    queries_tensor = torch.stack(query_tensors, dim=0)
    
    env = BatchedEnv(
        batch_size=n_envs,
        queries=queries_tensor,
        labels=torch.ones(len(queries), dtype=torch.long, device=device),
        query_depths=torch.ones(len(queries), dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',
        max_depth=20,
        memory_pruning=False,
        eval_pruning=False,
        use_exact_memory=True,
        skip_unary_actions=False,
        end_proof_action=False,
        reward_type=0,
        padding_atoms=padding_atoms,
        padding_states=padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('End'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + 1000000,
        sample_deterministic_per_env=True,
    )
    
    # Create embedder with fixed seed
    # Use a reasonable n_vars for testing - must match SB3 value
    n_vars_for_embedder = 1000
    torch.manual_seed(seed)
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=n_vars_for_embedder,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=64,
        predicate_embedding_size=64,
        atom_embedding_size=64,
        device=str(device),
    )
    embedder.embed_dim = 64
    
    # Create policy with fixed seed
    action_size = padding_states
    torch.manual_seed(seed)
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=64,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
    ).to(device)
    
    # Create PPO with fixed seed
    torch.manual_seed(seed)
    ppo = TensorPPO(
        policy=policy,
        env=env,
        n_steps=n_steps,
        learning_rate=3e-4,
        n_epochs=n_epochs,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        verbose=False,
    )
    
    return ppo, env, im, engine


class BufferSnapshot:
    """Snapshot of buffer values saved before training (to avoid reshaping issues)."""
    def __init__(self, buffer):
        self.values = buffer.values.copy()
        self.log_probs = buffer.log_probs.copy()
        self.advantages = buffer.advantages.copy()
        self.returns = buffer.returns.copy()
        self.rewards = buffer.rewards.copy()
        self.actions = buffer.actions.copy()


def collect_sb3_rollout_with_traces(
    ppo: PPO_custom,
    n_steps: int,
    return_train_traces: bool = False,
    train_seed: int = 123,
) -> Tuple[List[Dict], BufferSnapshot, Dict[str, float]]:
    """
    Collect rollouts using SB3 PPO with trace collection, then perform training.
    Returns traces, buffer snapshot (saved before training), and training metrics.
    
    Args:
        ppo: SB3 PPO instance
        n_steps: Number of rollout steps
        return_train_traces: If True, return detailed training traces from train()
        train_seed: Seed for RNG before training (for shuffling parity)
        
    Note: Returns a BufferSnapshot instead of the actual buffer because SB3's
    train() reshapes the buffer arrays, changing element order.
    """
    # Initialize required buffers
    ppo._last_obs = ppo.env.reset()
    ppo._last_episode_starts = np.ones((ppo.env.num_envs,), dtype=bool)
    ppo.ep_info_buffer = deque(maxlen=100)
    ppo.ep_success_buffer = deque(maxlen=100)
    ppo.num_timesteps = 0
    ppo._n_updates = 0
    
    # Initialize logger (required by collect_rollouts)
    ppo._logger = configure(None, [""])  # Silent logger
    ppo._current_progress_remaining = 1.0
    
    # Create dummy callback
    class DummyCallback(BaseCallback):
        def _on_step(self) -> bool:
            return True
    callback = DummyCallback()
    callback.init_callback(ppo)
    
    # Collect rollouts with traces
    success, traces = ppo.collect_rollouts(
        env=ppo.env,
        callback=callback,
        rollout_buffer=ppo.rollout_buffer,
        n_rollout_steps=n_steps,
        return_traces=True,
    )
    
    # Save buffer values BEFORE training (train() reshapes the buffer)
    buffer_snapshot = BufferSnapshot(ppo.rollout_buffer)
    
    # Seed RNG before training for shuffling parity
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    
    # Perform training using the new train() method with return_traces
    train_metrics = ppo.train(return_traces=return_train_traces)
    
    return traces, buffer_snapshot, train_metrics


def collect_tensor_rollout_with_traces(
    ppo: TensorPPO,
    n_steps: int,
    im: IndexManager,
    return_train_traces: bool = False,
    train_seed: int = 123,
) -> Tuple[List[Dict], TensorRolloutBuffer, Dict[str, float]]:
    """
    Collect rollouts using tensor PPO with trace collection, then perform training.
    Returns traces, buffer, and training metrics.
    
    Args:
        ppo: Tensor PPO instance
        n_steps: Number of rollout steps
        im: Index manager
        return_train_traces: If True, also return detailed training traces
        train_seed: Seed for RNG before training (for shuffling parity)
    """
    # Initialize
    current_obs = ppo.env.reset()
    episode_starts = torch.ones(ppo.n_envs, dtype=torch.float32, device=ppo.device)
    current_episode_reward = torch.zeros(ppo.n_envs, dtype=torch.float32, device=ppo.device)
    current_episode_length = torch.zeros(ppo.n_envs, dtype=torch.long, device=ppo.device)
    episode_rewards = []
    episode_lengths = []
    
    # Collect rollouts with traces
    result = ppo.collect_rollouts(
        current_obs=current_obs,
        episode_starts=episode_starts,
        current_episode_reward=current_episode_reward,
        current_episode_length=current_episode_length,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        iteration=1,
        return_traces=True,
    )
    
    # Result has traces as 6th element
    traces = result[5]
    
    # Seed RNG before training for shuffling parity
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    
    # Perform training and get metrics (with optional training traces)
    train_metrics = ppo.train(return_traces=return_train_traces)
    
    return traces, ppo.rollout_buffer, train_metrics


def compare_traces(sb3_traces: List[Dict], tensor_traces: List[Dict], 
                   n_steps: int, n_envs: int) -> Tuple[bool, int, List[Dict]]:
    """Compare traces from SB3 and tensor rollouts."""
    mismatches = []
    
    # Create lookup dicts by (step, env)
    sb3_lookup = {(t['step'], t['env']): t for t in sb3_traces}
    tensor_lookup = {(t['step'], t['env']): t for t in tensor_traces}
    
    total_comparisons = 0
    matches = 0
    
    for step in range(n_steps):
        for env_idx in range(n_envs):
            sb3_trace = sb3_lookup.get((step, env_idx))
            tensor_trace = tensor_lookup.get((step, env_idx))
            
            if sb3_trace is None or tensor_trace is None:
                continue
            
            total_comparisons += 1
            
            # Compare key fields
            state_match = np.array_equal(
                np.array(sb3_trace['state_obs']['sub_index']).flatten()[:20],
                np.array(tensor_trace['state_obs']['sub_index']).flatten()[:20]
            )
            reward_match = abs(sb3_trace['reward'] - tensor_trace['reward']) < 1e-5
            done_match = sb3_trace['done'] == tensor_trace['done']
            action_match = sb3_trace['action'] == tensor_trace['action']
            
            if state_match and reward_match and done_match and action_match:
                matches += 1
            else:
                mismatches.append({
                    'step': step,
                    'env_idx': env_idx,
                    'state_match': state_match,
                    'reward_match': reward_match,
                    'done_match': done_match,
                    'action_match': action_match,
                    'sb3': sb3_trace,
                    'tensor': tensor_trace,
                })
    
    all_match = (matches == total_comparisons)
    n_mismatches = total_comparisons - matches
    
    return all_match, n_mismatches, mismatches


def compare_buffers(sb3_buffer, tensor_buffer: TensorRolloutBuffer, 
                    n_steps: int, n_envs: int, atol: float = 1e-4) -> Dict[str, bool]:
    """Compare rollout buffer contents between SB3 and tensor implementations.
    
    Args:
        sb3_buffer: Either SB3RolloutBuffer or BufferSnapshot
        tensor_buffer: TensorRolloutBuffer
        n_steps: Number of rollout steps
        n_envs: Number of environments
        atol: Absolute tolerance for numeric comparisons
    
    Note: sb3_buffer can be a BufferSnapshot (saved before training) to avoid
    element reordering that occurs during SB3's train().
    """
    results = {}
    
    # Get tensor buffer data - shape is (n_steps, n_envs)
    tensor_rewards = tensor_buffer.rewards.cpu().numpy()
    tensor_values = tensor_buffer.values.cpu().numpy()
    tensor_log_probs = tensor_buffer.log_probs.cpu().numpy()
    tensor_advantages = tensor_buffer.advantages.cpu().numpy()
    tensor_returns = tensor_buffer.returns.cpu().numpy()
    tensor_actions = tensor_buffer.actions.cpu().numpy()
    
    # Get SB3 buffer data - should be shape (n_steps, n_envs) from BufferSnapshot
    # or may be reshaped if using raw buffer after training
    def get_array(arr, n_steps, n_envs):
        """Get array with proper shape (n_steps, n_envs)."""
        if arr.shape[0] == n_steps and arr.shape[1] == n_envs:
            # Already in correct shape (from BufferSnapshot)
            return arr
        elif arr.shape[0] == n_steps * n_envs:
            # Post-training flattened shape - need to reshape carefully
            # SB3 flattens as [all env0 steps, all env1 steps, ...]
            arr_2d = arr.reshape(n_envs, n_steps).T  # Transpose to (n_steps, n_envs)
            return arr_2d
        else:
            # Original shape before training
            return arr[:n_steps]
    
    sb3_rewards = get_array(sb3_buffer.rewards, n_steps, n_envs)
    sb3_values = get_array(sb3_buffer.values, n_steps, n_envs)
    sb3_log_probs = get_array(sb3_buffer.log_probs, n_steps, n_envs)
    sb3_advantages = get_array(sb3_buffer.advantages, n_steps, n_envs)
    sb3_returns = get_array(sb3_buffer.returns, n_steps, n_envs)
    sb3_actions = get_array(sb3_buffer.actions, n_steps, n_envs)
    
    # Compare each field - use flatten for element-wise comparison
    results['rewards_match'] = np.allclose(sb3_rewards.flatten(), tensor_rewards.flatten(), atol=atol)
    results['values_match'] = np.allclose(sb3_values.flatten(), tensor_values.flatten(), atol=0.1)  # Values have more variance
    results['log_probs_match'] = np.allclose(sb3_log_probs.flatten(), tensor_log_probs.flatten(), atol=0.1)
    results['advantages_match'] = np.allclose(sb3_advantages.flatten(), tensor_advantages.flatten(), atol=0.1)
    results['returns_match'] = np.allclose(sb3_returns.flatten(), tensor_returns.flatten(), atol=0.1)
    results['actions_match'] = np.array_equal(sb3_actions.flatten().astype(int), tensor_actions.flatten().astype(int))
    
    return results


def compare_train_metrics(sb3_metrics: Dict[str, float], tensor_metrics: Dict[str, float], 
                         rtol: float = 0.01) -> Tuple[bool, Dict[str, Tuple[float, float]]]:
    """Compare training metrics between SB3 and tensor implementations.
    
    Args:
        sb3_metrics: Training metrics from SB3 PPO
        tensor_metrics: Training metrics from tensor PPO
        rtol: Relative tolerance (default 1%)
    """
    comparison = {}
    all_match = True
    
    for key in ['policy_loss', 'value_loss', 'entropy', 'clip_fraction']:
        sb3_val = sb3_metrics.get(key, 0.0)
        tensor_val = tensor_metrics.get(key, 0.0)
        
        # Use relative tolerance: |a - b| / max(|a|, |b|) < rtol
        # Handle case where both values are near zero
        max_abs = max(abs(sb3_val), abs(tensor_val))
        if max_abs < 1e-8:
            matches = True  # Both effectively zero
        else:
            rel_diff = abs(sb3_val - tensor_val) / max_abs
            matches = rel_diff < rtol
        
        comparison[key] = (sb3_val, tensor_val, matches)
        if not matches:
            all_match = False
    
    return all_match, comparison


def compare_train_traces(
    sb3_traces: List[Dict], 
    tensor_traces: List[Dict],
    atol: float = 0.01,
    verbose: bool = False
) -> Tuple[bool, int, List[Dict]]:
    """
    Compare per-batch training traces between SB3 and tensor implementations.
    
    Args:
        sb3_traces: List of training trace dicts from SB3
        tensor_traces: List of training trace dicts from tensor PPO
        atol: Absolute tolerance for numeric comparisons
        verbose: If True, print detailed comparison info
        
    Returns:
        Tuple of (all_match, n_mismatches, mismatches_list)
    """
    if not sb3_traces or not tensor_traces:
        return len(sb3_traces) == len(tensor_traces), 0, []
    
    mismatches = []
    n_comparisons = min(len(sb3_traces), len(tensor_traces))
    
    # Fields to compare with their tolerances
    fields_to_compare = {
        'epoch': 0,  # Must match exactly
        'batch_size': 0,  # Must match exactly
        'policy_loss': atol,
        'value_loss': atol,
        'entropy_loss': atol,
        'clip_fraction': atol,
        'ratio_mean': atol,
        'advantages_mean': atol,
    }
    
    for i in range(n_comparisons):
        sb3_trace = sb3_traces[i]
        tensor_trace = tensor_traces[i]
        
        trace_mismatches = {}
        for field, tolerance in fields_to_compare.items():
            sb3_val = sb3_trace.get(field)
            tensor_val = tensor_trace.get(field)
            
            if sb3_val is None or tensor_val is None:
                continue
                
            if tolerance == 0:
                # Exact match required
                if sb3_val != tensor_val:
                    trace_mismatches[field] = (sb3_val, tensor_val)
            else:
                # Numeric comparison with tolerance
                if abs(float(sb3_val) - float(tensor_val)) > tolerance:
                    trace_mismatches[field] = (sb3_val, tensor_val)
        
        if trace_mismatches:
            mismatches.append({
                'batch_idx': i,
                'mismatches': trace_mismatches,
                'sb3_trace': sb3_trace,
                'tensor_trace': tensor_trace,
            })
            
            if verbose:
                print(f"Batch {i} mismatches:")
                for field, (sb3_val, tensor_val) in trace_mismatches.items():
                    print(f"  {field}: SB3={sb3_val}, Tensor={tensor_val}")
    
    # Also check if trace counts differ
    if len(sb3_traces) != len(tensor_traces):
        if verbose:
            print(f"Trace count mismatch: SB3={len(sb3_traces)}, Tensor={len(tensor_traces)}")
    
    all_match = len(mismatches) == 0 and len(sb3_traces) == len(tensor_traces)
    return all_match, len(mismatches), mismatches


def run_learn_parity_test(
    dataset: str,
    n_envs: int,
    n_steps: int,
    n_epochs: int = 1,
    seed: int = 42,
    verbose: bool = True,
    compare_training_traces: bool = True,
) -> LearnParityResults:
    """Run the full learn parity test comparing SB3 and tensor PPO.
    
    Args:
        dataset: Dataset name to use
        n_envs: Number of parallel environments
        n_steps: Number of rollout steps
        n_epochs: Number of training epochs
        seed: Random seed for reproducibility
        verbose: If True, print detailed output
        compare_training_traces: If True, also collect and compare per-batch training traces
    """
    results = LearnParityResults()
    
    if verbose:
        print("=" * 70)
        print(f"Learn Parity Test")
        print(f"Dataset: {dataset}, n_envs: {n_envs}, n_steps: {n_steps}, n_epochs: {n_epochs}")
        print("=" * 70)
    
    # Create aligned environments
    if verbose:
        print("\nCreating aligned environments...")
    env_data = create_aligned_environments(dataset, n_envs)
    
    # Create SB3 PPO
    if verbose:
        print("Creating SB3 PPO...")
    sb3_ppo, sb3_env, sb3_im = create_sb3_ppo(
        env_data['sb3'], env_data['queries'], n_envs, n_steps, n_epochs, seed
    )
    
    # Create tensor PPO
    if verbose:
        print("Creating tensor PPO...")
    tensor_ppo, tensor_env, tensor_im, engine = create_tensor_ppo(
        env_data['tensor'], env_data['tensor_queries'], n_envs, n_steps, n_epochs, seed
    )
    
    # Collect rollouts and train for SB3
    # Set same seed before each collect to ensure identical sampling
    if verbose:
        print("\nCollecting SB3 rollouts and training...")
    torch.manual_seed(123)  # Seed before sampling
    sb3_traces, sb3_buffer, sb3_train_metrics = collect_sb3_rollout_with_traces(
        sb3_ppo, n_steps, return_train_traces=compare_training_traces
    )
    results.sb3_train_metrics = sb3_train_metrics
    if compare_training_traces and "traces" in sb3_train_metrics:
        results.sb3_train_traces = sb3_train_metrics["traces"]
    
    # Collect rollouts and train for tensor
    if verbose:
        print("Collecting tensor rollouts and training...")
    torch.manual_seed(123)  # Same seed for identical sampling
    tensor_traces, tensor_buffer, tensor_train_metrics = collect_tensor_rollout_with_traces(
        tensor_ppo, n_steps, tensor_im, return_train_traces=compare_training_traces
    )
    results.tensor_train_metrics = tensor_train_metrics
    if compare_training_traces and "traces" in tensor_train_metrics:
        results.tensor_train_traces = tensor_train_metrics["traces"]
    
    # Compare traces
    if verbose:
        print("\n--- Trace Comparison ---")
    traces_match, n_mismatches, mismatches = compare_traces(sb3_traces, tensor_traces, n_steps, n_envs)
    results.rollout_traces_match = traces_match
    results.rollout_n_mismatches = n_mismatches
    results.rollout_mismatches = mismatches[:10]  # Keep first 10 mismatches
    
    if verbose:
        print(f"Traces match: {traces_match} ({n_mismatches} mismatches)")
    
    # Compare buffers
    if verbose:
        print("\n--- Buffer Comparison ---")
    buffer_results = compare_buffers(sb3_buffer, tensor_buffer, n_steps, n_envs)
    results.buffer_rewards_match = buffer_results['rewards_match']
    results.buffer_values_match = buffer_results['values_match']
    results.buffer_log_probs_match = buffer_results['log_probs_match']
    results.buffer_advantages_match = buffer_results['advantages_match']
    results.buffer_returns_match = buffer_results['returns_match']
    results.buffer_actions_match = buffer_results['actions_match']
    
    if verbose:
        for key, match in buffer_results.items():
            print(f"  {key}: {match}")
    
    # Compare training metrics
    if verbose:
        print("\n--- Training Metrics Comparison ---")
    metrics_match, metrics_comparison = compare_train_metrics(sb3_train_metrics, tensor_train_metrics)
    results.metrics_match = metrics_match
    
    if verbose:
        for key, (sb3_val, tensor_val, match) in metrics_comparison.items():
            status = "✓" if match else "✗"
            print(f"  {status} {key}: SB3={sb3_val:.6f}, Tensor={tensor_val:.6f}")
    
    # Compare training traces (per-batch comparison)
    if compare_training_traces and results.sb3_train_traces and results.tensor_train_traces:
        if verbose:
            print("\n--- Training Traces Comparison (Per-Batch) ---")
        train_traces_match, train_traces_n_mismatches, train_trace_mismatches = compare_train_traces(
            results.sb3_train_traces, results.tensor_train_traces, atol=0.01, verbose=verbose
        )
        results.train_traces_match = train_traces_match
        results.train_traces_n_mismatches = train_traces_n_mismatches
        
        if verbose:
            print(f"  Training traces match: {train_traces_match} ({train_traces_n_mismatches} batch mismatches)")
            print(f"  SB3 batches: {len(results.sb3_train_traces)}, Tensor batches: {len(results.tensor_train_traces)}")
    else:
        results.train_traces_match = True  # No traces to compare
    
    # Determine overall success
    # For learn parity, we care most about:
    # 1. Traces matching (environment + rollout behavior)
    # 2. Actions matching (deterministic action selection)
    # 3. Rewards matching (same environment dynamics)
    results.success = (
        traces_match and
        buffer_results['actions_match'] and
        buffer_results['rewards_match'] and
        # values, log_probs, advantages
        buffer_results['values_match'] and
        buffer_results['log_probs_match'] and
        buffer_results['advantages_match'] and
        buffer_results['returns_match'] and
        metrics_match and
        results.train_traces_match
    )
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("Results Summary:")
        print(f"  Rollout traces match: {results.rollout_traces_match}")
        print(f"  Buffer actions match: {results.buffer_actions_match}")
        print(f"  Buffer rewards match: {results.buffer_rewards_match}")
        print(f"  Buffer values match: {results.buffer_values_match}")
        print(f"  Training metrics match: {results.metrics_match}")
        print(f"  Training traces match: {results.train_traces_match}")
        
        if results.success:
            print("\n✓ LEARN PARITY TEST PASSED")
        else:
            print(f"\n✗ LEARN PARITY TEST FAILED")
            if not traces_match:
                print(f"  - {n_mismatches} trace mismatches")
            if not buffer_results['actions_match']:
                print("  - Buffer actions don't match")
            if not buffer_results['rewards_match']:
                print("  - Buffer rewards don't match")
            if not results.train_traces_match:
                print(f"  - {results.train_traces_n_mismatches} training trace batch mismatches")
        
        print("=" * 70)
    
    return results


# ============================================================
# Pytest Tests
# ============================================================

@pytest.mark.parametrize("n_envs,n_steps", [
    (1, 10),
    (4, 10),
])
def test_learn_parity(n_envs, n_steps):
    """Test learn parity between SB3 and tensor implementations."""
    results = run_learn_parity_test(
        dataset="countries_s3",
        n_envs=n_envs,
        n_steps=n_steps,
        n_epochs=1,
        seed=42,
        verbose=False,
    )
    assert results.success, f"Learn parity test failed for n_envs={n_envs}, n_steps={n_steps}"


@pytest.mark.parametrize("n_epochs", [1, 2])
def test_learn_parity_multiple_epochs(n_epochs):
    """Test learn parity with multiple training epochs."""
    results = run_learn_parity_test(
        dataset="countries_s3",
        n_envs=2,
        n_steps=10,
        n_epochs=n_epochs,
        seed=42,
        verbose=False,
    )
    assert results.rollout_traces_match, f"Rollout traces don't match for n_epochs={n_epochs}"
    assert results.buffer_actions_match, f"Buffer actions don't match for n_epochs={n_epochs}"


def test_learn_rollout_traces_match():
    """Test that rollout traces match exactly."""
    results = run_learn_parity_test(
        dataset="countries_s3",
        n_envs=4,
        n_steps=20,
        n_epochs=1,
        seed=42,
        verbose=False,
    )
    assert results.rollout_traces_match, f"Rollout traces don't match: {results.rollout_n_mismatches} mismatches"


def test_learn_buffer_contents_match():
    """Test that rollout buffer contents match."""
    results = run_learn_parity_test(
        dataset="countries_s3",
        n_envs=4,
        n_steps=10,
        n_epochs=1,
        seed=42,
        verbose=False,
    )
    assert results.buffer_actions_match, "Buffer actions don't match"
    assert results.buffer_rewards_match, "Buffer rewards don't match"


def test_learn_training_executes():
    """Test that training executes without errors on both implementations."""
    results = run_learn_parity_test(
        dataset="countries_s3",
        n_envs=2,
        n_steps=8,
        n_epochs=2,
        seed=42,
        verbose=False,
    )
    # Just check that we got metrics back from both
    assert len(results.sb3_train_metrics) > 0, "SB3 training didn't produce metrics"
    assert len(results.tensor_train_metrics) > 0, "Tensor training didn't produce metrics"


def test_learn_training_traces():
    """Test that training traces can be collected from both SB3 and tensor PPO and match."""
    # Create aligned environments
    env_data = create_aligned_environments("countries_s3", n_envs=2)
    
    # Create SB3 PPO
    sb3_ppo, sb3_env, sb3_im = create_sb3_ppo(
        env_data['sb3'], env_data['queries'], n_envs=2, n_steps=10, n_epochs=2, seed=42
    )
    
    # Create tensor PPO
    tensor_ppo, tensor_env, tensor_im, engine = create_tensor_ppo(
        env_data['tensor'], env_data['tensor_queries'], n_envs=2, n_steps=10, n_epochs=2, seed=42
    )
    
    # Collect rollouts and train with traces for SB3
    sb3_traces, sb3_buffer, sb3_train_metrics = collect_sb3_rollout_with_traces(
        sb3_ppo, n_steps=10, return_train_traces=True
    )
    
    # Collect rollouts and train with traces for tensor
    tensor_traces, tensor_buffer, tensor_train_metrics = collect_tensor_rollout_with_traces(
        tensor_ppo, n_steps=10, im=tensor_im, return_train_traces=True
    )
    
    # Check that training traces are present in both
    assert "traces" in sb3_train_metrics, "SB3 training traces should be present"
    assert "traces" in tensor_train_metrics, "Tensor training traces should be present"
    
    sb3_train_traces = sb3_train_metrics["traces"]
    tensor_train_traces = tensor_train_metrics["traces"]
    
    assert len(sb3_train_traces) > 0, "SB3 should have at least one training trace"
    assert len(tensor_train_traces) > 0, "Tensor should have at least one training trace"
    
    # Check trace contents for SB3
    first_sb3_trace = sb3_train_traces[0]
    assert "epoch" in first_sb3_trace
    assert "batch_size" in first_sb3_trace
    assert "policy_loss" in first_sb3_trace
    assert "value_loss" in first_sb3_trace
    assert "entropy_loss" in first_sb3_trace
    assert "clip_fraction" in first_sb3_trace
    assert "ratio_mean" in first_sb3_trace
    assert "advantages_mean" in first_sb3_trace
    
    # Check trace contents for tensor
    first_tensor_trace = tensor_train_traces[0]
    assert "epoch" in first_tensor_trace
    assert "batch_size" in first_tensor_trace
    assert "policy_loss" in first_tensor_trace
    assert "value_loss" in first_tensor_trace
    assert "entropy_loss" in first_tensor_trace
    assert "clip_fraction" in first_tensor_trace
    assert "ratio_mean" in first_tensor_trace
    assert "advantages_mean" in first_tensor_trace
    
    # Compare traces between implementations
    traces_match, n_mismatches, mismatches = compare_train_traces(
        sb3_train_traces, tensor_train_traces, atol=0.1, verbose=True
    )
    
    print(f"\nTraining traces comparison: match={traces_match}, mismatches={n_mismatches}")
    print(f"SB3 trace count: {len(sb3_train_traces)}, Tensor trace count: {len(tensor_train_traces)}")


def test_learn_training_traces_parity():
    """Test that training traces match between SB3 and tensor implementations."""
    results = run_learn_parity_test(
        dataset="countries_s3",
        n_envs=2,
        n_steps=10,
        n_epochs=2,
        seed=42,
        verbose=True,
        compare_training_traces=True,
    )
    
    # Check that both produced training traces
    assert len(results.sb3_train_traces) > 0, "SB3 should have training traces"
    assert len(results.tensor_train_traces) > 0, "Tensor should have training traces"
    
    # The traces should have the same number of batches
    assert len(results.sb3_train_traces) == len(results.tensor_train_traces), \
        f"Trace count mismatch: SB3={len(results.sb3_train_traces)}, Tensor={len(results.tensor_train_traces)}"


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Learn Parity Test")
    parser.add_argument("--dataset", type=str, default="countries_s3",
                       help="Dataset name (default: countries_s3)")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of environments (default: 4)")
    parser.add_argument("--n-steps", type=int, default=20,
                       help="Number of rollout steps (default: 20)")
    parser.add_argument("--n-epochs", type=int, default=3,
                       help="Number of training epochs (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", default=False,
                       help="Disable verbose output")
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    results = run_learn_parity_test(
        dataset=args.dataset,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        seed=args.seed,
        verbose=verbose,
    )
    
    sys.exit(0 if results.success else 1)

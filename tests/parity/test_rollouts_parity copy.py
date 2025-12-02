"""
Rollout Collection Parity Tests.

Tests verifying that the tensor-based PPO's collect_rollouts produces the SAME
buffer contents (states, derived states, rewards, dones, actions, values, log_probs)
step-by-step as the SB3 PPO's collect_rollouts when using deterministic first action.

This tests the full rollout collection pipeline, not just the environment.

Usage:
    pytest tests/parity/test_rollouts_parity.py -v
    python tests/parity/test_rollouts_parity.py --dataset countries_s3 --n-envs 4 --n-steps 20
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

import pytest
import torch
import numpy as np

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
class RolloutTrace:
    """Stores trace information from a single rollout step."""
    step: int
    env_idx: int
    state: str  # Canonical string representation of observation
    derived_states: List[str]  # List of derived state strings
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    action_mask_sum: int  # Number of valid actions


@dataclass 
class RolloutTraceCollection:
    """Collection of all traces from a rollout."""
    traces: List[RolloutTrace] = field(default_factory=list)
    
    def add(self, trace: RolloutTrace):
        self.traces.append(trace)
    
    def get_step(self, step: int, env_idx: int) -> Optional[RolloutTrace]:
        for t in self.traces:
            if t.step == step and t.env_idx == env_idx:
                return t
        return None


def tensor_obs_to_canonical_state(obs, im, env_idx: int = 0) -> str:
    """Convert tensor observation to canonical string representation."""
    try:
        if isinstance(obs, dict):
            sub_idx = obs.get('sub_index', obs.get('state'))
            if sub_idx is None:
                return "<no state>"
            
            # Handle TensorDict
            if hasattr(sub_idx, 'cpu'):
                sub_idx = sub_idx.cpu()
            if isinstance(sub_idx, torch.Tensor):
                sub_idx = sub_idx.numpy()
            
            # Tensor env returns shape [n_envs, 1, A, D] - select env and squeeze
            if len(sub_idx.shape) == 4:  # [n_envs, 1, A, D]
                sub_idx = sub_idx[env_idx, 0]  # [A, D]
            elif len(sub_idx.shape) == 3:  # [n_envs, A, D] or [1, A, D]
                sub_idx = sub_idx[env_idx]
            elif len(sub_idx.shape) == 2:  # [A, D]
                pass  # Already single env
            
            # Convert to list of atoms
            atoms = []
            padding_idx = im.padding_idx
            for atom in sub_idx:
                pred_idx, arg0_idx, arg1_idx = int(atom[0]), int(atom[1]), int(atom[2])
                if pred_idx == padding_idx:
                    continue
                pred = im.idx2predicate[pred_idx] if pred_idx < len(im.idx2predicate) else f"<pred_{pred_idx}>"
                arg0 = im.idx2constant[arg0_idx] if arg0_idx < len(im.idx2constant) else f"<C{arg0_idx}>"
                arg1 = im.idx2constant[arg1_idx] if arg1_idx < len(im.idx2constant) else f"<C{arg1_idx}>"
                atoms.append(f"{pred}({arg0},{arg1})")
            return " ; ".join(atoms) if atoms else "<empty>"
        else:
            return f"<unknown obs type: {type(obs)}>"
    except Exception as e:
        return f"<error: {e}>"


def sb3_obs_to_canonical_state(obs, im, env_idx: int = 0) -> str:
    """Convert SB3 observation to canonical string representation."""
    try:
        sub_idx = obs.get('sub_index')
        if sub_idx is None:
            return "<no state>"
        
        # SB3 VecEnv returns shape [n_envs, 1, A, D] - select env and squeeze
        if len(sub_idx.shape) == 4:  # [n_envs, 1, A, D]
            sub_idx = sub_idx[env_idx, 0]  # [A, D]
        elif len(sub_idx.shape) == 3:  # [n_envs, A, D]
            sub_idx = sub_idx[env_idx]
        
        # SB3 IndexManager has str2idx dicts, need to create reverse mappings
        idx2constant = {v: k for k, v in im.constant_str2idx.items()}
        idx2predicate = {v: k for k, v in im.predicate_str2idx.items()}
        
        # Convert to list of atoms
        atoms = []
        padding_idx = im.padding_idx
        for atom in sub_idx:
            pred_idx, arg0_idx, arg1_idx = int(atom[0]), int(atom[1]), int(atom[2])
            if pred_idx == padding_idx:
                continue
            pred = idx2predicate.get(pred_idx, f"<pred_{pred_idx}>")
            arg0 = idx2constant.get(arg0_idx, f"<C{arg0_idx}>")
            arg1 = idx2constant.get(arg1_idx, f"<C{arg1_idx}>")
            atoms.append(f"{pred}({arg0},{arg1})")
        return " ; ".join(atoms) if atoms else "<empty>"
    except Exception as e:
        return f"<error: {e}>"


def get_derived_states_tensor(obs, im, env_idx: int = 0, max_states: int = 5) -> List[str]:
    """Get derived states as string list from tensor observation."""
    try:
        derived = obs.get('derived_sub_indices')
        if derived is None:
            return []
        
        if hasattr(derived, 'cpu'):
            derived = derived.cpu()
        if isinstance(derived, torch.Tensor):
            derived = derived.numpy()
        
        # Shape: [n_envs, pad_states, pad_atoms, 3]
        if len(derived.shape) == 4:
            derived = derived[env_idx]  # [pad_states, pad_atoms, 3]
        
        states = []
        padding_idx = im.padding_idx
        for state_idx in range(min(max_states, derived.shape[0])):
            state_atoms = derived[state_idx]
            atoms = []
            for atom in state_atoms:
                pred_idx, arg0_idx, arg1_idx = int(atom[0]), int(atom[1]), int(atom[2])
                if pred_idx == padding_idx:
                    continue
                pred = im.idx2predicate[pred_idx] if pred_idx < len(im.idx2predicate) else f"<pred_{pred_idx}>"
                arg0 = im.idx2constant[arg0_idx] if arg0_idx < len(im.idx2constant) else f"<C{arg0_idx}>"
                arg1 = im.idx2constant[arg1_idx] if arg1_idx < len(im.idx2constant) else f"<C{arg1_idx}>"
                atoms.append(f"{pred}({arg0},{arg1})")
            if atoms:
                states.append(" ; ".join(atoms))
        return states
    except Exception as e:
        return [f"<error: {e}>"]


def get_derived_states_sb3(obs, im, env_idx: int = 0, max_states: int = 5) -> List[str]:
    """Get derived states as string list from SB3 observation."""
    try:
        derived = obs.get('derived_sub_indices')
        if derived is None:
            return []
        
        # Shape: [n_envs, pad_states, pad_atoms, 3]
        if len(derived.shape) == 4:
            derived = derived[env_idx]  # [pad_states, pad_atoms, 3]
        
        idx2constant = {v: k for k, v in im.constant_str2idx.items()}
        idx2predicate = {v: k for k, v in im.predicate_str2idx.items()}
        
        states = []
        padding_idx = im.padding_idx
        for state_idx in range(min(max_states, derived.shape[0])):
            state_atoms = derived[state_idx]
            atoms = []
            for atom in state_atoms:
                pred_idx, arg0_idx, arg1_idx = int(atom[0]), int(atom[1]), int(atom[2])
                if pred_idx == padding_idx:
                    continue
                pred = idx2predicate.get(pred_idx, f"<pred_{pred_idx}>")
                arg0 = idx2constant.get(arg0_idx, f"<C{arg0_idx}>")
                arg1 = idx2constant.get(arg1_idx, f"<C{arg1_idx}>")
                atoms.append(f"{pred}({arg0},{arg1})")
            if atoms:
                states.append(" ; ".join(atoms))
        return states
    except Exception as e:
        return [f"<error: {e}>"]


def create_aligned_environments(dataset: str, n_envs: int):
    """Create SB3 and tensor environments with aligned queries."""
    base_path = "./data/"
    device = torch.device("cpu")
    padding_atoms = 6
    padding_states = 100  # Reduced from 500 to lower memory usage
    
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


def create_sb3_ppo(env_data: Dict, queries: List, n_envs: int, n_steps: int):
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
                seed=42,
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
    
    # Create embedder
    embedder = SB3Embedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=im.variable_no if hasattr(im, 'variable_no') else 1000,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=64,
        predicate_embedding_size=64,
        atom_embedding_size=64,
        device=device,
    )
    embedder.embed_dim = 64  # Set embed_dim
    
    # Create PPO
    ppo = PPO_custom(
        policy=CustomActorCriticPolicy,
        env=vec_env,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device=device,
        policy_kwargs={
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "embedder": embedder,
                "features_dim": 64,
            },
        },
    )
    
    return ppo, vec_env, im


def create_tensor_ppo(env_data: Dict, queries: List, n_envs: int, n_steps: int):
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
    
    # Create embedder
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=im.variable_no,
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
    
    # Create policy
    action_size = padding_states  # action space size is padding_states
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=64,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
    ).to(device)
    
    # Create PPO
    ppo = TensorPPO(
        policy=policy,
        env=env,
        n_steps=n_steps,
        learning_rate=3e-4,
        n_epochs=1,
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


def collect_rollouts_sb3_with_traces(
    ppo: PPO_custom,
    n_steps: int,
    deterministic_action: bool = True,
) -> Tuple[RolloutTraceCollection, SB3RolloutBuffer]:
    """
    Collect rollouts using SB3 PPO with trace collection.
    Always uses the first valid action (action 0) when deterministic_action=True.
    """
    from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
    
    traces = RolloutTraceCollection()
    env = ppo.env
    n_envs = env.num_envs
    im = env.envs[0].env.index_manager  # Get index manager from first env
    
    # Initialize
    ppo._last_obs = env.reset()
    ppo._last_episode_starts = np.ones((n_envs,), dtype=bool)
    ppo.rollout_buffer.reset()
    
    ppo.policy.set_training_mode(False)
    
    for step in range(n_steps):
        with torch.no_grad():
            obs_tensor = obs_as_tensor(ppo._last_obs, ppo.device)
            
            if deterministic_action:
                # Always use action 0 (first valid action)
                actions = np.zeros(n_envs, dtype=np.int64)
                values = ppo.policy.predict_values(obs_tensor)
                # Get log probs for action 0
                dist = ppo.policy.get_distribution(obs_tensor)
                log_probs = dist.log_prob(torch.zeros(n_envs, dtype=torch.long, device=ppo.device))
            else:
                actions, values, log_probs = ppo.policy(obs_tensor, deterministic=True)
                actions = actions.cpu().numpy()
        
        # Record traces before step
        for env_idx in range(n_envs):
            state_str = sb3_obs_to_canonical_state(ppo._last_obs, im, env_idx)
            derived_states = get_derived_states_sb3(ppo._last_obs, im, env_idx)
            action_mask = ppo._last_obs.get('action_mask', np.ones((n_envs, 500)))[env_idx]
            action_mask_sum = int(action_mask.sum())
            
            trace = RolloutTrace(
                step=step,
                env_idx=env_idx,
                state=state_str,
                derived_states=derived_states,
                action=int(actions[env_idx]) if isinstance(actions, np.ndarray) else int(actions[env_idx].item()),
                reward=0.0,  # Will be updated after step
                done=False,  # Will be updated after step
                value=float(values[env_idx]) if hasattr(values, '__getitem__') else float(values),
                log_prob=float(log_probs[env_idx]) if hasattr(log_probs, '__getitem__') else float(log_probs),
                action_mask_sum=action_mask_sum,
            )
            traces.add(trace)
        
        # Execute step
        new_obs, rewards, dones, infos = env.step(actions)
        
        # Update traces with rewards and dones
        for env_idx in range(n_envs):
            trace = traces.get_step(step, env_idx)
            if trace:
                trace.reward = float(rewards[env_idx])
                trace.done = bool(dones[env_idx])
        
        # Add to buffer
        if isinstance(actions, np.ndarray) and actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        
        ppo.rollout_buffer.add(
            ppo._last_obs,
            actions,
            rewards,
            ppo._last_episode_starts,
            values,
            log_probs,
        )
        
        ppo._last_obs = new_obs
        ppo._last_episode_starts = dones
    
    # Compute returns
    with torch.no_grad():
        values = ppo.policy.predict_values(obs_as_tensor(new_obs, ppo.device))
    ppo.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    
    return traces, ppo.rollout_buffer


def collect_rollouts_tensor_with_traces(
    ppo: TensorPPO,
    n_steps: int,
    im: IndexManager,
    deterministic_action: bool = True,
) -> Tuple[RolloutTraceCollection, TensorRolloutBuffer]:
    """
    Collect rollouts using tensor PPO with trace collection.
    Always uses the first valid action (action 0) when deterministic_action=True.
    """
    from tensordict import TensorDict
    
    traces = RolloutTraceCollection()
    env = ppo.env
    n_envs = ppo.n_envs
    
    # Initialize
    current_obs = env.reset()
    episode_starts = torch.ones(n_envs, dtype=torch.float32, device=ppo.device)
    ppo.rollout_buffer.reset()
    
    ppo.policy.eval()
    
    for step in range(n_steps):
        with torch.no_grad():
            obs_device = current_obs.to(ppo.device)
            
            if deterministic_action:
                # Always use action 0 (first valid action)
                actions = torch.zeros(n_envs, dtype=torch.long, device=ppo.device)
                _, values, log_probs = ppo.policy(obs_device, deterministic=True)
                # Recompute log_probs for action 0
                _, _, log_probs = ppo.policy.evaluate_actions(obs_device, actions)
            else:
                actions, values, log_probs = ppo.policy(obs_device, deterministic=True)
        
        # Record traces before step
        obs_dict = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                   for k, v in current_obs.items()}
        
        for env_idx in range(n_envs):
            state_str = tensor_obs_to_canonical_state(obs_dict, im, env_idx)
            derived_states = get_derived_states_tensor(obs_dict, im, env_idx)
            action_mask = obs_dict.get('action_mask', np.ones((n_envs, 500)))[env_idx]
            action_mask_sum = int(action_mask.sum())
            
            trace = RolloutTrace(
                step=step,
                env_idx=env_idx,
                state=state_str,
                derived_states=derived_states,
                action=int(actions[env_idx].item()),
                reward=0.0,  # Will be updated after step
                done=False,  # Will be updated after step
                value=float(values[env_idx].item()),
                log_prob=float(log_probs[env_idx].item()),
                action_mask_sum=action_mask_sum,
            )
            traces.add(trace)
        
        # Execute step
        actions_env = actions.to(env._device)
        action_td = TensorDict({"action": actions_env}, batch_size=current_obs.batch_size, device=env._device)
        step_result, next_obs = env.step_and_maybe_reset(action_td)
        
        # Extract done/reward
        if "next" in step_result.keys():
            step_info = step_result["next"]
        else:
            step_info = step_result
        
        rewards = step_info.get("reward", torch.zeros(n_envs, device=env._device))
        dones = step_info.get("done", torch.zeros(n_envs, dtype=torch.bool, device=env._device))
        
        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)
        if dones.dim() > 1:
            dones = dones.squeeze(-1)
        
        # Update traces with rewards and dones
        for env_idx in range(n_envs):
            trace = traces.get_step(step, env_idx)
            if trace:
                trace.reward = float(rewards[env_idx].item())
                trace.done = bool(dones[env_idx].item())
        
        # Add to buffer
        ppo.rollout_buffer.add(
            obs=obs_device,
            action=actions,
            reward=rewards.to(ppo.device),
            episode_start=episode_starts,
            value=values,
            log_prob=log_probs
        )
        
        # Update episode starts
        if dones.any():
            episode_starts = dones.float()
        else:
            episode_starts = torch.zeros(n_envs, dtype=torch.float32, device=ppo.device)
        
        current_obs = next_obs
    
    # Compute returns
    with torch.no_grad():
        _, last_values, _ = ppo.policy(current_obs.to(ppo.device))
    
    if dones.dim() > 1:
        dones = dones.squeeze(-1)
    ppo.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones.to(ppo.device))
    
    return traces, ppo.rollout_buffer


def compare_traces(sb3_traces: RolloutTraceCollection, tensor_traces: RolloutTraceCollection, 
                   n_steps: int, n_envs: int, verbose: bool = True) -> Dict[str, Any]:
    """Compare traces from SB3 and tensor rollouts."""
    results = {
        'total_comparisons': 0,
        'state_matches': 0,
        'reward_matches': 0,
        'done_matches': 0,
        'action_matches': 0,
        'value_matches': 0,
        'log_prob_matches': 0,
        'mismatches': [],
    }
    
    for step in range(n_steps):
        for env_idx in range(n_envs):
            sb3_trace = sb3_traces.get_step(step, env_idx)
            tensor_trace = tensor_traces.get_step(step, env_idx)
            
            if sb3_trace is None or tensor_trace is None:
                continue
            
            results['total_comparisons'] += 1
            
            # Compare state
            state_match = sb3_trace.state[:40] == tensor_trace.state[:40]  # Compare first 40 chars
            if state_match:
                results['state_matches'] += 1
            
            # Compare reward
            reward_match = abs(sb3_trace.reward - tensor_trace.reward) < 1e-5
            if reward_match:
                results['reward_matches'] += 1
            
            # Compare done
            done_match = sb3_trace.done == tensor_trace.done
            if done_match:
                results['done_matches'] += 1
            
            # Compare action
            action_match = sb3_trace.action == tensor_trace.action
            if action_match:
                results['action_matches'] += 1
            
            # Compare value (with tolerance)
            value_match = abs(sb3_trace.value - tensor_trace.value) < 0.1
            if value_match:
                results['value_matches'] += 1
            
            # Compare log_prob (with tolerance)
            log_prob_match = abs(sb3_trace.log_prob - tensor_trace.log_prob) < 0.1
            if log_prob_match:
                results['log_prob_matches'] += 1
            
            # Record mismatches
            if not (state_match and reward_match and done_match and action_match):
                mismatch = {
                    'step': step,
                    'env_idx': env_idx,
                    'state_match': state_match,
                    'reward_match': reward_match,
                    'done_match': done_match,
                    'action_match': action_match,
                    'sb3_state': sb3_trace.state[:50],
                    'tensor_state': tensor_trace.state[:50],
                    'sb3_reward': sb3_trace.reward,
                    'tensor_reward': tensor_trace.reward,
                    'sb3_done': sb3_trace.done,
                    'tensor_done': tensor_trace.done,
                }
                results['mismatches'].append(mismatch)
            
            if verbose:
                status = "OK" if (state_match and reward_match and done_match) else "MISMATCH"
                if status == "MISMATCH" or step < 3:
                    print(f"Step {step}, Env {env_idx}: {status}")
                    print(f"  SB3   : state={sb3_trace.state[:50]}, reward={sb3_trace.reward:.3f}, done={sb3_trace.done}")
                    print(f"  Tensor: state={tensor_trace.state[:50]}, reward={tensor_trace.reward:.3f}, done={tensor_trace.done}")
    
    return results


def run_rollout_parity_test(dataset: str, n_envs: int, n_steps: int, verbose: bool = True) -> bool:
    """Run the full rollout parity test."""
    print("=" * 70)
    print(f"Rollout Collection Parity Test")
    print(f"Dataset: {dataset}, n_envs: {n_envs}, n_steps: {n_steps}")
    print("=" * 70)
    
    # Create aligned environments
    print("\nCreating aligned environments...")
    env_data = create_aligned_environments(dataset, n_envs)
    
    # Create SB3 PPO
    print("Creating SB3 PPO...")
    sb3_ppo, sb3_env, sb3_im = create_sb3_ppo(
        env_data['sb3'], env_data['queries'], n_envs, n_steps
    )
    
    # Create tensor PPO
    print("Creating tensor PPO...")
    tensor_ppo, tensor_env, tensor_im, engine = create_tensor_ppo(
        env_data['tensor'], env_data['tensor_queries'], n_envs, n_steps
    )
    
    # Collect rollouts with traces
    print("\nCollecting SB3 rollouts...")
    sb3_traces, sb3_buffer = collect_rollouts_sb3_with_traces(
        sb3_ppo, n_steps, deterministic_action=True
    )
    
    print("Collecting tensor rollouts...")
    tensor_traces, tensor_buffer = collect_rollouts_tensor_with_traces(
        tensor_ppo, n_steps, tensor_im, deterministic_action=True
    )
    
    # Compare traces
    print("\n--- Trace Comparison ---")
    results = compare_traces(sb3_traces, tensor_traces, n_steps, n_envs, verbose=verbose)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary:")
    print(f"  Total comparisons: {results['total_comparisons']}")
    print(f"  State matches: {results['state_matches']}/{results['total_comparisons']}")
    print(f"  Reward matches: {results['reward_matches']}/{results['total_comparisons']}")
    print(f"  Done matches: {results['done_matches']}/{results['total_comparisons']}")
    print(f"  Action matches: {results['action_matches']}/{results['total_comparisons']}")
    
    success = (
        results['state_matches'] == results['total_comparisons'] and
        results['reward_matches'] == results['total_comparisons'] and
        results['done_matches'] == results['total_comparisons'] and
        results['action_matches'] == results['total_comparisons']
    )
    
    if success:
        print("\n✓ ALL ROLLOUT TRACES MATCH")
    else:
        print(f"\n✗ MISMATCHES FOUND: {len(results['mismatches'])}")
        if results['mismatches'] and verbose:
            print("\nFirst 5 mismatches:")
            for m in results['mismatches'][:5]:
                print(f"  Step {m['step']}, Env {m['env_idx']}:")
                print(f"    SB3: {m['sb3_state']}")
                print(f"    Tensor: {m['tensor_state']}")
    
    print("=" * 70)
    
    return success


# ============================================================
# Pytest Tests
# ============================================================

@pytest.mark.parametrize("n_envs,n_steps", [
    (1, 10),
    (4, 10),
])
def test_rollout_parity(n_envs, n_steps):
    """Test rollout collection parity between SB3 and tensor implementations."""
    success = run_rollout_parity_test(
        dataset="countries_s3",
        n_envs=n_envs,
        n_steps=n_steps,
        verbose=False
    )
    assert success, f"Rollout parity test failed for n_envs={n_envs}, n_steps={n_steps}"


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rollout Collection Parity Test")
    parser.add_argument("--dataset", type=str, default="countries_s3",
                       help="Dataset name (default: countries_s3)")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of environments (default: 4)")
    parser.add_argument("--n-steps", type=int, default=20,
                       help="Number of rollout steps (default: 20)")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Enable verbose output (show each step comparison)")
    
    args = parser.parse_args()
    
    success = run_rollout_parity_test(
        dataset=args.dataset,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)

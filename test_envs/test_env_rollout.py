"""
Test module for environment evaluation using PPO rollout collector.

Tests the environment using the RolloutCollector class from ppo_rollout.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import random
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List
from types import SimpleNamespace
from tensordict import TensorDict

from test_envs.test_env_tensor import setup_tensor_env
from ppo.rollout import RolloutCollector
from ppo.ppo import PPO
from ppo.model import create_actor_critic
from ppo.rollout import RolloutBuffer


class LogitsProducingActor(nn.Module):
    """Actor that produces logits for rollout collector."""
    
    def __init__(self, max_actions: int = 500, deterministic: bool = True, seed: int = 42):
        super().__init__()
        self.max_actions = max_actions
        self.deterministic = deterministic
        self.rng = random.Random(seed)
        self._debug_step = 0
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Produce logits based on action mask."""
        batch_size = td.batch_size[0] if isinstance(td.batch_size, torch.Size) else td.batch_size
        action_mask = td.get("action_mask")
        
        if action_mask is None:
            # No mask, create uniform logits
            logits = torch.zeros(batch_size, self.max_actions, device=td.device)
        else:
            # Get number of actions from mask
            n_actions = action_mask.shape[-1]
            logits = torch.zeros(batch_size, n_actions, device=td.device)
            
            if self.deterministic:
                # IMPORTANT: In deterministic mode, we must select the first VALID action
                # in canonical order. The MaskedPolicyWrapper will mask invalid actions,
                # but if we set high logit for an invalid action, the sampler becomes random.
                # So we MUST check the mask and select the first valid action index.
                for i in range(batch_size):
                    mask = action_mask[i]
                    # Find first valid action (should be action 0 in canonical order)
                    valid_indices = torch.where(mask)[0]
                    if len(valid_indices) > 0:
                        # Set high logit for first valid action
                        first_valid = valid_indices[0]
                        logits[i, first_valid] = 10.0
                        if i == 0 and hasattr(self, '_debug_step'):
                            self._debug_step += 1
                            if self._debug_step <= 3:
                                print(f"    [LogitsProducingActor] Batch {i}: first_valid={first_valid}, n_valid={len(valid_indices)}")
            else:
                # Uniform logits for valid actions (will be sampled randomly)
                logits = torch.where(action_mask, torch.zeros_like(logits), torch.full_like(logits, float('-inf')))
        
        td["logits"] = logits
        return td


class DummyCritic(nn.Module):
    """Dummy critic for rollout collector."""
    
    def forward(self, td):
        # Handle nested batch dimensions from collector
        if isinstance(td.batch_size, torch.Size):
            shape = td.batch_size
        else:
            shape = (td.batch_size,)
        td["state_value"] = torch.zeros(shape, device=td.device)
        return td


class RandomActorCritic(nn.Module):
    """Random actor-critic policy that mimics LogitsProducingActor behavior."""
    
    def __init__(self, max_actions: int = 500, deterministic: bool = True, seed: int = 42):
        super().__init__()
        self.max_actions = max_actions
        self.deterministic = deterministic
        self.rng = random.Random(seed)
        self._debug_step = 0
        
        # Add dummy parameters so PPO optimizer doesn't fail
        self.dummy_param = nn.Parameter(torch.zeros(1))
    
    def forward(self, obs_td):
        """Produce actions and values based on observation."""
        batch_size = obs_td.batch_size[0] if isinstance(obs_td.batch_size, torch.Size) else obs_td.batch_size
        action_mask = obs_td.get("action_mask")
        
        if action_mask is None:
            # No mask, create uniform logits
            logits = torch.zeros(batch_size, self.max_actions, device=obs_td.device)
        else:
            # Get number of actions from mask
            n_actions = action_mask.shape[-1]
            logits = torch.zeros(batch_size, n_actions, device=obs_td.device)
            
            if self.deterministic:
                # Select first valid action
                for i in range(batch_size):
                    mask = action_mask[i]
                    valid_indices = torch.where(mask)[0]
                    if len(valid_indices) > 0:
                        first_valid = valid_indices[0]
                        logits[i, first_valid] = 10.0
            else:
                # Uniform random for valid actions
                logits = torch.where(action_mask, torch.zeros_like(logits), torch.full_like(logits, float('-inf')))
        
        # Sample action from logits
        if self.deterministic:
            # For deterministic, take argmax
            actions = torch.argmax(logits, dim=-1)
            log_probs = torch.zeros(batch_size, device=obs_td.device)
        else:
            # For random, sample from categorical
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        
        # Dummy values
        values = torch.zeros(batch_size, device=obs_td.device)
        
        return actions, values, log_probs
    
    def predict_values(self, obs_td):
        """Predict values for observations (dummy implementation)."""
        batch_size = obs_td.batch_size[0] if isinstance(obs_td.batch_size, torch.Size) else obs_td.batch_size
        return torch.zeros(batch_size, device=obs_td.device)
    
    def evaluate_actions(self, obs_td, actions):
        """Evaluate actions (dummy implementation for training)."""
        batch_size = obs_td.batch_size[0] if isinstance(obs_td.batch_size, torch.Size) else obs_td.batch_size
        log_probs = torch.zeros(batch_size, device=obs_td.device)
        entropy = torch.zeros(batch_size, device=obs_td.device)
        values = torch.zeros(batch_size, device=obs_td.device)
        return values, log_probs, entropy


def test_rollout_env(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    config: SimpleNamespace
) -> Dict:
    """
    Test environment using PPO rollout collector.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        config: Configuration namespace with dataset, deterministic, seed, verbose, collect_action_stats, etc.
        
    Returns:
        Dict with keys:
            - total_queries: int
            - successful: int
            - avg_reward: float
            - avg_steps: float
            - avg_actions: float (0.0 if collect_action_stats=False)
            - traces: List of trace dicts
    """
    print("Setting up environment for rollout collector...")
    
    dataset = config.dataset
    deterministic = config.deterministic
    seed = config.seed
    verbose = config.verbose
    collect_action_stats = config.collect_action_stats
    
    env_data = setup_tensor_env(dataset=dataset, seed=seed, batch_size=len(queries), config=config)
    batched_env, debug_helper, constant_no, im_batched, dh_batched = env_data
    
    # Prepare queries for eval mode
    all_query_tensors = []
    for split, (p, h, t) in queries:
        query_atom = im_batched.atom_to_tensor(p, h, t)
        query_padded = torch.full((batched_env.padding_atoms, 3), batched_env.padding_idx,
                                   dtype=torch.long, device='cpu')
        query_padded[0] = query_atom
        all_query_tensors.append(query_padded)
    
    queries_tensor = torch.stack(all_query_tensors, dim=0)
    labels_tensor = torch.ones(len(queries), dtype=torch.long, device='cpu')
    depths_tensor = torch.ones(len(queries), dtype=torch.long, device='cpu')
    per_slot_lengths = torch.ones(len(queries), dtype=torch.long, device='cpu')
    
    batched_env.set_eval_dataset(
        queries=queries_tensor,
        labels=labels_tensor,
        query_depths=depths_tensor,
        per_slot_lengths=per_slot_lengths
    )
    
    if verbose:
        print(f"  Environment mode after set_eval_dataset: {batched_env.mode}")
        print(f"  Eval slot lengths: {batched_env._eval_slot_lengths}")
    
    # Create logits-producing actor for rollout collector
    # Note: RolloutCollector will wrap this with MaskedPolicyWrapper internally
    actor = LogitsProducingActor(
        max_actions=batched_env.padding_states,
        deterministic=deterministic,
        seed=seed
    )
    critic = DummyCritic()
    
    print(f"Running rollout collector with {'deterministic' if deterministic else 'random'} actor...")
    # Create rollout collector
    rollout_collector = RolloutCollector(
        env=batched_env,
        actor=actor,
        n_envs=len(queries),
        n_steps=config.n_steps,
        device=torch.device(config.device),
        debug=verbose,
        debug_action_space=False,
    )
    
    # Collect rollouts
    results = rollout_collector.collect(critic=critic, return_processed_results=True, collect_action_stats=collect_action_stats)
    
    if verbose:
        successful = results['successful']
        n_total = results['total_queries']
        print(f"  Rollout summary: {successful}/{n_total} successful")
    
    return results


def test_rolloutsb3_env(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    config: SimpleNamespace
) -> Dict:
    """
    Test environment using PPO SB3-style rollout collection.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        config: Configuration namespace with dataset, deterministic, seed, verbose, collect_action_stats, etc.
        
    Returns:
        Dict with keys:
            - total_queries: int
            - successful: int
            - avg_reward: float
            - avg_steps: float
            - avg_actions: float (0.0 if collect_action_stats=False)
            - traces: List of trace dicts
    """
    print("Setting up environment for PPO SB3-style rollout...")
    
    dataset = config.dataset
    deterministic = config.deterministic
    seed = config.seed
    verbose = config.verbose
    collect_action_stats = config.collect_action_stats
    
    env_data = setup_tensor_env(dataset=dataset, seed=seed, batch_size=len(queries), config=config)
    batched_env, debug_helper, constant_no, im_batched, dh_batched = env_data
    
    # Prepare queries for eval mode
    all_query_tensors = []
    for split, (p, h, t) in queries:
        query_atom = im_batched.atom_to_tensor(p, h, t)
        query_padded = torch.full((batched_env.padding_atoms, 3), batched_env.padding_idx,
                                   dtype=torch.long, device='cpu')
        query_padded[0] = query_atom
        all_query_tensors.append(query_padded)
    
    queries_tensor = torch.stack(all_query_tensors, dim=0)
    labels_tensor = torch.ones(len(queries), dtype=torch.long, device='cpu')
    depths_tensor = torch.ones(len(queries), dtype=torch.long, device='cpu')
    per_slot_lengths = torch.ones(len(queries), dtype=torch.long, device='cpu')
    
    batched_env.set_eval_dataset(
        queries=queries_tensor,
        labels=labels_tensor,
        query_depths=depths_tensor,
        per_slot_lengths=per_slot_lengths
    )
    
    if verbose:
        print(f"  Environment mode after set_eval_dataset: {batched_env.mode}")
        print(f"  Eval slot lengths: {batched_env._eval_slot_lengths}")
    
    # Create random actor-critic policy
    random_policy = RandomActorCritic(
        max_actions=batched_env.padding_states,
        deterministic=deterministic,
        seed=seed
    )
    
    print(f"Running PPO SB3-style rollout with {'deterministic' if deterministic else 'random'} policy...")
    
    # Create PPO agent with random policy
    ppo_agent = PPO(
        policy=random_policy,
        env=batched_env,
        n_steps=config.n_steps,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        learning_rate=config.learning_rate,
        device=torch.device(config.device),
        verbose=0,
    )
    
    # Setup and collect rollouts
    ppo_agent._setup_model()
    
    # Collect rollouts and get processed results
    results = ppo_agent.collect_rollouts_with_results(return_processed_results=True, collect_action_stats=collect_action_stats)
    
    if verbose:
        successful = results['successful']
        n_total = results['total_queries']
        print(f"  PPO SB3 rollout summary: {successful}/{n_total} successful")
    
    return results

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
from typing import Tuple, Dict, List
from tensordict import TensorDict

from test_envs.test_env_tensor import setup_tensor_env
from ppo.ppo_rollout import RolloutCollector


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


def test_rollout_env(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    dataset: str,
    deterministic: bool = True,
    max_depth: int = 20,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Test environment using PPO rollout collector.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        dataset: Dataset name
        deterministic: If True, use deterministic actor; if False, random
        max_depth: Maximum proof depth (controls n_steps for rollout)
        seed: Random seed
        verbose: Print detailed information
        
    Returns:
        Dict with keys:
            - total_queries: int
            - successful: int
            - avg_reward: float
            - avg_steps: float
            - traces: List of trace dicts
    """
    print("Setting up environment for rollout collector...")
    env_data = setup_tensor_env(dataset=dataset, seed=seed, batch_size=len(queries))
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
        n_steps=max_depth,
        device=torch.device('cpu'),
        debug=verbose,
        debug_action_space=False,
    )
    
    # Collect rollouts
    experiences, stats = rollout_collector.collect(critic=critic)
    
    # Extract results from rollout by tracking per-slot episodes
    # The RolloutCollector doesn't track which slot each episode belongs to,
    # so we need to manually extract per-slot results from the experiences
    n_total = len(queries)
    
    if verbose:
        print(f"  Processing {len(experiences)} experience steps...")
    
    # Track per-slot cumulative rewards, steps, and episode counts
    slot_rewards = [0.0 for _ in range(n_total)]
    slot_steps = [0 for _ in range(n_total)]
    slot_done = [False for _ in range(n_total)]
    slot_success = [False for _ in range(n_total)]
    slot_episode_count = [0 for _ in range(n_total)]  # Track how many episodes completed per slot
    
    # Process experiences step by step to track per-slot episodes
    for step_idx, step_td in enumerate(experiences):
        # Access the 'next' dict which contains reward, done, etc
        next_td = step_td.get('next')
        if next_td is None:
            continue
        
        # Extract values - they have batch dimension [n_envs]
        rewards = next_td.get('reward', torch.zeros(n_total))
        dones = next_td.get('done', torch.zeros(n_total, dtype=torch.bool))
        is_success = next_td.get('is_success', torch.zeros(n_total, dtype=torch.bool))
        
        # Squeeze extra dimensions if present
        if rewards.dim() > 1:
            rewards = rewards.squeeze()
        if dones.dim() > 1:
            dones = dones.squeeze()
        if is_success.dim() > 1:
            is_success = is_success.squeeze()
        
        # Update per-slot tracking (only for first episode per slot)
        for i in range(n_total):
            if slot_episode_count[i] == 0:  # Only track first episode per slot
                slot_rewards[i] += float(rewards[i])
                slot_steps[i] += 1
                
                if bool(dones[i]):
                    slot_done[i] = True
                    slot_success[i] = bool(is_success[i])
                    slot_episode_count[i] = 1
                    
                    if verbose and step_idx < 100 and i < 5:
                        print(f"  Slot {i} completed: success={slot_success[i]}, "
                              f"reward={slot_rewards[i]:.2f}, steps={slot_steps[i]}")
    
    # Create traces from per-slot results
    traces = []
    for i in range(n_total):
        traces.append({
            'success': slot_success[i],
            'steps': slot_steps[i],
            'reward': slot_rewards[i],
            'trace': []  # No detailed trace from rollout collector
        })
    
    # Calculate statistics
    successful = sum(1 for t in traces if t['success'])
    total_reward = sum(t['reward'] for t in traces)
    total_steps = sum(t['steps'] for t in traces)
    
    if verbose:
        print(f"  Rollout summary: {successful}/{n_total} successful")
        print(f"  Completed episodes: {n_completed}/{n_total}")
        if n_completed > 0:
            print(f"  Rewards (first 5): {[f'{r:.2f}' for r in rewards_list[:5]]}")
            print(f"  Steps (first 5): {lengths_list[:5]}")
            print(f"  Success (first 5): {success_list[:5]}")
    
    return {
        'total_queries': n_total,
        'successful': successful,
        'avg_reward': total_reward / n_total if n_total > 0 else 0.0,
        'avg_steps': total_steps / n_total if n_total > 0 else 0.0,
        'traces': traces
    }

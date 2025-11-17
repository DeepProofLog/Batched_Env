"""
Test module for environment evaluation using evaluate_policy.

Tests the environment using the evaluate_policy function from model_eval.py.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import torch
import torch.nn as nn
from typing import Tuple, Dict, List
from tensordict import TensorDict

from test_envs.test_env_tensor import setup_tensor_env
from model_eval import evaluate_policy


class DeterministicActorForEval(nn.Module):
    """Simple deterministic actor that always chooses action 0 (or first valid action)."""
    
    def __init__(self, collect_action_stats: bool = False):
        super().__init__()
        self.collect_action_stats = collect_action_stats
        self.action_counts = [] if collect_action_stats else None
        self.step_count = 0
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Select action 0 for all batch items."""
        batch_size = td.batch_size[0] if isinstance(td.batch_size, torch.Size) else td.batch_size
        action = torch.zeros(batch_size, dtype=torch.long, device=td.device)
        
        if self.collect_action_stats:
            action_mask = td.get("action_mask")
            if action_mask is not None:
                # Only count actions where mask has valid actions (sum > 0)
                # This filters out inactive/reset states
                num_actions = action_mask.sum(dim=-1).cpu().numpy()
                for n in num_actions:
                    if n > 0:  # Only count if there are valid actions
                        self.action_counts.append(int(n))
            self.step_count += 1
        
        td["action"] = action
        return td


class RandomActorForEval(nn.Module):
    """Random actor that chooses valid actions uniformly at random."""
    
    def __init__(self, seed: int = 42, collect_action_stats: bool = False):
        super().__init__()
        import random
        self.rng = random.Random(seed)
        self.collect_action_stats = collect_action_stats
        self.action_counts = [] if collect_action_stats else None
        self.step_count = 0
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Select random valid action for each batch item."""
        batch_size = td.batch_size[0] if isinstance(td.batch_size, torch.Size) else td.batch_size
        action_mask = td.get("action_mask")
        actions = torch.zeros(batch_size, dtype=torch.long, device=td.device)
        
        for i in range(batch_size):
            mask = action_mask[i] if action_mask is not None else None
            if mask is not None:
                valid_actions = torch.where(mask)[0].tolist()
                if valid_actions:
                    actions[i] = self.rng.choice(valid_actions)
                    
                    if self.collect_action_stats:
                        # Only count if there are valid actions (active state)
                        self.action_counts.append(len(valid_actions))
        
        if self.collect_action_stats:
            self.step_count += 1
        
        td["action"] = actions
        return td


def test_eval_env(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    dataset: str,
    deterministic: bool = True,
    max_depth: int = 20,
    seed: int = 42,
    verbose: bool = False,
    collect_action_stats: bool = False
) -> Dict:
    """
    Test environment using evaluate_policy function.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        dataset: Dataset name
        deterministic: If True, use deterministic actor; if False, random
        seed: Random seed
        verbose: Print detailed information
        collect_action_stats: If True, collect action statistics (branching factor)
        
    Returns:
        Dict with keys:
            - total_queries: int
            - successful: int
            - avg_reward: float
            - avg_steps: float
            - avg_actions: float (0.0 if collect_action_stats=False)
            - traces: List of trace dicts
    """
    print("Setting up environment for evaluate_policy...")
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
    
    # Use deterministic or random actor
    if deterministic:
        actor = DeterministicActorForEval(collect_action_stats=collect_action_stats)
    else:
        actor = RandomActorForEval(seed=seed, collect_action_stats=collect_action_stats)
    
    print(f"Running evaluate_policy with {'deterministic' if deterministic else 'random'} actor...")
    eval_results = evaluate_policy(
        actor=actor,
        env=batched_env,
        n_eval_episodes=len(queries),
        deterministic=deterministic,
        verbose=verbose
    )
    
    # Extract results
    success = eval_results['success'].numpy()  # [B, T]
    rewards = eval_results['rewards'].numpy()
    lengths = eval_results['lengths'].numpy()
    mask = eval_results['mask'].numpy()
    
    # Aggregate per query (take first episode for each slot)
    successful = 0
    total_reward = 0.0
    total_steps = 0
    traces = []
    
    for i in range(len(queries)):
        if mask[i, 0]:  # First episode for this slot
            is_success = bool(success[i, 0])
            reward = float(rewards[i, 0])
            steps = int(lengths[i, 0])
            
            successful += int(is_success)
            total_reward += reward
            total_steps += steps
            
            # Create minimal trace for compatibility
            traces.append({
                'success': is_success,
                'steps': steps,
                'reward': reward,
                'trace': []  # No detailed trace from evaluate_policy
            })
    
    # Compute average actions if collected
    if collect_action_stats and hasattr(actor, 'action_counts') and actor.action_counts:
        avg_actions = sum(actor.action_counts) / len(actor.action_counts)
        if verbose:
            print(f"  Action stats: {len(actor.action_counts)} samples, avg={avg_actions:.2f}")
            print(f"  Forward calls: {actor.step_count}")
            print(f"  Action count samples (first 20): {actor.action_counts[:20]}")
    else:
        avg_actions = 0.0
    
    return {
        'total_queries': len(queries),
        'successful': successful,
        'avg_reward': total_reward / len(queries) if queries else 0.0,
        'avg_steps': total_steps / len(queries) if queries else 0.0,
        'avg_actions': avg_actions,
        'traces': traces
    }

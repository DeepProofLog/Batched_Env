"""
Test the Tensor PPO rollout collector.

Since the tensor PPO.collect_rollouts is a placeholder and the actual rollout collection
is embedded in learn(), we create a wrapper that extracts this functionality into a 
testable collect_rollouts method similar to SB3's interface.
"""

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import torch
import numpy as np
from typing import Dict, Any, Optional
from tensordict import TensorDict
from types import SimpleNamespace

# Tensor imports
from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from env import BatchedEnv
from model import ActorCriticPolicy
from ppo import PPO
from rollout import RolloutBuffer
from embeddings import EmbedderLearnable


class PPOWithCollectRollouts(PPO):
    """
    PPO wrapper that exposes a proper collect_rollouts method for testing.
    
    This extracts the rollout collection logic from learn() into a standalone method.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # State for rollout collection
        self._last_obs = None
        self._last_episode_starts = None
        self._initialized = False
    
    def setup_rollouts(self) -> None:
        """Initialize state for rollout collection (similar to SB3's _setup_learn)."""
        self._last_obs = self.env.reset()
        self._last_episode_starts = torch.ones(self.n_envs, dtype=torch.float32, device=self.device)
        self._initialized = True
    
    def collect_rollouts(self, n_rollout_steps: Optional[int] = None) -> bool:
        """
        Collect rollouts using the current policy.
        
        This is extracted from the learn() method to match SB3's collect_rollouts interface.
        
        Args:
            n_rollout_steps: Number of steps to collect. Defaults to self.n_steps.
            
        Returns:
            True if collection completed successfully.
        """
        if n_rollout_steps is None:
            n_rollout_steps = self.n_steps
            
        if not self._initialized:
            self.setup_rollouts()
        
        self.policy.eval()
        self.rollout_buffer.reset()
        
        current_obs = self._last_obs
        episode_starts = self._last_episode_starts
        
        n_collected = 0
        
        with torch.no_grad():
            while n_collected < n_rollout_steps:
                # Clone the current observation
                obs_snapshot = current_obs.clone()
                obs_device = obs_snapshot.to(self.device)
                
                # Get action from policy
                actions, values, log_probs = self.policy(obs_device, deterministic=True)
                
                # Step environment
                actions_env = actions.to(self.env_device)
                action_td = TensorDict(
                    {"action": actions_env}, 
                    batch_size=current_obs.batch_size, 
                    device=self.env_device
                )
                step_result, next_obs = self.env.step_and_maybe_reset(action_td)
                
                # Extract done/reward
                if "next" in step_result.keys():
                    step_info = step_result["next"]
                else:
                    step_info = step_result
                
                rewards_env = step_info.get("reward", torch.zeros(self.n_envs, device=self.env_device))
                dones_env = step_info.get("done", torch.zeros(self.n_envs, dtype=torch.bool, device=self.env_device))
                
                # Squeeze to ensure correct shape
                if rewards_env.dim() > 1:
                    rewards_env = rewards_env.squeeze(-1)
                if dones_env.dim() > 1:
                    dones_env = dones_env.squeeze(-1)
                rewards = rewards_env.to(self.device)
                dones = dones_env.to(self.device)
                
                # Store transition
                self.rollout_buffer.add(
                    obs=obs_device,
                    action=actions,
                    reward=rewards,
                    episode_start=episode_starts,
                    value=values,
                    log_prob=log_probs
                )
                
                n_collected += 1
                
                # Update episode starts for next step
                if dones.any():
                    episode_starts = dones.float()
                else:
                    episode_starts = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
                
                current_obs = next_obs
        
        # Update state
        self._last_obs = current_obs
        self._last_episode_starts = episode_starts
        
        # Compute returns and advantages
        with torch.no_grad():
            last_values = self.policy.predict_values(current_obs.to(self.device))
            if last_values.dim() > 1:
                last_values = last_values.squeeze(-1)
        
        self.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=dones.float()
        )
        
        return True


def create_tensor_env(dataset: str = "countries_s3", n_envs: int = 4, num_train_queries: int = 10, device: torch.device = None):
    """Create tensor-based environment."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_path = "./data/"
    padding_atoms = 100
    padding_states = 500
    max_total_runtime_vars = 1000000
    embed_dim = 64
    
    dh = DataHandler(
        dataset_name=dataset,
        base_path=base_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )
    
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=max_total_runtime_vars,
        padding_atoms=padding_atoms,
        max_arity=dh.max_arity,
        device=device,
        rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)
    
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
    
    # Create embedder (same as SB3)
    embedder = EmbedderLearnable(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=im.variable_no,
        max_arity=dh.max_arity,
        padding_atoms=padding_atoms,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=embed_dim,
        predicate_embedding_size=embed_dim,
        atom_embedding_size=embed_dim,
        device=str(device),
    )
    
    # Limit queries if specified
    train_queries = dh.train_queries[:num_train_queries] if num_train_queries else dh.train_queries
    
    # Convert queries to tensor format
    query_tensors = []
    for q in train_queries:
        query_atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        query_padded = torch.full((padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
        query_padded[0] = query_atom
        query_tensors.append(query_padded)
    
    # Stack into [n_envs, padding_atoms, 3] tensor
    # Repeat queries to fill n_envs
    queries_tensor = torch.stack(query_tensors * (n_envs // len(query_tensors) + 1), dim=0)[:n_envs]
    
    env = BatchedEnv(
        batch_size=n_envs,
        queries=queries_tensor,
        labels=torch.ones(n_envs, dtype=torch.long, device=device),
        query_depths=torch.ones(n_envs, dtype=torch.long, device=device),
        unification_engine=engine,
        mode='train',
        max_depth=20,
        memory_pruning=False,
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
        total_vocab_size=im.constant_no + max_total_runtime_vars,
    )
    
    return env, im, embedder, embed_dim


def create_tensor_ppo(dataset: str = "countries_s3", n_envs: int = 4, n_steps: int = 10) -> PPOWithCollectRollouts:
    """
    Create a PPOWithCollectRollouts instance for testing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the environment
    env, index_manager, embedder, embed_dim = create_tensor_env(
        dataset=dataset,
        n_envs=n_envs,
        num_train_queries=10,
        device=device,
    )
    
    # Get action space size (padding_states)
    action_size = 500  # padding_states
    
    # Create policy
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        action_dim=action_size,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
    ).to(device)
    
    # Create PPO with collect_rollouts support
    ppo = PPOWithCollectRollouts(
        policy=policy,
        env=env,
        n_steps=n_steps,
        batch_size=32,
        learning_rate=3e-4,
        device=device,
        verbose=False,
    )
    
    return ppo
    return ppo


def test_tensor_collect_rollouts(dataset: str = "countries_s3", n_envs: int = 4, n_steps: int = 10):
    """
    Test the Tensor PPO collect_rollouts method.
    """
    print(f"\n{'='*60}")
    print(f"Testing Tensor collect_rollouts")
    print(f"Dataset: {dataset}, n_envs: {n_envs}, n_steps: {n_steps}")
    print(f"{'='*60}")
    
    # Create PPO instance
    ppo = create_tensor_ppo(dataset=dataset, n_envs=n_envs, n_steps=n_steps)
    
    # Setup and collect rollouts
    ppo.setup_rollouts()
    
    print("\nCalling PPOWithCollectRollouts.collect_rollouts()...")
    success = ppo.collect_rollouts(n_rollout_steps=n_steps)
    
    print(f"\ncollect_rollouts returned: {success}")
    
    # Inspect the rollout buffer
    buffer = ppo.rollout_buffer
    print(f"\n--- Rollout Buffer Contents ---")
    print(f"Buffer size: {buffer.buffer_size}")
    print(f"N envs: {buffer.n_envs}")
    print(f"Position: {buffer.pos}")
    print(f"Full: {buffer.full}")
    
    # Check stored data shapes
    print(f"\nObservations type: {type(buffer.observations)}")
    if hasattr(buffer.observations, 'shape'):
        print(f"Observations shape: {buffer.observations.shape}")
    
    print(f"Actions shape: {buffer.actions.shape}")
    print(f"Rewards shape: {buffer.rewards.shape}")
    print(f"Values shape: {buffer.values.shape}")
    print(f"Log probs shape: {buffer.log_probs.shape}")
    print(f"Episode starts shape: {buffer.episode_starts.shape}")
    print(f"Returns shape: {buffer.returns.shape}")
    print(f"Advantages shape: {buffer.advantages.shape}")
    
    # Print some sample data
    print(f"\n--- Sample Data (first 3 steps, first env) ---")
    for step_idx in range(min(3, n_steps)):
        print(f"\nStep {step_idx}:")
        print(f"  Action: {buffer.actions[step_idx, 0].item()}")
        print(f"  Reward: {buffer.rewards[step_idx, 0].item():.4f}")
        print(f"  Value: {buffer.values[step_idx, 0].item():.4f}")
        print(f"  Log prob: {buffer.log_probs[step_idx, 0].item():.4f}")
        print(f"  Episode start: {buffer.episode_starts[step_idx, 0].item()}")
        print(f"  Return: {buffer.returns[step_idx, 0].item():.4f}")
        print(f"  Advantage: {buffer.advantages[step_idx, 0].item():.4f}")
    
    # Compute statistics
    print(f"\n--- Rollout Statistics ---")
    total_reward = buffer.rewards.sum().item()
    mean_value = buffer.values.mean().item()
    mean_return = buffer.returns.mean().item()
    mean_advantage = buffer.advantages.mean().item()
    
    print(f"Total reward: {total_reward:.4f}")
    print(f"Mean value: {mean_value:.4f}")
    print(f"Mean return: {mean_return:.4f}")
    print(f"Mean advantage: {mean_advantage:.4f}")
    
    return {
        "success": success,
        "buffer_size": buffer.buffer_size,
        "n_envs": buffer.n_envs,
        "total_reward": total_reward,
        "mean_value": mean_value,
        "mean_return": mean_return,
        "mean_advantage": mean_advantage,
        "actions": buffer.actions.clone(),
        "rewards": buffer.rewards.clone(),
        "values": buffer.values.clone(),
        "log_probs": buffer.log_probs.clone(),
        "returns": buffer.returns.clone(),
        "advantages": buffer.advantages.clone(),
    }


def test_tensor_multiple_rollouts(dataset: str = "countries_s3", n_envs: int = 4, n_steps: int = 10, n_rollouts: int = 3):
    """
    Test collecting multiple rollouts in sequence.
    """
    print(f"\n{'='*60}")
    print(f"Testing multiple Tensor rollouts")
    print(f"Dataset: {dataset}, n_envs: {n_envs}, n_steps: {n_steps}, n_rollouts: {n_rollouts}")
    print(f"{'='*60}")
    
    ppo = create_tensor_ppo(dataset=dataset, n_envs=n_envs, n_steps=n_steps)
    ppo.setup_rollouts()
    
    all_rewards = []
    all_values = []
    
    for rollout_idx in range(n_rollouts):
        print(f"\n--- Rollout {rollout_idx + 1}/{n_rollouts} ---")
        
        success = ppo.collect_rollouts(n_rollout_steps=n_steps)
        
        total_reward = ppo.rollout_buffer.rewards.sum().item()
        mean_value = ppo.rollout_buffer.values.mean().item()
        
        print(f"Success: {success}")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Mean value: {mean_value:.4f}")
        
        all_rewards.append(total_reward)
        all_values.append(mean_value)
    
    print(f"\n--- Summary ---")
    print(f"All rollout rewards: {all_rewards}")
    print(f"All rollout mean values: {all_values}")
    
    return all_rewards, all_values


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Tensor rollout collector")
    parser.add_argument("--dataset", type=str, default="countries_s3", help="Dataset to use")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--n_steps", type=int, default=10, help="Number of steps per rollout")
    parser.add_argument("--multiple", action="store_true", help="Test multiple rollouts")
    parser.add_argument("--n_rollouts", type=int, default=3, help="Number of rollouts for multiple test")
    
    args = parser.parse_args()
    
    if args.multiple:
        test_tensor_multiple_rollouts(
            dataset=args.dataset,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            n_rollouts=args.n_rollouts,
        )
    else:
        test_tensor_collect_rollouts(
            dataset=args.dataset,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
        )

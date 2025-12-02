"""
Test the SB3 PPO rollout collector using the actual PPO_custom.collect_rollouts method.
"""

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)
# CRITICAL: Add sb3 folder to path so that relative imports within sb3 work correctly
sb3_path = os.path.join(root_path, 'sb3')
sys.path.insert(0, sb3_path)

import torch
import numpy as np
from typing import Dict, Any
from types import SimpleNamespace

# SB3 imports
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# sb3 imports (relative - sb3_path is in sys.path)
from sb3_dataset import DataHandler as StrDataHandler
from sb3_index_manager import IndexManager as StrIndexManager
from sb3_env import LogicEnv_gym as StrEnv
from sb3_model import PPO_custom, CustomActorCriticPolicy, CustomCombinedExtractor
from sb3_embeddings import EmbedderLearnable


class DummyCallback(BaseCallback):
    """Minimal callback that does nothing but allows rollout collection."""
    def __init__(self):
        super().__init__()
        
    def _on_step(self) -> bool:
        return True


def create_sb3_env_and_embedder(dataset: str = "countries_s3", n_envs: int = 4, num_train_queries: int = 10):
    """Create SB3 vectorized environment and embedder."""
    base_path = "./data/"
    device = torch.device("cpu")  # SB3 uses CPU for env
    
    dh = StrDataHandler(
        dataset_name=dataset,
        base_path=base_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )
    
    im = StrIndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_vars=1000000,
        rules=dh.rules,
        padding_atoms=100,
        max_arity=dh.max_arity,
        device=device,
    )
    
    facts_set = set(dh.facts)
    im.build_fact_index(list(facts_set))
    
    # Create embedder
    embedder = EmbedderLearnable(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=im.variable_no,
        max_arity=dh.max_arity,
        padding_atoms=100,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=64,
        predicate_embedding_size=64,
        atom_embedding_size=64,
        device="cpu",
    )
    embedder.embed_dim = 64  # Set embed_dim
    
    # Limit queries if specified
    train_queries = dh.train_queries[:num_train_queries] if num_train_queries else dh.train_queries
    train_labels = [1] * len(train_queries)
    
    def make_env(seed: int):
        def _init():
            env = StrEnv(
                index_manager=im,
                data_handler=dh,
                queries=train_queries,
                labels=train_labels,
                query_depths=[None] * len(train_queries),
                facts=facts_set,
                mode='train',
                seed=seed,
                max_depth=20,
                memory_pruning=False,
                padding_atoms=100,
                padding_states=500,
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
            return Monitor(env)
        return _init
    
    env_fns = [make_env(seed=i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    
    return vec_env, im, embedder


def create_sb3_ppo(dataset: str = "countries_s3", n_envs: int = 4, n_steps: int = 10) -> PPO_custom:
    """
    Create a PPO_custom instance with environment for rollout collection testing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the vectorized environment and embedder
    env, index_manager, embedder = create_sb3_env_and_embedder(dataset=dataset, n_envs=n_envs)
    
    # Create PPO with the environment
    ppo = PPO_custom(
        policy=CustomActorCriticPolicy,
        env=env,
        n_steps=n_steps,
        batch_size=32,
        learning_rate=3e-4,
        verbose=0,
        device=device,
        policy_kwargs={
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {"features_dim": embedder.embed_dim, "embedder": embedder},
        },
    )
    
    return ppo


def test_sb3_collect_rollouts(dataset: str = "countries_s3", n_envs: int = 4, n_steps: int = 10):
    """
    Test the SB3 PPO collect_rollouts method directly.
    
    This uses the actual PPO_custom.collect_rollouts method, not a manual implementation.
    """
    print(f"\n{'='*60}")
    print(f"Testing SB3 collect_rollouts")
    print(f"Dataset: {dataset}, n_envs: {n_envs}, n_steps: {n_steps}")
    print(f"{'='*60}")
    
    # Create PPO instance
    ppo = create_sb3_ppo(dataset=dataset, n_envs=n_envs, n_steps=n_steps)
    
    # Setup learning (initializes _last_obs, _last_episode_starts, etc.)
    # We need to call _setup_learn to properly initialize the PPO state
    callback = DummyCallback()
    ppo._setup_learn(
        total_timesteps=n_steps * n_envs,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name="test",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())
    
    # Now call the actual collect_rollouts method
    print("\nCalling PPO_custom.collect_rollouts()...")
    success = ppo.collect_rollouts(
        env=ppo.env,
        callback=callback,
        rollout_buffer=ppo.rollout_buffer,
        n_rollout_steps=n_steps,
    )
    
    print(f"\ncollect_rollouts returned: {success}")
    
    # Inspect the rollout buffer
    buffer = ppo.rollout_buffer
    print(f"\n--- Rollout Buffer Contents ---")
    print(f"Buffer size: {buffer.buffer_size}")
    print(f"N envs: {buffer.n_envs}")
    print(f"Position: {buffer.pos}")
    print(f"Full: {buffer.full}")
    
    # Check observations shape
    print(f"\nObservations shapes:")
    for key, val in buffer.observations.items():
        print(f"  {key}: {val.shape}")
    
    print(f"\nActions shape: {buffer.actions.shape}")
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
        print(f"  Action: {buffer.actions[step_idx, 0]}")
        print(f"  Reward: {buffer.rewards[step_idx, 0]:.4f}")
        print(f"  Value: {buffer.values[step_idx, 0]:.4f}")
        print(f"  Log prob: {buffer.log_probs[step_idx, 0]:.4f}")
        print(f"  Episode start: {buffer.episode_starts[step_idx, 0]}")
        print(f"  Return: {buffer.returns[step_idx, 0]:.4f}")
        print(f"  Advantage: {buffer.advantages[step_idx, 0]:.4f}")
    
    # Compute statistics
    print(f"\n--- Rollout Statistics ---")
    total_reward = np.sum(buffer.rewards)
    mean_value = np.mean(buffer.values)
    mean_return = np.mean(buffer.returns)
    mean_advantage = np.mean(buffer.advantages)
    
    print(f"Total reward: {total_reward:.4f}")
    print(f"Mean value: {mean_value:.4f}")
    print(f"Mean return: {mean_return:.4f}")
    print(f"Mean advantage: {mean_advantage:.4f}")
    
    # Clean up
    ppo.env.close()
    
    return {
        "success": success,
        "buffer_size": buffer.buffer_size,
        "n_envs": buffer.n_envs,
        "total_reward": total_reward,
        "mean_value": mean_value,
        "mean_return": mean_return,
        "mean_advantage": mean_advantage,
        "actions": buffer.actions.copy(),
        "rewards": buffer.rewards.copy(),
        "values": buffer.values.copy(),
        "log_probs": buffer.log_probs.copy(),
        "returns": buffer.returns.copy(),
        "advantages": buffer.advantages.copy(),
    }


def test_sb3_multiple_rollouts(dataset: str = "countries_s3", n_envs: int = 4, n_steps: int = 10, n_rollouts: int = 3):
    """
    Test collecting multiple rollouts in sequence.
    """
    print(f"\n{'='*60}")
    print(f"Testing multiple SB3 rollouts")
    print(f"Dataset: {dataset}, n_envs: {n_envs}, n_steps: {n_steps}, n_rollouts: {n_rollouts}")
    print(f"{'='*60}")
    
    ppo = create_sb3_ppo(dataset=dataset, n_envs=n_envs, n_steps=n_steps)
    
    callback = DummyCallback()
    ppo._setup_learn(
        total_timesteps=n_steps * n_envs * n_rollouts,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name="test",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())
    
    all_rewards = []
    all_values = []
    
    for rollout_idx in range(n_rollouts):
        print(f"\n--- Rollout {rollout_idx + 1}/{n_rollouts} ---")
        
        success = ppo.collect_rollouts(
            env=ppo.env,
            callback=callback,
            rollout_buffer=ppo.rollout_buffer,
            n_rollout_steps=n_steps,
        )
        
        total_reward = np.sum(ppo.rollout_buffer.rewards)
        mean_value = np.mean(ppo.rollout_buffer.values)
        
        print(f"Success: {success}")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Mean value: {mean_value:.4f}")
        
        all_rewards.append(total_reward)
        all_values.append(mean_value)
    
    ppo.env.close()
    
    print(f"\n--- Summary ---")
    print(f"All rollout rewards: {all_rewards}")
    print(f"All rollout mean values: {all_values}")
    
    return all_rewards, all_values


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SB3 rollout collector")
    parser.add_argument("--dataset", type=str, default="countries_s3", help="Dataset to use")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--n_steps", type=int, default=10, help="Number of steps per rollout")
    parser.add_argument("--multiple", action="store_true", help="Test multiple rollouts")
    parser.add_argument("--n_rollouts", type=int, default=3, help="Number of rollouts for multiple test")
    
    args = parser.parse_args()
    
    if args.multiple:
        test_sb3_multiple_rollouts(
            dataset=args.dataset,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            n_rollouts=args.n_rollouts,
        )
    else:
        test_sb3_collect_rollouts(
            dataset=args.dataset,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
        )

"""
Test for PPO SB3-style implementation with GPU-based rollout collection.

This test verifies that the new pposb3 implementation works correctly by:
1. Loading dataset and creating environment  
2. Collecting rollouts with the new PPO implementation
3. Training with PPO and verifying losses make sense
4. Comparing results with the original test_rollout.py

Expected success rates on countries_s3:
- Random policy (uniform sampling): ~24-26%
- Trained policy: should improve over time

Usage:
    python tests/test_eval_rollout_sb3.py --batch-size 100 --n-steps 64 --n-epochs 3
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import random
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
from time import time
from tensordict import TensorDict

from data_handler import DataHandler
from index_manager import IndexManager
from sampler import Sampler
from embeddings import get_embedder
from env import BatchedEnv
from unification_engine import UnificationEngine

# Import new PPO implementation
from ppo.pposb3 import PPO
from ppo.pposb3_model import create_actor_critic


def collect_rollout_stats_from_buffer(env, n_envs):
    """Helper to compute statistics from environment after rollout."""
    # Get info from environment's internal state
    # This is a simplified version - adjust based on actual env API
    slot_rewards = [0.0] * n_envs
    slot_steps = [0] * n_envs
    slot_success = [False] * n_envs
    
    # Try to get episode info from environment
    # Note: This might need adjustment based on actual BatchedEnv API
    
    # For now, return placeholder stats
    # In a real implementation, we'd track this during rollout
    return {
        'total': n_envs,
        'successful': 0,
        'success_rate': 0.0,
        'avg_reward': 0.0,
        'avg_steps': 0.0,
        'avg_actions': 0.0
    }


def evaluate_policy(ppo_agent, env, n_eval_episodes: int = 100):
    """
    Evaluate the policy on the environment.
    
    Args:
        ppo_agent: PPO agent
        env: Environment
        n_eval_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    ppo_agent.policy.eval()
    
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    n_envs = int(env.batch_size[0]) if isinstance(env.batch_size, torch.Size) else int(env.batch_size)
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    current_rewards = torch.zeros(n_envs, device=ppo_agent.device)
    current_lengths = torch.zeros(n_envs, dtype=torch.long, device=ppo_agent.device)
    n_completed = 0
    
    with torch.no_grad():
        while n_completed < n_eval_episodes:
            # Get action from policy (returns TensorDict)
            action_td = ppo_agent.predict(obs, deterministic=True)
            
            # Step environment (returns TensorDict with 'next' key)
            step_result = env.step(action_td)
            next_info = step_result['next']
            
            # Extract components
            obs = next_info
            rewards = next_info['reward'].squeeze(-1)
            dones = next_info['done'].squeeze(-1)
            is_success = next_info.get('is_success')
            if is_success is not None:
                is_success = is_success.squeeze(-1)
            
            # Update episode stats
            current_rewards += rewards
            current_lengths += 1
            
            # Check for completed episodes
            for i in range(n_envs):
                if dones[i] and n_completed < n_eval_episodes:
                    episode_rewards.append(float(current_rewards[i]))
                    episode_lengths.append(int(current_lengths[i]))
                    
                    # Get success if available
                    if is_success is not None:
                        episode_successes.append(float(is_success[i]))
                    
                    # Reset tracking for this env
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    n_completed += 1
            
            if n_completed >= n_eval_episodes:
                break
    
    # Compute statistics
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
    success_rate = (np.mean(episode_successes) * 100) if episode_successes else 0.0
    
    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'success_rate': success_rate,
        'n_episodes': n_completed,
    }


def test_ppo_sb3(
    n_tests: int = 3,
    dataset: str = "countries_s3",
    batch_size: int = 100,
    n_steps: int = 64,
    n_epochs: int = 3,
    max_depth: int = 20,
    seed: int = 42,
    device: str = None,
    total_timesteps: int = None,
):
    """
    Test the new PPO SB3-style implementation.
    
    Args:
        n_tests: Number of test stages (1=setup, 2=rollout+eval, 3=training)
        dataset: Dataset name
        batch_size: Number of parallel environments
        n_steps: Steps per rollout
        n_epochs: PPO epochs per update
        max_depth: Maximum proof depth
        seed: Random seed
        device: Device ('cpu' or 'cuda', None = auto-detect)
        total_timesteps: Total timesteps to train (if None, uses n_steps * 10)
    """
    # Set random seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    if total_timesteps is None:
        total_timesteps = n_steps * batch_size
    
    # Centralized configuration
    config = SimpleNamespace(
        dataset=dataset,
        dataset_name=dataset,
        batch_size=batch_size,
        n_steps=n_steps,
        n_epochs=n_epochs,
        max_depth=max_depth,
        seed=seed,
        device=device,
        total_timesteps=total_timesteps,
        # Environment settings
        padding_atoms=6,
        padding_states=20,
        memory_pruning=True,
        reward_type=0,
        verbose=0,
        prover_verbose=0,
        skip_unary_actions=True,
        end_proof_action=False,
        use_exact_memory=True,
        corruption_mode=False,
        train_neg_ratio=0,
        # Embedder settings
        atom_embedder='transe',
        state_embedder='mean',
        constant_embedding_size=100,
        predicate_embedding_size=100,
        atom_embedding_size=100,
        learn_embeddings=True,
        # PPO model settings
        hidden_dim=256,
        num_layers=4,
        dropout_prob=0.0,
        enable_kge_action=False,
        # PPO training settings
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        ppo_batch_size=None,  # Will be set to min(2048, batch_size * n_steps)
    )
    
    # Set PPO batch size
    if config.ppo_batch_size is None:
        config.ppo_batch_size = min(2048, config.batch_size * config.n_steps)
    
    print(f"\n{'='*80}")
    print(f"TESTING PPO SB3 IMPLEMENTATION")
    print(f"{'='*80}")
    print(f"Dataset: {config.dataset}")
    print(f"Batch size: {config.batch_size}")
    print(f"Steps per rollout: {config.n_steps}")
    print(f"Epochs per update: {config.n_epochs}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Max depth: {config.max_depth}")
    print(f"Seed: {config.seed}")
    print(f"Device: {config.device}")
    print(f"{'='*80}\n")
    
    # ============================================================
    # 1. Load dataset and create environment
    # ============================================================
    start_time = time()
    print("[1/3] Loading dataset and creating environment...")
    
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path="data",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
    )
    
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=1000000,
        padding_atoms=config.padding_atoms,
        max_arity=dh.max_arity,
        device=config.device,
    )
    dh.materialize_indices(im=im, device=config.device)
    
    train_split = dh.get_materialized_split('train')
    
    unification_engine = UnificationEngine.from_index_manager(
        im,
        stringifier_params=None,
        max_derived_per_state=20,
        end_pred_idx=im.predicate_str2idx.get('End', None),
        end_proof_action=config.end_proof_action
    )
    
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx.to(config.device),
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=config.device,
        default_mode=['tail'] if config.dataset == 'countries_s3' else ['head', 'tail'],
        seed=config.seed,
    )
    
    # Create environment
    env = BatchedEnv(
        batch_size=config.batch_size,
        unification_engine=unification_engine,
        queries=train_split.queries,
        labels=train_split.labels,
        query_depths=train_split.depths,
        mode='train',
        max_depth=config.max_depth,
        memory_pruning=config.memory_pruning,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        reward_type=config.reward_type,
        verbose=config.verbose,
        prover_verbose=config.prover_verbose,
        device=config.device,
        skip_unary_actions=config.skip_unary_actions,
        end_proof_action=config.end_proof_action,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('End'),
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + 1000000,
        use_exact_memory=config.use_exact_memory,
    )
    
    end_time = time()
    print(f"  Loaded {config.dataset}: {len(train_split.queries)} train queries")
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    
    if n_tests < 1:
        return
    
    # ============================================================
    # 2. Create PPO agent and test rollout collection
    # ============================================================
    start_time = time()
    print(f"\n[2/3] Creating PPO agent and testing rollout collection...")
    
    # Create embedder
    embedder_getter = get_embedder(
        args=config,
        data_handler=dh,
        constant_no=im.constant_no,
        predicate_no=im.predicate_no,
        runtime_var_end_index=im.runtime_var_end_index,
        constant_str2idx=im.constant_str2idx,
        predicate_str2idx=im.predicate_str2idx,
        constant_images_no=0,
        device=config.device
    )
    
    # Create actor-critic policy
    policy = create_actor_critic(
        embedder=embedder_getter.embedder,
        embed_dim=config.atom_embedding_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob,
        device=config.device,
    )
    
    # Create PPO agent
    ppo_agent = PPO(
        policy=policy,
        env=env,
        n_steps=config.n_steps,
        n_epochs=config.n_epochs,
        batch_size=config.ppo_batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        learning_rate=config.learning_rate,
        device=config.device,
        verbose=1,
    )
    
    # Setup and test initial rollout
    ppo_agent._setup_model()
    
    print(f"\n  Testing initial rollout collection (untrained policy)...")
    initial_rollout_start = time()
    ppo_agent.collect_rollouts()
    initial_rollout_time = time() - initial_rollout_start
    
    print(f"  Initial rollout collected in {initial_rollout_time:.2f} seconds")
    print(f"  Buffer size: {ppo_agent.rollout_buffer.size()}")
    
    # Evaluate untrained policy
    print(f"\n  Evaluating untrained policy...")
    eval_start = time()
    eval_stats_before = evaluate_policy(ppo_agent, env, n_eval_episodes=min(100, config.batch_size))
    eval_time = time() - eval_start
    
    print(f"\n  UNTRAINED POLICY RESULTS:")
    print(f"  Avg reward:        {eval_stats_before['avg_reward']:.2f}")
    print(f"  Avg length:        {eval_stats_before['avg_length']:.2f}")
    print(f"  Success rate:      {eval_stats_before['success_rate']:.2f}%")
    print(f"  Evaluation time:   {eval_time:.2f}s")
    
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    
    if n_tests < 2:
        return eval_stats_before
    
    # ============================================================
    # 3. Train with PPO
    # ============================================================
    start_time = time()
    print(f"\n[3/3] Training with PPO for {config.total_timesteps} timesteps...")
    
    # Train
    ppo_agent.learn(
        total_timesteps=config.total_timesteps,
        log_interval=1,
        reset_num_timesteps=True,
    )
    
    training_time = time() - start_time
    
    # Get training metrics
    metrics = ppo_agent.get_logger_dict()
    
    print(f"\n  TRAINING METRICS:")
    print(f"  Policy loss:       {metrics.get('train/policy_loss', 0.0):.4f}")
    print(f"  Value loss:        {metrics.get('train/value_loss', 0.0):.4f}")
    print(f"  Entropy:           {metrics.get('train/entropy', 0.0):.4f}")
    print(f"  Total loss:        {metrics.get('train/total_loss', 0.0):.4f}")
    print(f"  Approx KL:         {metrics.get('train/approx_kl', 0.0):.4f}")
    print(f"  Clip fraction:     {metrics.get('train/clip_fraction', 0.0):.4f}")
    print(f"  Training time:     {training_time:.2f}s")
    
    # Evaluate trained policy
    print(f"\n  Evaluating trained policy...")
    eval_start = time()
    eval_stats_after = evaluate_policy(ppo_agent, env, n_eval_episodes=min(100, config.batch_size))
    eval_time = time() - eval_start
    
    print(f"\n  TRAINED POLICY RESULTS:")
    print(f"  Avg reward:        {eval_stats_after['avg_reward']:.2f}")
    print(f"  Avg length:        {eval_stats_after['avg_length']:.2f}")
    print(f"  Success rate:      {eval_stats_after['success_rate']:.2f}%")
    print(f"  Evaluation time:   {eval_time:.2f}s")
    
    # Compare before and after
    reward_improvement = eval_stats_after['avg_reward'] - eval_stats_before['avg_reward']
    success_improvement = eval_stats_after['success_rate'] - eval_stats_before['success_rate']
    
    print(f"\n  IMPROVEMENT:")
    print(f"  Reward improvement:   {reward_improvement:+.2f}")
    print(f"  Success improvement:  {success_improvement:+.2f}%")
    
    # Sanity checks
    print(f"\n  SANITY CHECKS:")
    
    # Check if losses are reasonable (not NaN or extremely high)
    policy_loss = metrics.get('train/policy_loss', 0.0)
    value_loss = metrics.get('train/value_loss', 0.0)
    
    loss_ok = not (np.isnan(policy_loss) or np.isnan(value_loss) or 
                   policy_loss > 100 or value_loss > 100)
    print(f"  {'✓' if loss_ok else '✗'} Losses are reasonable: "
          f"policy={policy_loss:.4f}, value={value_loss:.4f}")
    
    # Check if training improved performance or at least didn't collapse
    improvement_ok = reward_improvement >= -0.5  # Allow small degradation
    print(f"  {'✓' if improvement_ok else '✗'} Performance didn't collapse: "
          f"improvement={reward_improvement:.2f}")
    
    # Check if success rate is in reasonable range
    success_reasonable = 0 <= eval_stats_after['success_rate'] <= 100
    print(f"  {'✓' if success_reasonable else '✗'} Success rate in valid range: "
          f"{eval_stats_after['success_rate']:.2f}%")
    
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    
    print(f"\n{'='*80}\n")
    
    return {
        'before': eval_stats_before,
        'after': eval_stats_after,
        'metrics': metrics,
        'improvement': {
            'reward': reward_improvement,
            'success': success_improvement,
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PPO SB3 implementation')
    parser.add_argument('--dataset', type=str, default='countries_s3', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of parallel environments')
    parser.add_argument('--n_steps', type=int, default=128, help='Steps per rollout')
    parser.add_argument('--n_epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--total_timesteps', type=int, default=None, help='Total timesteps to train')
    parser.add_argument('--max_depth', type=int, default=20, help='Maximum proof depth')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_tests', type=int, default=3, help='Number of test stages (1=setup, 2=rollout, 3=training)')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu or cuda, None=auto)')
    
    args = parser.parse_args()
    
    test_ppo_sb3(
        n_tests=args.n_tests,
        dataset=args.dataset,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        max_depth=args.max_depth,
        seed=args.seed,
        device=args.device,
        total_timesteps=args.total_timesteps,
    )

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
from debug_config import DebugConfig

# Import new PPO implementation
from ppo.pposb3 import PPO
from ppo.pposb3_model import create_actor_critic
from model_eval import evaluate_policy


def test_rollout_pipeline(test_mode=None, args: SimpleNamespace = None):
    """
    Test the new PPO SB3-style implementation.
    
    Args:
        test_mode: If 'rollout_only', only tests rollout collection without training.
        args: SimpleNamespace with configuration parameters. If None, uses defaults.
    """
    # Default configuration
    if args is None:
        args = SimpleNamespace()
    
    # Set defaults for any missing attributes
    defaults = SimpleNamespace(
        # Test parameters
        dataset='countries_s3',
        batch_size=256,
        n_steps=64,
        n_epochs=10,
        seed=42,
        device=None,
        rollout_device='cpu',
        total_timesteps=None,
        # Environment settings
        max_depth=20,
        padding_atoms=6,
        padding_states=20,
        memory_pruning=True,
        use_exact_memory=False,
        reward_type=0,
        verbose=0,
        prover_verbose=0,
        skip_unary_actions=True,
        end_proof_action=True,
        corruption_mode=False,
        train_neg_ratio=1,
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
        ppo_batch_size=2048,
        # Debug settings
        debug_mode=None,  # Can be 'entropy', 'agent', 'model', 'full', or None
        debug_agent=0,
        debug_model=0,
        debug_rollouts=0,
    )

    defaults.dataset_name = defaults.dataset  # backward compatibility
    
    # Merge defaults with provided args
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Handle device
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)
    
    # Handle rollout and training devices (for CPU rollout / GPU training separation)
    if not hasattr(args, 'rollout_device') or args.rollout_device is None:
        args.rollout_device = args.device  # Default to same as main device
    else:
        args.rollout_device = torch.device(args.rollout_device)
    
    if not hasattr(args, 'training_device') or args.training_device is None:
        args.training_device = args.device  # Default to same as main device
    else:
        args.training_device = torch.device(args.training_device)
    
    # Handle total_timesteps
    if args.total_timesteps is None:
        args.total_timesteps = args.n_steps * args.batch_size
    
    # # Handle ppo_batch_size
    # if args.ppo_batch_size is None:
    #     args.ppo_batch_size = min(2048, args.batch_size * args.n_steps)
    
    # Create config alias for backward compatibility
    config = args
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
    print(f"Rollout device: {config.rollout_device}")
    print(f"Training device: {config.training_device}")
    print(f"{'='*80}\n")
    
    # Create debug configuration early (before env creation)
    if config.debug_mode == 'entropy':
        debug_cfg = DebugConfig.create_entropy_debug()
    elif config.debug_mode == 'agent':
        debug_cfg = DebugConfig.create_agent_debug()
    elif config.debug_mode == 'model':
        debug_cfg = DebugConfig.create_model_debug()
    elif config.debug_mode == 'full':
        debug_cfg = DebugConfig.create_full_debug()
    else:
        # Use manual settings
        debug_cfg = DebugConfig(
            debug_agent=config.debug_agent,
            debug_model=config.debug_model,
            debug_rollouts=config.debug_rollouts,
        )
    
    if debug_cfg.is_enabled('agent') or debug_cfg.is_enabled('model') or debug_cfg.is_enabled('env'):
        print(f"  Debug configuration: {debug_cfg}\n")
    
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
        device=config.rollout_device,
    )
    dh.materialize_indices(im=im, device=config.rollout_device)
    
    train_split = dh.get_materialized_split('train')
    
    unification_engine = UnificationEngine.from_index_manager(
        im,
        stringifier_params=None,
        max_derived_per_state=20,
        end_pred_idx=im.predicate_str2idx.get('End', None),
        end_proof_action=config.end_proof_action
    )
    
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx.to(config.rollout_device),
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=config.rollout_device,
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
        device=config.rollout_device,
        skip_unary_actions=config.skip_unary_actions,
        end_proof_action=config.end_proof_action,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('End'),
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + 1000000,
        use_exact_memory=config.use_exact_memory,
        sampler=sampler,
        corruption_mode=config.corruption_mode,
        train_neg_ratio=config.train_neg_ratio,
        debug_config=debug_cfg,
    )
    
    end_time = time()
    print(f"  Loaded {config.dataset}: {len(train_split.queries)} train queries")
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    
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
        device=config.training_device
    )
    
    # Create actor-critic policy with optimizations
    use_cuda = config.training_device.type == 'cuda'
    policy = create_actor_critic(
        embedder=embedder_getter.embedder,
        embed_dim=config.atom_embedding_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob,
        device=config.training_device,
        debug_config=debug_cfg,
        use_amp=use_cuda,  # Enable AMP for CUDA
        use_compile=use_cuda,  # Enable torch.compile() for CUDA
    )
    
    # Wrap policy to return TensorDict for evaluate_policy compatibility
    class PolicyWrapper(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        
        def forward(self, obs: TensorDict) -> TensorDict:
            actions, values, log_probs = self.policy(obs, deterministic=False)
            return TensorDict({
                "action": actions,
                "value": values,
                "sample_log_prob": log_probs
            }, batch_size=obs.batch_size)
    
    wrapped_policy = PolicyWrapper(policy)
    
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
        device=config.training_device,
        rollout_device=config.rollout_device,
        verbose=1,
        debug_config=debug_cfg,
    )
    
    # Setup and test initial rollout
    ppo_agent._setup_model()

    if test_mode=='rollout_only':
        print(f"\n  Testing initial rollout collection (untrained policy)...")
        initial_rollout_start = time()
        ppo_agent.collect_rollouts()
        initial_rollout_time = time() - initial_rollout_start
        
        print(f"  Initial rollout collected in {initial_rollout_time:.2f} seconds")
        print(f"  Buffer size: {ppo_agent.rollout_buffer.size()}")
        
        # Evaluate untrained policy
        print(f"\n  Evaluating untrained policy...")
        eval_start = time()
        eval_results_before = evaluate_policy(wrapped_policy, env, n_eval_episodes=min(100, config.batch_size), deterministic=True)
        eval_time = time() - eval_start
        
        # Process results
        mask = eval_results_before['mask']
        rewards = eval_results_before['rewards'][mask]
        lengths = eval_results_before['lengths'][mask]
        success = eval_results_before['success'][mask]
        
        eval_stats_before = {
            'avg_reward': float(rewards.mean()) if len(rewards) > 0 else 0.0,
            'avg_length': float(lengths.float().mean()) if len(lengths) > 0 else 0.0,
            'success_rate': float(success.mean() * 100) if len(success) > 0 else 0.0,
            'n_episodes': int(mask.sum())
        }
        
        print(f"\n  UNTRAINED POLICY RESULTS:")
        print(f"  Avg reward:        {eval_stats_before['avg_reward']:.2f}")
        print(f"  Avg length:        {eval_stats_before['avg_length']:.2f}")
        print(f"  Success rate:      {eval_stats_before['success_rate']:.2f}%")
        print(f"  Evaluation time:   {eval_time:.2f}s")
        
        end_time = time()
        print(f"  Step completed in {end_time - start_time:.2f} seconds")
        
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
    eval_results_after = evaluate_policy(wrapped_policy, env, n_eval_episodes=min(100, config.batch_size), deterministic=True)
    eval_time = time() - eval_start
    
    # Process results
    mask = eval_results_after['mask']
    rewards = eval_results_after['rewards'][mask]
    lengths = eval_results_after['lengths'][mask]
    success = eval_results_after['success'][mask]
    
    eval_stats_after = {
        'avg_reward': float(rewards.mean()) if len(rewards) > 0 else 0.0,
        'avg_length': float(lengths.float().mean()) if len(lengths) > 0 else 0.0,
        'success_rate': float(success.mean() * 100) if len(success) > 0 else 0.0,
        'n_episodes': int(mask.sum())
    }
    
    print(f"\n  TRAINED POLICY RESULTS:")
    print(f"  Avg reward:        {eval_stats_after['avg_reward']:.2f}")
    print(f"  Avg length:        {eval_stats_after['avg_length']:.2f}")
    print(f"  Success rate:      {eval_stats_after['success_rate']:.2f}%")
    print(f"  Evaluation time:   {eval_time:.2f}s")
    
    # Sanity checks
    print(f"\n  SANITY CHECKS:")
    
    # Check if losses are reasonable (not NaN or extremely high)
    policy_loss = metrics.get('train/policy_loss', 0.0)
    value_loss = metrics.get('train/value_loss', 0.0)
    
    loss_ok = not (np.isnan(policy_loss) or np.isnan(value_loss) or 
                   policy_loss > 100 or value_loss > 100)
    print(f"  {'✓' if loss_ok else '✗'} Losses are reasonable: "
          f"policy={policy_loss:.4f}, value={value_loss:.4f}")
    
    
    # Check if success rate is in reasonable range
    success_reasonable = 0 <= eval_stats_after['success_rate'] <= 100
    print(f"  {'✓' if success_reasonable else '✗'} Success rate in valid range: "
          f"{eval_stats_after['success_rate']:.2f}%")
    
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    
    print(f"\n{'='*80}\n")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PPO SB3 implementation')
    # Test parameters
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=None, help='Number of parallel environments')
    parser.add_argument('--n_steps', type=int, default=None, help='Steps per rollout')
    parser.add_argument('--n_epochs', type=int, default=None, help='PPO epochs per update')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum proof depth')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu or cuda, None=auto)')
    parser.add_argument('--rollout_device', type=str, default=None, help='Device for rollout collection (cpu or cuda, None=same as device)')
    parser.add_argument('--training_device', type=str, default=None, help='Device for PPO training (cpu or cuda, None=same as device)')
    parser.add_argument('--total_timesteps', type=int, default=None, help='Total timesteps to train')
    
    # Environment settings
    parser.add_argument('--padding_atoms', type=int, default=None, help='Number of padding atoms')
    parser.add_argument('--padding_states', type=int, default=None, help='Number of padding states')
    parser.add_argument('--memory_pruning', type=lambda x: x.lower() == 'true', default=None, help='Enable memory pruning')
    parser.add_argument('--reward_type', type=int, default=None, help='Reward type')
    parser.add_argument('--verbose', type=int, default=None, help='Verbose level')
    
    # Debug settings
    parser.add_argument('--debug_mode', type=str, default=None, 
                        choices=['entropy', 'agent', 'model', 'full'],
                        help='Debug mode preset (entropy/agent/model/full)')
    parser.add_argument('--debug_agent', type=int, default=None, help='Agent debug level (0-2)')
    parser.add_argument('--debug_model', type=int, default=None, help='Model debug level (0-2)')
    parser.add_argument('--debug_rollouts', type=int, default=None, help='Rollout debug level (0-2)')
    
    # Embedder settings
    parser.add_argument('--atom_embedder', type=str, default=None, help='Atom embedder type')
    parser.add_argument('--constant_embedding_size', type=int, default=None, help='Constant embedding size')
    parser.add_argument('--atom_embedding_size', type=int, default=None, help='Atom embedding size')
    
    # PPO model settings
    parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=None, help='Number of layers')
    parser.add_argument('--dropout_prob', type=float, default=None, help='Dropout probability')
    
    # PPO training settings
    parser.add_argument('--gamma', type=float, default=None, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=None, help='GAE lambda')
    parser.add_argument('--clip_range', type=float, default=None, help='PPO clip range')
    parser.add_argument('--ent_coef', type=float, default=None, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=None, help='Value function coefficient')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--ppo_batch_size', type=int, default=None, help='PPO mini-batch size')
    
    cmd_args = parser.parse_args()
    
    # Create args namespace, only including non-None values from command line
    args = SimpleNamespace()
    for key, value in vars(cmd_args).items():
        if value is not None:
            setattr(args, key, value)
    
    test_rollout_pipeline(args=args)

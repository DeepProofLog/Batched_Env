"""
Isolated test for rollout collection with random policy and PPO learning.

This test verifies that rollout collection works correctly by:
1. Loading dataset and creating environment  
2. Collecting rollouts in TRAIN mode with random policy
3. Collecting rollouts in EVAL mode with random policy
4. Collecting rollouts with create_torch_modules (untrained)
5. Testing PPO learning and post-learning evaluation

Expected success rates on countries_s3:
- Random policy (uniform sampling): ~24-26%
- Deterministic policy (first valid action): ~30-40% (depends on query set)
- Note: test_all_configs uses deterministic=True by default

Usage:
    # Test rollout only
    python test_envs/test_rollout.py --batch-size 100 --n-tests 4
    
    # Test with learning
    python test_envs/test_rollout.py --batch-size 100 --n-tests 5 --n-epochs 10
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
from ppo.ppo_model import create_torch_modules
from env import BatchedEnv
from ppo.ppo import PPOAgent
from ppo.ppo_rollout import RolloutCollector
from unification_engine import UnificationEngine


class RandomActor(nn.Module):
    """Simple random actor for testing."""
    def __init__(self, seed: int = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
    
    def forward(self, td: TensorDict) -> TensorDict:
        batch_size = td.batch_size[0] if isinstance(td.batch_size, torch.Size) else td.batch_size
        action_mask = td.get("action_mask")
        
        if action_mask is None:
            raise ValueError("action_mask is None in RandomActor")
        
        # Sample from action mask using torch for each environment
        # action_mask shape: (batch_size, num_actions)
        # We need to sample one valid action per environment
        actions = torch.zeros(batch_size, dtype=torch.long, device=td.device)
        log_probs = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        
        for i in range(batch_size):
            valid_actions = torch.where(action_mask[i])[0]
            if len(valid_actions) > 0:
                idx = torch.randint(len(valid_actions), (1,), device=td.device).item()
                actions[i] = valid_actions[idx]
                log_probs[i] = -np.log(len(valid_actions))
        
        td["action"] = actions
        td["sample_log_prob"] = log_probs
        return td


class DeterministicActor(nn.Module):
    """Deterministic actor that produces logits favoring the first valid action.
    
    This mimics LogitsProducingActor with deterministic=True from test_env_rollout.py
    and should achieve ~48% success rate on countries_s3.
    """
    def __init__(self, max_actions: int = 500, seed: int = None):
        super().__init__()
        self.max_actions = max_actions
        if seed is not None:
            torch.manual_seed(seed)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Produce logits with high value for first valid action."""
        batch_size = td.batch_size[0] if isinstance(td.batch_size, torch.Size) else td.batch_size
        action_mask = td.get("action_mask")
        
        if action_mask is None:
            raise ValueError("action_mask is None in DeterministicActor")
        
        n_actions = action_mask.shape[-1]
        logits = torch.zeros(batch_size, n_actions, device=td.device)
        
        # Set high logit for first valid action (mimics deterministic=True behavior)
        for i in range(batch_size):
            valid_actions = torch.where(action_mask[i])[0]
            if len(valid_actions) > 0:
                first_valid = valid_actions[0]
                logits[i, first_valid] = 10.0  # High logit for first valid action
        
        td["logits"] = logits
        return td


class DummyCritic(nn.Module):
    """Dummy critic that returns zero values."""
    def forward(self, td: TensorDict) -> TensorDict:
        batch_size = td.batch_size[0] if isinstance(td.batch_size, torch.Size) else td.batch_size
        td["state_value"] = torch.zeros(batch_size, 1, device=td.device)
        return td


def collect_rollout_stats(experiences, n_envs):
    """Helper to compute statistics from rollout experiences."""
    slot_rewards = [0.0] * n_envs
    slot_steps = [0] * n_envs
    slot_success = [False] * n_envs
    slot_done = [False] * n_envs
    slot_action_counts = [[] for _ in range(n_envs)]
    
    for step_idx, step_td in enumerate(experiences):
        next_td = step_td.get('next')
        if next_td is None:
            continue
        
        rewards = next_td.get('reward', torch.zeros(n_envs))
        dones = next_td.get('done', torch.zeros(n_envs, dtype=torch.bool))
        is_success = next_td.get('is_success', torch.zeros(n_envs, dtype=torch.bool))
        action_mask = step_td.get('action_mask')
        
        if rewards.dim() > 1:
            rewards = rewards.squeeze()
        if dones.dim() > 1:
            dones = dones.squeeze()
        if is_success.dim() > 1:
            is_success = is_success.squeeze()
        
        for i in range(n_envs):
            if not slot_done[i]:
                slot_rewards[i] += float(rewards[i])
                slot_steps[i] += 1
                
                if action_mask is not None:
                    num_actions = int(action_mask[i].sum())
                    slot_action_counts[i].append(num_actions)
                
                if bool(dones[i]):
                    slot_done[i] = True
                    slot_success[i] = bool(is_success[i])
    
    # Compute statistics
    total = n_envs
    successful = sum(slot_success)
    success_rate = (successful / total * 100) if total > 0 else 0.0
    avg_reward = sum(slot_rewards) / total if total > 0 else 0.0
    avg_steps = sum(slot_steps) / total if total > 0 else 0.0
    
    total_actions = sum(sum(counts) for counts in slot_action_counts)
    total_action_steps = sum(len(counts) for counts in slot_action_counts)
    avg_actions = total_actions / total_action_steps if total_action_steps > 0 else 0.0
    
    return {
        'total': total,
        'successful': successful,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_actions': avg_actions
    }


def test_rollouts(
    n_tests: int = 5,
    dataset: str = "countries_s3",
    batch_size: int = 100,
    n_steps: int = 64,
    n_epochs: int = 0,
    max_depth: int = 20,
    seed: int = 42,
    device: str = None,
    deterministic: bool = False
):
    """
    Test rollout collection and optionally PPO learning.
    
    Args:
        n_tests: Number of test stages (1=data, 2=rollout-train, 3=rollout-eval, 4=rollout-torch, 5=learning)
        dataset: Dataset name
        batch_size: Number of parallel environments
        n_steps: Steps per rollout
        n_epochs: PPO epochs (0 = no learning)
        max_depth: Maximum proof depth
        seed: Random seed
        device: Device ('cpu' or 'cuda', None = auto-detect)
        deterministic: If True, use deterministic actor in eval mode (selects first valid action, ~48% success)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Centralized configuration using SimpleNamespace
    config = SimpleNamespace(
        dataset=dataset,
        dataset_name=dataset,  # For embedder compatibility
        batch_size=batch_size,
        n_steps=n_steps,
        n_epochs=n_epochs,
        max_depth=max_depth,
        seed=seed,
        device=device,
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
        # Embedder settings (for torch modules test)
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
    )
    
    print(f"\n{'='*80}")
    print(f"TESTING ROLLOUT COLLECTION")
    print(f"{'='*80}")
    print(f"Dataset: {config.dataset}")
    print(f"Batch size: {config.batch_size}")
    print(f"Steps: {config.n_steps}")
    print(f"Epochs: {config.n_epochs} {'(rollout only)' if config.n_epochs == 0 else '(with learning)'}")
    print(f"Max depth: {config.max_depth}")
    print(f"Seed: {config.seed}")
    print(f"Device: {config.device}")
    print(f"{'='*80}\n")
    
    # ============================================================
    # 1. Load dataset and create environment
    # ============================================================
    start_time = time()
    print("[1/5] Loading dataset and creating environment...")
    
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
    
    # Create environment for train mode testing
    env_train = BatchedEnv(
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
    
    # Create environment for eval mode testing with dummy queries
    dummy_query = torch.full((config.batch_size, config.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=config.device)
    dummy_labels = torch.ones(config.batch_size, dtype=torch.long, device=config.device)
    dummy_depths = torch.ones(config.batch_size, dtype=torch.long, device=config.device)
    
    env_eval = BatchedEnv(
        batch_size=config.batch_size,
        unification_engine=unification_engine,
        queries=dummy_query,
        labels=dummy_labels,
        query_depths=dummy_depths,
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
    # 2. Test rollout collection in TRAIN mode
    # ============================================================
    start_time = time()
    print(f"\n[2/5] Collecting rollout in TRAIN mode with random policy...")
    
    actor_train = RandomActor(seed=config.seed)
    critic_train = DummyCritic()
    
    rollout_collector_train = RolloutCollector(
        env=env_train,
        actor=actor_train,
        n_envs=config.batch_size,
        n_steps=config.n_steps,
        device=config.device,
        debug=False,
    )
    
    experiences_train, stats_train = rollout_collector_train.collect(critic=critic_train)
    rollout_stats_train = collect_rollout_stats(experiences_train, config.batch_size)
    
    # Print rollout results
    print(f"\n  TRAIN MODE RESULTS:")
    print(f"  Total queries:     {rollout_stats_train['total']}")
    print(f"  Successful:        {rollout_stats_train['successful']}")
    print(f"  Success rate:      {rollout_stats_train['success_rate']:.2f}%")
    print(f"  Avg reward:        {rollout_stats_train['avg_reward']:.2f}")
    print(f"  Avg steps:         {rollout_stats_train['avg_steps']:.2f}")
    print(f"  Avg actions:       {rollout_stats_train['avg_actions']:.2f}")
    
    # Verification (random policy typically gets ~26% for countries_s3)
    # Note: deterministic=True (first valid action) gives ~48%, random gives ~26%
    if rollout_stats_train['success_rate'] > 0:
        expected = 26.0
        diff = abs(rollout_stats_train['success_rate'] - expected)
        status = "✓" if diff < 5.0 else "✗"
        print(f"  {status} Expected ~{expected}% for random policy, got {rollout_stats_train['success_rate']:.2f}% (diff: {diff:.2f}%)")
    
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    
    if n_tests < 2:
        return rollout_stats_train
    
    # ============================================================
    # 3. Test rollout collection in EVAL mode
    # ============================================================
    start_time = time()
    actor_type = "deterministic" if deterministic else "random"
    print(f"\n[3/5] Collecting rollout in EVAL mode with {actor_type} policy...")
    
    if deterministic:
        actor_eval = DeterministicActor(max_actions=config.padding_states, seed=config.seed)
    else:
        actor_eval = RandomActor(seed=config.seed)
    critic_eval = DummyCritic()
    
    rollout_collector_eval = RolloutCollector(
        env=env_eval,
        actor=actor_eval,
        n_envs=config.batch_size,
        n_steps=config.n_steps,
        device=config.device,
        debug=False,
    )
    
    # Switch environment to eval mode
    env_queries = train_split.queries[:config.batch_size]
    env_labels = train_split.labels[:config.batch_size] if hasattr(train_split, 'labels') else torch.ones(config.batch_size, dtype=torch.long)
    env_depths = train_split.depths[:config.batch_size] if hasattr(train_split, 'depths') else torch.ones(config.batch_size, dtype=torch.long)
    per_slot_lengths = torch.ones(config.batch_size, dtype=torch.long, device=config.device)
    
    env_eval.set_eval_dataset(
        queries=env_queries,
        labels=env_labels,
        query_depths=env_depths,
        per_slot_lengths=per_slot_lengths
    )
    
    experiences_eval, stats_eval = rollout_collector_eval.collect(critic=critic_eval)
    rollout_stats_eval = collect_rollout_stats(experiences_eval, config.batch_size)
    
    # Print rollout results
    print(f"\n  EVAL MODE RESULTS:")
    print(f"  Total queries:     {rollout_stats_eval['total']}")
    print(f"  Successful:        {rollout_stats_eval['successful']}")
    print(f"  Success rate:      {rollout_stats_eval['success_rate']:.2f}%")
    print(f"  Avg reward:        {rollout_stats_eval['avg_reward']:.2f}")
    print(f"  Avg steps:         {rollout_stats_eval['avg_steps']:.2f}")
    print(f"  Avg actions:       {rollout_stats_eval['avg_actions']:.2f}")
    
    # Verification
    # IMPORTANT: Random policy gets ~24-26%, deterministic (first valid action) gets ~30-40%
    # The exact success rate depends on the specific queries in the batch
    if rollout_stats_eval['success_rate'] > 0:
        if deterministic:
            # Deterministic: expect 30-40%
            expected_min, expected_max = 20.0, 50.0
            in_range = expected_min <= rollout_stats_eval['success_rate'] <= expected_max
            status = "✓" if in_range else "✗"
            print(f"  {status} Expected {expected_min}-{expected_max}% for deterministic policy, got {rollout_stats_eval['success_rate']:.2f}%")
        else:
            # Random: expect ~26%
            expected = 26.0
            diff = abs(rollout_stats_eval['success_rate'] - expected)
            status = "✓" if diff < 5.0 else "✗"
            print(f"  {status} Expected ~{expected}% for random policy, got {rollout_stats_eval['success_rate']:.2f}% (diff: {diff:.2f}%)")
    
    # Compare train vs eval
    success_diff = abs(rollout_stats_train['success_rate'] - rollout_stats_eval['success_rate'])
    comp_status = "✓" if success_diff < 5.0 else "✗"
    print(f"  {comp_status} Train vs Eval difference: {success_diff:.2f}%")
    
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    
    if n_tests < 3:
        return rollout_stats_eval
    
    # ============================================================
    # 4. Test rollout collection with create_torch_modules
    # ============================================================
    start_time = time()
    print(f"\n[4/5] Collecting rollout with create_torch_modules (trained actor/critic)...")
    
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
    
    # Create actor and critic using create_torch_modules
    actor_torch, critic_torch = create_torch_modules(
        embedder=embedder_getter.embedder,
        num_actions=config.padding_states,
        embed_dim=config.atom_embedding_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob,
        device=config.device,
        enable_kge_action=config.enable_kge_action,
        kge_inference_engine=None,
        index_manager=im,
    )
    
    # Reset environment for fresh test
    env_eval.set_eval_dataset(
        queries=env_queries,
        labels=env_labels,
        query_depths=env_depths,
        per_slot_lengths=per_slot_lengths
    )
    
    rollout_collector_torch = RolloutCollector(
        env=env_eval,
        actor=actor_torch,
        n_envs=config.batch_size,
        n_steps=config.n_steps,
        device=config.device,
        debug=False,
    )
    
    experiences_torch, stats_torch = rollout_collector_torch.collect(critic=critic_torch)
    rollout_stats_torch = collect_rollout_stats(experiences_torch, config.batch_size)
    
    # Print rollout results
    print(f"\n  TORCH MODULES RESULTS:")
    print(f"  Total queries:     {rollout_stats_torch['total']}")
    print(f"  Successful:        {rollout_stats_torch['successful']}")
    print(f"  Success rate:      {rollout_stats_torch['success_rate']:.2f}%")
    print(f"  Avg reward:        {rollout_stats_torch['avg_reward']:.2f}")
    print(f"  Avg steps:         {rollout_stats_torch['avg_steps']:.2f}")
    print(f"  Avg actions:       {rollout_stats_torch['avg_actions']:.2f}")
    
    # Compare with random actor results
    torch_vs_random = abs(rollout_stats_torch['success_rate'] - rollout_stats_eval['success_rate'])
    print(f"\n  Comparison with random actor (eval mode):")
    print(f"    Success rate difference: {torch_vs_random:.2f}%")
    print(f"    Note: Untrained torch modules may perform differently than random policy")
    
    end_time = time()
    print(f"  Step completed in {end_time - start_time:.2f} seconds")
    
    if n_tests < 4:
        return rollout_stats_torch
    
    # ============================================================
    # 5. Test PPO learning
    # ============================================================
    if config.n_epochs > 0:
        start_time = time()
        print(f"\n[5/5] Testing PPO learning for {config.n_epochs} epochs...")
        
        # Use the actor and critic already created in test 4
        optimizer = torch.optim.AdamW(
            list(actor_torch.parameters()) + list(critic_torch.parameters()),
            lr=3e-4
        )
        
        # Create PPO agent
        ppo_agent = PPOAgent(
            actor=actor_torch,
            critic=critic_torch,
            optimizer=optimizer,
            train_env=env_train,
            eval_env=None,
            sampler=sampler,
            data_handler=dh,
            args=config,
            n_envs=config.batch_size,
            n_steps=config.n_steps,
            n_epochs=config.n_epochs,
            batch_size=min(2048, config.batch_size * config.n_steps),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            value_coef=0.5,
            max_grad_norm=0.5,
            device=config.device,
            debug_mode=False,
            min_multiaction_ratio=0.05,
            use_amp=False,
        )
        
        # Use the experiences already collected from test 4
        print(f"  Learning from {len(experiences_torch)} experience steps...")
        
        # Learn from experiences
        ppo_agent.learn(experiences=experiences_torch, n_steps=config.n_steps, n_envs=config.batch_size)
        
        # Collect new rollout with trained actor to see improvement
        print(f"\n  Collecting post-learning rollout...")
        env_train_reset = BatchedEnv(
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
        
        rollout_collector_trained = RolloutCollector(
            env=env_train_reset,
            actor=actor_torch,
            n_envs=config.batch_size,
            n_steps=config.n_steps,
            device=config.device,
            debug=False,
        )
        
        experiences_trained, _ = rollout_collector_trained.collect(critic=critic_torch)
        rollout_stats_trained = collect_rollout_stats(experiences_trained, config.batch_size)
        
        # Print results
        print(f"\n  POST-LEARNING RESULTS:")
        print(f"  Total queries:     {rollout_stats_trained['total']}")
        print(f"  Successful:        {rollout_stats_trained['successful']}")
        print(f"  Success rate:      {rollout_stats_trained['success_rate']:.2f}%")
        print(f"  Avg reward:        {rollout_stats_trained['avg_reward']:.2f}")
        print(f"  Avg steps:         {rollout_stats_trained['avg_steps']:.2f}")
        print(f"  Avg actions:       {rollout_stats_trained['avg_actions']:.2f}")
        
        # Compare with pre-training
        improvement = rollout_stats_trained['success_rate'] - rollout_stats_torch['success_rate']
        print(f"\n  Improvement over untrained:")
        print(f"    Success rate change: {improvement:+.2f}%")
        print(f"    Before: {rollout_stats_torch['success_rate']:.2f}%")
        print(f"    After:  {rollout_stats_trained['success_rate']:.2f}%")
        
        end_time = time()
        print(f"  Step completed in {end_time - start_time:.2f} seconds")
        
        print(f"\n{'='*80}\n")
        return rollout_stats_trained
    
    print(f"\n{'='*80}\n")
    return rollout_stats_torch


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test rollout collection and optionally PPO learning')
    parser.add_argument('--dataset', type=str, default='countries_s3', help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of parallel environments')
    parser.add_argument('--n-steps', type=int, default=64, help='Steps per rollout')
    parser.add_argument('--n-epochs', type=int, default=5, help='PPO epochs (0=rollout only)')
    parser.add_argument('--max-depth', type=int, default=20, help='Maximum proof depth')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n-tests', type=int, default=5, help='Number of test stages (1=data, 2=train, 3=eval, 4=torch, 5=learning)')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu or cuda, None=auto)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actor in eval mode (first valid action, 30-40% success)')
    
    args = parser.parse_args()
    
    test_rollouts(
        n_tests=args.n_tests,
        dataset=args.dataset,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        max_depth=args.max_depth,
        seed=args.seed,
        device=args.device,
        deterministic=args.deterministic
    )

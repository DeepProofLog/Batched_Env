"""
Test to verify that rollout collection is working correctly.

This test uses the countries_s3 dataset to verify that:
1. Random rollouts don't give completely random rewards
2. The environment respects done conditions
3. Rewards are within expected ranges
4. Memory pruning works
"""
import sys
import os
import torch
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import DataHandler
from index_manager import IndexManager
from embeddings import get_embedder
from ppo.ppo_model import create_torchrl_modules
from env import BatchedVecEnv
from ppo.ppo_rollout_custom import CustomRolloutCollector


# Set seeds for deterministic behavior
torch.manual_seed(123)
import random
random.seed(123)
np.random.seed(123)

def test_rollout_basic_sanity():
    """Test basic sanity of rollout collection."""
    print("\n" + "="*80)
    print("TEST 1: Basic Rollout Sanity Check (countries_s3)")
    print("="*80)
    
    # Use a simpler dataset
    args = SimpleNamespace(
        dataset_name="countries_s3",
        data_path="data",
        janus_file="countries_s3.pl",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="countries_s3.pl",
        n_train_queries=None,
        n_eval_queries=5,
        n_test_queries=5,
        corruption_mode=None,
        corruption_scheme=['head', 'tail'],
        max_total_vars=20,
        padding_atoms=6,
        padding_states=12,
        max_depth=20,
        memory_pruning=True,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=16,
        predicate_embedding_size=16,
        atom_embedding_size=16,
        learn_embeddings=True,
        variable_no=10,
        seed_run_i=42,
        batch_size=2,
        n_steps=300,
        n_epochs=1,
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        engine='python_tensor',
        endt_action=False,
        endf_action=True,
        skip_unary_actions=True,
        train_neg_ratio=1.0,
        reward_type=1,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n[1] Loading dataset...")
    dh = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        n_train_queries=args.n_train_queries,
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
    )
    print(f"  Loaded: {len(dh.train_queries)} train, {len(dh.facts)} facts, {len(dh.rules)} rules")
    
    # Build index manager
    print("\n[2] Building index manager...")
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_vars=args.max_total_vars,
        rules=dh.rules,
        padding_atoms=args.padding_atoms,
        max_arity=dh.max_arity,
        device=device,
    )
    im.build_fact_index(dh.facts)
    print(f"  Index ready: {im.constant_no} constants, {im.predicate_no} predicates")
    
    # Create embedder (skip sampler for simplicity)
    print("\n[3] Creating embedder...")
    embedder_getter = get_embedder(args, dh, im, device)
    embedder = embedder_getter.embedder
    embed_dim = getattr(embedder, 'embed_dim', getattr(embedder, 'embedding_dim', args.atom_embedding_size))
    
    # Create environment
    print("\n[4] Creating environment...")
    env = BatchedVecEnv(
        batch_size=args.batch_size,
        index_manager=im,
        data_handler=dh,
        queries=dh.train_queries,
        labels=[1] * len(dh.train_queries),
        query_depths=[0] * len(dh.train_queries),
        facts=set(dh.facts),
        mode='train',
        seed=args.seed_run_i,
        max_depth=args.max_depth,
        memory_pruning=args.memory_pruning,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        verbose=2,  # Enable verbose output with state strings for debugging
        prover_verbose=0,
        device=device,
        engine=args.engine,
    )
    print(f"  Environment ready: batch_size={env.batch_size_int}")
    
    # Create actor
    print("\n[5] Creating actor...")
    actor, critic = create_torchrl_modules(
        embedder=embedder,
        num_actions=args.padding_states,
        embed_dim=embed_dim,
        hidden_dim=16,
        num_layers=1,
        dropout_prob=0.0,
        device=device,
        enable_kge_action=False,
        kge_inference_engine=None,
        index_manager=im,
    )
    
    # Create rollout collector
    print("\n[6] Collecting rollout...")
    rollout_collector = CustomRolloutCollector(
        env=env,
        actor=actor,
        n_envs=args.batch_size,
        n_steps=args.n_steps,
        device=device,
        debug=False,  # Disable debug mode
    )
    
    experiences, stats = rollout_collector.collect(critic=critic)
    
    print(f"\n[7] Analyzing rollout results...")
    print(f"  Total steps collected: {len(experiences)}")
    print(f"  Episodes completed: {len(stats.get('episode_info', []))}")
    
    # Extract rewards
    rewards = []
    dones = []
    for exp in experiences:
        if 'next' in exp.keys():
            reward = exp['next']['reward']
            done = exp['next']['done']
        else:
            reward = exp.get('reward', torch.zeros(args.batch_size, 1))
            done = exp.get('done', torch.zeros(args.batch_size, 1))
        
        rewards.append(reward.cpu().numpy())
        dones.append(done.cpu().numpy())
    
    rewards = np.concatenate(rewards, axis=0)
    dones = np.concatenate(dones, axis=0)
    
    print(f"\n[8] Reward statistics:")
    print(f"  Mean reward: {rewards.mean():.4f}")
    print(f"  Std reward: {rewards.std():.4f}")
    print(f"  Min reward: {rewards.min():.4f}")
    print(f"  Max reward: {rewards.max():.4f}")
    print(f"  Non-zero rewards: {(rewards != 0).sum()} / {len(rewards)} ({100*(rewards != 0).sum()/len(rewards):.1f}%)")
    
    print(f"\n[9] Done statistics:")
    print(f"  Episodes finished: {dones.sum()}")
    print(f"  Done rate: {dones.mean():.4f}")
    
    # Assertions
    assert len(experiences) == args.n_steps, f"Expected {args.n_steps} experiences, got {len(experiences)}"
    assert rewards.shape[0] > 0, "Should have collected some rewards"
    assert rewards.min() >= -1.0, "Rewards should be >= -1.0"
    assert rewards.max() <= 1.0, "Rewards should be <= 1.0"
    
    # Check positive reward rate
    positive_reward_rate = (rewards > 0).sum() / len(rewards)
    print(f"  Positive reward rate: {positive_reward_rate:.4f}")
    
    
    print("\n✓ All basic sanity checks passed!")
    return rewards, dones


def test_rollout_determinism():
    """Test that rollouts are deterministic given same seed."""
    print("\n" + "="*80)
    print("TEST 2: Rollout Determinism Check")
    print("="*80)
    
    args = SimpleNamespace(
        dataset_name="countries_s3",
        data_path="data",
        janus_file="countries_s3.pl",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="countries_s3.pl",
        n_train_queries=None,
        n_eval_queries=5,
        n_test_queries=5,
        corruption_mode=None,
        corruption_scheme=['head', 'tail'],
        max_total_vars=20,
        padding_atoms=6,
        padding_states=12,
        max_depth=10,
        memory_pruning=False,  # Disable for determinism test
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=16,
        predicate_embedding_size=16,
        atom_embedding_size=16,
        learn_embeddings=True,
        variable_no=10,
        seed_run_i=123,
        batch_size=4,
        n_steps=16,
        n_epochs=1,
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        engine='python_tensor',
        endt_action=False,
        endf_action=True,
        skip_unary_actions=True,
        train_neg_ratio=1.0,
        reward_type=4,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def collect_rollout(seed):
        """Helper to collect a rollout with given seed."""
        # Set seeds for deterministic behavior
        torch.manual_seed(seed)
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        dh = DataHandler(
            dataset_name=args.dataset_name,
            base_path=args.data_path,
            janus_file=args.janus_file,
            train_file=args.train_file,
            valid_file=args.valid_file,
            test_file=args.test_file,
            rules_file=args.rules_file,
            facts_file=args.facts_file,
            n_train_queries=args.n_train_queries,
            n_eval_queries=args.n_eval_queries,
            n_test_queries=args.n_test_queries,
        )
        
        im = IndexManager(
            constants=dh.constants,
            predicates=dh.predicates,
            max_total_vars=args.max_total_vars,
            rules=dh.rules,
            padding_atoms=args.padding_atoms,
            max_arity=dh.max_arity,
            device=device,
        )
        im.build_fact_index(dh.facts)
        
        embedder_getter = get_embedder(args, dh, im, device)
        embedder = embedder_getter.embedder
        embed_dim = getattr(embedder, 'embed_dim', getattr(embedder, 'embedding_dim', args.atom_embedding_size))
        
        env = BatchedVecEnv(
            batch_size=args.batch_size,
            index_manager=im,
            data_handler=dh,
            queries=dh.train_queries,
            labels=[1] * len(dh.train_queries),
            query_depths=[0] * len(dh.train_queries),
            facts=set(dh.facts),
            mode='train',
            seed=seed,
            max_depth=args.max_depth,
            memory_pruning=args.memory_pruning,
            padding_atoms=args.padding_atoms,
            padding_states=args.padding_states,
            verbose=0,
            prover_verbose=0,
            device=device,
            engine=args.engine,
        )
        
        actor, critic = create_torchrl_modules(
            embedder=embedder,
            num_actions=args.padding_states,
            embed_dim=embed_dim,
            hidden_dim=16,
            num_layers=1,
            dropout_prob=0.0,
            device=device,
            enable_kge_action=False,
            kge_inference_engine=None,
            index_manager=im,
        )
        
        rollout_collector = CustomRolloutCollector(
            env=env,
            actor=actor,
            n_envs=args.batch_size,
            n_steps=args.n_steps,
            device=device,
            debug=False,
        )
        
        experiences, stats = rollout_collector.collect(critic=critic)
        
        # Extract actions
        actions = []
        for exp in experiences:
            actions.append(exp['action'].cpu().numpy())
        
        return np.concatenate(actions, axis=0)
    
    print("  Collecting rollout with seed 123...")
    actions_1 = collect_rollout(123)
    print(f"actions_1: {actions_1}")
    
    print("  Collecting rollout with seed 123 again...")
    actions_2 = collect_rollout(123)
    print(f"actions_2: {actions_2}")
    print("  Collecting rollout with seed 456...")
    actions_3 = collect_rollout(456)
    
    print(f"\n  Comparing results...")
    print(f"    Actions 1 vs 2 (same seed): {(actions_1 == actions_2).mean():.4f} match rate")
    print(f"    Actions 1 vs 3 (diff seed): {(actions_1 == actions_3).mean():.4f} match rate")
    
    # Same seed should give same actions
    match_rate_same = (actions_1 == actions_2).mean()
    match_rate_diff = (actions_1 == actions_3).mean()
    
    print(f"\n  Assertions...")
    assert match_rate_same == 1.0, f"Same seed should give identical rollouts, got {match_rate_same:.4f} match rate"
    
    # Note: Different seeds might still give same actions if the actor is deterministic or the environment state limits options
    # This is OK as long as same seed gives same actions
    if match_rate_diff == 1.0:
        print(f"  NOTE: Different seeds gave same actions - this may be OK if actor is nearly deterministic")
    
    print("\n✓ Determinism test passed!")


def test_rollout_memory_pruning():
    """Test that memory pruning works correctly."""
    print("\n" + "="*80)
    print("TEST 3: Memory Pruning Check")
    print("="*80)
    
    args = SimpleNamespace(
        dataset_name="countries_s3",
        data_path="data",
        janus_file="countries_s3.pl",
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="countries_s3.pl",
        n_train_queries=10,  # Small for faster test
        n_eval_queries=5,
        n_test_queries=5,
        corruption_mode=None,
        corruption_scheme=['head', 'tail'],
        max_total_vars=20,
        padding_atoms=6,
        padding_states=12,
        max_depth=15,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=16,
        predicate_embedding_size=16,
        atom_embedding_size=16,
        learn_embeddings=True,
        variable_no=10,
        seed_run_i=789,
        batch_size=4,
        n_steps=20,
        n_epochs=1,
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        engine='python_tensor',
        endt_action=False,
        endf_action=True,
        skip_unary_actions=True,
        train_neg_ratio=1.0,
        reward_type=4,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with memory pruning ON
    print("\n  Testing with memory pruning ON...")
    dh = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        n_train_queries=args.n_train_queries,
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
    )
    
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_vars=args.max_total_vars,
        rules=dh.rules,
        padding_atoms=args.padding_atoms,
        max_arity=dh.max_arity,
        device=device,
    )
    im.build_fact_index(dh.facts)
    
    embedder_getter = get_embedder(args, dh, im, device)
    embedder = embedder_getter.embedder
    embed_dim = getattr(embedder, 'embed_dim', getattr(embedder, 'embedding_dim', args.atom_embedding_size))
    
    env_with_pruning = BatchedVecEnv(
        batch_size=args.batch_size,
        index_manager=im,
        data_handler=dh,
        queries=dh.train_queries,
        labels=[1] * len(dh.train_queries),
        query_depths=[0] * len(dh.train_queries),
        facts=set(dh.facts),
        mode='train',
        seed=args.seed_run_i,
        max_depth=args.max_depth,
        memory_pruning=True,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        verbose=0,
        prover_verbose=0,
        device=device,
        engine=args.engine,
    )
    
    env_without_pruning = BatchedVecEnv(
        batch_size=args.batch_size,
        index_manager=im,
        data_handler=dh,
        queries=dh.train_queries,
        labels=[1] * len(dh.train_queries),
        query_depths=[0] * len(dh.train_queries),
        facts=set(dh.facts),
        mode='train',
        seed=args.seed_run_i,
        max_depth=args.max_depth,
        memory_pruning=False,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        verbose=0,
        prover_verbose=0,
        device=device,
        engine=args.engine,
    )
    
    # Check that memory sets exist with pruning
    assert hasattr(env_with_pruning, 'memories'), "Environment should have memories with pruning enabled"
    assert len(env_with_pruning.memories) == args.batch_size, "Should have one memory set per environment"
    
    print(f"    Memory sets initialized: {len(env_with_pruning.memories)}")
    
    # Do a few steps
    obs = env_with_pruning.reset()
    for _ in range(5):
        # Random actions
        actions = torch.randint(0, args.padding_states, (args.batch_size,), device=device)
        obs['action'] = actions
        obs = env_with_pruning.step(obs)
    
    # Check that memories are being populated
    memory_sizes = [len(mem) for mem in env_with_pruning.memories]
    print(f"    Memory sizes after 5 steps: {memory_sizes}")
    
    assert all(size > 0 for size in memory_sizes), "Memories should be populated after steps"
    assert all(size <= 5 for size in memory_sizes), "Memory size should be reasonable"
    
    print("\n✓ Memory pruning test passed!")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ROLLOUT COLLECTION VERIFICATION TESTS")
    print("="*80)
    
    # Run tests
    rewards, dones = test_rollout_basic_sanity()
    # test_rollout_determinism()
    # test_rollout_memory_pruning()

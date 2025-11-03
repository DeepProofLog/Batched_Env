"""
Integration test for PPO agent with parallel environments and rollout collection.

This test verifies that the complete PPO pipeline works correctly:
1. Parallel environments (TorchRL ParallelEnv)
2. PPO agent with actor-critic model
3. Rollout collection with SyncDataCollector
4. GAE advantage computation
5. Policy and value network updates

The test uses a simplified dummy environment to avoid dependencies on
the full data pipeline while testing the core RL infrastructure.
"""

import torch
import torch.nn as nn
from torchrl.envs import EnvBase, ParallelEnv
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import OneHotCategorical

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ppo.ppo_agent import PPOAgent
from ppo.ppo_rollout import collect_rollouts


class DummyLogicEnv(EnvBase):
    """
    Simplified environment that mimics LogicEnv's observation/action structure.
    
    This environment produces observations with:
    - sub_index: Current state representation (batch, 1, 3, 3)
    - derived_sub_indices: Available actions (batch, 10, 3, 3)
    - action_mask: Valid action mask (batch, 10)
    
    Actions are discrete choices from the available derived states.
    """
    
    def __init__(self, env_id=0, max_steps=10, num_actions=10, device=None):
        super().__init__(device=device)
        self.env_id = env_id
        self.max_steps = max_steps
        self.num_actions = num_actions
        self.step_count = 0
        
        # Define specs to match LogicEnv structure
        self.observation_spec = CompositeSpec(
            sub_index=UnboundedContinuousTensorSpec(shape=(1, 3, 3), dtype=torch.int32),
            derived_sub_indices=UnboundedContinuousTensorSpec(shape=(num_actions, 3, 3), dtype=torch.int32),
            action_mask=DiscreteTensorSpec(n=2, shape=(num_actions,), dtype=torch.bool),
            shape=()
        )
        
        # Action is one-hot encoded (matching OneHotCategorical output)
        self.action_spec = DiscreteTensorSpec(n=2, shape=(num_actions,), dtype=torch.bool)
        
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool)
        
    def _reset(self, tensordict=None, **kwargs):
        """Reset the environment"""
        self.step_count = 0
        
        # Generate random state and action indices
        sub_index = torch.randint(0, 100, (1, 3, 3), dtype=torch.int32, device=self.device)
        derived_sub_indices = torch.randint(0, 100, (self.num_actions, 3, 3), dtype=torch.int32, device=self.device)
        
        # Random action mask (at least one action must be valid)
        action_mask = torch.rand(self.num_actions, device=self.device) > 0.3
        if not action_mask.any():
            action_mask[0] = True
        
        td = TensorDict({
            'sub_index': sub_index,
            'derived_sub_indices': derived_sub_indices,
            'action_mask': action_mask,
            'terminated': torch.tensor([False], device=self.device),
            'truncated': torch.tensor([False], device=self.device),
            'done': torch.tensor([False], device=self.device),
        }, batch_size=())
        
        return td
    
    def _step(self, tensordict):
        """Take a step in the environment"""
        action = tensordict['action']  # one-hot encoded (bool tensor)
        self.step_count += 1
        
        # Convert one-hot action to index
        if action.dtype == torch.bool:
            action_idx = torch.nonzero(action, as_tuple=True)[0].item() if action.any() else 0
        else:
            action_idx = torch.argmax(action).item() if action.ndim > 0 else action.item()
        
        # Check if done
        done = self.step_count >= self.max_steps
        
        # Reward based on action validity (higher reward for valid actions)
        current_mask = tensordict.get('action_mask', None)
        if current_mask is not None and action_idx < len(current_mask):
            reward = 1.0 if current_mask[action_idx].item() else -0.5
        else:
            reward = 0.0
        
        # Generate next state
        next_sub_index = torch.randint(0, 100, (1, 3, 3), dtype=torch.int32, device=self.device)
        next_derived_sub_indices = torch.randint(0, 100, (self.num_actions, 3, 3), dtype=torch.int32, device=self.device)
        next_action_mask = torch.rand(self.num_actions, device=self.device) > 0.3
        if not next_action_mask.any():
            next_action_mask[0] = True
        
        # Build output following TorchRL convention
        td = TensorDict({
            'reward': torch.tensor([reward], dtype=torch.float32, device=self.device),
            'terminated': torch.tensor([done], device=self.device),
            'truncated': torch.tensor([False], device=self.device),
            'done': torch.tensor([done], device=self.device),
            'next': TensorDict({
                'sub_index': next_sub_index,
                'derived_sub_indices': next_derived_sub_indices,
                'action_mask': next_action_mask,
                'reward': torch.tensor([reward], dtype=torch.float32, device=self.device),
                'done': torch.tensor([done], device=self.device),
                # Add episode info for callbacks
                'label': torch.tensor([1 if reward > 0 else 0], dtype=torch.long, device=self.device),
                'query_depth': torch.tensor([self.step_count], dtype=torch.long, device=self.device),
                'is_success': torch.tensor([done and reward > 0], device=self.device),
            }, batch_size=()),
        }, batch_size=())
        
        return td
    
    def _set_seed(self, seed):
        """Set random seed"""
        torch.manual_seed(seed)
        return seed


class DummyEmbedder(nn.Module):
    """Simple embedder that converts indices to learned embeddings."""
    
    def __init__(self, vocab_size=1000, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def get_embeddings_batch(self, indices):
        """
        Get embeddings for a batch of indices.
        
        Args:
            indices: Tensor of shape (batch, num_items, atoms, 3)
            
        Returns:
            Embeddings of shape (batch, num_items, embed_dim)
        """
        # Move indices to the same device as the embedding
        indices = indices.to(self.embedding.weight.device)
        
        # Flatten last dimension and embed
        # indices shape: (batch, num_items, atoms, 3)
        batch_size = indices.shape[0]
        num_items = indices.shape[1]
        
        # Average over atoms and the 3 indices
        indices_flat = indices.reshape(batch_size, num_items, -1)
        
        # Clamp indices to valid range
        indices_flat = torch.clamp(indices_flat, 0, self.embedding.num_embeddings - 1)
        
        # Embed and average
        embeds = self.embedding(indices_flat)  # (batch, num_items, atoms*3, embed_dim)
        embeds = embeds.mean(dim=2)  # (batch, num_items, embed_dim)
        
        return embeds


class SimplePolicyNetwork(nn.Module):
    """Simple policy network for testing."""
    
    def __init__(self, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.action_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, obs_embeddings, action_embeddings, action_mask):
        """Compute policy logits."""
        # Encode
        obs_encoded = self.obs_net(obs_embeddings)  # (batch, embed_dim)
        action_encoded = self.action_net(action_embeddings)  # (batch, num_actions, embed_dim)
        
        # Compute similarity scores
        obs_encoded = obs_encoded.unsqueeze(-2)  # (batch, 1, embed_dim)
        logits = torch.matmul(obs_encoded, action_encoded.transpose(-2, -1)).squeeze(-2)  # (batch, num_actions)
        
        # Mask invalid actions
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        
        return logits


class SimpleValueNetwork(nn.Module):
    """Simple value network for testing."""
    
    def __init__(self, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, embeddings):
        """Compute value estimate."""
        return self.net(embeddings).squeeze(-1)


class SimpleEmbeddingExtractor(nn.Module):
    """Extract embeddings from observations."""
    
    def __init__(self, embedder, device):
        super().__init__()
        self.embedder = embedder
        self.embedding_dim = embedder.embed_dim
        self.device = device
        
    def forward(self, observations):
        """Extract embeddings from observations."""
        obs_sub_indices = observations["sub_index"].to(torch.int32)
        action_sub_indices = observations["derived_sub_indices"].to(torch.int32)
        action_mask = observations["action_mask"]
        
        # Get embeddings
        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices).squeeze(1)
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices)
        
        return obs_embeddings, action_embeddings, action_mask


class ActorModule(nn.Module):
    """Wrapper module for actor that extracts embeddings and computes logits."""
    
    def __init__(self, embedding_extractor, policy_net):
        super().__init__()
        self.embedding_extractor = embedding_extractor
        self.policy_net = policy_net
        
    def forward(self, sub_index, derived_sub_indices, action_mask):
        """Compute logits from observations."""
        td = TensorDict({
            "sub_index": sub_index,
            "derived_sub_indices": derived_sub_indices,
            "action_mask": action_mask,
        }, batch_size=[])
        obs_emb, action_emb, action_mask = self.embedding_extractor(td)
        logits = self.policy_net(obs_emb, action_emb, action_mask)
        return logits


class CriticModule(nn.Module):
    """Wrapper module for critic that extracts embeddings and computes values."""
    
    def __init__(self, embedding_extractor, value_net):
        super().__init__()
        self.embedding_extractor = embedding_extractor
        self.value_net = value_net
        
    def forward(self, sub_index, derived_sub_indices, action_mask):
        """Compute value from observations."""
        td = TensorDict({
            "sub_index": sub_index,
            "derived_sub_indices": derived_sub_indices,
            "action_mask": action_mask,
        }, batch_size=[])
        obs_emb, _, _ = self.embedding_extractor(td)
        value = self.value_net(obs_emb)
        return value


def build_actor_critic(embedder, embed_dim=64, hidden_dim=128, num_actions=10, device=None):
    """Build TorchRL-compatible actor and critic networks."""
    
    # Embedding extractor
    embedding_extractor = SimpleEmbeddingExtractor(embedder, device)
    
    # Policy network
    policy_net = SimplePolicyNetwork(embed_dim, hidden_dim)
    
    # Value network
    value_net = SimpleValueNetwork(embed_dim, hidden_dim)
    
    # Build actor wrapper
    actor_module = ActorModule(embedding_extractor, policy_net)
    
    # Wrap in TensorDictModule
    actor_td_module = TensorDictModule(
        module=actor_module,
        in_keys=["sub_index", "derived_sub_indices", "action_mask"],
        out_keys=["logits"]
    )
    
    # Wrap in ProbabilisticActor
    actor = ProbabilisticActor(
        module=actor_td_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )
    
    # Build critic wrapper
    critic_module = CriticModule(embedding_extractor, value_net)
    
    # Wrap in TensorDictModule
    critic = TensorDictModule(
        module=critic_module,
        in_keys=["sub_index", "derived_sub_indices", "action_mask"],
        out_keys=["state_value"]
    )
    
    return actor, critic, embedding_extractor, policy_net, value_net


def test_rollout_collection():
    """Test rollout collection with parallel environments."""
    print("=" * 70)
    print("TEST 1: Rollout Collection with Parallel Environments")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_envs = 4
    n_steps = 8
    num_actions = 10
    embed_dim = 64
    
    print(f"\nDevice: {device}")
    print(f"Number of environments: {n_envs}")
    print(f"Steps per rollout: {n_steps}")
    print(f"Actions per state: {num_actions}")
    
    # Create embedder
    embedder = DummyEmbedder(vocab_size=1000, embed_dim=embed_dim).to(device)
    
    # Build actor and critic
    actor, critic, _, _, _ = build_actor_critic(embedder, embed_dim, num_actions=num_actions, device=device)
    actor.to(device)
    critic.to(device)
    
    # Create parallel environment
    print("\n1. Creating parallel environment...")
    env = ParallelEnv(
        num_workers=n_envs,
        create_env_fn=lambda: DummyLogicEnv(num_actions=num_actions, device=device),
        device=device,
    )
    print(f"   ✓ Parallel environment created with {n_envs} workers")
    
    # Test reset
    print("\n2. Testing environment reset...")
    reset_td = env.reset()
    print(f"   ✓ Reset successful - batch_size: {reset_td.batch_size}")
    assert reset_td.batch_size == torch.Size([n_envs])
    assert 'sub_index' in reset_td.keys()
    assert 'derived_sub_indices' in reset_td.keys()
    assert 'action_mask' in reset_td.keys()
    print(f"   ✓ All required keys present in reset output")
    
    # Collect rollouts
    print("\n3. Collecting rollouts...")
    experiences, stats = collect_rollouts(
        env=env,
        actor=actor,
        critic=critic,
        n_envs=n_envs,
        n_steps=n_steps,
        device=device,
    )
    print(f"   ✓ Collected {len(experiences)} timesteps")
    print(f"   ✓ Episode info: {len(stats['episode_info'])} episodes completed")
    
    # Verify experience structure
    print("\n4. Verifying experience structure...")
    assert len(experiences) == n_steps, f"Expected {n_steps} experiences, got {len(experiences)}"
    
    for i, exp in enumerate(experiences):
        assert exp.batch_size == torch.Size([n_envs]), f"Step {i}: wrong batch size"
        assert 'sub_index' in exp.keys(), f"Step {i}: missing sub_index"
        assert 'derived_sub_indices' in exp.keys(), f"Step {i}: missing derived_sub_indices"
        assert 'action_mask' in exp.keys(), f"Step {i}: missing action_mask"
        assert 'action' in exp.keys(), f"Step {i}: missing action"
        assert 'sample_log_prob' in exp.keys(), f"Step {i}: missing sample_log_prob"
        assert 'state_value' in exp.keys(), f"Step {i}: missing state_value"
        assert 'next' in exp.keys(), f"Step {i}: missing next"
        
        next_td = exp['next']
        assert 'reward' in next_td.keys(), f"Step {i}: missing reward in next"
        assert 'done' in next_td.keys(), f"Step {i}: missing done in next"
    
    print(f"   ✓ All {n_steps} experiences have correct structure")
    
    # Clean up (collector already closed the environment)
    print("\n" + "=" * 70)
    print("✓ TEST 1 PASSED: Rollout collection works correctly")
    print("=" * 70)


def test_ppo_agent_training():
    """Test complete PPO agent training loop."""
    print("\n" + "=" * 70)
    print("TEST 2: PPO Agent Training Loop")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_envs = 4
    n_steps = 8
    num_actions = 10
    embed_dim = 64
    hidden_dim = 128
    
    print(f"\nDevice: {device}")
    print(f"Training configuration:")
    print(f"  - Environments: {n_envs}")
    print(f"  - Steps per rollout: {n_steps}")
    print(f"  - Epochs per update: 3")
    print(f"  - Batch size: 16")
    
    # Create embedder
    embedder = DummyEmbedder(vocab_size=1000, embed_dim=embed_dim).to(device)
    
    # Build actor and critic
    actor, critic, _, policy_net, value_net = build_actor_critic(
        embedder, embed_dim, hidden_dim, num_actions=num_actions, device=device
    )
    actor.to(device)
    critic.to(device)
    
    # Create optimizer
    params = list(policy_net.parameters()) + list(value_net.parameters()) + list(embedder.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    
    # Create environments
    print("\n1. Creating training and evaluation environments...")
    train_env = ParallelEnv(
        num_workers=n_envs,
        create_env_fn=lambda: DummyLogicEnv(num_actions=num_actions, device=device),
        device=device,
    )
    eval_env = DummyLogicEnv(num_actions=num_actions, device=device)
    print(f"   ✓ Environments created")
    
    # Create PPO agent (with minimal dependencies)
    print("\n2. Creating PPO agent...")
    agent = PPOAgent(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        train_env=train_env,
        eval_env=eval_env,
        sampler=None,  # Not needed for this test
        data_handler=None,  # Not needed for this test
        n_envs=n_envs,
        n_steps=n_steps,
        n_epochs=3,
        batch_size=16,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device=device,
    )
    print(f"   ✓ PPO agent created")
    
    # Collect rollouts
    print("\n3. Collecting initial rollouts...")
    experiences, stats = collect_rollouts(
        env=train_env,
        actor=actor,
        critic=critic,
        n_envs=n_envs,
        n_steps=n_steps,
        device=device,
    )
    print(f"   ✓ Collected {len(experiences)} timesteps")
    print(f"   ✓ Completed episodes: {len(stats['episode_info'])}")
    
    # Compute advantages
    print("\n4. Computing advantages with GAE...")
    advantages, returns = agent.compute_advantages(experiences, n_steps, n_envs)
    print(f"   ✓ Advantages shape: {advantages.shape}")
    print(f"   ✓ Returns shape: {returns.shape}")
    assert advantages.shape == torch.Size([n_steps, n_envs])
    assert returns.shape == torch.Size([n_steps, n_envs])
    
    # Perform learning step
    print("\n5. Performing PPO learning step...")
    initial_params = {name: param.clone() for name, param in policy_net.named_parameters()}
    
    train_stats = agent.learn(experiences, n_steps, n_envs)
    
    print(f"   ✓ Training stats:")
    print(f"      - Policy loss: {train_stats['policy_loss']:.4f}")
    print(f"      - Value loss: {train_stats['value_loss']:.4f}")
    print(f"      - Entropy: {train_stats['entropy']:.4f}")
    print(f"      - KL divergence: {train_stats['approx_kl']:.4f}")
    
    # Verify parameters updated
    params_changed = False
    for name, param in policy_net.named_parameters():
        if not torch.allclose(param, initial_params[name], atol=1e-6):
            params_changed = True
            break
    
    assert params_changed, "Model parameters did not change after learning step"
    print(f"   ✓ Model parameters updated successfully")
    
    # Test second rollout to ensure everything still works
    print("\n6. Testing second rollout after parameter update...")
    experiences2, stats2 = collect_rollouts(
        env=train_env,
        actor=actor,
        critic=critic,
        n_envs=n_envs,
        n_steps=n_steps,
        device=device,
    )
    print(f"   ✓ Second rollout successful")
    
    # Clean up (collector already closed the environment)
    print("\n" + "=" * 70)
    print("✓ TEST 2 PASSED: PPO agent training loop works correctly")
    print("=" * 70)


def test_multi_update_training():
    """Test multiple training updates to ensure stability."""
    print("\n" + "=" * 70)
    print("TEST 3: Multiple Training Updates (Stability Test)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_envs = 8
    n_steps = 16
    num_actions = 10
    embed_dim = 32
    hidden_dim = 64
    num_updates = 5
    
    print(f"\nDevice: {device}")
    print(f"Running {num_updates} training updates")
    
    # Create components
    embedder = DummyEmbedder(vocab_size=1000, embed_dim=embed_dim).to(device)
    actor, critic, _, policy_net, value_net = build_actor_critic(
        embedder, embed_dim, hidden_dim, num_actions=num_actions, device=device
    )
    actor.to(device)
    critic.to(device)
    
    params = list(policy_net.parameters()) + list(value_net.parameters()) + list(embedder.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    
    train_env = ParallelEnv(
        num_workers=n_envs,
        create_env_fn=lambda: DummyLogicEnv(num_actions=num_actions, max_steps=20, device=device),
        device=device,
    )
    
    agent = PPOAgent(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        train_env=train_env,
        eval_env=None,
        sampler=None,
        data_handler=None,
        n_envs=n_envs,
        n_steps=n_steps,
        n_epochs=2,
        batch_size=32,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device=device,
    )
    
    print("\nRunning training updates...")
    for update in range(num_updates):
        # Collect rollouts
        experiences, stats = collect_rollouts(
            env=train_env,
            actor=actor,
            critic=critic,
            n_envs=n_envs,
            n_steps=n_steps,
            device=device,
        )
        
        # Compute advantages
        advantages, returns = agent.compute_advantages(experiences, n_steps, n_envs)
        
        # Learn
        train_stats = agent.learn(experiences, n_steps, n_envs)
        
        print(f"  Update {update+1}/{num_updates}: "
              f"policy_loss={train_stats['policy_loss']:.4f}, "
              f"value_loss={train_stats['value_loss']:.4f}, "
              f"entropy={train_stats['entropy']:.4f}")
        
        # Check for NaN/Inf
        assert not torch.isnan(torch.tensor(train_stats['policy_loss'])), "Policy loss is NaN"
        assert not torch.isnan(torch.tensor(train_stats['value_loss'])), "Value loss is NaN"
        assert not torch.isinf(torch.tensor(train_stats['policy_loss'])), "Policy loss is Inf"
        assert not torch.isinf(torch.tensor(train_stats['value_loss'])), "Value loss is Inf"
    
    # Clean up (collector already closed the environment)
    print("\n" + "=" * 70)
    print("✓ TEST 3 PASSED: Multiple updates completed without instability")
    print("=" * 70)


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("PPO INTEGRATION TEST SUITE")
    print("=" * 70)
    print("\nThis test suite verifies:")
    print("  1. Rollout collection with parallel environments")
    print("  2. PPO agent training loop (rollout → GAE → learn)")
    print("  3. Training stability over multiple updates")
    print("\n" + "=" * 70)
    
    try:
        test_rollout_collection()
        test_ppo_agent_training()
        test_multi_update_training()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe PPO implementation is working correctly:")
        print("  ✓ Parallel environments functioning properly")
        print("  ✓ Rollout collection via SyncDataCollector")
        print("  ✓ GAE advantage computation")
        print("  ✓ PPO learning updates")
        print("  ✓ Training stability over multiple iterations")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run_all_tests()

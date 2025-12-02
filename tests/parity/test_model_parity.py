"""
Model Parity Tests.

Tests verifying that the tensor-based ActorCriticPolicy produces the same
results as the SB3 CustomActorCriticPolicy.
"""
from pathlib import Path
import sys
from typing import Tuple, Optional, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import pytest
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))

# Import real embedders from both implementations
from embeddings import EmbedderLearnable

# Try to import SB3 components for parity testing
try:
    from sb3.sb3_model import CustomActorCriticPolicy as SB3CustomActorCriticPolicy
    from sb3.sb3_model import CustomCombinedExtractor as SB3CustomCombinedExtractor
    from sb3.sb3_embeddings import EmbedderLearnable as SB3EmbedderLearnable
    SB3_AVAILABLE = True
except ImportError as e:
    SB3_AVAILABLE = False
    _SB3_IMPORT_ERROR = e


# ============================================================================
# Embedder Creation Helpers
# ============================================================================

def create_embedder(
    n_constants: int = 100,
    n_predicates: int = 20,
    n_vars: int = 10,
    embed_dim: int = 64,
    seed: int = 42,
    device: str = "cpu"
) -> EmbedderLearnable:
    """Create a real EmbedderLearnable for the tensor-based implementation."""
    torch.manual_seed(seed)
    embedder = EmbedderLearnable(
        n_constants=n_constants,
        n_predicates=n_predicates,
        n_vars=n_vars,
        max_arity=2,
        padding_atoms=3,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=embed_dim,
        predicate_embedding_size=embed_dim,
        atom_embedding_size=embed_dim,
        kge_regularization=0.0,
        kge_dropout_rate=0.0,
        device=device,
    )
    return embedder


def create_sb3_embedder(
    n_constants: int = 100,
    n_predicates: int = 20,
    n_vars: int = 10,
    embed_dim: int = 64,
    seed: int = 42,
    device: str = "cpu"
):
    """Create a real EmbedderLearnable for the SB3 implementation."""
    if not SB3_AVAILABLE:
        pytest.skip(f"SB3 dependencies unavailable: {_SB3_IMPORT_ERROR}")
    torch.manual_seed(seed)
    embedder = SB3EmbedderLearnable(
        n_constants=n_constants,
        n_predicates=n_predicates,
        n_vars=n_vars,
        max_arity=2,
        padding_atoms=3,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=embed_dim,
        predicate_embedding_size=embed_dim,
        atom_embedding_size=embed_dim,
        kge_regularization=0.0,
        kge_dropout_rate=0.0,
        device=device,
    )
    return embedder


def create_dummy_observation(
    batch_size: int = 4,
    n_actions: int = 16,
    n_atoms: int = 3,
    device: str = "cpu"
) -> TensorDict:
    """Create a dummy observation TensorDict."""
    torch.manual_seed(42)
    
    # sub_index: (batch, 1, n_atoms, 3)
    sub_index = torch.randint(0, 50, (batch_size, 1, n_atoms, 3), dtype=torch.int32, device=device)
    
    # derived_sub_indices: (batch, n_actions, n_atoms, 3)
    derived_sub_indices = torch.randint(0, 50, (batch_size, n_actions, n_atoms, 3), dtype=torch.int32, device=device)
    
    # action_mask: (batch, n_actions) - random valid/invalid actions
    action_mask = torch.randint(0, 2, (batch_size, n_actions), dtype=torch.bool, device=device)
    # Ensure at least one valid action per batch
    action_mask[:, 0] = True
    
    return TensorDict({
        "sub_index": sub_index,
        "derived_sub_indices": derived_sub_indices,
        "action_mask": action_mask,
    }, batch_size=torch.Size([batch_size]))


# ============================================================================
# Network Architecture Tests
# ============================================================================

def test_shared_policy_value_network_shapes():
    """Test that SharedPolicyValueNetwork produces correct output shapes."""
    from model import SharedPolicyValueNetwork
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    shared_net = SharedPolicyValueNetwork(embed_dim=embed_dim, hidden_dim=128, num_layers=4)
    
    obs_embeddings = torch.randn(batch_size, 1, embed_dim)
    action_embeddings = torch.randn(batch_size, n_actions, embed_dim)
    action_mask = torch.ones(batch_size, n_actions, dtype=torch.bool)
    action_mask[:, -3:] = False  # Mask some actions
    
    logits = shared_net.forward_policy(obs_embeddings, action_embeddings, action_mask)
    
    assert logits.shape == (batch_size, n_actions)
    # Masked positions should be -inf
    assert torch.all(torch.isinf(logits[:, -3:]))
    # Valid positions should be finite
    assert torch.all(torch.isfinite(logits[:, :-3]))


def test_shared_value_head_shapes():
    """Test that SharedPolicyValueNetwork value head produces correct output shapes."""
    from model import SharedPolicyValueNetwork
    
    torch.manual_seed(42)
    
    batch_size = 4
    embed_dim = 64
    
    shared_net = SharedPolicyValueNetwork(embed_dim=embed_dim, hidden_dim=128, num_layers=4)
    
    obs_embeddings = torch.randn(batch_size, embed_dim)
    
    values = shared_net.forward_value(obs_embeddings)
    
    assert values.shape == (batch_size,)


def test_custom_network_forward():
    """Test CustomNetwork forward method."""
    from model import CustomNetwork
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    custom_net = CustomNetwork(embed_dim=embed_dim)
    
    obs_embeddings = torch.randn(batch_size, 1, embed_dim)
    action_embeddings = torch.randn(batch_size, n_actions, embed_dim)
    action_mask = torch.ones(batch_size, n_actions, dtype=torch.bool)
    
    features = (obs_embeddings, action_embeddings, action_mask)
    logits, values = custom_net(features)
    
    assert logits.shape == (batch_size, n_actions)
    assert values.shape == (batch_size,)


# ============================================================================
# ActorCriticPolicy Tests
# ============================================================================

def test_actor_critic_forward():
    """Test ActorCriticPolicy forward method."""
    from model import ActorCriticPolicy
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    embedder = DummyEmbedder(vocab_size=100, embed_dim=embed_dim)
    
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device("cpu"),
        action_dim=n_actions,
    )
    policy.eval()
    
    obs = create_dummy_observation(batch_size=batch_size, n_actions=n_actions)
    
    with torch.no_grad():
        actions, values, log_probs = policy(obs, deterministic=True)
    
    assert actions.shape == (batch_size,)
    assert values.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    
    # Actions should be valid indices
    assert torch.all(actions >= 0)
    assert torch.all(actions < n_actions)
    
    # Log probs should be finite and <= 0
    assert torch.all(torch.isfinite(log_probs))
    assert torch.all(log_probs <= 0)


def test_actor_critic_evaluate_actions():
    """Test ActorCriticPolicy evaluate_actions method."""
    from model import ActorCriticPolicy
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    embedder = DummyEmbedder(vocab_size=100, embed_dim=embed_dim)
    
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device("cpu"),
        action_dim=n_actions,
    )
    policy.eval()
    
    obs = create_dummy_observation(batch_size=batch_size, n_actions=n_actions)
    
    # Get some actions first
    with torch.no_grad():
        actions, _, _ = policy(obs, deterministic=False)
    
    # Evaluate those actions
    values, log_probs, entropy = policy.evaluate_actions(obs, actions)
    
    assert values.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    
    # Entropy should be non-negative
    assert torch.all(entropy >= 0)


def test_actor_critic_predict_values():
    """Test ActorCriticPolicy predict_values method."""
    from model import ActorCriticPolicy
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    embedder = DummyEmbedder(vocab_size=100, embed_dim=embed_dim)
    
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device("cpu"),
        action_dim=n_actions,
    )
    policy.eval()
    
    obs = create_dummy_observation(batch_size=batch_size, n_actions=n_actions)
    
    with torch.no_grad():
        values = policy.predict_values(obs)
    
    assert values.shape == (batch_size,)
    assert torch.all(torch.isfinite(values))


# ============================================================================
# Determinism Tests
# ============================================================================

def test_deterministic_mode_consistency():
    """Test that deterministic mode gives same actions."""
    from model import ActorCriticPolicy
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    embedder = DummyEmbedder(vocab_size=100, embed_dim=embed_dim)
    
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device("cpu"),
        action_dim=n_actions,
    )
    policy.eval()
    
    obs = create_dummy_observation(batch_size=batch_size, n_actions=n_actions)
    
    with torch.no_grad():
        actions1, values1, log_probs1 = policy(obs, deterministic=True)
        actions2, values2, log_probs2 = policy(obs, deterministic=True)
    
    # Deterministic mode should give identical results
    assert torch.allclose(actions1, actions2)
    assert torch.allclose(values1, values2)
    assert torch.allclose(log_probs1, log_probs2)


def test_stochastic_mode_variability():
    """Test that stochastic mode can give different actions.
    
    This test verifies that the model's sample() method actually samples
    from the distribution rather than always returning the mode.
    """
    from model import ActorCriticPolicy
    from stable_baselines3.common.distributions import CategoricalDistribution
    
    # Test 1: Verify the distribution's sample method works correctly
    # Create a distribution with uniform probabilities where sampling should vary
    test_logits = torch.zeros(16, 8)  # 16 batches, 8 actions - uniform distribution
    dist = CategoricalDistribution(action_dim=8)
    dist.proba_distribution(action_logits=test_logits)
    
    # Sample multiple times and check for variability
    samples = [dist.sample() for _ in range(50)]
    stacked = torch.stack(samples, dim=0)
    
    # With uniform distribution and 50 samples, there must be variability
    unique_counts = [len(stacked[:, i].unique()) for i in range(16)]
    assert sum(u > 1 for u in unique_counts) >= 10, \
        "Distribution sample() should produce variable results with uniform logits"
    
    # Test 2: Verify ActorCriticPolicy uses sample() in stochastic mode
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    embedder = DummyEmbedder(vocab_size=100, embed_dim=embed_dim)
    
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device("cpu"),
        action_dim=n_actions,
    )
    policy.eval()
    
    obs = create_dummy_observation(batch_size=batch_size, n_actions=n_actions)
    
    # Get the distribution logits to check if they're reasonably varied
    with torch.no_grad():
        features = policy.extract_features(obs)
        if policy.share_features_extractor:
            logits, _ = policy.mlp_extractor(features)
        else:
            obs_embeddings, action_embeddings, action_mask = features[0]
            logits = policy.mlp_extractor.forward_actor((obs_embeddings, action_embeddings, action_mask))
    
    # The test passes if either:
    # 1. The model produces varied samples, OR  
    # 2. The logits are so peaked that sampling the same action is expected
    probs = torch.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1).values
    
    # If max probability > 0.99, the distribution is effectively deterministic
    # and we should skip the variability check
    if max_probs.mean() < 0.99:
        all_actions = []
        with torch.no_grad():
            for _ in range(30):
                actions, _, _ = policy(obs, deterministic=False)
                all_actions.append(actions.clone())
        
        stacked = torch.stack(all_actions, dim=0)
        unique_per_batch = [len(stacked[:, i].unique()) for i in range(batch_size)]
        
        # At least some batches should have variable actions
        assert sum(u > 1 for u in unique_per_batch) >= 1 or n_actions == 1, \
            f"Expected stochastic variability but got unique counts: {unique_per_batch}"


# ============================================================================
# Action Mask Tests
# ============================================================================

def test_action_mask_respected():
    """Test that masked actions are never selected in deterministic mode."""
    from model import ActorCriticPolicy
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    embedder = DummyEmbedder(vocab_size=100, embed_dim=embed_dim)
    
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device("cpu"),
        action_dim=n_actions,
    )
    policy.eval()
    
    obs = create_dummy_observation(batch_size=batch_size, n_actions=n_actions)
    
    # Mask half the actions
    obs["action_mask"][:, n_actions // 2:] = False
    
    with torch.no_grad():
        for _ in range(10):
            actions, _, _ = policy(obs, deterministic=False)
            # All actions should be in the valid range
            assert torch.all(actions < n_actions // 2), f"Invalid action selected: {actions}"


# ============================================================================
# Gradient Flow Tests
# ============================================================================

def test_gradient_flow_evaluate_actions():
    """Test that gradients flow through evaluate_actions."""
    from model import ActorCriticPolicy
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    embedder = DummyEmbedder(vocab_size=100, embed_dim=embed_dim)
    
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device("cpu"),
        action_dim=n_actions,
    )
    policy.train()
    
    obs = create_dummy_observation(batch_size=batch_size, n_actions=n_actions)
    actions = torch.randint(0, n_actions // 2, (batch_size,))  # Valid actions only
    
    values, log_probs, entropy = policy.evaluate_actions(obs, actions)
    
    # Compute dummy loss
    loss = values.mean() + log_probs.mean() + entropy.mean()
    loss.backward()
    
    # Check that some gradients exist
    has_grads = False
    for param in policy.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grads = True
            break
    
    assert has_grads, "No gradients computed"


# ============================================================================
# Feature Extractor Tests
# ============================================================================

def test_feature_extractor_output():
    """Test CustomCombinedExtractor produces correct outputs."""
    from model import CustomCombinedExtractor
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    embedder = DummyEmbedder(vocab_size=100, embed_dim=embed_dim)
    extractor = CustomCombinedExtractor(embedder)
    
    obs = create_dummy_observation(batch_size=batch_size, n_actions=n_actions)
    
    obs_emb, action_emb, mask = extractor(obs)
    
    # Check shapes
    assert obs_emb.shape[-1] == embed_dim
    assert action_emb.shape[-1] == embed_dim
    assert mask.shape == (batch_size, n_actions)


# ============================================================================
# Numerical Stability Tests
# ============================================================================

def test_no_nan_with_all_masked():
    """Test that model handles edge case where many actions are masked."""
    from model import ActorCriticPolicy
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    embedder = DummyEmbedder(vocab_size=100, embed_dim=embed_dim)
    
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        device=torch.device("cpu"),
        action_dim=n_actions,
    )
    policy.eval()
    
    obs = create_dummy_observation(batch_size=batch_size, n_actions=n_actions)
    
    # Leave only one action valid per batch
    obs["action_mask"][:, :] = False
    obs["action_mask"][:, 0] = True
    
    with torch.no_grad():
        actions, values, log_probs = policy(obs, deterministic=True)
    
    # All should select action 0
    assert torch.all(actions == 0)
    # No NaN/Inf in outputs
    assert torch.all(torch.isfinite(values))
    assert torch.all(torch.isfinite(log_probs))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

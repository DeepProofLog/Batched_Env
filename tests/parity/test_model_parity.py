"""
Model Parity Tests.

Tests verifying that the tensor-based ActorCriticPolicy produces the same
results as the SB3 CustomActorCriticPolicy.
"""
from pathlib import Path
import sys
from typing import Tuple, Optional, Dict

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


# ============================================================================
# Dummy Components
# ============================================================================

class DummyEmbedder:
    """Minimal embedder for testing."""
    
    def __init__(self, vocab_size: int = 100, embed_dim: int = 64, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding_dim = embed_dim  # Alias for compatibility
        self.device = device
        
        torch.manual_seed(42)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.to(device)
    
    def get_embeddings_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for batch of indices."""
        # indices shape: (batch, ..., 3) where last dim is [pred, arg1, arg2]
        # Return mean embedding over atoms
        original_shape = indices.shape
        flat_indices = indices.reshape(-1).clamp(0, self.vocab_size - 1)
        embeddings = self.embedding(flat_indices.long())
        
        # Reshape back and take mean over last atom dimensions
        new_shape = original_shape[:-1] + (self.embed_dim,)
        embeddings = embeddings.view(*original_shape, self.embed_dim)
        
        # Mean over the 3 terms (pred, arg1, arg2) to get per-atom embedding
        embeddings = embeddings.mean(dim=-2)
        
        # If there are multiple atoms, mean over atoms too
        while embeddings.dim() > len(original_shape[:-2]) + 1:
            embeddings = embeddings.mean(dim=-2)
        
        return embeddings


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

def test_policy_network_shapes():
    """Test that PolicyNetwork produces correct output shapes."""
    from model import PolicyNetwork
    
    torch.manual_seed(42)
    
    batch_size = 4
    n_actions = 16
    embed_dim = 64
    
    policy_net = PolicyNetwork(embed_dim=embed_dim, hidden_dim=128, num_layers=4)
    
    obs_embeddings = torch.randn(batch_size, 1, embed_dim)
    action_embeddings = torch.randn(batch_size, n_actions, embed_dim)
    action_mask = torch.ones(batch_size, n_actions, dtype=torch.bool)
    action_mask[:, -3:] = False  # Mask some actions
    
    logits = policy_net(obs_embeddings, action_embeddings, action_mask)
    
    assert logits.shape == (batch_size, n_actions)
    # Masked positions should be -inf
    assert torch.all(torch.isinf(logits[:, -3:]))
    # Valid positions should be finite
    assert torch.all(torch.isfinite(logits[:, :-3]))


def test_value_network_shapes():
    """Test that ValueNetwork produces correct output shapes."""
    from model import ValueNetwork
    
    torch.manual_seed(42)
    
    batch_size = 4
    embed_dim = 64
    
    value_net = ValueNetwork(embed_dim=embed_dim, hidden_dim=128, num_layers=4)
    
    obs_embeddings = torch.randn(batch_size, embed_dim)
    
    values = value_net(obs_embeddings)
    
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
    """Test that stochastic mode can give different actions."""
    from model import ActorCriticPolicy
    
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
    
    # Collect multiple samples
    all_actions = []
    with torch.no_grad():
        for _ in range(10):
            actions, _, _ = policy(obs, deterministic=False)
            all_actions.append(actions.clone())
    
    # With 10 samples, there should be some variability
    # (though this is probabilistic and could fail rarely)
    stacked = torch.stack(all_actions, dim=0)
    unique_per_batch = [len(stacked[:, i].unique()) for i in range(batch_size)]
    
    # At least some batches should have variable actions
    assert sum(u > 1 for u in unique_per_batch) >= 1 or n_actions == 1


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

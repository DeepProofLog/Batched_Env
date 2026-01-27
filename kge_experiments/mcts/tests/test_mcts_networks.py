"""
Tests for MCTS neural networks.

Tests MCTSPolicy forward pass shapes and gradient flow.
"""

import sys
from pathlib import Path

# Add paths for imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "kge_experiments") not in sys.path:
    sys.path.insert(0, str(_ROOT / "kge_experiments"))

import pytest
import torch
import torch.nn as nn

from mcts.networks import MCTSPolicy, MCTSEmbedder, MCTSBackbone


class TestMCTSEmbedder:
    """Tests for MCTSEmbedder."""

    def test_embedder_creation(self):
        """Test embedder creation with basic params."""
        embedder = MCTSEmbedder(
            n_constants=100,
            n_predicates=50,
            n_vars=100,
            embedding_dim=64,
            padding_idx=0,
        )

        assert embedder.embedding_dim == 64

    def test_embedder_forward_3d(self):
        """Test embedder forward pass with [B, A, 3] input."""
        embedder = MCTSEmbedder(
            n_constants=100,
            n_predicates=50,
            n_vars=100,
            embedding_dim=64,
        )

        # [B, A, 3] input
        indices = torch.randint(1, 50, (4, 8, 3))
        output = embedder(indices)

        # Should aggregate to [B, E]
        assert output.shape == (4, 64)

    def test_embedder_forward_4d(self):
        """Test embedder forward pass with [B, S, A, 3] input."""
        embedder = MCTSEmbedder(
            n_constants=100,
            n_predicates=50,
            n_vars=100,
            embedding_dim=64,
        )

        # [B, S, A, 3] input
        indices = torch.randint(1, 50, (4, 10, 8, 3))
        output = embedder(indices)

        # Should aggregate to [B, S, E]
        assert output.shape == (4, 10, 64)


class TestMCTSBackbone:
    """Tests for MCTSBackbone."""

    def test_backbone_creation(self):
        """Test backbone creation."""
        backbone = MCTSBackbone(
            input_dim=64,
            hidden_dim=128,
            num_layers=4,
        )

        assert backbone.input_dim == 64
        assert backbone.hidden_dim == 128

    def test_backbone_forward(self):
        """Test backbone forward pass."""
        backbone = MCTSBackbone(input_dim=64, hidden_dim=128, num_layers=4)

        x = torch.randn(4, 10, 64)
        output = backbone(x)

        assert output.shape == (4, 10, 128)

    def test_backbone_gradient_flow(self):
        """Test gradients flow through backbone."""
        backbone = MCTSBackbone(input_dim=64, hidden_dim=128, num_layers=4)

        x = torch.randn(4, 10, 64, requires_grad=True)
        output = backbone(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestMCTSPolicy:
    """Tests for MCTSPolicy."""

    @pytest.fixture
    def policy(self):
        """Create test policy."""
        embedder = MCTSEmbedder(
            n_constants=100,
            n_predicates=50,
            n_vars=100,
            embedding_dim=64,
        )
        return MCTSPolicy(
            embedder=embedder,
            embed_dim=64,
            hidden_dim=128,
            num_layers=2,
            device=torch.device("cpu"),
        )

    def test_policy_creation(self, policy):
        """Test policy creation."""
        assert policy.embed_dim == 64
        assert policy.hidden_dim == 128

    def test_get_logits(self, policy):
        """Test get_logits method."""
        obs = {
            "sub_index": torch.randint(1, 50, (4, 1, 8, 3)),
            "derived_sub_indices": torch.randint(1, 50, (4, 10, 8, 3)),
            "action_mask": torch.ones(4, 10, dtype=torch.bool),
        }

        logits = policy.get_logits(obs)

        assert logits.shape == (4, 10)

    def test_get_logits_with_mask(self, policy):
        """Test get_logits respects action mask."""
        obs = {
            "sub_index": torch.randint(1, 50, (4, 1, 8, 3)),
            "derived_sub_indices": torch.randint(1, 50, (4, 10, 8, 3)),
            "action_mask": torch.zeros(4, 10, dtype=torch.bool),
        }
        # Enable only action 5
        obs["action_mask"][:, 5] = True

        logits = policy.get_logits(obs)

        # Masked actions should have -inf
        for i in range(4):
            for j in range(10):
                if j == 5:
                    assert logits[i, j] != float("-inf")
                else:
                    assert logits[i, j] == float("-inf")

    def test_predict_values(self, policy):
        """Test predict_values method."""
        obs = {
            "sub_index": torch.randint(1, 50, (4, 1, 8, 3)),
            "derived_sub_indices": torch.randint(1, 50, (4, 10, 8, 3)),
            "action_mask": torch.ones(4, 10, dtype=torch.bool),
        }

        values = policy.predict_values(obs)

        assert values.shape == (4,)

    def test_forward_action_selection(self, policy):
        """Test forward pass for action selection."""
        obs = {
            "sub_index": torch.randint(1, 50, (4, 1, 8, 3)),
            "derived_sub_indices": torch.randint(1, 50, (4, 10, 8, 3)),
            "action_mask": torch.ones(4, 10, dtype=torch.bool),
        }

        actions, values, log_probs = policy(obs, deterministic=False)

        assert actions.shape == (4,)
        assert values.shape == (4,)
        assert log_probs.shape == (4,)

        # Actions should be in valid range
        assert (actions >= 0).all()
        assert (actions < 10).all()

    def test_forward_deterministic(self, policy):
        """Test deterministic forward pass."""
        obs = {
            "sub_index": torch.randint(1, 50, (4, 1, 8, 3)),
            "derived_sub_indices": torch.randint(1, 50, (4, 10, 8, 3)),
            "action_mask": torch.ones(4, 10, dtype=torch.bool),
        }

        # Run twice with deterministic=True
        actions1, _, _ = policy(obs, deterministic=True)
        actions2, _, _ = policy(obs, deterministic=True)

        # Should be identical
        assert (actions1 == actions2).all()

    def test_evaluate_actions(self, policy):
        """Test evaluate_actions for training."""
        sub_index = torch.randint(1, 50, (4, 1, 8, 3))
        derived = torch.randint(1, 50, (4, 10, 8, 3))
        mask = torch.ones(4, 10, dtype=torch.bool)
        actions = torch.randint(0, 10, (4,))

        values, log_probs, entropy = policy.evaluate_actions(
            sub_index, derived, mask, actions
        )

        assert values.shape == (4,)
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)

        # Entropy should be non-negative
        assert (entropy >= 0).all()

    def test_gradient_flow(self, policy):
        """Test gradients flow through policy."""
        obs = {
            "sub_index": torch.randint(1, 50, (4, 1, 8, 3)),
            "derived_sub_indices": torch.randint(1, 50, (4, 10, 8, 3)),
            "action_mask": torch.ones(4, 10, dtype=torch.bool),
        }
        actions = torch.randint(0, 10, (4,))

        values, log_probs, entropy = policy.evaluate_actions(
            obs["sub_index"],
            obs["derived_sub_indices"],
            obs["action_mask"],
            actions,
        )

        loss = values.sum() + log_probs.sum() + entropy.sum()
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in policy.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"


class TestMCTSPolicyGPU:
    """GPU-specific tests for MCTSPolicy."""

    @pytest.fixture
    def gpu_policy(self):
        """Create GPU policy if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        embedder = MCTSEmbedder(
            n_constants=100,
            n_predicates=50,
            n_vars=100,
            embedding_dim=64,
        )
        return MCTSPolicy(
            embedder=embedder,
            embed_dim=64,
            hidden_dim=128,
            num_layers=2,
            device=torch.device("cuda"),
        )

    def test_gpu_forward(self, gpu_policy):
        """Test forward pass on GPU."""
        device = torch.device("cuda")
        obs = {
            "sub_index": torch.randint(1, 50, (4, 1, 8, 3), device=device),
            "derived_sub_indices": torch.randint(1, 50, (4, 10, 8, 3), device=device),
            "action_mask": torch.ones(4, 10, dtype=torch.bool, device=device),
        }

        logits = gpu_policy.get_logits(obs)

        assert logits.device.type == "cuda"
        assert logits.shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

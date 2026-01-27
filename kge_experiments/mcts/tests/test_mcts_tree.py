"""
Tests for MCTS tree data structures.

Tests Node, MinMaxStats, and MCTS UCB formula correctness.
"""

import sys
from pathlib import Path

# Add paths for imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "kge_experiments") not in sys.path:
    sys.path.insert(0, str(_ROOT / "kge_experiments"))

import math
import pytest
import torch

from mcts.config import MCTSConfig
from mcts.tree import Node, MinMaxStats, MCTS


class TestNode:
    """Tests for MCTS Node class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(prior=0.5)
        assert node.prior == 0.5
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert not node.is_expanded()

    def test_node_value(self):
        """Test node value (mean Q-value) computation."""
        node = Node()
        assert node.value() == 0.0  # No visits

        node.visit_count = 10
        node.value_sum = 5.0
        assert node.value() == 0.5

    def test_node_expand(self):
        """Test node expansion with children."""
        node = Node()
        actions = [0, 1, 2]
        priors = torch.tensor([0.2, 0.5, 0.3])

        node.expand(actions, priors, reward=0.0)

        assert node.is_expanded()
        assert len(node.children) == 3
        assert 0 in node.children
        assert 1 in node.children
        assert 2 in node.children

        assert node.children[0].prior == pytest.approx(0.2)
        assert node.children[1].prior == pytest.approx(0.5)
        assert node.children[2].prior == pytest.approx(0.3)

    def test_node_exploration_noise(self):
        """Test Dirichlet noise addition at root."""
        node = Node()
        node.expand([0, 1, 2], torch.tensor([0.33, 0.33, 0.34]))

        original_priors = [node.children[a].prior for a in [0, 1, 2]]

        node.add_exploration_noise(
            dirichlet_alpha=0.3,
            exploration_fraction=0.25,
        )

        new_priors = [node.children[a].prior for a in [0, 1, 2]]

        # Priors should change
        assert new_priors != original_priors

        # Sum should still be ~1 (approximately, due to mixing)
        assert sum(new_priors) == pytest.approx(1.0, abs=0.1)


class TestMinMaxStats:
    """Tests for MinMaxStats Q-value normalization."""

    def test_initial_state(self):
        """Test initial min/max are infinity."""
        stats = MinMaxStats()
        assert stats.min_value == float("inf")
        assert stats.max_value == float("-inf")

    def test_update(self):
        """Test min/max tracking."""
        stats = MinMaxStats()

        stats.update(5.0)
        assert stats.min_value == 5.0
        assert stats.max_value == 5.0

        stats.update(3.0)
        assert stats.min_value == 3.0
        assert stats.max_value == 5.0

        stats.update(7.0)
        assert stats.min_value == 3.0
        assert stats.max_value == 7.0

    def test_normalize(self):
        """Test value normalization to [0, 1]."""
        stats = MinMaxStats()

        # Before any updates, normalize should return 0
        assert stats.normalize(5.0) == 0.0

        stats.update(0.0)
        stats.update(10.0)

        assert stats.normalize(0.0) == 0.0
        assert stats.normalize(10.0) == 1.0
        assert stats.normalize(5.0) == 0.5
        assert stats.normalize(2.5) == 0.25

    def test_normalize_same_min_max(self):
        """Test normalization when min == max."""
        stats = MinMaxStats()
        stats.update(5.0)

        # When min == max, should return 0
        assert stats.normalize(5.0) == 0.0


class TestMCTSConfig:
    """Tests for MCTS configuration."""

    def test_default_config(self):
        """Test default config values."""
        config = MCTSConfig()

        assert config.num_simulations == 50
        assert config.pb_c_base == 19652.0
        assert config.pb_c_init == 1.25
        assert config.discount == 0.99

    def test_temperature_schedule_linear(self):
        """Test linear temperature decay."""
        config = MCTSConfig(
            temperature_init=1.0,
            temperature_final=0.0,
            temperature_decay_steps=100,
            temperature_schedule="linear",
        )

        assert config.get_temperature(0) == 1.0
        assert config.get_temperature(50) == 0.5
        assert config.get_temperature(100) == 0.0
        assert config.get_temperature(200) == 0.0  # Clamped

    def test_temperature_schedule_constant(self):
        """Test constant temperature."""
        config = MCTSConfig(
            temperature_init=1.0,
            temperature_schedule="constant",
        )

        assert config.get_temperature(0) == 1.0
        assert config.get_temperature(1000) == 1.0

    def test_puct_constant(self):
        """Test PUCT exploration constant formula."""
        config = MCTSConfig(pb_c_base=19652.0, pb_c_init=1.25)

        # c(s) = log((1 + N(s) + pb_c_base) / pb_c_base) + pb_c_init
        # For N(s) = 0: log((1 + 0 + 19652) / 19652) + 1.25 ≈ 1.25
        c0 = config.get_puct_constant(0)
        expected_c0 = math.log((1 + 0 + 19652) / 19652) + 1.25
        assert c0 == pytest.approx(expected_c0)

        # For higher visit counts, c should increase slowly
        c100 = config.get_puct_constant(100)
        assert c100 > c0


class TestMCTSUCB:
    """Tests for MCTS UCB/PUCT formula."""

    def test_ucb_favors_high_prior_unvisited(self):
        """Test that UCB favors high-prior unvisited nodes."""
        config = MCTSConfig(num_simulations=1)
        mcts = MCTS(config)

        # Create parent with two children: one high prior, one low
        parent = Node(visit_count=10)
        parent.children = {
            0: Node(prior=0.8, visit_count=0),
            1: Node(prior=0.2, visit_count=0),
        }

        min_max_stats = MinMaxStats()

        action, child = mcts._select_child(parent, min_max_stats)

        # Should select high-prior action
        assert action == 0

    def test_ucb_explores_less_visited(self):
        """Test that UCB explores less-visited nodes."""
        config = MCTSConfig(num_simulations=1)
        mcts = MCTS(config)

        # Create parent with equal priors but different visit counts
        parent = Node(visit_count=100)
        parent.children = {
            0: Node(prior=0.5, visit_count=90, value_sum=45),
            1: Node(prior=0.5, visit_count=10, value_sum=5),
        }

        min_max_stats = MinMaxStats()
        min_max_stats.update(0.5)

        action, child = mcts._select_child(parent, min_max_stats)

        # Should explore less-visited action (1) due to higher exploration bonus
        assert action == 1

    def test_backpropagation(self):
        """Test value backpropagation through search path."""
        config = MCTSConfig(discount=0.99)
        mcts = MCTS(config)

        # Create a simple search path
        nodes = [Node(), Node(), Node()]

        min_max_stats = MinMaxStats()

        # Backpropagate value of 1.0
        mcts._backpropagate(nodes, 1.0, min_max_stats)

        # All nodes should have visit_count = 1
        for node in nodes:
            assert node.visit_count == 1

        # Values should be discounted
        # Last node: 1.0
        # Middle node: 0 + 0.99 * 1.0 = 0.99
        # First node: 0 + 0.99 * 0.99 = 0.9801
        assert nodes[2].value() == pytest.approx(1.0)
        assert nodes[1].value() == pytest.approx(0.99)
        assert nodes[0].value() == pytest.approx(0.9801)


class TestMCTSActionSelection:
    """Tests for MCTS action selection from visit counts."""

    def test_greedy_selection(self):
        """Test greedy (temperature=0) action selection."""
        config = MCTSConfig()
        mcts = MCTS(config)

        root = Node()
        root.children = {
            0: Node(visit_count=10),
            1: Node(visit_count=50),  # Most visited
            2: Node(visit_count=30),
        }

        action = mcts._select_action(root, temperature=0.0)
        assert action == 1

    def test_proportional_selection(self):
        """Test proportional (temperature=1) action selection."""
        config = MCTSConfig()
        mcts = MCTS(config)

        root = Node()
        root.children = {
            0: Node(visit_count=10),
            1: Node(visit_count=50),
            2: Node(visit_count=40),
        }

        # Run multiple selections to check distribution
        selections = [mcts._select_action(root, temperature=1.0) for _ in range(1000)]

        counts = {0: 0, 1: 0, 2: 0}
        for a in selections:
            counts[a] += 1

        # Action 1 should be selected most often (~50%)
        assert counts[1] > counts[0]
        assert counts[1] > counts[2] * 0.8  # Allow some variance

    def test_get_action_probs(self):
        """Test action probability computation."""
        config = MCTSConfig()
        mcts = MCTS(config)

        root = Node()
        root.children = {
            0: Node(visit_count=25),
            1: Node(visit_count=50),
            2: Node(visit_count=25),
        }

        probs = mcts.get_action_probs(root, temperature=1.0)

        assert probs[0] == pytest.approx(0.25)
        assert probs[1] == pytest.approx(0.50)
        assert probs[2] == pytest.approx(0.25)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

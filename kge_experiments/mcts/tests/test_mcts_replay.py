"""
Tests for MCTS replay buffer.

Tests trajectory storage, n-step return computation, and batch sampling.
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

from mcts.replay_buffer import (
    MCTSTransition,
    MCTSTrajectory,
    MCTSReplayBuffer,
)


class TestMCTSTransition:
    """Tests for MCTSTransition dataclass."""

    def test_transition_creation(self):
        """Test basic transition creation."""
        obs = {"sub_index": torch.zeros(1, 1, 4, 3)}
        trans = MCTSTransition(
            obs=obs,
            action=5,
            reward=1.0,
            done=False,
            visit_counts={5: 10, 3: 5},
            root_value=0.8,
        )

        assert trans.action == 5
        assert trans.reward == 1.0
        assert trans.done is False
        assert trans.visit_counts[5] == 10
        assert trans.root_value == 0.8
        assert trans.policy_target is None
        assert trans.value_target is None


class TestMCTSTrajectory:
    """Tests for MCTSTrajectory storage."""

    def test_trajectory_add(self):
        """Test adding transitions to trajectory."""
        trajectory = MCTSTrajectory()

        for i in range(5):
            trans = MCTSTransition(
                obs={"sub_index": torch.zeros(1, 1, 4, 3)},
                action=i,
                reward=1.0,
                done=(i == 4),
                visit_counts={i: 10},
                root_value=0.5,
            )
            trajectory.add(trans)

        assert trajectory.length == 5
        assert trajectory.total_reward == 5.0

    def test_compute_targets_policy(self):
        """Test policy target computation (normalized visit counts)."""
        trajectory = MCTSTrajectory()

        trans = MCTSTransition(
            obs={"sub_index": torch.zeros(1, 1, 4, 3)},
            action=0,
            reward=1.0,
            done=True,
            visit_counts={0: 25, 1: 50, 2: 25},
            root_value=0.5,
        )
        trajectory.add(trans)
        trajectory.compute_targets(discount=0.99, n_step=5)

        # Policy target should be normalized visit counts
        pt = trajectory.transitions[0].policy_target
        assert pt is not None
        assert pt[0] == pytest.approx(0.25)
        assert pt[1] == pytest.approx(0.50)
        assert pt[2] == pytest.approx(0.25)

    def test_compute_targets_value_nstep(self):
        """Test n-step value target computation."""
        trajectory = MCTSTrajectory()

        # Create 5-step trajectory with rewards [1, 1, 1, 1, 1]
        for i in range(5):
            trans = MCTSTransition(
                obs={"sub_index": torch.zeros(1, 1, 4, 3)},
                action=i,
                reward=1.0,
                done=(i == 4),
                visit_counts={i: 10},
                root_value=0.0,  # No bootstrap for done trajectory
            )
            trajectory.add(trans)

        trajectory.compute_targets(discount=0.99, n_step=3)

        # For step 0 with 3-step returns: r0 + 0.99*r1 + 0.99^2*r2 + 0.99^3*V3
        # = 1.0 + 0.99*1.0 + 0.99^2*1.0 + 0.99^3*0.0 = 2.9701
        vt0 = trajectory.transitions[0].value_target
        expected = 1.0 + 0.99 + 0.99**2
        assert vt0 == pytest.approx(expected, rel=0.01)

        # For step 4 (terminal): just reward 1.0
        vt4 = trajectory.transitions[4].value_target
        assert vt4 == pytest.approx(1.0)


class TestMCTSReplayBuffer:
    """Tests for MCTSReplayBuffer."""

    def _create_dummy_trajectory(self, length: int = 5, reward: float = 1.0) -> MCTSTrajectory:
        """Create a dummy trajectory for testing."""
        trajectory = MCTSTrajectory()
        for i in range(length):
            trans = MCTSTransition(
                obs={
                    "sub_index": torch.randn(1, 1, 4, 3),
                    "derived_sub_indices": torch.randn(1, 10, 4, 3),
                    "action_mask": torch.ones(1, 10, dtype=torch.bool),
                },
                action=i % 10,
                reward=reward,
                done=(i == length - 1),
                visit_counts={j: max(1, 10 - j) for j in range(10)},
                root_value=0.5,
            )
            trajectory.add(trans)
        return trajectory

    def test_buffer_add_trajectory(self):
        """Test adding trajectories to buffer."""
        buffer = MCTSReplayBuffer(max_size=100)

        assert len(buffer) == 0
        assert buffer.total_transitions == 0

        traj = self._create_dummy_trajectory(length=10)
        buffer.add_trajectory(traj)

        assert len(buffer) == 1
        assert buffer.total_transitions == 10

    def test_buffer_max_size(self):
        """Test buffer respects max_size."""
        buffer = MCTSReplayBuffer(max_size=5)

        for _ in range(10):
            buffer.add_trajectory(self._create_dummy_trajectory(length=3))

        assert len(buffer) == 5  # Only keeps last 5

    def test_buffer_sample_batch(self):
        """Test batch sampling from buffer."""
        buffer = MCTSReplayBuffer(max_size=100, device="cpu")

        # Add enough data
        for _ in range(20):
            buffer.add_trajectory(self._create_dummy_trajectory(length=5))

        batch = buffer.sample_batch(batch_size=8, sequence_length=1)

        assert "obs" in batch
        assert "actions" in batch
        assert "policy_targets" in batch
        assert "value_targets" in batch
        assert "rewards" in batch

        assert batch["actions"].shape[0] == 8
        assert batch["value_targets"].shape[0] == 8

    def test_buffer_sample_insufficient_data(self):
        """Test sampling with insufficient data raises error."""
        buffer = MCTSReplayBuffer(max_size=100)

        buffer.add_trajectory(self._create_dummy_trajectory(length=3))

        with pytest.raises(ValueError):
            buffer.sample_batch(batch_size=100)  # More than available

    def test_buffer_statistics(self):
        """Test buffer statistics computation."""
        buffer = MCTSReplayBuffer(max_size=100)

        buffer.add_trajectory(self._create_dummy_trajectory(length=5, reward=1.0))
        buffer.add_trajectory(self._create_dummy_trajectory(length=10, reward=2.0))

        stats = buffer.get_statistics()

        assert stats["num_trajectories"] == 2
        assert stats["total_transitions"] == 15
        assert stats["avg_trajectory_length"] == 7.5
        assert stats["avg_trajectory_reward"] == 12.5  # (5*1 + 10*2) / 2 = 12.5 avg reward
        assert stats["min_trajectory_length"] == 5
        assert stats["max_trajectory_length"] == 10

    def test_buffer_clear(self):
        """Test buffer clearing."""
        buffer = MCTSReplayBuffer(max_size=100)

        for _ in range(5):
            buffer.add_trajectory(self._create_dummy_trajectory())

        assert len(buffer) == 5

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.total_transitions == 0

    def test_buffer_add_from_lists(self):
        """Test convenience add method from lists."""
        buffer = MCTSReplayBuffer(max_size=100)

        obs_list = [{"sub_index": torch.zeros(1, 1, 4, 3)} for _ in range(5)]
        actions = [0, 1, 2, 3, 4]
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        dones = [False, False, False, False, True]
        visit_counts = [{0: 10, 1: 5} for _ in range(5)]
        root_values = [0.5, 0.5, 0.5, 0.5, 0.5]

        buffer.add(obs_list, actions, rewards, dones, visit_counts, root_values)

        assert len(buffer) == 1
        assert buffer.total_transitions == 5


class TestNStepReturns:
    """Tests for n-step return computation edge cases."""

    def test_nstep_longer_than_trajectory(self):
        """Test n-step when n > trajectory length."""
        trajectory = MCTSTrajectory()

        for i in range(3):
            trans = MCTSTransition(
                obs={"sub_index": torch.zeros(1, 1, 4, 3)},
                action=i,
                reward=1.0,
                done=(i == 2),
                visit_counts={i: 10},
                root_value=0.0,
            )
            trajectory.add(trans)

        # n_step=5 but trajectory length=3
        trajectory.compute_targets(discount=0.99, n_step=5)

        # Should still compute correctly
        vt0 = trajectory.transitions[0].value_target
        expected = 1.0 + 0.99 + 0.99**2  # All 3 rewards, no bootstrap (terminal)
        assert vt0 == pytest.approx(expected, rel=0.01)

    def test_nstep_with_bootstrap(self):
        """Test n-step with value bootstrap."""
        trajectory = MCTSTrajectory()

        for i in range(5):
            trans = MCTSTransition(
                obs={"sub_index": torch.zeros(1, 1, 4, 3)},
                action=i,
                reward=1.0,
                done=False,  # Not terminal - need bootstrap
                visit_counts={i: 10},
                root_value=0.5,  # Bootstrap value
            )
            trajectory.add(trans)

        trajectory.compute_targets(discount=0.99, n_step=3)

        # Step 0: r0 + 0.99*r1 + 0.99^2*r2 + 0.99^3*V3
        # = 1.0 + 0.99 + 0.99^2 + 0.99^3 * 0.5
        vt0 = trajectory.transitions[0].value_target
        expected = 1.0 + 0.99 + 0.99**2 + 0.99**3 * 0.5
        assert vt0 == pytest.approx(expected, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

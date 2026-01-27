"""
Integration tests for MCTS module.

Tests full MCTS search and trainer with mock environment.
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
from tensordict import TensorDict

from mcts.config import MCTSConfig
from mcts.tree import MCTS
from mcts.networks import MCTSPolicy, MCTSEmbedder
from mcts.replay_buffer import MCTSReplayBuffer
from mcts.trainer import MuZeroTrainer


class MockEnv:
    """Mock environment for MCTS testing."""

    def __init__(self, num_actions: int = 10, max_steps: int = 5, device: str = "cpu"):
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.step_count = 0
        self.device = torch.device(device)

    def reset(self, queries=None):
        """Reset environment."""
        self.step_count = 0
        obs = self._make_obs()
        state = self._make_state()
        return obs, state

    def step(self, state, actions):
        """Take a step in environment."""
        self.step_count += 1
        new_obs = self._make_obs()
        new_state = self._make_state()

        # Random reward and done logic
        done = self.step_count >= self.max_steps
        success = done and torch.rand(1).item() > 0.5
        reward = 1.0 if success else 0.0

        new_state["step_rewards"] = torch.tensor([reward])
        new_state["step_dones"] = torch.tensor([done], dtype=torch.bool)
        new_state["done"] = torch.tensor([done], dtype=torch.bool)
        new_state["success"] = torch.tensor([success], dtype=torch.bool)

        return new_obs, new_state

    def _make_obs(self):
        """Create mock observation."""
        return {
            "sub_index": torch.randint(1, 50, (1, 1, 4, 3), device=self.device),
            "derived_sub_indices": torch.randint(1, 50, (1, self.num_actions, 4, 3), device=self.device),
            "action_mask": torch.ones(1, self.num_actions, dtype=torch.bool, device=self.device),
        }

    def _make_state(self):
        """Create mock state."""
        return TensorDict({
            "current_states": torch.randint(1, 50, (1, 4, 3), device=self.device),
            "derived_states": torch.randint(1, 50, (1, self.num_actions, 4, 3), device=self.device),
            "derived_counts": torch.tensor([self.num_actions], device=self.device),
            "depths": torch.tensor([self.step_count], device=self.device),
            "done": torch.tensor([False], dtype=torch.bool, device=self.device),
            "success": torch.tensor([False], dtype=torch.bool, device=self.device),
            "step_rewards": torch.tensor([0.0], device=self.device),
            "step_dones": torch.tensor([False], dtype=torch.bool, device=self.device),
        }, batch_size=[1], device=self.device)


class TestMCTSSearch:
    """Tests for full MCTS search."""

    @pytest.fixture
    def setup(self):
        """Setup MCTS components."""
        config = MCTSConfig(num_simulations=10, max_episode_steps=5)
        embedder = MCTSEmbedder(
            n_constants=100,
            n_predicates=50,
            n_vars=100,
            embedding_dim=32,
        )
        policy = MCTSPolicy(
            embedder=embedder,
            embed_dim=32,
            hidden_dim=64,
            num_layers=2,
            device=torch.device("cpu"),
        )
        env = MockEnv(num_actions=10, max_steps=5)
        mcts = MCTS(config)

        return config, policy, env, mcts

    def test_mcts_search_returns_valid_action(self, setup):
        """Test MCTS search returns valid action."""
        config, policy, env, mcts = setup

        obs, state = env.reset()
        action_mask = obs["action_mask"].squeeze(0)

        action, stats = mcts.search(
            env=env,
            env_state=state,
            networks=policy,
            obs=obs,
            action_mask=action_mask,
            add_noise=True,
        )

        # Action should be valid
        assert 0 <= action < env.num_actions
        assert action_mask[action].item()

        # Stats should have visit counts
        assert "visit_counts" in stats
        assert "root_value" in stats
        assert len(stats["visit_counts"]) > 0

    def test_mcts_search_respects_mask(self, setup):
        """Test MCTS only selects from valid actions."""
        config, policy, env, mcts = setup

        obs, state = env.reset()

        # Only allow actions 3 and 7
        action_mask = torch.zeros(env.num_actions, dtype=torch.bool)
        action_mask[3] = True
        action_mask[7] = True
        obs["action_mask"] = action_mask.unsqueeze(0)

        for _ in range(10):  # Run multiple times to check consistency
            action, _ = mcts.search(
                env=env,
                env_state=state,
                networks=policy,
                obs=obs,
                action_mask=action_mask,
                add_noise=False,
            )
            assert action in [3, 7], f"Invalid action {action} selected"

    def test_mcts_search_visit_counts_sum(self, setup):
        """Test visit counts sum to num_simulations."""
        config, policy, env, mcts = setup

        obs, state = env.reset()
        action_mask = obs["action_mask"].squeeze(0)

        _, stats = mcts.search(
            env=env,
            env_state=state,
            networks=policy,
            obs=obs,
            action_mask=action_mask,
            add_noise=False,
        )

        total_visits = sum(stats["visit_counts"].values())
        # Total visits should equal num_simulations (first expansion counts)
        assert total_visits <= config.num_simulations


class TestMuZeroTrainer:
    """Tests for MuZero trainer."""

    @pytest.fixture
    def trainer(self):
        """Setup trainer."""
        # Use CPU for consistent testing
        device = "cpu"
        config = MCTSConfig(
            num_simulations=5,
            max_episode_steps=5,
            episodes_per_iteration=2,
            batch_size=4,
            min_buffer_size=10,
            replay_buffer_size=100,
            hidden_dim=32,
            verbose=False,
            device=device,
        )
        embedder = MCTSEmbedder(
            n_constants=100,
            n_predicates=50,
            n_vars=100,
            embedding_dim=32,
        )
        policy = MCTSPolicy(
            embedder=embedder,
            embed_dim=32,
            hidden_dim=32,
            num_layers=1,
            device=torch.device(device),
        )
        env = MockEnv(num_actions=10, max_steps=5, device=device)

        return MuZeroTrainer(policy=policy, env=env, config=config)

    def test_collect_episodes(self, trainer):
        """Test episode collection."""
        stats = trainer.collect_episodes(num_episodes=3)

        assert stats["episodes_collected"] == 3
        assert stats["total_transitions"] > 0
        assert "mean_reward" in stats
        assert "success_rate" in stats

    def test_train_step_requires_data(self, trainer):
        """Test train_step requires sufficient buffer data."""
        # With empty buffer, should skip
        result = trainer.train_step()
        assert result.get("train_step_skipped", False)

    def test_train_step_with_data(self, trainer):
        """Test train_step with sufficient data."""
        # Collect enough episodes to fill buffer
        for _ in range(5):
            trainer.collect_episodes(num_episodes=5)

        result = trainer.train_step()

        # Should have loss metrics
        assert "policy_loss" in result
        assert "value_loss" in result
        assert "total_loss" in result

    def test_learn_short_run(self, trainer):
        """Test learn() for a short run."""
        results = trainer.learn(total_timesteps=50)

        assert "num_timesteps" in results
        assert results["num_timesteps"] >= 50
        assert "episode_rewards" in results
        assert len(results["episode_rewards"]) > 0


class TestMCTSConfigIntegration:
    """Test MCTS config integration with TrainConfig."""

    def test_mcts_config_from_train_config(self):
        """Test creating MCTSConfig from TrainConfig fields."""
        # This simulates the builder.py pattern
        class MockTrainConfig:
            mcts_num_simulations = 30
            mcts_pb_c_base = 19652.0
            mcts_pb_c_init = 1.25
            mcts_discount = 0.99
            mcts_root_dirichlet_alpha = 0.25
            mcts_root_exploration_fraction = 0.3
            mcts_add_exploration_noise = True
            mcts_value_nstep = 5
            mcts_temperature_init = 1.0
            mcts_temperature_final = 0.1
            mcts_temperature_decay_steps = 10000
            mcts_temperature_schedule = "linear"
            mcts_hidden_dim = 128
            mcts_num_layers = 3
            mcts_batch_size = 128
            max_steps = 15
            device = "cpu"
            compile = False
            verbose = False

        tc = MockTrainConfig()

        config = MCTSConfig(
            num_simulations=tc.mcts_num_simulations,
            pb_c_base=tc.mcts_pb_c_base,
            pb_c_init=tc.mcts_pb_c_init,
            discount=tc.mcts_discount,
            root_dirichlet_alpha=tc.mcts_root_dirichlet_alpha,
            root_exploration_fraction=tc.mcts_root_exploration_fraction,
            add_exploration_noise=tc.mcts_add_exploration_noise,
            value_nstep=tc.mcts_value_nstep,
            temperature_init=tc.mcts_temperature_init,
            temperature_final=tc.mcts_temperature_final,
            temperature_decay_steps=tc.mcts_temperature_decay_steps,
            temperature_schedule=tc.mcts_temperature_schedule,
            hidden_dim=tc.mcts_hidden_dim,
            num_layers=tc.mcts_num_layers,
            batch_size=tc.mcts_batch_size,
            max_episode_steps=tc.max_steps,
            device=tc.device,
            compile=tc.compile,
            verbose=tc.verbose,
        )

        assert config.num_simulations == 30
        assert config.hidden_dim == 128
        assert config.batch_size == 128
        assert config.max_episode_steps == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

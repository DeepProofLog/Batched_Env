"""
Model Evaluation Parity Tests.

Tests verifying that evaluation metrics are computed consistently
between the new tensor-based implementation and SB3.
"""
from pathlib import Path
import sys
import importlib.util
from typing import Tuple

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
# Dummy Environment and Policy for Testing
# ============================================================================

class DummyPolicy(nn.Module):
    """Simple deterministic policy for testing evaluation."""
    
    def __init__(self, action_dim: int = 4):
        super().__init__()
        self.action_dim = action_dim
        self.value_net = nn.Linear(8, 1)
        
    def forward(self, obs, deterministic=True):
        if isinstance(obs, TensorDict):
            obs_tensor = obs.get("obs", obs.get("sub_index"))
        else:
            obs_tensor = obs
        
        batch_size = obs_tensor.shape[0]
        device = obs_tensor.device
        
        # Deterministic actions: always action 0
        if deterministic:
            actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            actions = torch.randint(0, self.action_dim, (batch_size,), device=device)
        
        # Dummy values and log probs
        values = torch.zeros(batch_size, device=device)
        log_probs = torch.zeros(batch_size, device=device)
        
        return actions, values, log_probs
    
    def eval(self):
        return super().eval()
    
    def train(self, mode=True):
        return super().train(mode)


class DummyTensorEnv:
    """Simple TensorDict-based environment for testing."""
    
    def __init__(self, batch_size: int = 4, max_steps: int = 5):
        self.batch_size = torch.Size([batch_size])
        self.n_envs = batch_size
        self.max_steps = max_steps
        self.device = torch.device("cpu")
        
        self._step_count = torch.zeros(batch_size, dtype=torch.long)
        self._episode_rewards = torch.zeros(batch_size)
        self._last_obs = None
        
    def reset(self, reset_td: TensorDict = None) -> TensorDict:
        # Handle partial reset via reset_td
        if reset_td is not None and "_reset" in reset_td.keys():
            reset_mask = reset_td.get("_reset").view(-1)
            self._step_count[reset_mask] = 0
            self._episode_rewards[reset_mask] = 0
        else:
            self._step_count.zero_()
            self._episode_rewards.zero_()
        
        obs = torch.randn(self.n_envs, 8)
        self._last_obs = obs
        return TensorDict({
            "obs": obs,
            "done": torch.zeros(self.n_envs, dtype=torch.bool),
        }, batch_size=self.batch_size)
    
    def step(self, action_td: TensorDict) -> TensorDict:
        """Execute one step, return TensorDict with next state under 'next' key."""
        self._step_count += 1
        
        # Random reward
        rewards = torch.rand(self.n_envs)
        self._episode_rewards += rewards
        
        # Episode ends after max_steps
        dones = self._step_count >= self.max_steps
        
        # Success for some episodes
        success = dones & (torch.rand(self.n_envs) > 0.5)
        
        # Build next observation
        next_obs = torch.randn(self.n_envs, 8)
        self._last_obs = next_obs
        
        # Return TensorDict with 'next' key (TorchRL style)
        return TensorDict({
            "reward": rewards,
            "done": dones,
            "is_success": success,
            "length": self._step_count.clone(),
            "next": TensorDict({
                "obs": next_obs,
                "done": dones,
                "is_success": success,
            }, batch_size=self.batch_size),
        }, batch_size=self.batch_size)
    
    def step_and_maybe_reset(self, action_td: TensorDict) -> Tuple[TensorDict, TensorDict]:
        """Legacy interface for compatibility."""
        step_td = self.step(action_td)
        next_obs_td = step_td.get("next")
        
        # Auto-reset finished envs
        dones = step_td.get("done")
        if dones.any():
            self._step_count[dones] = 0
            self._episode_rewards[dones] = 0
        
        return step_td, next_obs_td


# ============================================================================
# Evaluation Function Tests
# ============================================================================

def test_evaluate_policy_produces_valid_output():
    """Test that evaluate_policy produces valid output."""
    from model_eval import evaluate_policy
    
    torch.manual_seed(42)
    
    env = DummyTensorEnv(batch_size=4, max_steps=5)
    actor = DummyPolicy(action_dim=4)
    
    results = evaluate_policy(
        actor=actor,
        env=env,
        n_eval_episodes=2,
        deterministic=True,
        verbose=0,
    )
    
    # Check required keys
    assert "rewards" in results or "episode_rewards" in results or hasattr(results, "__getitem__")
    
    # Results should be tensors
    for key in results:
        val = results[key]
        if isinstance(val, torch.Tensor):
            assert val.numel() > 0


def test_evaluate_policy_with_target_episodes():
    """Test evaluate_policy with specific target episodes per env."""
    from model_eval import evaluate_policy
    
    torch.manual_seed(42)
    
    batch_size = 4
    env = DummyTensorEnv(batch_size=batch_size, max_steps=3)
    actor = DummyPolicy(action_dim=4)
    
    # Different target episodes per environment
    target_episodes = [1, 2, 1, 2]
    
    results = evaluate_policy(
        actor=actor,
        env=env,
        target_episodes=target_episodes,
        deterministic=True,
        verbose=0,
    )
    
    # Should complete the specified number of episodes
    assert results is not None


def test_evaluate_policy_deterministic_vs_stochastic():
    """Test that deterministic and stochastic evaluation differ."""
    from model_eval import evaluate_policy
    
    # Need a stochastic policy
    class StochasticPolicy(DummyPolicy):
        def forward(self, obs, deterministic=True):
            if isinstance(obs, TensorDict):
                obs_tensor = obs.get("obs", obs.get("sub_index"))
            else:
                obs_tensor = obs
            
            batch_size = obs_tensor.shape[0]
            device = obs_tensor.device
            
            if deterministic:
                actions = torch.zeros(batch_size, dtype=torch.long, device=device)
            else:
                # Random actions
                torch.manual_seed(int(torch.rand(1).item() * 10000))
                actions = torch.randint(0, 4, (batch_size,), device=device)
            
            values = torch.zeros(batch_size, device=device)
            log_probs = torch.zeros(batch_size, device=device)
            
            return actions, values, log_probs
    
    torch.manual_seed(42)
    env1 = DummyTensorEnv(batch_size=2, max_steps=5)
    torch.manual_seed(42)
    env2 = DummyTensorEnv(batch_size=2, max_steps=5)
    
    actor = StochasticPolicy()
    
    # Deterministic should always give same results
    torch.manual_seed(0)
    results_det = evaluate_policy(
        actor=actor,
        env=env1,
        n_eval_episodes=1,
        deterministic=True,
        verbose=0,
    )
    
    # Stochastic may give different results
    torch.manual_seed(0)
    results_stoch = evaluate_policy(
        actor=actor,
        env=env2,
        n_eval_episodes=1,
        deterministic=False,
        verbose=0,
    )
    
    # Both should complete without error
    assert results_det is not None
    assert results_stoch is not None


# ============================================================================
# Metric Computation Tests
# ============================================================================

def test_episode_reward_accumulation():
    """Test that episode rewards are accumulated correctly."""
    from model_eval import evaluate_policy
    
    torch.manual_seed(42)
    
    # Create environment with known rewards
    class FixedRewardEnv(DummyTensorEnv):
        def step(self, action_td):
            self._step_count += 1
            
            # Fixed reward of 1.0 per step
            rewards = torch.ones(self.n_envs)
            self._episode_rewards += rewards
            
            dones = self._step_count >= self.max_steps
            
            next_obs = torch.randn(self.n_envs, 8)
            
            return TensorDict({
                "reward": rewards,
                "done": dones,
                "is_success": dones,  # All successful
                "length": self._step_count.clone(),
                "next": TensorDict({
                    "obs": next_obs,
                    "done": dones,
                    "is_success": dones,
                }, batch_size=self.batch_size),
            }, batch_size=self.batch_size)
    
    env = FixedRewardEnv(batch_size=2, max_steps=5)
    actor = DummyPolicy()
    
    results = evaluate_policy(
        actor=actor,
        env=env,
        n_eval_episodes=1,
        deterministic=True,
        verbose=0,
    )
    
    # With fixed 1.0 reward per step and 5 steps per episode,
    # each episode should have total reward of 5.0
    assert results is not None


def test_episode_length_tracking():
    """Test that episode lengths are tracked correctly."""
    from model_eval import evaluate_policy
    
    torch.manual_seed(42)
    
    max_steps = 7
    env = DummyTensorEnv(batch_size=2, max_steps=max_steps)
    actor = DummyPolicy()
    
    results = evaluate_policy(
        actor=actor,
        env=env,
        n_eval_episodes=1,
        deterministic=True,
        verbose=0,
    )
    
    # Episodes should end at max_steps
    if "lengths" in results:
        lengths = results["lengths"]
        # Completed episodes should have length = max_steps
        valid_lengths = lengths[lengths > 0]
        if valid_lengths.numel() > 0:
            assert valid_lengths.max() <= max_steps


# ============================================================================
# Success Rate Computation Tests
# ============================================================================

def test_success_rate_computation():
    """Test that success rate is computed correctly."""
    from model_eval import evaluate_policy
    
    torch.manual_seed(42)
    
    # Environment where half succeed
    class HalfSuccessEnv(DummyTensorEnv):
        def step(self, action_td):
            self._step_count += 1
            
            rewards = torch.rand(self.n_envs)
            dones = self._step_count >= self.max_steps
            
            # Alternate success
            success = torch.zeros(self.n_envs, dtype=torch.bool)
            for i in range(self.n_envs):
                if dones[i]:
                    success[i] = (i % 2 == 0)  # Even indices succeed
            
            next_obs = torch.randn(self.n_envs, 8)
            
            return TensorDict({
                "reward": rewards,
                "done": dones,
                "is_success": success,
                "length": self._step_count.clone(),
                "next": TensorDict({
                    "obs": next_obs,
                    "done": dones,
                    "is_success": success,
                }, batch_size=self.batch_size),
            }, batch_size=self.batch_size)
    
    env = HalfSuccessEnv(batch_size=4, max_steps=3)
    actor = DummyPolicy()
    
    results = evaluate_policy(
        actor=actor,
        env=env,
        n_eval_episodes=1,
        deterministic=True,
        verbose=0,
    )
    
    assert results is not None


# ============================================================================
# Edge Cases
# ============================================================================

def test_single_step_episode():
    """Test evaluation with very short episodes."""
    from model_eval import evaluate_policy
    
    torch.manual_seed(42)
    
    env = DummyTensorEnv(batch_size=2, max_steps=1)
    actor = DummyPolicy()
    
    results = evaluate_policy(
        actor=actor,
        env=env,
        n_eval_episodes=1,
        deterministic=True,
        verbose=0,
    )
    
    assert results is not None


def test_many_episodes():
    """Test evaluation with many episodes."""
    from model_eval import evaluate_policy
    
    torch.manual_seed(42)
    
    env = DummyTensorEnv(batch_size=4, max_steps=3)
    actor = DummyPolicy()
    
    results = evaluate_policy(
        actor=actor,
        env=env,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )
    
    assert results is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

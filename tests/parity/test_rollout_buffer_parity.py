"""
Comprehensive parity tests for RolloutBuffer.

Verifies that the new tensor-based RolloutBuffer produces exactly the same
GAE advantages and returns as the SB3 RolloutBuffer.
"""
from pathlib import Path
import sys
import importlib.util

import gymnasium as gym
import numpy as np
import torch
import pytest
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"parity_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NewRolloutBuffer = _load_module("rollout", ROOT / "rollout.py").RolloutBuffer

try:
    from stable_baselines3.common.buffers import RolloutBuffer as SB3RolloutBuffer
except Exception as exc:  # pragma: no cover - helps when sb3 is missing
    SB3RolloutBuffer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _build_obs(step: int, n_envs: int, device: torch.device):
    """Build observation tensors for both implementations."""
    obs = torch.full((n_envs, 1), float(step), device=device)
    td = TensorDict({"obs": obs}, batch_size=[n_envs], device=device)
    return td, obs.cpu().numpy()


# ============================================================================
# Basic Parity Tests
# ============================================================================

def test_rollout_buffer_advantages_match_sb3():
    """Test that GAE advantages match SB3 exactly."""
    if SB3RolloutBuffer is None:
        pytest.skip(f"stable_baselines3 unavailable: {_IMPORT_ERROR}")

    torch.manual_seed(0)
    np.random.seed(0)

    buffer_size = 4
    n_envs = 2
    device = torch.device("cpu")

    new_buf = NewRolloutBuffer(
        buffer_size=buffer_size,
        n_envs=n_envs,
        device=device,
        gamma=0.99,
        gae_lambda=0.95,
    )

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)
    sb3_buf = SB3RolloutBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        gamma=0.99,
        gae_lambda=0.95,
        n_envs=n_envs,
    )

    episode_starts = torch.ones(n_envs, device=device)
    for step in range(buffer_size):
        obs_td, obs_np = _build_obs(step, n_envs, device)
        actions = torch.full((n_envs,), step % action_space.n, device=device, dtype=torch.long)
        rewards = torch.linspace(0.1, 0.1 * n_envs, n_envs, device=device)
        values = torch.full((n_envs,), 0.5 + 0.1 * step, device=device)
        log_probs = torch.zeros(n_envs, device=device)
        dones = torch.zeros(n_envs, device=device, dtype=torch.bool)

        new_buf.add(
            obs=obs_td,
            action=actions,
            reward=rewards,
            episode_start=episode_starts,
            value=values,
            log_prob=log_probs,
        )
        sb3_buf.add(
            obs=obs_np,
            action=actions.cpu().numpy(),
            reward=rewards.cpu().numpy(),
            episode_start=episode_starts.cpu().numpy(),
            value=values.cpu(),
            log_prob=log_probs.cpu(),
        )
        episode_starts = torch.zeros(n_envs, device=device)

    last_values = torch.full((n_envs,), 1.23, device=device)
    dones_np = np.zeros(n_envs, dtype=bool)
    new_buf.compute_returns_and_advantage(
        last_values=last_values, dones=torch.zeros(n_envs, device=device, dtype=torch.bool)
    )
    sb3_buf.compute_returns_and_advantage(last_values=last_values, dones=dones_np)

    assert torch.allclose(
        new_buf.advantages, torch.as_tensor(sb3_buf.advantages, device=device)
    ), "Advantages diverged from SB3 computation"
    assert torch.allclose(
        new_buf.returns, torch.as_tensor(sb3_buf.returns, device=device)
    ), "Returns diverged from SB3 computation"


def test_rollout_buffer_returns_match_sb3():
    """Test that returns (advantages + values) match SB3."""
    if SB3RolloutBuffer is None:
        pytest.skip(f"stable_baselines3 unavailable: {_IMPORT_ERROR}")

    torch.manual_seed(42)
    np.random.seed(42)

    buffer_size = 8
    n_envs = 4
    device = torch.device("cpu")

    new_buf = NewRolloutBuffer(
        buffer_size=buffer_size,
        n_envs=n_envs,
        device=device,
        gamma=0.99,
        gae_lambda=0.95,
    )

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(5)
    sb3_buf = SB3RolloutBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        gamma=0.99,
        gae_lambda=0.95,
        n_envs=n_envs,
    )

    episode_starts = torch.ones(n_envs, device=device)
    for step in range(buffer_size):
        obs_td, obs_np = _build_obs(step, n_envs, device)
        actions = torch.randint(0, action_space.n, (n_envs,), device=device)
        rewards = torch.rand(n_envs, device=device)
        values = torch.rand(n_envs, device=device)
        log_probs = torch.randn(n_envs, device=device)
        
        # Simulate some episode endings
        dones = torch.zeros(n_envs, device=device, dtype=torch.bool)
        if step == 3:
            dones[0] = True
            dones[2] = True

        new_buf.add(
            obs=obs_td,
            action=actions,
            reward=rewards,
            episode_start=episode_starts,
            value=values,
            log_prob=log_probs,
        )
        sb3_buf.add(
            obs=obs_np,
            action=actions.cpu().numpy(),
            reward=rewards.cpu().numpy(),
            episode_start=episode_starts.cpu().numpy(),
            value=values.cpu(),
            log_prob=log_probs.cpu(),
        )
        episode_starts = dones.float()

    last_values = torch.rand(n_envs, device=device)
    new_buf.compute_returns_and_advantage(
        last_values=last_values, 
        dones=torch.zeros(n_envs, device=device, dtype=torch.bool)
    )
    sb3_buf.compute_returns_and_advantage(
        last_values=last_values, 
        dones=np.zeros(n_envs, dtype=bool)
    )

    # Check returns match
    assert torch.allclose(
        new_buf.returns, 
        torch.as_tensor(sb3_buf.returns, device=device),
        rtol=1e-5, atol=1e-5
    ), "Returns diverged from SB3 computation"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_rollout_buffer_with_episode_boundaries():
    """Test buffer behavior with episode boundaries."""
    if SB3RolloutBuffer is None:
        pytest.skip(f"stable_baselines3 unavailable: {_IMPORT_ERROR}")

    buffer_size = 6
    n_envs = 2
    device = torch.device("cpu")

    new_buf = NewRolloutBuffer(
        buffer_size=buffer_size,
        n_envs=n_envs,
        device=device,
        gamma=0.99,
        gae_lambda=0.95,
    )

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)
    sb3_buf = SB3RolloutBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        gamma=0.99,
        gae_lambda=0.95,
        n_envs=n_envs,
    )

    # Simulate episodes ending at different times
    episode_end_pattern = [
        [False, False],  # Step 0
        [False, True],   # Step 1 - env 1 ends
        [True, False],   # Step 2 - env 0 ends (starts new episode)
        [False, False],  # Step 3
        [False, False],  # Step 4
        [True, True],    # Step 5 - both end
    ]

    episode_starts = torch.ones(n_envs, device=device)
    for step in range(buffer_size):
        obs_td, obs_np = _build_obs(step, n_envs, device)
        actions = torch.randint(0, action_space.n, (n_envs,), device=device)
        rewards = torch.ones(n_envs, device=device) * (step + 1)  # Increasing rewards
        values = torch.ones(n_envs, device=device) * 0.5

        new_buf.add(
            obs=obs_td,
            action=actions,
            reward=rewards,
            episode_start=episode_starts,
            value=values,
            log_prob=torch.zeros(n_envs, device=device),
        )
        sb3_buf.add(
            obs=obs_np,
            action=actions.cpu().numpy(),
            reward=rewards.cpu().numpy(),
            episode_start=episode_starts.cpu().numpy(),
            value=values.cpu(),
            log_prob=torch.zeros(n_envs),
        )
        
        # Update episode starts for next step
        dones = torch.tensor(episode_end_pattern[step], dtype=torch.bool, device=device)
        episode_starts = dones.float()

    # Compute advantages
    last_values = torch.zeros(n_envs, device=device)
    final_dones = torch.tensor(episode_end_pattern[-1], dtype=torch.bool, device=device)
    new_buf.compute_returns_and_advantage(last_values=last_values, dones=final_dones)
    sb3_buf.compute_returns_and_advantage(
        last_values=last_values, 
        dones=np.array(episode_end_pattern[-1], dtype=bool)
    )

    assert torch.allclose(
        new_buf.advantages, 
        torch.as_tensor(sb3_buf.advantages, device=device),
        rtol=1e-5, atol=1e-5
    ), "Advantages diverged with episode boundaries"


def test_rollout_buffer_different_gamma_lambda():
    """Test with different gamma and lambda values."""
    if SB3RolloutBuffer is None:
        pytest.skip(f"stable_baselines3 unavailable: {_IMPORT_ERROR}")

    test_cases = [
        (0.99, 0.95),  # Default
        (0.9, 0.8),    # Lower gamma and lambda
        (1.0, 1.0),    # Monte Carlo
        (0.99, 0.0),   # TD(0)
    ]

    for gamma, gae_lambda in test_cases:
        buffer_size = 4
        n_envs = 2
        device = torch.device("cpu")

        new_buf = NewRolloutBuffer(
            buffer_size=buffer_size,
            n_envs=n_envs,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(3)
        sb3_buf = SB3RolloutBuffer(
            buffer_size=buffer_size,
            observation_space=obs_space,
            action_space=action_space,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_envs=n_envs,
        )

        episode_starts = torch.ones(n_envs, device=device)
        for step in range(buffer_size):
            obs_td, obs_np = _build_obs(step, n_envs, device)
            actions = torch.zeros(n_envs, device=device, dtype=torch.long)
            rewards = torch.ones(n_envs, device=device)
            values = torch.full((n_envs,), 0.5, device=device)

            new_buf.add(
                obs=obs_td,
                action=actions,
                reward=rewards,
                episode_start=episode_starts,
                value=values,
                log_prob=torch.zeros(n_envs, device=device),
            )
            sb3_buf.add(
                obs=obs_np,
                action=actions.cpu().numpy(),
                reward=rewards.cpu().numpy(),
                episode_start=episode_starts.cpu().numpy(),
                value=values.cpu(),
                log_prob=torch.zeros(n_envs),
            )
            episode_starts = torch.zeros(n_envs, device=device)

        last_values = torch.ones(n_envs, device=device)
        new_buf.compute_returns_and_advantage(
            last_values=last_values, 
            dones=torch.zeros(n_envs, device=device, dtype=torch.bool)
        )
        sb3_buf.compute_returns_and_advantage(
            last_values=last_values, 
            dones=np.zeros(n_envs, dtype=bool)
        )

        assert torch.allclose(
            new_buf.advantages, 
            torch.as_tensor(sb3_buf.advantages, device=device),
            rtol=1e-4, atol=1e-4
        ), f"Advantages diverged for gamma={gamma}, lambda={gae_lambda}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

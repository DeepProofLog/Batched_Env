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
    obs = torch.full((n_envs, 1), float(step), device=device)
    td = TensorDict({"obs": obs}, batch_size=[n_envs], device=device)
    return td, obs.cpu().numpy()


def test_rollout_buffer_advantages_match_sb3():
    if SB3RolloutBuffer is None:
        raise pytest.SkipTest(f"stable_baselines3 unavailable: {_IMPORT_ERROR}")

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

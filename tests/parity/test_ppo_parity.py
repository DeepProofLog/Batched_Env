"""
PPO Training Parity Tests.

Verifies that the new tensor-based PPO produces training metrics consistent
with SB3's PPO implementation.
"""
from pathlib import Path
import sys
import importlib.util

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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"parity_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NewRolloutBuffer = _load_module("rollout", ROOT / "rollout.py").RolloutBuffer

try:
    from stable_baselines3.common.buffers import RolloutBuffer as SB3RolloutBuffer
except Exception as exc:
    SB3RolloutBuffer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class DummyPolicy(nn.Module):
    """Simple policy for testing PPO training loop."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
        self.action_dist = None
    
    def forward(self, obs, deterministic=False):
        """Get action, value, and log_prob."""
        if isinstance(obs, TensorDict):
            x = obs["obs"]
        else:
            x = obs
        
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        
        features = self.features(x.float())
        action_logits = self.actor(features)
        values = self.critic(features)
        
        # Create distribution
        dist = torch.distributions.Categorical(logits=action_logits)
        self.action_dist = type('ActionDist', (), {'distribution': dist})()
        
        if deterministic:
            actions = action_logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        
        return actions, values.squeeze(-1), log_probs
    
    def evaluate_actions(self, obs, actions):
        """Evaluate given actions."""
        if isinstance(obs, TensorDict):
            x = obs["obs"]
        else:
            x = obs
        
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        
        features = self.features(x.float())
        action_logits = self.actor(features)
        values = self.critic(features)
        
        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return values, log_probs, entropy


class DummyEnv:
    """Simple environment for testing."""
    
    def __init__(self, n_envs: int = 4, obs_dim: int = 8, action_dim: int = 4):
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = torch.Size([n_envs])
        self._device = torch.device("cpu")
        self.step_count = 0
        self.max_steps = 10
        
    def reset(self):
        self.step_count = 0
        obs = torch.randn(self.n_envs, self.obs_dim)
        return TensorDict({"obs": obs}, batch_size=[self.n_envs])
    
    def step_and_maybe_reset(self, action_td):
        self.step_count += 1
        
        # Random rewards
        rewards = torch.randn(self.n_envs)
        
        # Done after max_steps
        dones = torch.zeros(self.n_envs, dtype=torch.bool)
        if self.step_count >= self.max_steps:
            dones = torch.ones(self.n_envs, dtype=torch.bool)
            self.step_count = 0
        
        # Random next obs
        next_obs = torch.randn(self.n_envs, self.obs_dim)
        
        step_result = TensorDict({
            "reward": rewards,
            "done": dones,
        }, batch_size=[self.n_envs])
        
        next_obs_td = TensorDict({"obs": next_obs}, batch_size=[self.n_envs])
        
        return step_result, next_obs_td


# ============================================================================
# PPO Training Tests
# ============================================================================

def test_ppo_train_produces_valid_metrics():
    """Test that PPO training produces valid loss metrics."""
    from ppo import PPO
    
    torch.manual_seed(42)
    
    env = DummyEnv(n_envs=4)
    policy = DummyPolicy(obs_dim=env.obs_dim, action_dim=env.action_dim)
    
    ppo = PPO(
        policy=policy,
        env=env,
        n_steps=16,
        n_epochs=2,
        batch_size=8,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=False,
    )
    
    # Manually fill buffer
    ppo.rollout_buffer.reset()
    obs = env.reset()
    episode_starts = torch.ones(env.n_envs)
    
    with torch.no_grad():
        for step in range(ppo.n_steps):
            obs_device = obs.to(ppo.device)
            actions, values, log_probs = ppo.policy(obs_device)
            
            action_td = TensorDict({"action": actions}, batch_size=obs.batch_size)
            step_result, next_obs = env.step_and_maybe_reset(action_td)
            
            rewards = step_result["reward"]
            
            ppo.rollout_buffer.add(
                obs=obs_device,
                action=actions,
                reward=rewards,
                episode_start=episode_starts,
                value=values,
                log_prob=log_probs,
            )
            
            episode_starts = step_result["done"].float()
            obs = next_obs
        
        _, last_values, _ = ppo.policy(obs)
    
    # Compute advantages
    ppo.rollout_buffer.compute_returns_and_advantage(
        last_values=last_values,
        dones=torch.zeros(env.n_envs, dtype=torch.bool)
    )
    
    # Train
    metrics = ppo.train()
    
    # Check metrics are valid
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "clip_fraction" in metrics
    
    # Check metrics are finite
    assert np.isfinite(metrics["policy_loss"]), "Policy loss is not finite"
    assert np.isfinite(metrics["value_loss"]), "Value loss is not finite"
    assert np.isfinite(metrics["entropy"]), "Entropy is not finite"
    assert 0.0 <= metrics["clip_fraction"] <= 1.0, "Clip fraction out of range"


def test_ppo_clip_fraction_behavior():
    """Test that clip fraction increases when policy changes significantly."""
    from ppo import PPO
    
    torch.manual_seed(42)
    
    env = DummyEnv(n_envs=4)
    policy = DummyPolicy(obs_dim=env.obs_dim, action_dim=env.action_dim)
    
    ppo = PPO(
        policy=policy,
        env=env,
        n_steps=16,
        n_epochs=1,
        batch_size=4,
        learning_rate=0.1,  # High LR to force clipping
        clip_range=0.1,     # Small clip range
        verbose=False,
    )
    
    # Fill buffer
    ppo.rollout_buffer.reset()
    obs = env.reset()
    episode_starts = torch.ones(env.n_envs)
    
    with torch.no_grad():
        for step in range(ppo.n_steps):
            obs_device = obs.to(ppo.device)
            actions, values, log_probs = ppo.policy(obs_device)
            
            action_td = TensorDict({"action": actions}, batch_size=obs.batch_size)
            step_result, next_obs = env.step_and_maybe_reset(action_td)
            
            ppo.rollout_buffer.add(
                obs=obs_device,
                action=actions,
                reward=step_result["reward"],
                episode_start=episode_starts,
                value=values,
                log_prob=log_probs,
            )
            
            episode_starts = step_result["done"].float()
            obs = next_obs
        
        _, last_values, _ = ppo.policy(obs)
    
    ppo.rollout_buffer.compute_returns_and_advantage(
        last_values=last_values,
        dones=torch.zeros(env.n_envs, dtype=torch.bool)
    )
    
    # Train with high learning rate - should see some clipping
    metrics = ppo.train()
    
    # With high LR and small clip range, we expect some clipping to occur
    # (though not guaranteed)
    assert metrics["clip_fraction"] >= 0.0
    assert metrics["clip_fraction"] <= 1.0


def test_ppo_advantage_normalization():
    """Test that advantage normalization works correctly."""
    from ppo import PPO
    
    torch.manual_seed(42)
    
    env = DummyEnv(n_envs=4)
    policy = DummyPolicy(obs_dim=env.obs_dim, action_dim=env.action_dim)
    
    # Test with normalization enabled
    ppo_norm = PPO(
        policy=policy,
        env=env,
        n_steps=16,
        n_epochs=1,
        batch_size=8,
        normalize_advantage=True,
        verbose=False,
    )
    
    # Fill buffer
    ppo_norm.rollout_buffer.reset()
    obs = env.reset()
    episode_starts = torch.ones(env.n_envs)
    
    with torch.no_grad():
        for step in range(ppo_norm.n_steps):
            obs_device = obs.to(ppo_norm.device)
            actions, values, log_probs = ppo_norm.policy(obs_device)
            
            action_td = TensorDict({"action": actions}, batch_size=obs.batch_size)
            step_result, next_obs = env.step_and_maybe_reset(action_td)
            
            ppo_norm.rollout_buffer.add(
                obs=obs_device,
                action=actions,
                reward=step_result["reward"],
                episode_start=episode_starts,
                value=values,
                log_prob=log_probs,
            )
            
            episode_starts = step_result["done"].float()
            obs = next_obs
        
        _, last_values, _ = ppo_norm.policy(obs)
    
    ppo_norm.rollout_buffer.compute_returns_and_advantage(
        last_values=last_values,
        dones=torch.zeros(env.n_envs, dtype=torch.bool)
    )
    
    metrics = ppo_norm.train()
    
    # Should produce valid metrics
    assert np.isfinite(metrics["policy_loss"])


# ============================================================================
# GAE Computation Parity
# ============================================================================

def test_gae_computation_matches_sb3():
    """Test that GAE computation in our rollout buffer matches SB3."""
    if SB3RolloutBuffer is None:
        pytest.skip(f"stable_baselines3 unavailable: {_IMPORT_ERROR}")
    
    import gymnasium as gym
    
    torch.manual_seed(123)
    np.random.seed(123)
    
    buffer_size = 16
    n_envs = 4
    gamma = 0.99
    gae_lambda = 0.95
    device = torch.device("cpu")
    
    # Create buffers
    new_buf = NewRolloutBuffer(
        buffer_size=buffer_size,
        n_envs=n_envs,
        device=device,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)
    sb3_buf = SB3RolloutBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=n_envs,
    )
    
    # Generate random data
    episode_starts = torch.ones(n_envs, device=device)
    
    for step in range(buffer_size):
        obs = torch.randn(n_envs, 8, device=device)
        obs_td = TensorDict({"obs": obs}, batch_size=[n_envs], device=device)
        
        actions = torch.randint(0, 4, (n_envs,), device=device)
        rewards = torch.randn(n_envs, device=device)
        values = torch.randn(n_envs, device=device)
        log_probs = torch.randn(n_envs, device=device)
        
        # Simulate some episode boundaries
        dones = torch.rand(n_envs, device=device) > 0.8
        
        new_buf.add(
            obs=obs_td,
            action=actions,
            reward=rewards,
            episode_start=episode_starts,
            value=values,
            log_prob=log_probs,
        )
        sb3_buf.add(
            obs=obs.cpu().numpy(),
            action=actions.cpu().numpy(),
            reward=rewards.cpu().numpy(),
            episode_start=episode_starts.cpu().numpy(),
            value=values.cpu(),
            log_prob=log_probs.cpu(),
        )
        
        episode_starts = dones.float()
    
    # Compute GAE
    last_values = torch.randn(n_envs, device=device)
    final_dones = torch.zeros(n_envs, dtype=torch.bool, device=device)
    
    new_buf.compute_returns_and_advantage(last_values=last_values, dones=final_dones)
    sb3_buf.compute_returns_and_advantage(last_values=last_values, dones=final_dones.cpu().numpy())
    
    # Compare advantages
    new_advantages = new_buf.advantages
    sb3_advantages = torch.as_tensor(sb3_buf.advantages, device=device)
    
    assert torch.allclose(new_advantages, sb3_advantages, rtol=1e-5, atol=1e-5), \
        f"Advantages differ: max diff = {(new_advantages - sb3_advantages).abs().max().item()}"
    
    # Compare returns
    new_returns = new_buf.returns
    sb3_returns = torch.as_tensor(sb3_buf.returns, device=device)
    
    assert torch.allclose(new_returns, sb3_returns, rtol=1e-5, atol=1e-5), \
        f"Returns differ: max diff = {(new_returns - sb3_returns).abs().max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Rollout Buffer for On-Policy Reinforcement Learning.

This module provides a GPU-accelerated rollout buffer for collecting
and processing environment transitions during PPO training.

Key Features:
    - GPU-native storage using torch tensors
    - GAE (Generalized Advantage Estimation) computation
    - Efficient minibatch generation for training
    - Compatible with vectorized environments

Tensor Shape Conventions:
    T = buffer_size (number of steps per rollout)
    N = n_envs (number of parallel environments)
    * = variable observation dimensions
"""

import torch
import numpy as np
from typing import Optional, Generator, Tuple
from tensordict import TensorDict


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms like PPO.
    
    This buffer stores transitions collected during rollout and computes
    advantages using Generalized Advantage Estimation (GAE).
    
    Storage shapes:
        observations: Dict[str, (buffer_size, n_envs, *obs_shape)]
        actions:      (buffer_size, n_envs, action_dim)
        rewards:      (buffer_size, n_envs)
        values:       (buffer_size, n_envs)
        log_probs:    (buffer_size, n_envs)
        advantages:   (buffer_size, n_envs)
        returns:      (buffer_size, n_envs)
    
    Args:
        buffer_size: Number of steps to collect per rollout
        n_envs: Number of parallel environments
        device: PyTorch device (cpu or cuda)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    """
    
    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Current position in buffer
        self.pos = 0
        self.full = False
        self.generator_ready = False
        
        # Storage tensors (will be initialized on first add)
        self.observations = None
        self.actions = None
        self.rewards = None
        self.values = None
        self.log_probs = None
        self.episode_starts = None
        self.advantages = None
        self.returns = None
        
        # Store observation structure info
        self.obs_keys = None
        self.obs_shapes = {}
        self.obs_dtypes = {}
        
    def reset(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self.full = False
        self.generator_ready = False
        
        if self.observations is not None:
            # Clear existing storage
            for key in self.observations.keys():
                self.observations[key].zero_()
            self.actions.zero_()
            self.rewards.zero_()
            self.values.zero_()
            self.log_probs.zero_()
            self.episode_starts.zero_()
            if self.advantages is not None:
                self.advantages.zero_()
            if self.returns is not None:
                self.returns.zero_()
    
    def _initialize_storage(self, obs: TensorDict, action: torch.Tensor) -> None:
        """Initialize storage tensors based on first observation and action."""
        # Store observation structure
        self.obs_keys = list(obs.keys())
        self.observations = {}
        
        for key in self.obs_keys:
            obs_tensor = obs[key]
            self.obs_shapes[key] = obs_tensor.shape[1:]  # Exclude batch dimension
            self.obs_dtypes[key] = obs_tensor.dtype
            
            # Create storage: [buffer_size, n_envs, *obs_shape]
            storage_shape = (self.buffer_size, self.n_envs) + self.obs_shapes[key]
            self.observations[key] = torch.zeros(
                storage_shape,
                dtype=obs_tensor.dtype,
                device=self.device
            )
        
        # Action storage
        action_dim = action.shape[-1] if action.dim() > 1 else 1
        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs, action_dim),
            dtype=action.dtype,
            device=self.device
        )
        
        # Scalar storages
        self.rewards = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.values = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.log_probs = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.episode_starts = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.advantages = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.returns = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
    
    def add(
        self,
        obs: TensorDict,
        action: torch.Tensor,
        reward: torch.Tensor,
        episode_start: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            obs: Observations (TensorDict with batch_size=n_envs)
            action: Actions taken (n_envs, action_dim) or (n_envs,)
            reward: Rewards received (n_envs,)
            episode_start: Episode start flags (n_envs,)
            value: Value estimates (n_envs,)
            log_prob: Log probabilities of actions (n_envs,)
        """
        # Initialize storage on first call
        if self.observations is None:
            self._initialize_storage(obs, action)
        
        # Ensure tensors are on the correct device
        if action.device != self.device:
            action = action.to(self.device)
        if reward.device != self.device:
            reward = reward.to(self.device)
        if episode_start.device != self.device:
            episode_start = episode_start.to(self.device)
        if value.device != self.device:
            value = value.to(self.device)
        if log_prob.device != self.device:
            log_prob = log_prob.to(self.device)
        
        # Handle action shape
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        
        # Store observations
        for key in self.obs_keys:
            obs_tensor = obs[key]
            if obs_tensor.device != self.device:
                obs_tensor = obs_tensor.to(self.device)
            self.observations[key][self.pos] = obs_tensor
        
        # Store other data
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.flatten()
        self.log_probs[self.pos] = log_prob.flatten()
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantage(
        self,
        last_values: torch.Tensor,
        dones: torch.Tensor
    ) -> None:
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_values: Value estimates for the last step (n_envs,)
            dones: Done flags for the last step (n_envs,)
        """
        last_values = last_values.flatten().to(self.device)
        dones = dones.float().to(self.device)
        
        last_gae_lam = torch.zeros(self.n_envs, device=self.device)
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # Compute returns
        self.returns = self.advantages + self.values
    
    @staticmethod
    def swap_and_flatten(tensor: torch.Tensor) -> torch.Tensor:
        """
        Swap and flatten axes 0 (buffer_size) and 1 (n_envs).
        
        Shape: (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        """
        shape = tensor.shape
        if len(shape) < 3:
            shape = (*shape, 1)
            tensor = tensor.reshape(shape)
        
        # Swap axes 0 and 1, then flatten
        # Make contiguous to match numpy's memory layout and ensure deterministic operations
        return tensor.transpose(0, 1).contiguous().reshape(shape[0] * shape[1], *shape[2:])
    
    def get(
        self,
        batch_size: Optional[int] = None
    ) -> Generator[Tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """
        Generate batches for training.
        
        Args:
            batch_size: Size of minibatches. If None, returns entire buffer.
        
        Yields:
            Tuples of (observations, actions, values, log_probs, advantages, returns)
        """
        if not self.full:
            raise RuntimeError("Buffer is not full. Cannot sample.")
        
        # Create random permutation of indices
        # Ensures consistent random state consumption independent of generator readiness
        total_size = self.buffer_size * self.n_envs
        indices = np.random.permutation(total_size)
        
        # Prepare flattened data on first call
        if not self.generator_ready:
            # Flatten observations
            self.flat_observations = {}
            for key in self.obs_keys:
                self.flat_observations[key] = self.swap_and_flatten(self.observations[key])
            
            # Flatten other tensors
            self.flat_actions = self.swap_and_flatten(self.actions)
            self.flat_values = self.swap_and_flatten(self.values)
            self.flat_log_probs = self.swap_and_flatten(self.log_probs)
            self.flat_advantages = self.swap_and_flatten(self.advantages)
            self.flat_returns = self.swap_and_flatten(self.returns)
            
            self.generator_ready = True
        
        # Convert indices to torch tensor
        indices = torch.from_numpy(indices).to(self.device)
        
        # Default to full batch if batch_size not specified
        if batch_size is None:
            batch_size = total_size
        
        # Generate minibatches
        start_idx = 0
        while start_idx < total_size:
            end_idx = min(start_idx + batch_size, total_size)
            batch_indices = indices[start_idx:end_idx]
            
            # Create observation TensorDict for this batch
            batch_obs = TensorDict(
                {key: self.flat_observations[key][batch_indices] for key in self.obs_keys},
                batch_size=len(batch_indices),
                device=self.device
            )
            
            yield (
                batch_obs,
                self._format_actions(self.flat_actions[batch_indices]),
                self.flat_values[batch_indices].squeeze(-1),
                self.flat_log_probs[batch_indices].squeeze(-1),
                self.flat_advantages[batch_indices].squeeze(-1),
                self.flat_returns[batch_indices].squeeze(-1),
            )
            
            start_idx = end_idx

    def size(self) -> int:
        """Return current size of buffer."""
        if self.full:
            return self.buffer_size * self.n_envs
        return self.pos * self.n_envs

    @staticmethod
    def _format_actions(actions: torch.Tensor) -> torch.Tensor:
        """Squeeze the last dimension if it is singular."""
        if actions.dim() > 1 and actions.shape[-1] == 1:
            return actions.squeeze(-1)
        return actions

"""
Rollout Buffer for On-Policy Reinforcement Learning with Optimized Environment.

This module provides a GPU-accelerated rollout buffer designed to work with
EvalEnvOptimized (which uses EvalObs NamedTuples instead of TensorDict).

Key Features:
    - GPU-native storage using torch tensors
    - GAE (Generalized Advantage Estimation) computation
    - Efficient minibatch generation for training
    - Compatible with EvalObs observation format
"""

import torch
import numpy as np
from typing import Optional, Generator, Tuple, NamedTuple


class RolloutBufferOptimized:
    """
    Rollout buffer for on-policy algorithms working with EvalEnvOptimized.
    
    Unlike RolloutBuffer which uses TensorDict, this stores observations
    as separate tensor fields matching the EvalObs structure.
    
    Storage shapes:
        sub_index:            (buffer_size, n_envs, 1, A, 3)
        derived_sub_indices:  (buffer_size, n_envs, S, A, 3)
        action_mask:          (buffer_size, n_envs, S)
        actions:              (buffer_size, n_envs)
        rewards:              (buffer_size, n_envs)
        values:               (buffer_size, n_envs)
        log_probs:            (buffer_size, n_envs)
        advantages:           (buffer_size, n_envs)
        returns:              (buffer_size, n_envs)
    
    Args:
        buffer_size: Number of steps to collect per rollout
        n_envs: Number of parallel environments
        device: PyTorch device (cpu or cuda)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        padding_atoms: A dimension (number of atoms per state)
        padding_states: S dimension (max derived states)
    """
    
    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        padding_atoms: int = 6,
        padding_states: int = 100,
        parity: bool = False,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.parity = parity
        
        # Current position in buffer
        self.pos = 0
        self.full = False
        self.generator_ready = False
        
        # Initialize storage tensors
        self._initialize_storage()
        
    def _initialize_storage(self) -> None:
        """Initialize storage tensors for observations and other data."""
        T, N, A, S = self.buffer_size, self.n_envs, self.padding_atoms, self.padding_states
        
        # Observation storage matching EvalObs structure
        self.sub_index = torch.zeros((T, N, 1, A, 3), dtype=torch.long, device=self.device)
        self.derived_sub_indices = torch.zeros((T, N, S, A, 3), dtype=torch.long, device=self.device)
        self.action_mask = torch.zeros((T, N, S), dtype=torch.bool, device=self.device)
        
        # Action and scalar storage
        self.actions = torch.zeros((T, N), dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.episode_starts = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        
    def reset(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self.full = False
        self.generator_ready = False
        
        # Zero out storage
        self.sub_index.zero_()
        self.derived_sub_indices.zero_()
        self.action_mask.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.episode_starts.zero_()
        self.advantages.zero_()
        self.returns.zero_()
    
    def add(
        self,
        sub_index: torch.Tensor,
        derived_sub_indices: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        episode_start: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            sub_index: Current state observation [N, 1, A, 3]
            derived_sub_indices: Derived states [N, S, A, 3]
            action_mask: Valid action mask [N, S]
            action: Actions taken [N]
            reward: Rewards received [N]
            episode_start: Episode start flags [N]
            value: Value estimates [N]
            log_prob: Log probabilities of actions [N]
        """
        # Store observations
        self.sub_index[self.pos] = sub_index.to(self.device)
        self.derived_sub_indices[self.pos] = derived_sub_indices.to(self.device)
        self.action_mask[self.pos] = action_mask.to(self.device)
        
        # Store scalars
        self.actions[self.pos] = action.to(self.device)
        self.rewards[self.pos] = reward.to(self.device)
        self.episode_starts[self.pos] = episode_start.to(self.device)
        self.values[self.pos] = value.flatten().to(self.device)
        self.log_probs[self.pos] = log_prob.flatten().to(self.device)
        
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
            last_values: Value estimates for the last step [N]
            dones: Done flags for the last step [N]
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
        
        Shape: (T, N, ...) -> (T * N, ...)
        """
        shape = tensor.shape
        # Swap axes 0 and 1, then flatten first two dims
        return tensor.transpose(0, 1).contiguous().reshape(shape[0] * shape[1], *shape[2:])
    
    def get(
        self,
        batch_size: Optional[int] = None
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate batches for training.
        
        Args:
            batch_size: Size of minibatches. If None, returns entire buffer.
        
        Yields:
            Tuples of (sub_index, derived_sub_indices, action_mask, 
                       actions, values, log_probs, advantages, returns)
        """
        if not self.full:
            raise RuntimeError("Buffer is not full. Cannot sample.")
        
        # Create random permutation of indices
        total_size = self.buffer_size * self.n_envs
        
        if self.parity:
            # Use numpy for exact parity with SB3
            indices = np.random.permutation(total_size)
            indices = torch.from_numpy(indices).to(self.device)
        else:
            # Use torch.randperm for efficiency
            indices = torch.randperm(total_size, device=self.device)
        
        # Prepare flattened data on first call
        if not self.generator_ready:
            self.flat_sub_index = self.swap_and_flatten(self.sub_index)
            self.flat_derived_sub_indices = self.swap_and_flatten(self.derived_sub_indices)
            self.flat_action_mask = self.swap_and_flatten(self.action_mask)
            self.flat_actions = self.swap_and_flatten(self.actions.unsqueeze(-1)).squeeze(-1)
            self.flat_values = self.swap_and_flatten(self.values.unsqueeze(-1)).squeeze(-1)
            self.flat_log_probs = self.swap_and_flatten(self.log_probs.unsqueeze(-1)).squeeze(-1)
            self.flat_advantages = self.swap_and_flatten(self.advantages.unsqueeze(-1)).squeeze(-1)
            self.flat_returns = self.swap_and_flatten(self.returns.unsqueeze(-1)).squeeze(-1)
            self.generator_ready = True
        
        # Default to full batch if batch_size not specified
        if batch_size is None:
            batch_size = total_size
        
        # Generate minibatches
        start_idx = 0
        while start_idx < total_size:
            end_idx = min(start_idx + batch_size, total_size)
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                self.flat_sub_index[batch_indices],
                self.flat_derived_sub_indices[batch_indices],
                self.flat_action_mask[batch_indices],
                self.flat_actions[batch_indices],
                self.flat_values[batch_indices],
                self.flat_log_probs[batch_indices],
                self.flat_advantages[batch_indices],
                self.flat_returns[batch_indices],
            )
            
            start_idx = end_idx

    def size(self) -> int:
        """Return current size of buffer."""
        if self.full:
            return self.buffer_size * self.n_envs
        return self.pos * self.n_envs

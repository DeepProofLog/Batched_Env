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
from typing import Optional, Generator, Tuple, NamedTuple, List


def _batch_index_select(
    tensors: List[torch.Tensor],
    indices: torch.Tensor,
    outputs: List[torch.Tensor],
) -> None:
    """
    Batch index_select operations for multiple tensors.
    
    This function performs index_select on multiple tensors simultaneously,
    using in-place operations to maintain stable memory addresses for CUDA graphs.
    
    Args:
        tensors: List of source tensors to index into
        indices: Index tensor [batch_size]
        outputs: List of pre-allocated output tensors (same order as tensors)
    """
    for src, dst in zip(tensors, outputs):
        torch.index_select(src, 0, indices, out=dst)


class RolloutBuffer:
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
        batch_size: int = 64,  # Pre-allocate batch tensors for this size
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.parity = parity
        self._default_batch_size = batch_size
        
        # Current position in buffer
        self.pos = 0
        self.full = False
        # generator_ready is persistent - never reset once True
        self.generator_ready = False
        
        # Initialize storage tensors
        self._initialize_storage()
        
        # Pre-allocate flattened tensors for CUDA graph stability
        # These must have stable memory addresses across resets
        self._initialize_flat_storage()
        
        # Pre-allocate batch tensors for CUDA graph stability
        # These are used in get() and must have stable memory addresses
        self._initialize_batch_storage(batch_size)
        
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
    
    def _initialize_flat_storage(self) -> None:
        """Pre-allocate flattened tensors for CUDA graph compatibility.
        
        These tensors maintain stable memory addresses across resets,
        which is required for CUDA graphs in reduce-overhead mode.
        """
        total_size = self.buffer_size * self.n_envs
        A = self.padding_atoms
        S = self.padding_states
        
        # Flattened observation storage
        self.flat_sub_index = torch.zeros((total_size, 1, A, 3), dtype=torch.long, device=self.device)
        self.flat_derived_sub_indices = torch.zeros((total_size, S, A, 3), dtype=torch.long, device=self.device)
        self.flat_action_mask = torch.zeros((total_size, S), dtype=torch.bool, device=self.device)
        
        # Flattened scalar storage
        self.flat_actions = torch.zeros(total_size, dtype=torch.long, device=self.device)
        self.flat_values = torch.zeros(total_size, dtype=torch.float32, device=self.device)
        self.flat_log_probs = torch.zeros(total_size, dtype=torch.float32, device=self.device)
        self.flat_advantages = torch.zeros(total_size, dtype=torch.float32, device=self.device)
        self.flat_returns = torch.zeros(total_size, dtype=torch.float32, device=self.device)
        
        # Mark as always ready since tensors are pre-allocated
        self.generator_ready = True
    
    def _initialize_batch_storage(self, batch_size: int) -> None:
        """Pre-allocate batch tensors for CUDA graph compatibility.
        
        These tensors are used in get() and must have stable memory addresses
        for CUDA graphs in reduce-overhead mode. Allocated once in __init__,
        never reallocated.
        
        Args:
            batch_size: The batch size to pre-allocate for.
        """
        A = self.padding_atoms
        S = self.padding_states
        total_size = self.buffer_size * self.n_envs
        
        # Batch output tensors - these hold the actual batch data
        self._batch_sub_index = torch.zeros((batch_size, 1, A, 3), dtype=torch.long, device=self.device)
        self._batch_derived = torch.zeros((batch_size, S, A, 3), dtype=torch.long, device=self.device)
        self._batch_mask = torch.zeros((batch_size, S), dtype=torch.bool, device=self.device)
        self._batch_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._batch_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self._batch_log_probs = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self._batch_advantages = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self._batch_returns = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self._batch_tensors_size = batch_size
        
        # Pre-allocate index tensor for stable memory addresses
        # This avoids creating new index tensors each batch
        self._batch_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Pre-allocate full permutation tensor
        self._permutation = torch.zeros(total_size, dtype=torch.long, device=self.device)
        
    def reset(self) -> None:
        """Reset the buffer.
        
        NOTE: Does NOT reset generator_ready or flattened tensors to maintain
        stable memory addresses for CUDA graph compatibility.
        """
        self.pos = 0
        self.full = False
        # DO NOT reset generator_ready - flattened tensors are pre-allocated
        # and must keep stable memory addresses for CUDA graphs
        
        # Zero out storage (in-place - preserves memory addresses)
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
    
    @staticmethod
    def _swap_and_flatten_into(src: torch.Tensor, dst: torch.Tensor) -> None:
        """
        Swap and flatten axes 0 and 1 of src, copying INTO dst.
        
        This preserves the memory address of dst, which is critical for
        CUDA graph compatibility in reduce-overhead mode.
        
        Args:
            src: Source tensor with shape (T, N, ...)
            dst: Pre-allocated destination tensor with shape (T*N, ...)
        """
        shape = src.shape
        # Create a view with swapped and flattened dims, then copy into dst
        flattened = src.transpose(0, 1).contiguous().reshape(shape[0] * shape[1], *shape[2:])
        dst.copy_(flattened)
    
    def get(
        self,
        batch_size: Optional[int] = None
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate batches for training.
        
        Args:
            batch_size: Size of minibatches. If None, uses default batch size.
        
        Yields:
            Tuples of (sub_index, derived_sub_indices, action_mask, 
                       actions, values, log_probs, advantages, returns)
        
        IMPORTANT: For CUDA graph compatibility (reduce-overhead mode), the batch_size
        should match the one used in __init__. If a different batch_size is passed,
        the method will still work but may trigger recompilation.
        """
        if not self.full:
            raise RuntimeError("Buffer is not full. Cannot sample.")
        
        total_size = self.buffer_size * self.n_envs
        
        # Default to pre-allocated batch size
        if batch_size is None:
            batch_size = self._default_batch_size
        
        # Check if we need to reallocate batch tensors (warning: breaks CUDA graphs)
        if batch_size != self._batch_tensors_size:
            import warnings
            warnings.warn(
                f"Batch size changed from {self._batch_tensors_size} to {batch_size}. "
                f"This may cause CUDA graph recompilation in reduce-overhead mode.",
                RuntimeWarning
            )
            self._initialize_batch_storage(batch_size)
        
        # Generate permutation INTO pre-allocated tensor (stable memory address)
        if self.parity:
            perm_np = np.random.permutation(total_size)
            self._permutation.copy_(torch.from_numpy(perm_np))
        else:
            torch.randperm(total_size, out=self._permutation)
        
        # Copy data INTO pre-allocated flattened tensors (preserves memory addresses)
        self._swap_and_flatten_into(self.sub_index, self.flat_sub_index)
        self._swap_and_flatten_into(self.derived_sub_indices, self.flat_derived_sub_indices)
        self._swap_and_flatten_into(self.action_mask, self.flat_action_mask)
        self._swap_and_flatten_into(self.actions, self.flat_actions)
        self._swap_and_flatten_into(self.values, self.flat_values)
        self._swap_and_flatten_into(self.log_probs, self.flat_log_probs)
        self._swap_and_flatten_into(self.advantages, self.flat_advantages)
        self._swap_and_flatten_into(self.returns, self.flat_returns)
        
        # Generate minibatches using stable index tensor
        start_idx = 0
        while start_idx < total_size:
            end_idx = min(start_idx + batch_size, total_size)
            actual_batch_size = end_idx - start_idx
            
            if actual_batch_size < batch_size:
                # Last smaller batch - this won't use CUDA graphs anyway
                # Create temporary slice (okay since it's the final batch)
                batch_idx = self._permutation[start_idx:end_idx]
                yield (
                    self.flat_sub_index[batch_idx],
                    self.flat_derived_sub_indices[batch_idx],
                    self.flat_action_mask[batch_idx],
                    self.flat_actions[batch_idx],
                    self.flat_values[batch_idx],
                    self.flat_log_probs[batch_idx],
                    self.flat_advantages[batch_idx],
                    self.flat_returns[batch_idx],
                )
            else:
                # Copy indices into pre-allocated index tensor (stable address)
                self._batch_indices.copy_(self._permutation[start_idx:end_idx])
                
                # Use batched index_select for better kernel scheduling
                # All index_selects use the same indices, so the GPU can overlap them
                _batch_index_select(
                    [self.flat_sub_index, self.flat_derived_sub_indices, self.flat_action_mask,
                     self.flat_actions, self.flat_values, self.flat_log_probs,
                     self.flat_advantages, self.flat_returns],
                    self._batch_indices,
                    [self._batch_sub_index, self._batch_derived, self._batch_mask,
                     self._batch_actions, self._batch_values, self._batch_log_probs,
                     self._batch_advantages, self._batch_returns]
                )
                
                yield (
                    self._batch_sub_index,
                    self._batch_derived,
                    self._batch_mask,
                    self._batch_actions,
                    self._batch_values,
                    self._batch_log_probs,
                    self._batch_advantages,
                    self._batch_returns,
                )
            
            start_idx = end_idx

    def size(self) -> int:
        """Return current size of buffer."""
        if self.full:
            return self.buffer_size * self.n_envs
        return self.pos * self.n_envs

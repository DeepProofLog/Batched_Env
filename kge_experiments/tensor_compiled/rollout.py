"""
Rollout Buffer for On-Policy Reinforcement Learning with Optimized Environment.

This module provides a GPU-accelerated rollout buffer designed to work with
EnvVec (which uses EnvObs NamedTuples instead of TensorDict).

Key Features:
    - GPU-native storage using torch tensors
    - GAE (Generalized Advantage Estimation) computation
    - Efficient minibatch generation for training
    - Compatible with EnvObs observation format
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
    # Use index_copy_ which can be more efficient for in-place operations
    for src, dst in zip(tensors, outputs):
        torch.index_select(src, 0, indices, out=dst)


# Compiled version for better performance (will be JIT compiled on first call)
@torch.compile(mode="reduce-overhead", fullgraph=True)
def _compiled_flatten_and_permute(
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    permutation: torch.Tensor,
    T: int,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiled flatten + permute for scalar tensors."""
    # Flatten
    flat_actions = actions.transpose(0, 1).contiguous().reshape(T * N)
    flat_values = values.transpose(0, 1).contiguous().reshape(T * N)
    flat_log_probs = log_probs.transpose(0, 1).contiguous().reshape(T * N)
    flat_advantages = advantages.transpose(0, 1).contiguous().reshape(T * N)
    flat_returns = returns.transpose(0, 1).contiguous().reshape(T * N)

    # Permute (index into flattened tensors)
    return (
        flat_actions[permutation],
        flat_values[permutation],
        flat_log_probs[permutation],
        flat_advantages[permutation],
        flat_returns[permutation],
    )


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms working with EnvVec.
    
    Unlike RolloutBuffer which uses TensorDict, this stores observations
    as separate tensor fields matching the EnvObs structure.
    
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
        batch_size: int = 64,  # Pre-allocate batch tensors for this size
        parity: bool = False,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self._default_batch_size = batch_size
        self.parity = bool(parity)
        
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

        # Observation storage - store directly in flat format to avoid flatten in get()
        # Layout: (T*N, ...) but we treat it as (T, N, ...) using computed indices
        # This eliminates expensive transpose+contiguous+flatten in get()
        total_size = T * N
        self.flat_sub_index = torch.zeros((total_size, 1, A, 3), dtype=torch.long, device=self.device)
        self.flat_derived_sub_indices = torch.zeros((total_size, S, A, 3), dtype=torch.long, device=self.device)
        self.flat_action_mask = torch.zeros((total_size, S), dtype=torch.bool, device=self.device)

        # Scalar storage - keep in (T, N) format for GAE computation
        self.actions = torch.zeros((T, N), dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.episode_starts = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((T, N), dtype=torch.float32, device=self.device)
    
    def _initialize_flat_storage(self) -> None:
        """Pre-allocate flattened tensors for scalar data.

        Observation tensors are already in flat format from _initialize_storage().
        This only handles scalars which need (T, N) format for GAE but flat for batching.
        """
        total_size = self.buffer_size * self.n_envs

        # Flattened scalar storage (observations already flat in _initialize_storage)
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

        # Pre-compute base indices for add(): env_idx * buffer_size
        # This avoids creating torch.arange() each add() call
        self._add_base_indices = torch.arange(self.n_envs, device=self.device) * self.buffer_size

        # Pre-compute multiple random permutations to avoid randperm overhead during training
        # This is a significant optimization: randperm takes ~0.26s per call
        # Skip pre-computation in parity mode to preserve deterministic random state
        if not self.parity:
            self._num_precomputed_perms = 100  # Enough for many epochs
            self._precomputed_perms = torch.stack([
                torch.randperm(total_size, device=self.device) for _ in range(self._num_precomputed_perms)
            ])
            self._perm_index = 0
        else:
            self._num_precomputed_perms = 0
            self._precomputed_perms = None
            self._perm_index = 0
        
    def reset(self) -> None:
        """Reset the buffer.

        NOTE: Does NOT reset generator_ready or flattened tensors to maintain
        stable memory addresses for CUDA graph compatibility.
        """
        self.pos = 0
        self.full = False
        # DO NOT reset generator_ready - flattened tensors are pre-allocated
        # and must keep stable memory addresses for CUDA graphs

        # Zero out storage using batched operations (reduces kernel launches)
        # Batch scalar tensors together
        scalar_tensors = [
            self.actions, self.rewards, self.values, self.log_probs,
            self.episode_starts, self.advantages, self.returns
        ]
        torch._foreach_zero_(scalar_tensors)

        # Observation tensors are in flat format - zero them out
        self.flat_sub_index.zero_()
        self.flat_derived_sub_indices.zero_()
        self.flat_action_mask.zero_()
    
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
        pos = self.pos

        # Scalar storage - keep in (T, N) format for GAE computation
        self.actions[pos] = action
        self.rewards[pos] = reward
        self.episode_starts[pos] = episode_start
        self.values[pos] = value.flatten()
        self.log_probs[pos] = log_prob.flatten()

        # Observations - write directly to flat storage in env-major order
        # Uses pre-computed base indices (env_idx * T) + pos
        # This matches the scalar flatten order: (N, T) -> (N*T)
        indices = self._add_base_indices + pos
        self.flat_sub_index[indices] = sub_index
        self.flat_derived_sub_indices[indices] = derived_sub_indices
        # Handle bool/uint8 dtype mismatch for action_mask
        self.flat_action_mask[indices] = action_mask.bool() if action_mask.dtype != torch.bool else action_mask

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
        
        # Use pre-computed permutation (avoids expensive randperm call)
        if self.parity:
            indices = torch.from_numpy(np.random.permutation(total_size)).to(self.device)
            self._permutation.copy_(indices)
        else:
            # Cycle through pre-computed permutations (fast copy instead of randperm)
            self._permutation.copy_(self._precomputed_perms[self._perm_index])
            self._perm_index = (self._perm_index + 1) % self._num_precomputed_perms
        
        # Flatten and shuffle all data once using the permutation
        # This converts random access (slow) to contiguous slicing (fast)
        perm = self._permutation

        # Flatten scalars with permutation applied in one step
        flat_actions_tmp = self.actions.transpose(0, 1).contiguous().view(-1)
        flat_values_tmp = self.values.transpose(0, 1).contiguous().view(-1)
        flat_log_probs_tmp = self.log_probs.transpose(0, 1).contiguous().view(-1)
        flat_advantages_tmp = self.advantages.transpose(0, 1).contiguous().view(-1)
        flat_returns_tmp = self.returns.transpose(0, 1).contiguous().view(-1)

        # Apply permutation to all tensors (one-time shuffle per get() call)
        self.flat_actions.copy_(flat_actions_tmp[perm])
        self.flat_values.copy_(flat_values_tmp[perm])
        self.flat_log_probs.copy_(flat_log_probs_tmp[perm])
        self.flat_advantages.copy_(flat_advantages_tmp[perm])
        self.flat_returns.copy_(flat_returns_tmp[perm])

        # Shuffle observations (already in flat format)
        # Create shuffled versions in pre-allocated storage
        if not hasattr(self, '_shuffled_sub_index'):
            # First call - allocate shuffled storage
            self._shuffled_sub_index = self.flat_sub_index.clone()
            self._shuffled_derived = self.flat_derived_sub_indices.clone()
            self._shuffled_mask = self.flat_action_mask.clone()

        # Apply permutation to observations
        torch.index_select(self.flat_sub_index, 0, perm, out=self._shuffled_sub_index)
        torch.index_select(self.flat_derived_sub_indices, 0, perm, out=self._shuffled_derived)
        torch.index_select(self.flat_action_mask, 0, perm, out=self._shuffled_mask)

        # Generate minibatches using contiguous slices (no random indexing per batch)
        start_idx = 0
        while start_idx < total_size:
            end_idx = min(start_idx + batch_size, total_size)
            actual_batch_size = end_idx - start_idx

            if actual_batch_size < batch_size:
                # Last smaller batch
                yield (
                    self._shuffled_sub_index[start_idx:end_idx],
                    self._shuffled_derived[start_idx:end_idx],
                    self._shuffled_mask[start_idx:end_idx],
                    self.flat_actions[start_idx:end_idx],
                    self.flat_values[start_idx:end_idx],
                    self.flat_log_probs[start_idx:end_idx],
                    self.flat_advantages[start_idx:end_idx],
                    self.flat_returns[start_idx:end_idx],
                )
            else:
                # Copy contiguous slice into pre-allocated batch tensors (stable addresses)
                self._batch_sub_index.copy_(self._shuffled_sub_index[start_idx:end_idx])
                self._batch_derived.copy_(self._shuffled_derived[start_idx:end_idx])
                self._batch_mask.copy_(self._shuffled_mask[start_idx:end_idx])
                self._batch_actions.copy_(self.flat_actions[start_idx:end_idx])
                self._batch_values.copy_(self.flat_values[start_idx:end_idx])
                self._batch_log_probs.copy_(self.flat_log_probs[start_idx:end_idx])
                self._batch_advantages.copy_(self.flat_advantages[start_idx:end_idx])
                self._batch_returns.copy_(self.flat_returns[start_idx:end_idx])

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

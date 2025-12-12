"""
Lightweight Evaluation-Only Environment for Ranking Metrics.

This is a simplified, high-performance environment specifically designed for
evaluating query corruptions to compute ranking metrics (MRR, Hits@K).

Key Optimizations over BatchedEnv:
    - No memory pruning (deterministic policy always takes argmax)
    - No negative sampling (handled externally)
    - No complex scheduling (simple sequential processing)
    - No debug tracing
    - Minimal state tracking
    - Pre-allocated buffers

Usage:
    from env_eval import EvalOnlyEnv
    
    eval_env = EvalOnlyEnv(
        unification_engine=engine,
        batch_size=1000,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        device=device,
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple

from tensordict import TensorDict


class EvalOnlyEnv:
    """
    Lightweight evaluation-only environment optimized for ranking metrics.
    
    This environment is a stripped-down version of BatchedEnv that:
    - Removes all training-specific overhead
    - Uses simple sequential episode processing
    - Pre-allocates all buffers for zero per-step allocation
    - Optimizes for throughput over flexibility
    
    Attributes:
        batch_size: Number of parallel environments
        unification_engine: The proof logic engine
        padding_atoms: Max atoms per state
        padding_states: Max successor states
        max_depth: Maximum proof depth
    """
    
    def __init__(
        self,
        unification_engine: Any,
        batch_size: int = 100,
        padding_atoms: int = 6,
        padding_states: int = 120,
        max_depth: int = 20,
        end_proof_action: bool = True,
        skip_unary_actions: bool = False,
        runtime_var_start_index: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.end_proof_action = end_proof_action
        self.skip_unary_actions = skip_unary_actions
        
        # Unification engine
        self.engine = unification_engine
        self.im = getattr(unification_engine, 'index_manager', None)
        
        # Key indices
        self.padding_idx = getattr(self.im, 'padding_idx', 0) if self.im else 0
        self.true_pred_idx = self.im.predicate_str2idx.get('True') if self.im else None
        self.false_pred_idx = self.im.predicate_str2idx.get('False') if self.im else None
        self.end_pred_idx = self.im.predicate_str2idx.get('Endf') if self.im else None
        
        # Runtime variable start index - critical for proper unification
        if runtime_var_start_index is not None:
            self.runtime_var_start_index = runtime_var_start_index
        elif self.im is not None:
            # Use constant_no + 1 as default (constants are 0..constant_no)
            self.runtime_var_start_index = getattr(self.im, 'constant_no', 0) + 1
        else:
            self.runtime_var_start_index = 1000  # Fallback
        
        # End tensor for end-proof action
        self.end_tensor = None
        if self.end_pred_idx is not None and self.end_pred_idx >= 0:
            self.end_tensor = torch.tensor(
                [[self.end_pred_idx, self.padding_idx, self.padding_idx]],
                dtype=torch.long,
                device=self.device,
            )
        
        # Pre-allocated buffers (lazily initialized per batch)
        self._current_states = None      # [B, A, D]
        self._original_queries = None    # [B, A, D] - for cycle prevention
        self._derived_states = None      # [B, S, A, D]
        self._derived_counts = None      # [B]
        self._depths = None              # [B]
        self._next_var_indices = None    # [B] - critical for variable tracking
        self._done = None                # [B]
        self._success = None             # [B]
        self._queries_dataset = None     # [M, A, D]
        self._query_ptr = 0              # Current position in dataset
        
    def set_queries(self, queries: Tensor) -> None:
        """
        Set the queries to evaluate.
        
        Args:
            queries: [M, A, D] or [M, 3] Query tensor
        """
        if queries.ndim == 2:
            # Pad [M, 3] to [M, A, 3]
            M = queries.shape[0]
            padded = torch.full((M, self.padding_atoms, 3), self.padding_idx, 
                               dtype=torch.long, device=self.device)
            padded[:, 0, :] = queries.to(self.device)
            queries = padded
        else:
            queries = queries.to(self.device)
            
        self._queries_dataset = queries
        self._query_ptr = 0
        
    def reset_batch(self, n_envs: Optional[int] = None) -> TensorDict:
        """
        Reset a batch of environments with the next n_envs queries.
        
        Args:
            n_envs: Number of environments to reset (default: batch_size)
            
        Returns:
            TensorDict with initial observations
        """
        n = n_envs or self.batch_size
        n = min(n, len(self._queries_dataset) - self._query_ptr)
        
        if n <= 0:
            return None  # No more queries
        
        # Get next batch of queries
        queries = self._queries_dataset[self._query_ptr:self._query_ptr + n]
        self._query_ptr += n
        
        # Initialize state buffers
        self._current_states = queries.clone()  # [n, A, D]
        self._original_queries = queries.clone()  # Keep original for cycle prevention
        self._depths = torch.zeros(n, dtype=torch.long, device=self.device)
        self._done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._success = torch.zeros(n, dtype=torch.bool, device=self.device)
        
        # Critical: Initialize variable indices from runtime_var_start_index
        self._next_var_indices = torch.full(
            (n,), self.runtime_var_start_index, 
            dtype=torch.long, device=self.device
        )
        
        # Compute initial derived states
        self._compute_derived_states()
        
        return self._make_obs()
    
    def step(self, actions: Tensor) -> Tuple[TensorDict, Tensor, Tensor, Tensor]:
        """
        Execute one step in all environments.
        
        Args:
            actions: [n] Action indices
            
        Returns:
            Tuple of (obs, rewards, dones, success)
        """
        n = len(actions)
        
        # Get selected next states
        batch_idx = torch.arange(n, device=self.device)
        next_states = self._derived_states[batch_idx, actions]  # [n, A, D]
        
        # Update current states for non-done envs
        active_mask = ~self._done
        self._current_states[active_mask] = next_states[active_mask]
        self._depths[active_mask] += 1
        
        # Check termination
        first_pred = next_states[:, 0, 0]  # Predicate of first atom
        
        is_true = (first_pred == self.true_pred_idx) if self.true_pred_idx is not None else torch.zeros_like(first_pred, dtype=torch.bool)
        is_false = (first_pred == self.false_pred_idx) if self.false_pred_idx is not None else torch.zeros_like(first_pred, dtype=torch.bool)
        is_end = (first_pred == self.end_pred_idx) if self.end_pred_idx is not None else torch.zeros_like(first_pred, dtype=torch.bool)
        is_depth_limit = (self._depths >= self.max_depth)
        
        # Update done/success
        newly_done = is_true | is_false | is_end | is_depth_limit
        self._done = self._done | newly_done
        self._success = self._success | is_true
        
        # Compute rewards
        rewards = torch.zeros(n, device=self.device)
        rewards[is_true & ~self._done] = 1.0  # Only reward first success
        
        # Compute derived states for non-done envs
        if (~self._done).any():
            self._compute_derived_states()
        
        return self._make_obs(), rewards, self._done.clone(), self._success.clone()
    
    def _compute_derived_states(self) -> None:
        """Compute successor states using unification engine."""
        n = self._current_states.shape[0]
        
        # Call unification engine with correct signature:
        # get_derived_states(current_states, next_var_indices, excluded_queries=None, verbose=0)
        # excluded_queries prevents cycles back to the original query
        # Shape should be [n, 1, 3] - just the first atom as a triple
        excluded = self._original_queries[:, 0:1, :]  # [n, 1, 3]
        
        result = self.engine.get_derived_states(
            self._current_states,
            self._next_var_indices,
            excluded_queries=excluded,
            verbose=0,
        )
        
        # Return is (derived_states, counts, new_next_var_indices)
        if len(result) == 3:
            derived, counts, new_var_indices = result
            # Update variable indices for next step
            self._next_var_indices = new_var_indices
        else:
            derived, counts = result
        
        # Derived shape is [B, K, M, 3] where K=successors, M=atoms
        # We need [B, S, A, 3] where S=padding_states, A=padding_atoms
        
        # First handle atoms dimension (M -> A)
        cur_atoms = derived.shape[2]
        if cur_atoms < self.padding_atoms:
            pad_a = torch.full(
                (n, derived.shape[1], self.padding_atoms - cur_atoms, 3),
                self.padding_idx, dtype=torch.long, device=self.device
            )
            derived = torch.cat([derived, pad_a], dim=2)
        elif cur_atoms > self.padding_atoms:
            derived = derived[:, :, :self.padding_atoms, :]
        
        # Then handle successors dimension (K -> S)
        cur_states = derived.shape[1]
        if cur_states < self.padding_states:
            pad_s = torch.full(
                (n, self.padding_states - cur_states, self.padding_atoms, 3),
                self.padding_idx, dtype=torch.long, device=self.device
            )
            derived = torch.cat([derived, pad_s], dim=1)
        elif cur_states > self.padding_states:
            derived = derived[:, :self.padding_states]
            counts = torch.clamp(counts, max=self.padding_states)
        
        # Add end action if enabled
        if self.end_proof_action and self.end_tensor is not None:
            derived, counts = self._add_end_action(derived, counts)
        
        self._derived_states = derived
        self._derived_counts = counts
        
    def _add_end_action(self, states: Tensor, counts: Tensor) -> Tuple[Tensor, Tensor]:
        """Add END action to available actions."""
        n = states.shape[0]
        
        # Check if current state is terminal
        first_pred = self._current_states[:, 0, 0]
        is_terminal = torch.zeros(n, dtype=torch.bool, device=self.device)
        if self.true_pred_idx is not None:
            is_terminal = is_terminal | (first_pred == self.true_pred_idx)
        if self.false_pred_idx is not None:
            is_terminal = is_terminal | (first_pred == self.false_pred_idx)
        
        # Add end action for non-terminal states with room
        can_add = (~is_terminal) & (counts < self.padding_states)
        
        if can_add.any():
            add_idx = can_add.nonzero(as_tuple=True)[0]
            slot_idx = counts[add_idx]
            
            end_state = torch.full((self.padding_atoms, 3), self.padding_idx, 
                                  dtype=torch.long, device=self.device)
            end_state[0] = self.end_tensor[0]
            
            for i, (env_i, slot_i) in enumerate(zip(add_idx, slot_idx)):
                states[env_i, slot_i] = end_state
            
            counts[add_idx] += 1
        
        return states, counts
    
    def _make_obs(self) -> TensorDict:
        """Construct observation tensordict."""
        n = self._current_states.shape[0]
        
        # Action mask: valid actions based on counts
        action_mask = torch.zeros(n, self.padding_states, dtype=torch.bool, device=self.device)
        for i in range(n):
            action_mask[i, :self._derived_counts[i]] = True
        
        return TensorDict({
            'sub_index': self._current_states.unsqueeze(1),  # [n, 1, A, D]
            'derived_sub_indices': self._derived_states,      # [n, S, A, D]
            'action_mask': action_mask,                       # [n, S]
        }, batch_size=[n], device=self.device)
    
    @property
    def remaining_queries(self) -> int:
        """Number of queries remaining to process."""
        if self._queries_dataset is None:
            return 0
        return len(self._queries_dataset) - self._query_ptr
    
    def reset_pointer(self) -> None:
        """Reset query pointer to beginning."""
        self._query_ptr = 0

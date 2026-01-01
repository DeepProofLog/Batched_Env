"""
Minimal Evaluation Environment.

Stripped-down env for pure evaluation benchmarking - no training overhead.
"""
import torch
from torch import Tensor
from typing import Tuple, Optional
from tensordict import TensorDict

from unification import UnificationEngineVectorized


class EnvEval:
    """Minimal evaluation environment - no training overhead.
    
    Key simplifications vs EnvVec:
    - No sampler/negative sampling  
    - No train/eval mode switching
    - No complex buffer allocation
    - No per_env_ptrs, neg_counters
    - Simpler state structure
    """
    
    def __init__(
        self,
        vec_engine: UnificationEngineVectorized,
        batch_size: int,
        padding_atoms: int,
        padding_states: int,
        max_depth: int,
        device: torch.device,
        runtime_var_start_index: int,
        end_proof_action: bool = True,
        memory_pruning: bool = True,
    ):
        self.engine = vec_engine
        self._batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.device = device
        self.runtime_var_start_index = runtime_var_start_index
        self.end_proof_action = end_proof_action
        self.memory_pruning = memory_pruning
        
        # Constants
        self.padding_idx = vec_engine.padding_idx
        self.max_history_size = 64
        
        # Pre-allocated buffers for static shapes
        B, S, A = batch_size, padding_states, padding_atoms
        self._arange_S = torch.arange(S, device=device)
        self._arange_B = torch.arange(B, device=device)
        self._ones_B = torch.ones(B, dtype=torch.long, device=device)
        
        # False state (empty state)
        self._false_state_base = torch.full((S, A, 3), self.padding_idx, dtype=torch.long, device=device)
        
        # End state for end proof action  
        self.end_state = torch.full((A, 3), self.padding_idx, dtype=torch.long, device=device)
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value
        # Re-allocate size-dependent buffers
        self._arange_B = torch.arange(value, device=self.device)
        self._ones_B = torch.ones(value, dtype=torch.long, device=self.device)
    
    def reset(self, queries: Tensor) -> Tuple[TensorDict, TensorDict]:
        """Initialize state from queries.
        
        Args:
            queries: [B, 3] tensor of queries
            
        Returns:
            (obs, state) tuple
        """
        device = self.device
        B = queries.shape[0]
        
        # Pad queries to (B, padding_atoms, 3)
        if queries.ndim == 2:
            padded = torch.full((B, self.padding_atoms, 3), self.padding_idx, dtype=torch.long, device=device)
            padded[:, 0, :] = queries.to(device)
            queries = padded
        else:
            queries = queries.to(device)
        
        # Initialize variables
        var_idx = torch.full((B,), self.runtime_var_start_index, dtype=torch.long, device=device)
        history = torch.zeros((B, self.max_history_size), dtype=torch.int64, device=device)
        history[:, 0] = self._compute_hash(queries)
        h_count = torch.ones(B, dtype=torch.long, device=device)
        
        # Compute initial derived states
        derived, counts, new_var = self._compute_derived(
            queries, var_idx, queries, history, h_count, queries[:, 0:1, :]
        )
        
        state = TensorDict({
            "current_states": queries,
            "derived_states": derived.clone(),
            "derived_counts": counts,
            "original_queries": queries,
            "next_var_indices": new_var,
            "depths": torch.zeros(B, dtype=torch.long, device=device),
            "done": torch.zeros(B, dtype=torch.uint8, device=device),
            "success": torch.zeros(B, dtype=torch.uint8, device=device),
            "current_labels": torch.ones(B, dtype=torch.long, device=device),
            "history_hashes": history,
            "history_count": h_count,
            "step_rewards": torch.zeros(B, dtype=torch.float32, device=device),
            "step_dones": torch.zeros(B, dtype=torch.uint8, device=device),
        }, batch_size=[B], device=device)
        
        return self._state_to_obs(state), state
    
    def step_core(self, state: TensorDict, actions: Tensor) -> Tuple[TensorDict, TensorDict]:
        """Single step without auto-reset - pure functional."""
        B, device = self._batch_size, self.device
        was_done = state['done'].bool()
        active = ~was_done
        
        # Select next states
        batch_idx = torch.arange(B, device=device)
        next_states = state['derived_states'][batch_idx, actions]
        new_current = torch.where(active.view(B, 1, 1), next_states, state['current_states'])
        new_depths = torch.where(active, state['depths'] + 1, state['depths'])
        
        # Compute reward/termination
        rewards, terminated, truncated, is_success = self._compute_reward(
            new_current, state['current_labels'], new_depths, B
        )
        newly_done = active & (terminated | truncated)
        new_done = was_done | newly_done
        new_success = state['success'].bool() | (active & is_success)
        rewards = torch.where(active, rewards, torch.zeros_like(rewards))
        
        # Update history
        write_pos = state['history_count'].clamp(max=self.max_history_size - 1)
        new_hash = self._compute_hash(new_current)
        update_val = torch.where(active, new_hash, state['history_hashes'][batch_idx, write_pos])
        new_history = state['history_hashes'].clone()
        new_history.scatter_(1, write_pos.unsqueeze(1), update_val.unsqueeze(1))
        new_h_count = torch.where(
            active, 
            (state['history_count'] + 1).clamp(max=self.max_history_size), 
            state['history_count']
        )
        
        # Compute derived for still-active
        still_active = ~new_done
        new_derived, new_counts, new_var = self._compute_derived(
            new_current, state['next_var_indices'], state['original_queries'], 
            new_history, new_h_count
        )
        new_derived = torch.where(still_active.view(B, 1, 1, 1), new_derived, state['derived_states'])
        new_counts = torch.where(still_active, new_counts, state['derived_counts'])
        new_var = torch.where(still_active, new_var, state['next_var_indices'])
        
        new_state = TensorDict({
            "current_states": new_current,
            "derived_states": new_derived,
            "derived_counts": new_counts,
            "original_queries": state['original_queries'],
            "next_var_indices": new_var,
            "depths": new_depths,
            "done": new_done.to(torch.uint8),
            "success": new_success.to(torch.uint8),
            "current_labels": state['current_labels'],
            "history_hashes": new_history,
            "history_count": new_h_count,
            "step_rewards": rewards,
            "step_dones": newly_done.to(torch.uint8),
        }, batch_size=[B], device=device)
        
        return self._state_to_obs(new_state), new_state
    
    def _state_to_obs(self, state: TensorDict) -> TensorDict:
        """Extract observation from state - matches EnvVec format."""
        B = state.batch_size[0]
        counts = state['derived_counts']
        action_mask = (self._arange_S < counts.unsqueeze(1)).to(torch.uint8)
        
        return TensorDict({
            'sub_index': state['current_states'].unsqueeze(1),  # [B, 1, A, 3]
            'derived_sub_indices': state['derived_states'],  # [B, S, A, 3]
            'action_mask': action_mask,  # [B, S]
        }, batch_size=[B], device=self.device)
    
    def _compute_hash(self, states: Tensor) -> Tensor:
        """Compute state hash for history tracking."""
        # Simple hash: sum of element products
        return states.view(states.size(0), -1).sum(dim=1)
    
    def _get_derived_raw(self, current_states, next_var_indices, excluded):
        """Get raw derived states from engine."""
        B, S, A = self._batch_size, self.padding_states, self.padding_atoms
        derived_raw, counts_raw, new_vars = self.engine.get_derived_states_compiled(
            current_states, next_var_indices, excluded
        )
        buf = torch.full((B, S, A, 3), self.padding_idx, dtype=torch.long, device=self.device)
        K, M = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
        buf[:, :K, :M, :] = derived_raw[:, :K, :M, :]
        return buf, counts_raw, new_vars
    
    def _compute_derived(
        self,
        current_states: Tensor,
        next_var_indices: Tensor,
        original_queries: Tensor,
        history_hashes: Optional[Tensor] = None,
        history_count: Optional[Tensor] = None,
        excluded_queries: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute derived states with static shapes."""
        B, S, A, pad = self._batch_size, self.padding_states, self.padding_atoms, self.padding_idx
        
        derived, raw_counts, new_var = self._get_derived_raw(current_states, next_var_indices, excluded_queries)
        
        within_count = self._arange_S.unsqueeze(0) < raw_counts.unsqueeze(1)
        valid_atom = derived[:, :, :, 0] != pad
        atom_counts = valid_atom.sum(dim=2)
        base_valid = within_count & (atom_counts <= A) & (atom_counts > 0)
        
        derived = torch.where(base_valid.unsqueeze(-1).unsqueeze(-1), derived, self._false_state_base.unsqueeze(0).expand(B, -1, -1, -1))
        new_counts = base_valid.sum(dim=1)
        
        # Compact
        flat_dim = A * 3
        target_pos = torch.cumsum(base_valid.long(), dim=1) - 1
        target_pos = torch.where(base_valid, target_pos.clamp(min=0, max=S-1), self._ones_B.unsqueeze(1) * (S-1))
        src = torch.where(base_valid.unsqueeze(-1), derived.reshape(B, S, flat_dim), torch.zeros(B, S, flat_dim, dtype=torch.long, device=self.device))
        compact = torch.full((B, S, flat_dim), pad, dtype=torch.long, device=self.device)
        compact.scatter_(1, target_pos.unsqueeze(-1).expand(B, S, flat_dim), src)
        derived = compact.view(B, S, A, 3)
        
        needs_false = new_counts == 0
        if self._false_state_base is not None:
            derived = torch.where(needs_false.view(-1,1,1,1), self._false_state_base.unsqueeze(0).expand(B,-1,-1,-1), derived)
            new_counts = torch.where(needs_false, self._ones_B, new_counts)
        
        if self.memory_pruning and history_hashes is not None:
            derived, new_counts = self._prune_visited(derived, new_counts, history_hashes, history_count)
            needs_false2 = new_counts == 0
            if self._false_state_base is not None:
                derived = torch.where(needs_false2.view(-1,1,1,1), self._false_state_base.unsqueeze(0).expand(B,-1,-1,-1), derived)
                new_counts = torch.where(needs_false2, self._ones_B, new_counts)
        
        if self.end_proof_action and self.end_state is not None:
            new_counts = new_counts.clamp(max=S-1)
            derived, new_counts = self._add_end_action(current_states, derived, new_counts)
        
        return derived, new_counts, new_var
    
    def _prune_visited(
        self, 
        derived: Tensor, 
        counts: Tensor, 
        history: Tensor, 
        h_count: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Remove visited states from derived."""
        B, S = derived.size(0), derived.size(1)
        
        # Compute hashes for derived states
        derived_hashes = derived.view(B, S, -1).sum(dim=2)  # [B, S]
        
        # Check against history
        history_valid_mask = torch.arange(history.size(1), device=self.device) < h_count.unsqueeze(1)
        history_masked = torch.where(history_valid_mask, history, torch.full_like(history, -1))
        
        # Mark visited
        is_visited = (derived_hashes.unsqueeze(2) == history_masked.unsqueeze(1)).any(dim=2)  # [B, S]
        
        # Clear visited states (set to padding)
        derived = torch.where(
            is_visited.unsqueeze(2).unsqueeze(3),
            self._false_state_base.unsqueeze(0).expand(B, -1, -1, -1),
            derived
        )
        
        # Recompute counts
        valid = (derived[:, :, 0, 0] != self.padding_idx)
        new_counts = valid.sum(dim=1)
        
        return derived, new_counts
    
    def _add_end_action(
        self,
        current: Tensor,
        states: Tensor,
        counts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Add END action slot."""
        B, S = current.size(0), self.padding_states
        
        # Check if terminal (first atom is padding)
        is_terminal = (current[:, 0, 0] == self.padding_idx)
        
        # Place end action at position counts (clamped)
        insert_pos = counts.clamp(max=S - 1)
        batch_idx = torch.arange(B, device=self.device)
        
        # Clone to avoid in-place modification
        new_states = states.clone()
        new_states[batch_idx, insert_pos] = self.end_state.unsqueeze(0).expand(B, -1, -1)
        
        # Increment counts for non-terminal
        new_counts = torch.where(~is_terminal, counts + 1, counts)
        new_counts = new_counts.clamp(max=S)
        
        return new_states, new_counts
    
    def _compute_reward(
        self,
        states: Tensor,
        labels: Tensor,
        depths: Tensor,
        B: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute reward and termination."""
        device = self.device
        
        # Check if terminal (first atom is padding = proof complete)
        is_terminal = (states[:, 0, 0] == self.padding_idx)
        
        # Success if terminal and label is positive (1)
        is_success = is_terminal & (labels == 1)
        
        # Truncate if max depth reached
        truncated = (depths >= self.max_depth)
        
        # Reward: +1 for success, -1 for failure at terminal, 0 otherwise
        rewards = torch.zeros(B, dtype=torch.float32, device=device)
        rewards = torch.where(is_success, torch.ones_like(rewards), rewards)
        rewards = torch.where(is_terminal & ~is_success, -torch.ones_like(rewards), rewards)
        
        return rewards, is_terminal, truncated, is_success

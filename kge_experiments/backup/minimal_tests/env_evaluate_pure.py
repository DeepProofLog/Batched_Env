"""
Minimal Evaluation Environment - Pure Functional Version.

Uses raw tensors with pure functional step for CUDA graph compatibility.
No in-place mutations inside compiled region.
"""
import torch
from torch import Tensor
from typing import Tuple, Optional


class EnvEvalPure:
    """Pure functional evaluation environment for CUDA graphs.
    
    Key design:
    - Step returns NEW tensors (no in-place mutations)
    - Persistent buffers exist but copying happens OUTSIDE compiled region
    - Compatible with zero-argument compiled step via closure
    """
    
    def __init__(
        self,
        vec_engine,
        batch_size: int,
        padding_atoms: int,
        padding_states: int,
        max_depth: int,
        device: torch.device,
        runtime_var_start_index: int,
        end_proof_action: bool = True,
    ):
        self.engine = vec_engine
        self._batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.device = device
        self.runtime_var_start_index = runtime_var_start_index
        self.end_proof_action = end_proof_action
        
        self.padding_idx = vec_engine.padding_idx
        self.max_history_size = 64
        
        B, S, A = batch_size, padding_states, padding_atoms
        self._arange_S = torch.arange(S, device=device)
        self._arange_B = torch.arange(B, device=device)
        self._ones_B = torch.ones(B, dtype=torch.long, device=device)
        self._false_state = torch.full((S, A, 3), self.padding_idx, dtype=torch.long, device=device)
        self.end_state = torch.full((A, 3), self.padding_idx, dtype=torch.long, device=device)
        
        # Persistent buffers for zero-copy between graph invocations
        self._current_states = torch.zeros(B, A, 3, dtype=torch.long, device=device)
        self._derived_states = torch.zeros(B, S, A, 3, dtype=torch.long, device=device)
        self._derived_counts = torch.zeros(B, dtype=torch.long, device=device)
        self._depths = torch.zeros(B, dtype=torch.long, device=device)
        self._done = torch.zeros(B, dtype=torch.bool, device=device)
        self._success = torch.zeros(B, dtype=torch.bool, device=device)
        self._step_rewards = torch.zeros(B, dtype=torch.float32, device=device)
        self._step_dones = torch.zeros(B, dtype=torch.bool, device=device)
        self._action_mask = torch.zeros(B, S, dtype=torch.uint8, device=device)
        self._original_queries = torch.zeros(B, A, 3, dtype=torch.long, device=device)
        self._next_var_indices = torch.full((B,), runtime_var_start_index, dtype=torch.long, device=device)
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value):
        if value != self._batch_size:
            self._batch_size = value
            self._reinit_buffers()
    
    def _reinit_buffers(self):
        B, S, A = self._batch_size, self.padding_states, self.padding_atoms
        device = self.device
        self._arange_B = torch.arange(B, device=device)
        self._ones_B = torch.ones(B, dtype=torch.long, device=device)
        self._current_states = torch.zeros(B, A, 3, dtype=torch.long, device=device)
        self._derived_states = torch.zeros(B, S, A, 3, dtype=torch.long, device=device)
        self._derived_counts = torch.zeros(B, dtype=torch.long, device=device)
        self._depths = torch.zeros(B, dtype=torch.long, device=device)
        self._done = torch.zeros(B, dtype=torch.bool, device=device)
        self._success = torch.zeros(B, dtype=torch.bool, device=device)
        self._step_rewards = torch.zeros(B, dtype=torch.float32, device=device)
        self._step_dones = torch.zeros(B, dtype=torch.bool, device=device)
        self._action_mask = torch.zeros(B, S, dtype=torch.uint8, device=device)
        self._original_queries = torch.zeros(B, A, 3, dtype=torch.long, device=device)
        self._next_var_indices = torch.full((B,), self.runtime_var_start_index, dtype=torch.long, device=device)
    
    def reset(self, queries: Tensor):
        """Reset - writes to persistent buffers (called OUTSIDE compiled region)."""
        B, A = queries.shape[0], self.padding_atoms
        device = self.device
        pad = self.padding_idx
        
        if queries.ndim == 2:
            padded = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
            padded[:, 0, :] = queries.to(device)
            queries = padded
        else:
            queries = queries.to(device)
        
        var_idx = torch.full((B,), self.runtime_var_start_index, dtype=torch.long, device=device)
        derived, counts, new_var = self._compute_derived(queries, var_idx, queries[:, 0:1, :])
        
        # Write to buffers
        self._current_states.copy_(queries)
        self._derived_states.copy_(derived)
        self._derived_counts.copy_(counts)
        self._original_queries.copy_(queries)
        self._next_var_indices.copy_(new_var)
        self._depths.zero_()
        self._done.zero_()
        self._success.zero_()
        self._step_rewards.zero_()
        self._step_dones.zero_()
        self._action_mask.copy_((self._arange_S < counts.unsqueeze(1)).to(torch.uint8))
    
    def step_pure(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pure functional step - returns NEW tensors, no mutations.
        
        Returns: (new_current, new_derived, new_counts, new_depths, new_done, 
                  new_success, step_rewards, step_dones, new_mask)
        """
        B = self._batch_size
        pad = self.padding_idx
        
        was_done = self._done
        active = ~was_done
        
        # Select next states
        next_states = self._derived_states[self._arange_B, actions]
        new_current = torch.where(active.view(B, 1, 1), next_states, self._current_states)
        new_depths = torch.where(active, self._depths + 1, self._depths)
        
        # Check termination
        is_terminal = (new_current[:, 0, 0] == pad)
        is_success = is_terminal & active
        truncated = (new_depths >= self.max_depth)
        newly_done = active & (is_terminal | truncated)
        new_done = was_done | newly_done
        new_success = self._success | (active & is_success)
        
        step_rewards = torch.where(is_success, torch.ones(B, device=self.device), torch.zeros(B, device=self.device))
        
        # Compute derived for still-active
        still_active = ~new_done
        new_derived, new_counts, new_var = self._compute_derived(new_current, self._next_var_indices, None)
        new_derived = torch.where(still_active.view(B, 1, 1, 1), new_derived, self._derived_states)
        new_counts = torch.where(still_active, new_counts, self._derived_counts)
        
        new_mask = (self._arange_S < new_counts.unsqueeze(1)).to(torch.uint8)
        
        return (new_current, new_derived, new_counts, new_depths, new_done, 
                new_success, step_rewards, newly_done, new_mask)
    
    def copy_step_results(self, results: Tuple):
        """Copy step results into persistent buffers (called OUTSIDE compiled region)."""
        (new_current, new_derived, new_counts, new_depths, new_done,
         new_success, step_rewards, step_dones, new_mask) = results
        
        self._current_states.copy_(new_current)
        self._derived_states.copy_(new_derived)
        self._derived_counts.copy_(new_counts)
        self._depths.copy_(new_depths)
        self._done.copy_(new_done)
        self._success.copy_(new_success)
        self._step_rewards.copy_(step_rewards)
        self._step_dones.copy_(step_dones)
        self._action_mask.copy_(new_mask)
    
    def _compute_derived(self, current, var_indices, excluded):
        """Compute derived states."""
        B, S, A, pad = self._batch_size, self.padding_states, self.padding_atoms, self.padding_idx
        
        derived_raw, counts_raw, new_vars = self.engine.get_derived_states_compiled(
            current, var_indices, excluded
        )
        
        buf = torch.full((B, S, A, 3), pad, dtype=torch.long, device=self.device)
        K, M = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
        buf[:, :K, :M, :] = derived_raw[:, :K, :M, :]
        
        within_count = self._arange_S.unsqueeze(0) < counts_raw.unsqueeze(1)
        valid = buf[:, :, :, 0] != pad
        atom_counts = valid.sum(dim=2)
        base_valid = within_count & (atom_counts <= A) & (atom_counts > 0)
        
        buf = torch.where(base_valid.unsqueeze(-1).unsqueeze(-1), buf,
                         self._false_state.unsqueeze(0).expand(B, -1, -1, -1))
        new_counts = base_valid.sum(dim=1)
        
        needs_false = new_counts == 0
        buf = torch.where(needs_false.view(-1,1,1,1), 
                         self._false_state.unsqueeze(0).expand(B,-1,-1,-1), buf)
        new_counts = torch.where(needs_false, self._ones_B, new_counts)
        
        if self.end_proof_action:
            new_counts = new_counts.clamp(max=S-1)
            idx = new_counts.clamp(max=S-1)
            buf[self._arange_B, idx] = self.end_state.unsqueeze(0).expand(B, -1, -1)
            is_terminal = (current[:, 0, 0] == pad)
            new_counts = torch.where(~is_terminal, new_counts + 1, new_counts).clamp(max=S)
        
        return buf, new_counts, new_vars

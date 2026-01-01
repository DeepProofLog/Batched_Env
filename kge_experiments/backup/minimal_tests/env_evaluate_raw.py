"""
Minimal Evaluation Environment - Raw Tensor Version.

Uses raw tensors instead of TensorDict for zero-overhead evaluation.
Designed for zero-argument compiled step pattern.
"""
import torch
from torch import Tensor
from typing import Tuple, Optional, NamedTuple

from unification import UnificationEngineVectorized


class EvalState(NamedTuple):
    """Raw tensor state for evaluation - no TensorDict overhead."""
    current_states: Tensor       # [B, A, 3]
    derived_states: Tensor       # [B, S, A, 3]
    derived_counts: Tensor       # [B]
    original_queries: Tensor     # [B, A, 3]
    next_var_indices: Tensor     # [B]
    depths: Tensor               # [B]
    done: Tensor                 # [B] bool
    success: Tensor              # [B] bool
    history_hashes: Tensor       # [B, H]
    history_count: Tensor        # [B]
    step_rewards: Tensor         # [B]
    step_dones: Tensor           # [B] bool


class EvalObs(NamedTuple):
    """Raw tensor observation for evaluation."""
    sub_index: Tensor            # [B, 1, A, 3]
    derived_sub_indices: Tensor  # [B, S, A, 3]
    action_mask: Tensor          # [B, S]


class EnvEvalRaw:
    """Minimal evaluation environment using raw tensors.
    
    Key optimizations:
    - Raw tensors instead of TensorDict (no dict overhead)
    - Persistent buffers updated in-place
    - Designed for zero-argument compiled step
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
        
        # Pre-allocated buffers
        B, S, A = batch_size, padding_states, padding_atoms
        self._arange_S = torch.arange(S, device=device)
        self._arange_B = torch.arange(B, device=device)
        self._ones_B = torch.ones(B, dtype=torch.long, device=device)
        self._false_state_base = torch.full((S, A, 3), self.padding_idx, dtype=torch.long, device=device)
        self.end_state = torch.full((A, 3), self.padding_idx, dtype=torch.long, device=device)
        
        # Persistent state buffers (updated in-place for zero-arg step)
        self._obs = None
        self._state = None
        self._init_persistent_buffers()
    
    def _init_persistent_buffers(self):
        """Allocate persistent buffers that will be updated in-place."""
        B, S, A, H = self._batch_size, self.padding_states, self.padding_atoms, self.max_history_size
        device = self.device
        pad = self.padding_idx
        
        # Observation buffers
        self._obs = EvalObs(
            sub_index=torch.full((B, 1, A, 3), pad, dtype=torch.long, device=device),
            derived_sub_indices=torch.full((B, S, A, 3), pad, dtype=torch.long, device=device),
            action_mask=torch.zeros(B, S, dtype=torch.uint8, device=device),
        )
        
        # State buffers  
        self._state = EvalState(
            current_states=torch.full((B, A, 3), pad, dtype=torch.long, device=device),
            derived_states=torch.full((B, S, A, 3), pad, dtype=torch.long, device=device),
            derived_counts=torch.zeros(B, dtype=torch.long, device=device),
            original_queries=torch.full((B, A, 3), pad, dtype=torch.long, device=device),
            next_var_indices=torch.full((B,), self.runtime_var_start_index, dtype=torch.long, device=device),
            depths=torch.zeros(B, dtype=torch.long, device=device),
            done=torch.zeros(B, dtype=torch.bool, device=device),
            success=torch.zeros(B, dtype=torch.bool, device=device),
            history_hashes=torch.zeros(B, H, dtype=torch.int64, device=device),
            history_count=torch.zeros(B, dtype=torch.long, device=device),
            step_rewards=torch.zeros(B, dtype=torch.float32, device=device),
            step_dones=torch.zeros(B, dtype=torch.bool, device=device),
        )
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @batch_size.setter  
    def batch_size(self, value: int):
        if value != self._batch_size:
            self._batch_size = value
            self._arange_B = torch.arange(value, device=self.device)
            self._ones_B = torch.ones(value, dtype=torch.long, device=self.device)
            self._init_persistent_buffers()
    
    def reset_into_buffers(self, queries: Tensor):
        """Reset environment - writes directly into persistent buffers.
        
        This is called OUTSIDE the compiled step.
        """
        device = self.device
        B = queries.shape[0]
        pad = self.padding_idx
        
        # Pad queries to (B, A, 3)
        if queries.ndim == 2:
            padded = torch.full((B, self.padding_atoms, 3), pad, dtype=torch.long, device=device)
            padded[:, 0, :] = queries.to(device)
            queries = padded
        else:
            queries = queries.to(device)
        
        # Initialize state buffers
        var_idx = torch.full((B,), self.runtime_var_start_index, dtype=torch.long, device=device)
        
        # Compute initial derived states
        derived, counts, new_var = self._compute_derived_external(
            queries, var_idx, queries[:, 0:1, :]
        )
        
        # Write to persistent state
        self._state.current_states.copy_(queries)
        self._state.derived_states.copy_(derived)
        self._state.derived_counts.copy_(counts)
        self._state.original_queries.copy_(queries)
        self._state.next_var_indices.copy_(new_var)
        self._state.depths.zero_()
        self._state.done.zero_()
        self._state.success.zero_()
        self._state.history_hashes.zero_()
        self._state.history_hashes[:, 0] = self._compute_hash(queries)
        self._state.history_count.fill_(1)
        self._state.step_rewards.zero_()
        self._state.step_dones.zero_()
        
        # Write to persistent obs
        self._update_obs_from_state()
    
    def _update_obs_from_state(self):
        """Update observation buffers from current state."""
        self._obs.sub_index[:, 0].copy_(self._state.current_states)
        self._obs.derived_sub_indices.copy_(self._state.derived_states)
        self._obs.action_mask.copy_(
            (self._arange_S < self._state.derived_counts.unsqueeze(1)).to(torch.uint8)
        )
    
    def step_inplace(self, actions: Tensor):
        """Single step - updates persistent buffers IN-PLACE.
        
        This is designed to be called from a zero-argument compiled step
        that captures self._obs and self._state via closure.
        """
        B, device = self._batch_size, self.device
        pad = self.padding_idx
        
        was_done = self._state.done
        active = ~was_done
        
        # Select next states
        batch_idx = self._arange_B
        next_states = self._state.derived_states[batch_idx, actions]
        new_current = torch.where(active.view(B, 1, 1), next_states, self._state.current_states)
        new_depths = torch.where(active, self._state.depths + 1, self._state.depths)
        
        # Compute reward/termination
        is_terminal = (new_current[:, 0, 0] == pad)
        is_success = is_terminal & active  # Success if terminal while active
        truncated = (new_depths >= self.max_depth)
        newly_done = active & (is_terminal | truncated)
        new_done = was_done | newly_done
        new_success = self._state.success | (active & is_success)
        
        rewards = torch.zeros(B, dtype=torch.float32, device=device)
        rewards = torch.where(is_success, torch.ones_like(rewards), rewards)
        
        # Update history (simplified - just track hash)
        write_pos = self._state.history_count.clamp(max=self.max_history_size - 1)
        new_hash = self._compute_hash(new_current)
        
        # Update state buffers in-place
        self._state.current_states.copy_(new_current)
        self._state.depths.copy_(new_depths)
        self._state.done.copy_(new_done)
        self._state.success.copy_(new_success)
        self._state.step_rewards.copy_(rewards)
        self._state.step_dones.copy_(newly_done)
        self._state.history_hashes.scatter_(1, write_pos.unsqueeze(1), new_hash.unsqueeze(1))
        self._state.history_count.copy_(
            torch.where(active, (self._state.history_count + 1).clamp(max=self.max_history_size), 
                       self._state.history_count)
        )
        
        # Compute derived for still-active (always compute, mask with where)
        still_active = ~new_done
        new_derived, new_counts, new_var = self._compute_derived_compiled(
            new_current, self._state.next_var_indices, 
            self._state.history_hashes, self._state.history_count
        )
        self._state.derived_states.copy_(
            torch.where(still_active.view(B, 1, 1, 1), new_derived, self._state.derived_states)
        )
        self._state.derived_counts.copy_(
            torch.where(still_active, new_counts, self._state.derived_counts)
        )
        self._state.next_var_indices.copy_(
            torch.where(still_active, new_var, self._state.next_var_indices)
        )
        
        # Update observation
        self._update_obs_from_state()
    
    def _compute_hash(self, states: Tensor) -> Tensor:
        """Compute state hash."""
        return states.view(states.size(0), -1).sum(dim=1)
    
    def _compute_derived_external(self, current_states, var_indices, excluded=None):
        """Compute derived - called OUTSIDE compiled region."""
        return self._compute_derived_impl(current_states, var_indices, None, None, excluded)
    
    def _compute_derived_compiled(self, current_states, var_indices, history=None, h_count=None):
        """Compute derived - called INSIDE compiled region."""
        return self._compute_derived_impl(current_states, var_indices, history, h_count, None)
    
    def _compute_derived_impl(self, current_states, var_indices, history, h_count, excluded):
        """Actual derived computation."""
        B, S, A, pad = self._batch_size, self.padding_states, self.padding_atoms, self.padding_idx
        
        derived_raw, counts_raw, new_vars = self.engine.get_derived_states_compiled(
            current_states, var_indices, excluded
        )
        
        # Ensure static shape
        buf = torch.full((B, S, A, 3), pad, dtype=torch.long, device=self.device)
        K, M = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
        buf[:, :K, :M, :] = derived_raw[:, :K, :M, :]
        
        within_count = self._arange_S.unsqueeze(0) < counts_raw.unsqueeze(1)
        valid_atom = buf[:, :, :, 0] != pad
        atom_counts = valid_atom.sum(dim=2)
        base_valid = within_count & (atom_counts <= A) & (atom_counts > 0)
        
        buf = torch.where(base_valid.unsqueeze(-1).unsqueeze(-1), buf, 
                         self._false_state_base.unsqueeze(0).expand(B, -1, -1, -1))
        new_counts = base_valid.sum(dim=1)
        
        # Ensure at least one action
        needs_false = new_counts == 0
        buf = torch.where(needs_false.view(-1,1,1,1), 
                         self._false_state_base.unsqueeze(0).expand(B,-1,-1,-1), buf)
        new_counts = torch.where(needs_false, self._ones_B, new_counts)
        
        # Add end action
        if self.end_proof_action:
            new_counts = new_counts.clamp(max=S-1)
            insert_pos = new_counts.clamp(max=S - 1)
            buf[self._arange_B, insert_pos] = self.end_state.unsqueeze(0).expand(B, -1, -1)
            is_terminal = (current_states[:, 0, 0] == pad)
            new_counts = torch.where(~is_terminal, new_counts + 1, new_counts)
            new_counts = new_counts.clamp(max=S)
        
        return buf, new_counts, new_vars

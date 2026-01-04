"""
Optimal Vectorized Knowledge Graph Environment.

Optimized implementation combining best practices from deep dive documents:
- TensorDict state management
- Buffer-copy pattern for CUDA graph stability
- Separate _step_core (eval) and _step_and_reset_core (train)
- torch.compile with mode='reduce-overhead', fullgraph=True

Public API:
    compile() - Compile step functions (required before use)
    train()   - Switch to training mode
    eval()    - Switch to evaluation mode  
    reset()   - Initialize environment state
    step()    - Take one step (with optional auto-reset)
"""
from __future__ import annotations

import torch
from typing import Any, Optional, Tuple
from tensordict import TensorDict

from unification import UnificationEngineVectorized

Tensor = torch.Tensor
EnvObs = TensorDict
EnvState = TensorDict


class EnvOptimal:
    """Optimal vectorized KG reasoning environment for CUDA graphs."""

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        vec_engine: UnificationEngineVectorized,
        batch_size: int = 100,
        padding_atoms: int = 6,
        padding_states: int = 120,
        max_depth: int = 20,
        end_proof_action: bool = True,
        runtime_var_start_index: Optional[int] = None,
        device: Optional[torch.device] = None,
        memory_pruning: bool = True,
        train_queries: Optional[Tensor] = None,
        valid_queries: Optional[Tensor] = None,
        sampler: Optional[Any] = None,
        negative_ratio: float = 1.0,
        corruption_scheme: Tuple[str, ...] = ('head', 'tail'),
        reward_type: int = 0,
    ) -> None:
        # Device & dimensions
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.end_proof_action = end_proof_action
        self.memory_pruning = memory_pruning
        self.max_history_size = max_depth + 1

        # Training config
        self.sampler = sampler
        self.default_negative_ratio = float(negative_ratio)
        self.negative_ratio = float(negative_ratio)
        self.default_order = False
        self.order = False
        self.corruption_scheme = corruption_scheme
        self.reward_type = reward_type
        self.sample_deterministic_per_env = False
        self.rejection_weight = 1.0 / negative_ratio if negative_ratio > 0 else 1.0

        # Pre-allocated reward scalars
        self._reward_pos = torch.tensor(1.0, device=self.device)
        self._reward_neg = torch.tensor(-1.0, device=self.device)
        self._reward_zero = torch.tensor(0.0, device=self.device)
        self._reward_neg_half = torch.tensor(-0.5, device=self.device)
        self._reward_neg_1_5 = torch.tensor(-1.5, device=self.device)

        # Hash constants
        self._hash_mix_const = 0x9E3779B97F4A7C15
        self._hash_mask63 = (1 << 63) - 1

        # Engine
        self.engine = vec_engine
        self.padding_idx = vec_engine.padding_idx
        self.true_pred_idx = vec_engine.true_pred_idx
        self.false_pred_idx = vec_engine.false_pred_idx
        self.end_pred_idx = vec_engine.end_pred_idx
        self.runtime_var_start_index = runtime_var_start_index or (vec_engine.constant_no + 1)
        self._hash_pack_base = getattr(vec_engine, 'pack_base', vec_engine.constant_no + 1001)

        # Pre-built special states
        self.end_state = None
        if self.end_pred_idx is not None and self.end_pred_idx >= 0:
            self.end_state = torch.full((padding_atoms, 3), self.padding_idx, dtype=torch.long, device=self.device)
            self.end_state[0, 0] = self.end_pred_idx

        self._false_state_base = None
        if self.false_pred_idx is not None:
            self._false_state_base = torch.full((padding_states, padding_atoms, 3), self.padding_idx, dtype=torch.long, device=self.device)
            self._false_state_base[0, 0, 0] = self.false_pred_idx

        # Index tensors
        B, S, A = batch_size, padding_states, padding_atoms
        self._positions_S = torch.arange(S, device=self.device).unsqueeze(0)
        self._arange_S = torch.arange(S, device=self.device)
        self._arange_A = torch.arange(A, device=self.device)
        self._arange_B = torch.arange(B, device=self.device)
        self._ones_B = torch.ones(B, dtype=torch.long, device=self.device)
        self._reset_ones_B = torch.ones(B, dtype=torch.long, device=self.device)
        self._reset_zeros_B = torch.zeros(B, dtype=torch.long, device=self.device)
        self._reset_mask_true = torch.ones(B, dtype=torch.bool, device=self.device)

        # Query pools
        self.train_queries = train_queries.to(self.device) if train_queries is not None else None
        self.valid_queries = valid_queries.to(self.device) if valid_queries is not None else None
        self._query_pool = None
        self._per_env_ptrs = None

        # Compiled function placeholders
        self._reset_fn = None
        self._step_fn = None
        self._step_and_reset_fn = None
        
        # State buffer for CUDA graph stability
        self._state_buffer = None
        self._allocate_buffers()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def train(self) -> None:
        """Switch to training mode."""
        if self.train_queries is None:
            raise ValueError("No train_queries provided")
        pool = self.train_queries
        self._set_queries_internal(pool)
        self.negative_ratio = self.default_negative_ratio
        self.order = self.default_order
        self.rejection_weight = 1.0 / self.negative_ratio if self.negative_ratio > 0 else 1.0

    def eval(self, queries: Optional[Tensor] = None) -> None:
        """Switch to evaluation mode."""
        pool = queries if queries is not None else self.valid_queries
        if pool is None:
            raise ValueError("No queries for eval")
        self._set_queries_internal(pool)
        self.negative_ratio = 0.0
        self.order = True
        self.rejection_weight = 1.0

    def compile(self, mode: str = 'reduce-overhead', fullgraph: bool = True) -> None:
        """Compile step functions for CUDA graph optimization."""
        self._reset_fn = torch.compile(self._reset_from_queries, mode=mode, fullgraph=fullgraph, dynamic=False)
        self._step_fn = torch.compile(self._step_core, mode=mode, fullgraph=fullgraph, dynamic=False)
        self._step_and_reset_fn = torch.compile(self._step_and_reset_core, mode=mode, fullgraph=fullgraph, dynamic=False)
        print(f"[EnvOptimal] Compiled with mode={mode}, fullgraph={fullgraph}")

    def reset(self, queries: Optional[Tensor] = None) -> Tuple[EnvObs, EnvState]:
        """Initialize environment state."""
        B, device = self.batch_size, self.device

        if queries is not None:
            torch.compiler.cudagraph_mark_step_begin()
            state = self._reset_fn(queries.to(device), torch.ones(B, dtype=torch.long, device=device))
            return self._state_to_obs(state), state

        if self._query_pool is None:
            raise RuntimeError("Call train()/eval() first")

        pool_size = self._query_pool.shape[0]
        indices = self._per_env_ptrs % pool_size
        init_q = self._query_pool[indices]
        init_labels = self._reset_ones_B

        init_q, init_labels, new_counters = self._sample_negatives(init_q, init_labels, self._reset_mask_true, self._reset_zeros_B)
        self._per_env_ptrs = (self._per_env_ptrs + 1) % pool_size

        torch.compiler.cudagraph_mark_step_begin()
        state = self._reset_fn(init_q, init_labels)
        state['per_env_ptrs'].copy_(self._per_env_ptrs)
        state['neg_counters'].copy_(new_counters)

        return self._state_to_obs(state), state

    def step(self, state: EnvState, actions: Tensor, auto_reset: bool = True) -> Tuple[EnvObs, EnvState]:
        """Take one step."""
        torch.compiler.cudagraph_mark_step_begin()

        if auto_reset and self._query_pool is not None:
            self._copy_state_to_buffer(state)
            new_obs, new_state = self._step_and_reset_fn(self._state_buffer, actions, self._query_pool, self._per_env_ptrs)
            new_obs = self._clone_obs(new_obs)
            self._copy_state_from_buffer(state, new_state)
            return new_obs, state
        else:
            return self._step_fn(state, actions)

    # =========================================================================
    # STEP LOGIC
    # =========================================================================

    def _reset_from_queries(self, queries: Tensor, labels: Tensor) -> EnvState:
        """Create initial state from queries."""
        device = self.device
        B, A, S, H, pad = self.batch_size, self.padding_atoms, self.padding_states, self.max_history_size, self.padding_idx

        current = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
        current[:, 0, :] = queries

        derived, counts, next_var = self._compute_derived(current, torch.zeros(B, dtype=torch.long, device=device), current.clone())
        h_hashes = torch.zeros(B, H, dtype=torch.long, device=device)
        h_count = torch.zeros(B, dtype=torch.long, device=device)

        return TensorDict({
            "current_states": current,
            "derived_states": derived,
            "derived_counts": counts,
            "original_queries": current.clone(),
            "next_var_indices": next_var,
            "depths": torch.zeros(B, dtype=torch.long, device=device),
            "done": torch.zeros(B, dtype=torch.uint8, device=device),
            "success": torch.zeros(B, dtype=torch.uint8, device=device),
            "current_labels": labels,
            "history_hashes": h_hashes,
            "history_count": h_count,
            "step_rewards": torch.zeros(B, dtype=torch.float32, device=device),
            "step_dones": torch.zeros(B, dtype=torch.uint8, device=device),
            "cumulative_rewards": torch.zeros(B, dtype=torch.float32, device=device),
            "per_env_ptrs": torch.zeros(B, dtype=torch.long, device=device),
            "neg_counters": torch.zeros(B, dtype=torch.int64, device=device),
        }, batch_size=[B], device=device)

    def _step_core(self, state: EnvState, actions: Tensor) -> Tuple[EnvObs, EnvState]:
        """Single step without auto-reset (for evaluation)."""
        B, device = self.batch_size, self.device
        was_done = state['done'].bool()
        active = ~was_done

        batch_idx = self._arange_B
        next_states = state['derived_states'][batch_idx, actions]
        new_current = torch.where(active.view(B, 1, 1), next_states, state['current_states'])
        new_depths = torch.where(active, state['depths'] + 1, state['depths'])

        rewards, terminated, truncated, is_success = self._compute_reward(new_current, state['current_labels'], new_depths, B)
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
        new_h_count = torch.where(active, (state['history_count'] + 1).clamp(max=self.max_history_size), state['history_count'])

        still_active = ~new_done
        new_derived, new_counts, new_var = self._compute_derived(new_current, state['next_var_indices'], state['original_queries'], new_history, new_h_count)
        new_derived = torch.where(still_active.view(B, 1, 1, 1), new_derived, state['derived_states'])
        new_counts = torch.where(still_active, new_counts, state['derived_counts'])
        new_var = torch.where(still_active, new_var, state['next_var_indices'])
        new_cumulative = torch.where(active, state['cumulative_rewards'] + rewards, state['cumulative_rewards'])

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
            "cumulative_rewards": new_cumulative,
            "per_env_ptrs": state['per_env_ptrs'],
            "neg_counters": state['neg_counters'],
        }, batch_size=[B], device=device)

        return self._state_to_obs(new_state), new_state

    def _step_and_reset_core(self, state: EnvState, actions: Tensor, query_pool: Tensor, per_env_ptrs: Tensor) -> Tuple[EnvObs, EnvState]:
        """Fused step + reset (for training with auto-reset)."""
        _, next_state = self._step_core(state, actions)
        rewards = next_state['step_rewards']
        done_mask = next_state['step_dones'].bool()
        B, device = state['current_states'].shape[0], self.device
        pool_size = query_pool.shape[0]

        if self.order:
            safe_idx = per_env_ptrs % pool_size
            next_ptrs = (per_env_ptrs + 1) % pool_size
        else:
            safe_idx = torch.randint(0, pool_size, (B,), device=device)
            next_ptrs = per_env_ptrs

        reset_q = query_pool[safe_idx]
        new_ptrs = torch.where(done_mask, next_ptrs, per_env_ptrs)

        padding = torch.full((3,), self.padding_idx, dtype=torch.long, device=device)
        reset_q = torch.where(done_mask.unsqueeze(-1).expand(-1, 3), reset_q, padding)

        labels = torch.ones(B, dtype=torch.long, device=device)
        reset_q, labels, new_counters = self._sample_negatives(reset_q, labels, done_mask, state['neg_counters'])
        reset_state = self._reset_from_queries(reset_q, labels)

        m_A3 = done_mask.view(-1, 1, 1).expand(-1, self.padding_atoms, 3)
        m_SA3 = done_mask.view(-1, 1, 1, 1).expand(-1, self.padding_states, self.padding_atoms, 3)
        m_H = done_mask.view(-1, 1).expand(-1, self.max_history_size)

        mixed = TensorDict({
            "current_states": torch.where(m_A3, reset_state['current_states'], next_state['current_states']),
            "derived_states": torch.where(m_SA3, reset_state['derived_states'], next_state['derived_states']),
            "derived_counts": torch.where(done_mask, reset_state['derived_counts'], next_state['derived_counts']),
            "original_queries": torch.where(m_A3, reset_state['original_queries'], next_state['original_queries']),
            "next_var_indices": torch.where(done_mask, reset_state['next_var_indices'], next_state['next_var_indices']),
            "depths": torch.where(done_mask, reset_state['depths'], next_state['depths']),
            "done": torch.where(done_mask, reset_state['done'], next_state['done']),
            "success": torch.where(done_mask, reset_state['success'], next_state['success']),
            "current_labels": torch.where(done_mask, reset_state['current_labels'], next_state['current_labels']),
            "history_hashes": torch.where(m_H, reset_state['history_hashes'], next_state['history_hashes']),
            "history_count": torch.where(done_mask, reset_state['history_count'], next_state['history_count']),
            "step_rewards": rewards,
            "step_dones": done_mask.to(torch.uint8),
            "cumulative_rewards": torch.where(done_mask, reset_state['cumulative_rewards'], next_state['cumulative_rewards']),
            "per_env_ptrs": new_ptrs,
            "neg_counters": new_counters,
        }, batch_size=[B], device=device)

        return self._state_to_obs(mixed), mixed

    # =========================================================================
    # UNIFICATION ENGINE
    # =========================================================================

    def _get_derived_raw(self, current_states, next_var_indices, excluded):
        """Get raw derived states from engine."""
        B, S, A = self.batch_size, self.padding_states, self.padding_atoms
        derived_raw, counts_raw, new_vars = self.engine.get_derived_states_compiled(current_states, next_var_indices, excluded)
        buf = torch.full((B, S, A, 3), self.padding_idx, dtype=torch.long, device=self.device)
        K, M = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
        buf[:, :K, :M, :] = derived_raw[:, :K, :M, :]
        return buf, counts_raw, new_vars

    def get_derived_simple(self, current_states: Tensor, history_hashes: Optional[Tensor] = None, 
                            history_count: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Get derived states for V10 evaluation (full logic including memory pruning).
        
        This uses the same logic as _compute_derived:
        - Full validation and compaction
        - Memory pruning (when history provided)
        - END action (if end_proof_action is True)
        
        Args:
            current_states: [B, A, 3] current states
            history_hashes: Optional [B, H] history hashes for memory pruning
            history_count: Optional [B] count of valid history entries
            
        Returns:
            derived: [B, S, A, 3] derived states
            counts: [B] number of valid derived states per batch
        """
        B = current_states.shape[0]
        S, A = self.padding_states, self.padding_atoms
        pad = self.padding_idx
        device = self.device
        
        # Get raw derived from engine
        next_var = torch.zeros(B, dtype=torch.long, device=device)
        derived_raw, counts_raw, _ = self.engine.get_derived_states_compiled(current_states, next_var, None)
        
        # Pad to fixed size
        derived = torch.full((B, S, A, 3), pad, dtype=torch.long, device=device)
        K, M = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
        derived[:, :K, :M, :] = derived_raw[:, :K, :M, :]
        
        # Validate: check within count and valid atoms
        arange_S = torch.arange(S, device=device)
        ones_B = torch.ones(B, dtype=torch.long, device=device)
        within_count = arange_S.unsqueeze(0) < counts_raw.unsqueeze(1)
        valid_atom = derived[:, :, :, 0] != pad
        atom_counts = valid_atom.sum(dim=2)
        base_valid = within_count & (atom_counts <= A) & (atom_counts > 0)
        
        # Apply false state for invalid positions
        if self._false_state_base is not None:
            derived = torch.where(base_valid.unsqueeze(-1).unsqueeze(-1), derived, 
                                  self._false_state_base.unsqueeze(0).expand(B, -1, -1, -1))
        new_counts = base_valid.sum(dim=1)
        
        # Compact valid states to front
        flat_dim = A * 3
        target_pos = torch.cumsum(base_valid.long(), dim=1) - 1
        target_pos = torch.where(base_valid, target_pos.clamp(min=0, max=S-1), ones_B.unsqueeze(1) * (S-1))
        src = torch.where(base_valid.unsqueeze(-1), derived.reshape(B, S, flat_dim), 
                          torch.zeros(B, S, flat_dim, dtype=torch.long, device=device))
        compact = torch.full((B, S, flat_dim), pad, dtype=torch.long, device=device)
        compact.scatter_(1, target_pos.unsqueeze(-1).expand(B, S, flat_dim), src)
        derived = compact.view(B, S, A, 3)
        
        # Handle zero counts - set to false state
        needs_false = new_counts == 0
        if self._false_state_base is not None:
            derived = torch.where(needs_false.view(-1,1,1,1), 
                                  self._false_state_base.unsqueeze(0).expand(B,-1,-1,-1), derived)
            new_counts = torch.where(needs_false, ones_B, new_counts)
        
        # Memory pruning (same as _compute_derived)
        if self.memory_pruning and history_hashes is not None and history_count is not None:
            derived, new_counts = self._prune_visited(derived, new_counts, history_hashes, history_count)
            needs_false2 = new_counts == 0
            if self._false_state_base is not None:
                derived = torch.where(needs_false2.view(-1,1,1,1), 
                                      self._false_state_base.unsqueeze(0).expand(B,-1,-1,-1), derived)
                new_counts = torch.where(needs_false2, ones_B, new_counts)
        
        # Add END action (same as _compute_derived)
        if self.end_proof_action and self.end_state is not None:
            new_counts = new_counts.clamp(max=S-1)
            safe_pos = new_counts.clamp(max=S-1)
            idx = safe_pos.view(B, 1, 1, 1).expand(B, 1, A, 3)
            end_exp = self.end_state.unsqueeze(0).expand(B, 1, -1, -1)
            derived = derived.scatter(1, idx, end_exp)
            new_counts = (new_counts + 1).clamp(max=S)
        
        return derived, new_counts

    def _compute_derived(self, current_states, next_var_indices, original_queries, history_hashes=None, history_count=None, excluded_queries=None):
        """Compute derived states with static shapes."""
        B, S, A, pad = self.batch_size, self.padding_states, self.padding_atoms, self.padding_idx

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

    # =========================================================================
    # HASH & PRUNING
    # =========================================================================

    def _compute_hash(self, states: Tensor) -> Tensor:
        """Compute state hash."""
        if states.dim() == 4:
            B, K, A, D = states.shape
            return self._compute_hash(states.view(B*K, A, D)).view(B, K)
        B, A, _ = states.shape
        s = states.long()
        preds = s[:, :, 0]
        valid = preds != self.padding_idx
        if self.true_pred_idx is not None: valid = valid & (preds != self.true_pred_idx)
        if self.false_pred_idx is not None: valid = valid & (preds != self.false_pred_idx)
        if self.end_pred_idx is not None: valid = valid & (preds != self.end_pred_idx)
        base = self._hash_pack_base
        packed = ((s[:,:,0] * base + s[:,:,1]) * base + s[:,:,2]) & self._hash_mask63
        mixed = (packed * self._hash_mix_const) & self._hash_mask63
        return torch.where(valid, mixed, torch.zeros_like(mixed)).sum(dim=1) & self._hash_mask63

    def _prune_visited(self, derived, counts, history, h_count):
        """Remove visited states from derived."""
        B, K, A, _ = derived.shape
        H, pad = history.shape[1], self.padding_idx
        hashes = self._compute_hash(derived)
        hist_exp = history.unsqueeze(1).expand(-1, K, -1)
        h_mask = self._arange_S[:H].unsqueeze(0).unsqueeze(0) < h_count.unsqueeze(1).unsqueeze(2)
        match = (hashes.unsqueeze(-1) == hist_exp) & h_mask
        is_visited = match.any(dim=-1)
        keep = ~is_visited

        within_count = self._arange_S.unsqueeze(0) < counts.unsqueeze(1)
        keep = keep & within_count

        flat_dim = A * 3
        new_pos = torch.cumsum(keep.long(), dim=1) - 1
        new_pos = torch.where(keep, new_pos.clamp(min=0, max=K-1), self._ones_B.unsqueeze(1) * (K-1))
        src = torch.where(keep.unsqueeze(-1), derived.view(B, K, flat_dim), torch.zeros(B, K, flat_dim, dtype=torch.long, device=self.device))
        out = torch.full((B, K, flat_dim), pad, dtype=torch.long, device=self.device)
        out.scatter_(1, new_pos.unsqueeze(-1).expand(B, K, flat_dim), src)
        new_counts = keep.sum(dim=1)
        return out.view(B, K, A, 3), new_counts

    def _add_end_action(self, current, states, counts):
        """Add END action at position counts."""
        B, S = self.batch_size, self.padding_states
        safe_pos = counts.clamp(max=S-1)
        idx = safe_pos.view(B, 1, 1, 1).expand(B, 1, self.padding_atoms, 3)
        end_exp = self.end_state.unsqueeze(0).expand(B, 1, -1, -1)
        states = states.scatter(1, idx, end_exp)
        return states, (counts + 1).clamp(max=S)

    # =========================================================================
    # REWARD
    # =========================================================================

    def _compute_reward(self, states, labels, depths, B):
        """Compute reward and termination signals."""
        first_pred = states[:, 0, 0]
        is_true = first_pred == self.true_pred_idx if self.true_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=self.device)
        is_false = first_pred == self.false_pred_idx if self.false_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=self.device)
        is_pad = first_pred == self.padding_idx
        is_end = first_pred == self.end_pred_idx if self.end_pred_idx is not None else torch.zeros(B, dtype=torch.bool, device=self.device)

        pos_label = labels == 1
        neg_label = labels == 0

        # Success: true proof for pos OR false proof for neg
        is_success = (is_true & pos_label) | (is_false & neg_label)
        terminated = is_true | is_false | is_pad | is_end
        truncated = depths >= self.max_depth

        # Reward based on type
        if self.reward_type == 0:
            # Simple: +1 for success, -1 for failure/truncation
            rewards = torch.where(is_success, self._reward_pos, 
                        torch.where(terminated | truncated, self._reward_neg, self._reward_zero))
        else:
            # Type 1: weighted negatives
            rewards = torch.where(is_success & pos_label, self._reward_pos,
                        torch.where(is_success & neg_label, self._reward_pos * self.rejection_weight,
                        torch.where(terminated | truncated, self._reward_neg, self._reward_zero)))

        return rewards, terminated, truncated, is_success

    # =========================================================================
    # NEGATIVE SAMPLING
    # =========================================================================

    def _sample_negatives(self, queries, labels, mask, counters):
        """Apply negative sampling for training (graph-safe, no branching)."""
        if self.negative_ratio <= 0.0 or self.sampler is None:
            return queries, labels, counters

        B = queries.shape[0]
        neg_threshold = int(1.0 / (1.0 + self.negative_ratio) * 1000)
        
        # Round-robin between positive and negative
        new_counters = torch.where(mask, counters + 1, counters)
        should_neg = (new_counters % 1000) >= neg_threshold
        should_neg = should_neg & mask

        # Always compute negative samples (no should_neg.any() branch for graph compatibility)
        # Use first mode unconditionally to avoid data-dependent indexing
        mode = self.corruption_scheme[0]
        neg_queries = self.sampler.corrupt(queries, num_negatives=1, mode=mode, device=self.device).squeeze(1)
        
        # Select based on should_neg mask
        queries = torch.where(should_neg.unsqueeze(-1), neg_queries, queries)
        labels = torch.where(should_neg, torch.zeros_like(labels), labels)

        return queries, labels, new_counters

    # =========================================================================
    # BUFFER MANAGEMENT
    # =========================================================================

    def _set_queries_internal(self, queries):
        """Set query pool for training/eval."""
        self._query_pool = queries.to(self.device)
        self._per_env_ptrs = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

    def _allocate_buffers(self):
        """Pre-allocate CUDA graph buffers."""
        B, A, S, H = self.batch_size, self.padding_atoms, self.padding_states, self.max_history_size
        device = self.device
        
        self._state_buffer = TensorDict({
            "current_states": torch.zeros(B, A, 3, dtype=torch.long, device=device),
            "derived_states": torch.zeros(B, S, A, 3, dtype=torch.long, device=device),
            "derived_counts": torch.zeros(B, dtype=torch.long, device=device),
            "original_queries": torch.zeros(B, A, 3, dtype=torch.long, device=device),
            "next_var_indices": torch.zeros(B, dtype=torch.long, device=device),
            "depths": torch.zeros(B, dtype=torch.long, device=device),
            "done": torch.zeros(B, dtype=torch.uint8, device=device),
            "success": torch.zeros(B, dtype=torch.uint8, device=device),
            "current_labels": torch.zeros(B, dtype=torch.long, device=device),
            "history_hashes": torch.zeros(B, H, dtype=torch.long, device=device),
            "history_count": torch.zeros(B, dtype=torch.long, device=device),
            "step_rewards": torch.zeros(B, dtype=torch.float32, device=device),
            "step_dones": torch.zeros(B, dtype=torch.uint8, device=device),
            "cumulative_rewards": torch.zeros(B, dtype=torch.float32, device=device),
            "per_env_ptrs": torch.zeros(B, dtype=torch.long, device=device),
            "neg_counters": torch.zeros(B, dtype=torch.int64, device=device),
        }, batch_size=[B], device=device)

    def _state_to_obs(self, state):
        """Convert state to observation dict."""
        mask = self._arange_S.unsqueeze(0) < state['derived_counts'].unsqueeze(1)
        return TensorDict({
            'sub_index': state['current_states'].unsqueeze(1),
            'derived_sub_indices': state['derived_states'],
            'action_mask': mask.to(torch.uint8),
        }, batch_size=[self.batch_size], device=self.device)

    def _copy_state_to_buffer(self, state):
        """Copy state to pre-allocated buffer."""
        for k in self._state_buffer.keys():
            self._state_buffer[k].copy_(state[k])

    def _copy_state_from_buffer(self, target, source):
        """Copy from source state to target."""
        for k in target.keys():
            if k in source.keys():
                target[k].copy_(source[k])

    def _clone_obs(self, obs):
        """Clone observation tensors."""
        return TensorDict({k: v.clone() for k, v in obs.items()}, batch_size=obs.batch_size, device=self.device)

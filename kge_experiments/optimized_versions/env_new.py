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


class EnvVec:
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
        order: bool = False,
        sample_deterministic_per_env: bool = False,
        compile: bool = True,
        compile_mode: str = 'reduce-overhead',
        compile_fullgraph: bool = True,
        use_exact_memory: bool = False,  # For parity tests: use exact atom matching instead of hashes
        skip_unary_actions: bool = False,  # Auto-advance when only 1 action available (AAAI26 parity)
    ) -> None:
        # Device & dimensions
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.end_proof_action = end_proof_action
        self.memory_pruning = memory_pruning
        self.use_exact_memory = use_exact_memory
        self.skip_unary_actions = skip_unary_actions
        self.max_unary_iterations = 5  # Reduced from AAAI26's 20 for faster compilation
        self.max_history_size = max_depth + 1
        
        # Vectorized state history buffer for exact memory (replaces Python sets)
        # Stores actual states [B, H, A, 3] for collision-free matching
        # This is now fully GPU-vectorized and torch.compile compatible
        self._state_history = None  # Allocated in _allocate_buffers
        self._state_history_count = None  # [B] count of valid entries per env

        # Training config
        self.sampler = sampler
        self.default_negative_ratio = float(negative_ratio)
        self.negative_ratio = float(negative_ratio)
        self.default_order = order
        self.order = order
        self.sample_deterministic_per_env = sample_deterministic_per_env
        self.corruption_scheme = corruption_scheme
        self.reward_type = reward_type
        self.rejection_weight = 1.0 / negative_ratio if negative_ratio > 0 else 1.0

        # Pre-allocated reward scalars
        self._reward_pos = torch.tensor(1.0, device=self.device)
        self._reward_neg = torch.tensor(-1.0, device=self.device)
        self._reward_zero = torch.tensor(0.0, device=self.device)
        self._reward_neg_half = torch.tensor(-0.5, device=self.device)
        self._reward_neg_1_5 = torch.tensor(-1.5, device=self.device)
        self._reward_rejection = torch.tensor(self.rejection_weight, device=self.device)

        # Hash constants
        self._hash_mix_const = 0x9E3779B97F4A7C15
        self._hash_mask63 = (1 << 63) - 1

        # Engine
        self.engine = vec_engine
        self.parity_mode = bool(getattr(vec_engine, "parity_mode", False))
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
        self._mark_static_buffers()

        # Auto-compile if requested (now works with exact memory too - fully vectorized)
        # Note: skip_unary_actions loop is expensive to compile - disable compilation for now
        if self.parity_mode and compile:
            print("[EnvOptimal] parity_mode enabled - skipping torch.compile for parity tests")
            compile = False

        if compile and not skip_unary_actions:
            self.compile(mode=compile_mode, fullgraph=compile_fullgraph)
        elif skip_unary_actions:
            print(f"[EnvOptimal] skip_unary_actions enabled - skipping torch.compile for compatibility")

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
            if self._reset_fn is not None:
                state = self._reset_fn(queries.to(device), torch.ones(B, dtype=torch.long, device=device))
            else:
                state = self._reset_from_queries(queries.to(device), torch.ones(B, dtype=torch.long, device=device))
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
        if self._reset_fn is not None:
            state = self._reset_fn(init_q, init_labels)
        else:
            state = self._reset_from_queries(init_q, init_labels)
        state['per_env_ptrs'].copy_(self._per_env_ptrs)
        state['neg_counters'].copy_(new_counters)

        return self._state_to_obs(state), state

    def step(self, state: EnvState, actions: Tensor, auto_reset: bool = True) -> Tuple[EnvObs, EnvState]:
        """Take one step."""
        torch.compiler.cudagraph_mark_step_begin()

        if auto_reset and self._query_pool is not None:
            self._copy_state_to_buffer(state)
            if self._step_and_reset_fn is not None:
                new_obs, new_state = self._step_and_reset_fn(self._state_buffer, actions, self._query_pool, self._per_env_ptrs)
            else:
                new_obs, new_state = self._step_and_reset_core(self._state_buffer, actions, self._query_pool, self._per_env_ptrs)
            new_obs = self._clone_obs(new_obs)
            self._copy_state_from_buffer(state, new_state)
            return new_obs, state
        else:
            # Fallback to uncompiled version if compile() wasn't called
            if self._step_fn is not None:
                return self._step_fn(state, actions)
            else:
                return self._step_core(state, actions)

    # =========================================================================
    # STEP LOGIC
    # =========================================================================

    def _reset_from_queries(self, queries: Tensor, labels: Optional[Tensor] = None) -> EnvState:
        """Create initial state from queries."""
        device = self.device
        A, S, H, pad = self.padding_atoms, self.padding_states, self.max_history_size, self.padding_idx

        # Handle input shape - can be [B, 3], [B, 1, 3], or [B, A, 3]
        if queries.ndim == 2:
            # [B, 3] -> [B, A, 3]
            B = queries.shape[0]
            padded = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
            padded[:, 0, :] = queries.to(device)
            queries = padded
        elif queries.ndim == 3 and queries.shape[1] == 1:
            # [B, 1, 3] -> [B, A, 3]  
            B = queries.shape[0]
            padded = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
            padded[:, 0, :] = queries.squeeze(1).to(device)
            queries = padded
        else:
            # Assume [B, A, 3]
            queries = queries.to(device)
            B = queries.shape[0]

        # Handle None labels - default to all positive (1)
        if labels is None:
            labels = torch.ones(B, dtype=torch.long, device=device)
        else:
            labels = labels.to(device)

        # Initialize runtime variable indices from start index (matching original)
        var_idx = torch.full((B,), self.runtime_var_start_index, dtype=torch.long, device=device)

        # Initialize history with the initial state hash (matching original)
        h_hashes = torch.zeros((B, H), dtype=torch.int64, device=device)
        h_hashes[:, 0] = self._compute_hash(queries)
        h_count = torch.ones(B, dtype=torch.long, device=device)
        
        # Reset and populate state history for exact memory (fully vectorized)
        if self.use_exact_memory:
            reset_mask = torch.ones(B, dtype=torch.bool, device=device)
            self._state_history_reset(reset_mask)
            self._state_history_add(queries, reset_mask)

        # Compute derived with excluded_queries being the first atom (matching original)
        excluded = queries[:, 0:1, :]  # [B, 1, 3]
        derived, counts, new_var = self._compute_derived(queries, var_idx, queries, h_hashes, h_count, excluded)

        # Skip unary actions: auto-advance when only 1 non-terminal action (AAAI26 parity)
        current_states = queries
        if self.skip_unary_actions:
            current_states, derived, counts, new_var, h_hashes, h_count = self._advance_through_unary(
                queries, derived, counts, new_var,
                queries, h_hashes, h_count, excluded
            )

        return TensorDict({
            "current_states": current_states,
            "derived_states": derived.clone(),
            "derived_counts": counts,
            "original_queries": queries,
            "next_var_indices": new_var,
            "depths": torch.zeros(B, dtype=torch.long, device=device),
            "done": torch.zeros(B, dtype=torch.uint8, device=device),
            "success": torch.zeros(B, dtype=torch.uint8, device=device),
            "current_labels": labels,
            "history_hashes": h_hashes,
            "history_count": h_count,
            "step_rewards": torch.zeros(B, dtype=torch.float32, device=device),
            "step_dones": torch.zeros(B, dtype=torch.uint8, device=device),
            "step_successes": torch.zeros(B, dtype=torch.uint8, device=device),
            "step_labels": torch.zeros(B, dtype=torch.long, device=device),
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

        rewards, terminated, truncated, is_success = self._get_done_reward(new_current, state['current_labels'], new_depths, B)
        newly_done = active & (terminated | truncated)
        new_done = was_done | newly_done
        new_success = state['success'].bool() | (active & is_success)
        rewards = torch.where(active, rewards, torch.zeros_like(rewards))

        # Update history - avoid clone by computing hash and scattering into existing tensor view
        write_pos = state['history_count'].clamp(max=self.max_history_size - 1)
        new_hash = self._compute_hash(new_current)
        # Create new history by starting from existing and doing masked scatter
        # Only update where active, keeping existing values otherwise
        new_history = state['history_hashes'].scatter(1, write_pos.unsqueeze(1),
            torch.where(active.unsqueeze(1), new_hash.unsqueeze(1), state['history_hashes'].gather(1, write_pos.unsqueeze(1))))
        new_h_count = torch.where(active, (state['history_count'] + 1).clamp(max=self.max_history_size), state['history_count'])

        still_active = ~new_done

        # Update state history for exact memory (vectorized - add new current states before computing derived)
        if self.use_exact_memory:
            self._state_history_add(new_current, active & ~was_done)

        # Pass excluded_queries = first atom of original query to match tensor env behavior
        excluded = state['original_queries'][:, 0:1, :] if state['original_queries'] is not None else None
        new_derived, new_counts, new_var = self._compute_derived(new_current, state['next_var_indices'], state['original_queries'], new_history, new_h_count, excluded)

        # Skip unary actions: auto-advance when only 1 non-terminal action (AAAI26 parity)
        if self.skip_unary_actions:
            new_current, new_derived, new_counts, new_var, new_history, new_h_count = self._advance_through_unary(
                new_current, new_derived, new_counts, new_var,
                state['original_queries'], new_history, new_h_count, excluded
            )

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

    def _step_and_reset_core(self, state: EnvState, actions: Tensor, query_pool: Tensor, per_env_ptrs: Tensor,
                              slot_lengths: Optional[Tensor] = None, slot_offsets: Optional[Tensor] = None,
                              ) -> Tuple[EnvObs, EnvState]:
        """Fused step + reset for training and evaluation.
        
        Query selection modes:
        - Training (slot_lengths=None): 
            - self.order=True: round-robin through query_pool (parity tests)
            - self.order=False: random selection from query_pool (production)
        - Evaluation (slot_lengths provided):
            - Per-slot scheduling: query_idx = slot_offsets + (per_env_ptrs % slot_lengths)
            - Each slot processes its own sequence of queries (positive + corruptions)
        """
        _, next_state = self._step_core(state, actions)
        rewards = next_state['step_rewards']
        done_mask = next_state['step_dones'].bool()
        B, device = state['current_states'].shape[0], self.device
        pool_size = query_pool.shape[0]

        if slot_lengths is not None and slot_offsets is not None:
            # Slot-based scheduling for evaluation
            # Each slot has its own query sequence: slot_offsets[i] + (ptr % slot_lengths[i])
            safe_slot_len = slot_lengths.clamp(min=1)  # Avoid div by zero
            local_idx = per_env_ptrs % safe_slot_len
            query_idx = (slot_offsets + local_idx).clamp(max=pool_size - 1)
            safe_idx = query_idx
            next_ptrs = per_env_ptrs + 1  # Per-slot counter, no wraparound needed (handled by caller)
        elif self.order:
            # Round-robin for parity tests
            safe_idx = per_env_ptrs % pool_size
            next_ptrs = (per_env_ptrs + 1) % pool_size
        else:
            # Random selection for production training
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

        # Capture step success and labels BEFORE reset overwrites them (for callback tracking)
        step_successes = next_state['success']
        step_labels = next_state['current_labels']

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
            "step_successes": step_successes,  # Success from completed step, not reset
            "step_labels": step_labels,  # Labels from completed step, not reset
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
        
        # Track original atom counts BEFORE truncation for budget rejection
        # A state with > A atoms should be rejected, not truncated
        raw_valid_atoms = derived_raw[:, :, :, 0] != self.padding_idx
        original_atom_counts = raw_valid_atoms.sum(dim=2)  # [B, K] - true atom count per state
        
        buf = torch.full((B, S, A, 3), self.padding_idx, dtype=torch.long, device=self.device)
        K, M = min(derived_raw.shape[1], S), min(derived_raw.shape[2], A)
        buf[:, :K, :M, :] = derived_raw[:, :K, :M, :]
        
        # Pad original_atom_counts to match S dimension
        atom_counts_buf = torch.zeros((B, S), dtype=torch.long, device=self.device)
        atom_counts_buf[:, :K] = original_atom_counts[:, :K]
        
        return buf, counts_raw, new_vars, atom_counts_buf


    def _compute_derived(self, current_states, next_var_indices, original_queries, history_hashes=None, history_count=None, excluded_queries=None):
        """Compute derived states with static shapes."""
        B, S, A, pad = self.batch_size, self.padding_states, self.padding_atoms, self.padding_idx

        derived, raw_counts, new_var, original_atom_counts = self._get_derived_raw(current_states, next_var_indices, excluded_queries)

        within_count = self._arange_S.unsqueeze(0) < raw_counts.unsqueeze(1)
        valid_atom = derived[:, :, :, 0] != pad
        # Use ORIGINAL atom counts (before truncation) to properly reject over-budget states
        # A 7-atom state truncated to 6 should be rejected, not kept
        base_valid = within_count & (original_atom_counts <= A) & (original_atom_counts > 0)

        derived = torch.where(base_valid.unsqueeze(-1).unsqueeze(-1), derived, self._false_state_base.unsqueeze(0).expand(B, -1, -1, -1))
        new_counts = base_valid.sum(dim=1)

        # Compact
        flat_dim = A * 3
        target_pos = torch.cumsum(base_valid.long(), dim=1) - 1
        target_pos = torch.where(base_valid, target_pos.clamp(min=0, max=S-1), self._ones_B.unsqueeze(1) * (S-1))
        src = torch.where(base_valid.unsqueeze(-1), derived.reshape(B, S, flat_dim), self._compact_zeros)
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

    # =========================================================================
    # VECTORIZED STATE HISTORY (replaces Python set-based exact memory)
    # =========================================================================
    
    def _state_history_reset(self, env_mask: Tensor) -> None:
        """Reset state history for environments specified by mask. Fully vectorized."""
        # Zero out history entries and reset counts for masked environments
        self._state_history_count = torch.where(env_mask, torch.zeros_like(self._state_history_count), self._state_history_count)
        # Fill history with padding for reset envs
        pad_fill = torch.full_like(self._state_history[0:1], self.padding_idx).expand(self.batch_size, -1, -1, -1)
        self._state_history = torch.where(env_mask.view(-1, 1, 1, 1), pad_fill, self._state_history)
    
    def _state_history_add(self, states: Tensor, active_mask: Optional[Tensor] = None) -> None:
        """Add states to history buffer. Fully vectorized.
        
        Args:
            states: [B, A, 3] current states to add
            active_mask: [B] optional mask of which envs to update
        """
        B, A = states.shape[0], self.padding_atoms
        H = self.max_history_size
        
        if active_mask is None:
            active_mask = torch.ones(B, dtype=torch.bool, device=self.device)
        
        # Compute write position (clamped to max)
        write_pos = self._state_history_count.clamp(max=H - 1)  # [B]
        
        # Create index for scatter: [B, 1, A, 3]
        idx = write_pos.view(B, 1, 1, 1).expand(B, 1, A, 3)
        
        # Scatter states into history at write position
        new_history = self._state_history.scatter(1, idx, states.unsqueeze(1))
        
        # Only update for active environments
        self._state_history = torch.where(active_mask.view(B, 1, 1, 1), new_history, self._state_history)
        
        # Increment count for active environments (clamped to max)
        new_count = (self._state_history_count + 1).clamp(max=H)
        self._state_history_count = torch.where(active_mask, new_count, self._state_history_count)
    
    def _normalize_state_for_comparison(self, states: Tensor) -> Tensor:
        """Normalize states by zeroing out terminal predicates for comparison.
        
        This ensures that states with terminals (True, False, Endf) are compared
        consistently - we ignore terminal atoms when checking membership.
        
        Args:
            states: [..., A, 3] tensor of states
            
        Returns:
            Normalized states with terminal atoms zeroed
        """
        pad = self.padding_idx
        preds = states[..., 0]  # [..., A]
        
        # Create mask for terminal predicates
        is_terminal = torch.zeros_like(preds, dtype=torch.bool)
        if self.true_pred_idx is not None:
            is_terminal = is_terminal | (preds == self.true_pred_idx)
        if self.false_pred_idx is not None:
            is_terminal = is_terminal | (preds == self.false_pred_idx)
        if self.end_pred_idx is not None:
            is_terminal = is_terminal | (preds == self.end_pred_idx)
        
        # Zero out terminal atoms (replace with padding)
        return torch.where(is_terminal.unsqueeze(-1), pad, states)

    def _prune_visited(self, derived, counts, history, h_count):
        """Remove visited states from derived. Vectorized for both hash and exact modes."""
        B, K, A, _ = derived.shape
        pad = self.padding_idx
        
        # Use vectorized exact memory if enabled
        if self.use_exact_memory:
            # Use hash-based comparison on the state history buffer
            # This gives us exact matching (no collisions) because we control what goes in
            # and the hash is computed consistently for both sides
            
            # Compute hashes based on normalized states (terminal atoms removed)
            # This matches the hash computation used when adding to history
            derived_norm = self._normalize_state_for_comparison(derived)  # [B, K, A, 3]
            history_norm = self._normalize_state_for_comparison(self._state_history)  # [B, H, A, 3]
            
            H = self._state_history.shape[1]
            
            # Compute hashes for derived states [B, K]
            derived_hashes = self._compute_hash(derived_norm)
            
            # Compute hashes for history states [B, H]
            history_hashes = self._compute_hash(history_norm)
            
            # Compare hashes: [B, K, 1] vs [B, 1, H] -> [B, K, H]
            hash_match = derived_hashes.unsqueeze(2) == history_hashes.unsqueeze(1)
            
            # Mask valid history entries
            h_valid = torch.arange(H, device=self.device).unsqueeze(0) < self._state_history_count.unsqueeze(1)  # [B, H]
            is_visited = (hash_match & h_valid.unsqueeze(1)).any(dim=2)  # [B, K]
            
            # Also check if derived state is padding
            is_padding = derived[:, :, 0, 0] == pad
            keep = ~is_visited & ~is_padding
            
            # Within count mask
            within_count = self._arange_S.unsqueeze(0) < counts.unsqueeze(1)
            keep = keep & within_count
        else:
            # Hash-based pruning (fast, but can have collisions)
            H = history.shape[1]
            hashes = self._compute_hash(derived)  # [B, K]
            hist_exp = history.unsqueeze(1).expand(-1, K, -1)  # [B, K, H]
            # Create valid history mask: [B, 1, H] - expanded to [B, K, H]
            arange_H = torch.arange(H, device=self.device)
            h_mask = arange_H.view(1, 1, H) < h_count.view(B, 1, 1)  # [B, 1, H] broadcasts to [B, K, H]
            match = (hashes.unsqueeze(-1) == hist_exp) & h_mask  # [B, K, H]
            is_visited = match.any(dim=-1)  # [B, K]
            keep = ~is_visited


            within_count = self._arange_S.unsqueeze(0) < counts.unsqueeze(1)
            keep = keep & within_count

        # Compact kept states (same for both modes)
        flat_dim = A * 3
        new_pos = torch.cumsum(keep.long(), dim=1) - 1
        ones_B = torch.ones(B, dtype=torch.long, device=self.device)
        new_pos = torch.where(keep, new_pos.clamp(min=0, max=K-1), ones_B.unsqueeze(1) * (K-1))
        src = torch.where(keep.unsqueeze(-1), derived.view(B, K, flat_dim), torch.zeros(B, K, flat_dim, dtype=torch.long, device=self.device))
        out = torch.full((B, K, flat_dim), pad, dtype=torch.long, device=self.device)
        out.scatter_(1, new_pos.unsqueeze(-1).expand(B, K, flat_dim), src)
        new_counts = keep.sum(dim=1)
        return out.view(B, K, A, 3), new_counts

    # =========================================================================
    # END ACTION
    # =========================================================================

    def _add_end_action(self, current, states, counts):
        """Add END action if no terminal state exists."""
        B, S = states.shape[:2]
        preds = states[:, :, 0, 0]
        valid = torch.arange(S, device=self.device).unsqueeze(0) < counts.unsqueeze(1)
        has_terminal = ((((preds == self.true_pred_idx) | (preds == self.false_pred_idx)) & valid).sum(dim=1) > 0) & (counts > 0)
        can_end = ~has_terminal & (counts < self.padding_states)
        pos = torch.arange(S, device=self.device).unsqueeze(0)
        at_end = pos == counts.unsqueeze(1)
        should_place = can_end.unsqueeze(1) & at_end
        new_states = states.clone()
        end_exp = self.end_state.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        new_states = torch.where(should_place.unsqueeze(-1).unsqueeze(-1), end_exp, new_states)
        return new_states, torch.where(can_end, counts + 1, counts)

    # =========================================================================
    # SKIP UNARY ACTIONS (AAAI26 parity)
    # =========================================================================

    def _advance_through_unary(
        self,
        current_states: Tensor,
        derived_states: Tensor,
        counts: Tensor,
        next_var_indices: Tensor,
        original_queries: Tensor,
        history_hashes: Tensor,
        history_count: Tensor,
        excluded_queries: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Auto-advance through unary actions (exactly 1 non-terminal derived state).

        This matches AAAI26 behavior where the agent doesn't need to pick the only
        available action - we advance automatically until there's a real choice.

        Returns updated: current_states, derived_states, counts, next_var_indices,
                        history_hashes, history_count
        """
        if not self.skip_unary_actions:
            return current_states, derived_states, counts, next_var_indices, history_hashes, history_count

        B, S, A = self.batch_size, self.padding_states, self.padding_atoms
        device = self.device

        # Fixed iterations with masking (torch.compile compatible - no data-dependent branching)
        for iteration in range(self.max_unary_iterations):
            # Check which envs have exactly 1 action
            has_one_action = counts == 1

            # Get the single derived state for each env (index 0)
            single_state = derived_states[:, 0, :, :]  # [B, A, 3]

            # Check if that state is terminal (True, False, End)
            first_pred = single_state[:, 0, 0]  # [B]
            is_terminal = torch.zeros(B, dtype=torch.bool, device=device)
            if self.true_pred_idx is not None:
                is_terminal = is_terminal | (first_pred == self.true_pred_idx)
            if self.false_pred_idx is not None:
                is_terminal = is_terminal | (first_pred == self.false_pred_idx)
            if self.end_pred_idx is not None:
                is_terminal = is_terminal | (first_pred == self.end_pred_idx)

            # Only advance if: has_one_action AND not terminal
            should_advance = has_one_action & ~is_terminal

            # Even if no envs should advance, continue the loop (torch.compile compatible)
            # The masking ensures no actual work is done when should_advance is all False

            # Update current states for envs that should advance
            new_current = torch.where(
                should_advance.view(B, 1, 1),
                single_state,
                current_states
            )

            # Update history (add the new current state)
            write_pos = history_count.clamp(max=self.max_history_size - 1)
            new_hash = self._compute_hash(new_current)
            new_history = history_hashes.scatter(
                1, write_pos.unsqueeze(1),
                torch.where(should_advance.unsqueeze(1), new_hash.unsqueeze(1),
                           history_hashes.gather(1, write_pos.unsqueeze(1)))
            )
            new_h_count = torch.where(
                should_advance,
                (history_count + 1).clamp(max=self.max_history_size),
                history_count
            )

            # Recompute derived for ALL envs (masking happens in the merge step)
            # This is less efficient but torch.compile compatible
            new_derived, new_counts, new_var = self._compute_derived_without_unary(
                new_current, next_var_indices, original_queries,
                new_history, new_h_count, excluded_queries
            )

            # Merge: only update envs that advanced
            current_states = new_current
            derived_states = torch.where(
                should_advance.view(B, 1, 1, 1),
                new_derived,
                derived_states
            )
            counts = torch.where(should_advance, new_counts, counts)
            next_var_indices = torch.where(should_advance, new_var, next_var_indices)
            history_hashes = new_history
            history_count = new_h_count

        return current_states, derived_states, counts, next_var_indices, history_hashes, history_count

    def _compute_derived_without_unary(self, current_states, next_var_indices, original_queries,
                                        history_hashes=None, history_count=None, excluded_queries=None):
        """Compute derived states without triggering unary advancement (to avoid recursion)."""
        # This is the same as _compute_derived but called during unary advancement loop
        B, S, A, pad = self.batch_size, self.padding_states, self.padding_atoms, self.padding_idx

        derived, raw_counts, new_var, original_atom_counts = self._get_derived_raw(current_states, next_var_indices, excluded_queries)

        within_count = self._arange_S.unsqueeze(0) < raw_counts.unsqueeze(1)
        valid_atom = derived[:, :, :, 0] != pad
        base_valid = within_count & (original_atom_counts <= A) & (original_atom_counts > 0)

        derived = torch.where(base_valid.unsqueeze(-1).unsqueeze(-1), derived, self._false_state_base.unsqueeze(0).expand(B, -1, -1, -1))
        new_counts = base_valid.sum(dim=1)

        # Compact
        flat_dim = A * 3
        target_pos = torch.cumsum(base_valid.long(), dim=1) - 1
        target_pos = torch.where(base_valid, target_pos.clamp(min=0, max=S-1), self._ones_B.unsqueeze(1) * (S-1))
        src = torch.where(base_valid.unsqueeze(-1), derived.reshape(B, S, flat_dim), self._compact_zeros)
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
    # REWARD
    # =========================================================================

    def _get_done_reward(
        self,
        states: Tensor,
        labels: Tensor,
        depths: Tensor,
        n: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute rewards and termination flags.
        Adapted from BatchedEnv._get_done_reward pure functional style.
        """
        device = self.device
        pad = self.padding_idx
        
        # states: [n, A, 3]
        non_pad = states[:, :, 0] != pad
        preds = states[:, :, 0]

        # 1. Compute 'is_end' condition (if enabled)
        is_end = torch.zeros(n, dtype=torch.bool, device=device)
        if self.end_proof_action:
            # single_pred: [n]
            single_pred = non_pad.sum(dim=1) == 1
            # first_pos: [n]
            first_pos = non_pad.long().argmax(dim=1)
            # first_pred: [n]
            first_pred = preds[torch.arange(n, device=device), first_pos]
            is_end = single_pred & (first_pred == self.end_pred_idx)
            
        # 2. Compute 'all_true' (success condition)
        if self.true_pred_idx is not None:
            # check if ALL non-pad atoms are TRUE
            true_mask = (preds == self.true_pred_idx) | ~non_pad
            all_true = true_mask.all(dim=1) & non_pad.any(dim=1)
        else:
            all_true = torch.zeros(n, dtype=torch.bool, device=device)
            
        # all_true must be false if is_end is true
        all_true = all_true & ~is_end
            
        # 3. Compute 'terminated' logic (all_true | any_false)
        terminated = all_true.clone()
        if self.false_pred_idx is not None:
             terminated = terminated | (preds == self.false_pred_idx).any(dim=1)
             
        # is_end implies terminated (failure)
        terminated = terminated | is_end
        
        # Termination on fully padded states (safety)
        terminated = terminated | ~non_pad.any(dim=1)
            
        # 4. Truncation
        depth_exceeded = depths >= self.max_depth
        truncated = depth_exceeded & ~terminated
        
        # Done Status
        done = terminated | truncated
        
        # 5. Success
        is_success = all_true

        # 6. Rewards logic (Matching BatchedEnv reward types)
        rewards = torch.zeros(n, dtype=torch.float32, device=device)
        pos = labels == 1
        neg = labels == 0
        
        if self.reward_type == 0:
            reward_mask = done & is_success & pos
            rewards = torch.where(reward_mask, self._reward_pos, self._reward_zero)
        elif self.reward_type == 1:
            tp = done & is_success & pos
            fp = done & is_success & neg
            rewards = torch.where(tp, self._reward_pos, rewards)
            rewards = torch.where(fp, self._reward_neg, rewards)
        elif self.reward_type == 2:
            reward_mask = done & ((is_success & pos) | (~is_success & neg))
            rewards = torch.where(reward_mask, self._reward_pos, self._reward_zero)
        elif self.reward_type == 3:
            tp = done & is_success & pos
            fn = done & ~is_success & pos
            fp = done & is_success & neg
            tn = done & ~is_success & neg
            rewards = torch.where(tp, self._reward_pos, rewards)
            rewards = torch.where(fn, self._reward_neg_half, rewards)
            rewards = torch.where(fp, self._reward_neg_1_5, rewards)
            rewards = torch.where(tn, self._reward_pos, rewards)
        elif self.reward_type == 4:
            tp = done & is_success & pos
            fn = done & ~is_success & pos
            fp = done & is_success & neg
            tn = done & ~is_success & neg
            rewards = torch.where(tp, self._reward_pos, rewards)
            rewards = torch.where(fn, self._reward_neg, rewards)
            rewards = torch.where(fp, self._reward_neg, rewards)
            rewards = torch.where(tn, self._reward_rejection, rewards)
            
        return rewards, terminated, truncated, is_success

    # =========================================================================
    # NEGATIVE SAMPLING
    # =========================================================================

    def _sample_negatives(self, queries, labels, mask, counters):
        """Apply negative sampling for training.
        
        When sample_deterministic_per_env=True, uses sequential corruption for RNG parity.
        Otherwise uses vectorized corruption for efficiency.
        """
        if self.negative_ratio <= 0.0 or self.sampler is None:
            return queries, labels, counters

        B = queries.shape[0]
        
        # Use same cycling logic as tensor env for parity
        # ratio=1 -> cycle=2
        # Counter 0=positive, counter 1=negative, counter 2=positive, etc.
        ratio = int(round(float(self.negative_ratio)))
        cycle = ratio + 1
        
        # Check CURRENT counter to determine if negative (before incrementing)
        # This matches tensor env's logic: (local_counters % cycle) != 0
        should_neg = (counters % cycle) != 0
        should_neg = should_neg & mask
        
        # Increment counters for next call
        new_counters = torch.where(mask, (counters + 1) % cycle, counters)

        if self.sample_deterministic_per_env:
            # Sequential corruption for RNG parity with tensor env
            # Only corrupt queries that need it, one at a time
            neg_indices = should_neg.nonzero(as_tuple=True)[0]
            if neg_indices.numel() > 0:
                queries = queries.clone()
                labels = labels.clone()
                for idx in neg_indices:
                    atom_to_corrupt = queries[idx:idx+1]  # [1, 3]
                    mode = self.corruption_scheme[0]
                    corrupted = self.sampler.corrupt(atom_to_corrupt, num_negatives=1, mode=mode, device=self.device)
                    if corrupted.dim() == 3:
                        corrupted = corrupted[:, 0, :]
                    queries[idx] = corrupted.squeeze(0)
                    labels[idx] = 0
        else:
            # Vectorized corruption (efficient but not RNG-deterministic per env)
            mode = self.corruption_scheme[0]
            neg_queries = self.sampler.corrupt(queries, num_negatives=1, mode=mode, device=self.device).squeeze(1)
            queries = torch.where(should_neg.unsqueeze(-1), neg_queries, queries)
            labels = torch.where(should_neg, torch.zeros_like(labels), labels)

        return queries, labels, new_counters

    def sample_negatives(self, queries, labels, mask, counters):
        """Public alias for _sample_negatives (for test compatibility)."""
        return self._sample_negatives(queries, labels, mask, counters)

    # =========================================================================
    # BUFFER MANAGEMENT
    # =========================================================================

    def _set_queries_internal(self, queries):
        """Set query pool for training/eval."""
        self._query_pool = queries.to(self.device)
        # Initialize per-env pointers with arange for deterministic per-env query assignment
        # This matches BatchedEnv's per-env round-robin behavior when sample_deterministic_per_env=True
        # Each env starts with a different query index: env0->query0, env1->query1, etc.
        self._per_env_ptrs = torch.arange(self.batch_size, dtype=torch.long, device=self.device)

    def _allocate_buffers(self):
        """Pre-allocate CUDA graph buffers."""
        B, A, S, H = self.batch_size, self.padding_atoms, self.padding_states, self.max_history_size
        device = self.device
        flat_dim = A * 3

        # State history buffer for vectorized exact memory
        # Stores actual states [B, H, A, 3] for collision-free matching
        self._state_history = torch.full((B, H, A, 3), self.padding_idx, dtype=torch.long, device=device)
        self._state_history_count = torch.zeros(B, dtype=torch.long, device=device)

        # Pre-allocated buffers for _compute_derived to avoid allocation per step
        self._compact_zeros = torch.zeros(B, S, flat_dim, dtype=torch.long, device=device)
        self._compact_full = torch.full((B, S, flat_dim), self.padding_idx, dtype=torch.long, device=device)
        
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
            "step_successes": torch.zeros(B, dtype=torch.uint8, device=device),
            "step_labels": torch.zeros(B, dtype=torch.long, device=device),
            "cumulative_rewards": torch.zeros(B, dtype=torch.float32, device=device),
            "per_env_ptrs": torch.zeros(B, dtype=torch.long, device=device),
            "neg_counters": torch.zeros(B, dtype=torch.int64, device=device),
        }, batch_size=[B], device=device)

    def _mark_static_buffers(self) -> None:
        """Mark persistent buffers as static to reduce cudagraph input copies."""
        if not hasattr(torch, "_dynamo"):
            return

        buffers = [
            self._reward_pos,
            self._reward_neg,
            self._reward_zero,
            self._reward_neg_half,
            self._reward_neg_1_5,
            self._reward_rejection,
            self._positions_S,
            self._arange_S,
            self._arange_A,
            self._arange_B,
            self._ones_B,
            self._reset_ones_B,
            self._reset_zeros_B,
            self._reset_mask_true,
            self._state_history,
            self._state_history_count,
            self._compact_zeros,
            self._compact_full,
            self.end_state,
            self._false_state_base,
        ]
        for buf in buffers:
            if isinstance(buf, torch.Tensor):
                torch._dynamo.mark_static_address(buf)

        engine = getattr(self, "engine", None)
        if engine is None:
            return

        engine_buffers = [
            "facts_idx",
            "rules_idx",
            "rule_lens",
            "rules_heads_idx",
            "rules_heads_sorted",
            "rules_idx_sorted",
            "rule_lens_sorted",
            "rule_seg_starts",
            "rule_seg_lens",
            "fact_seg_starts",
            "fact_seg_lens",
            "fact_hashes",
            "true_atom",
            "false_atom",
        ]
        for name in engine_buffers:
            buf = getattr(engine, name, None)
            if isinstance(buf, torch.Tensor):
                torch._dynamo.mark_static_address(buf)
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

    def set_eval_dataset(self, queries: Tensor, labels: Tensor, query_depths: Tensor,
                         per_slot_lengths: Optional[Tensor] = None):
        """Compatibility method for eval_corruptions. Simply calls eval()."""
        self.eval(queries)

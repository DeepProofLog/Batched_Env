"""
Vectorized Knowledge Graph Environment (Env_vec)

A high-performance, compilation-friendly environment for knowledge graph reasoning
using PyTorch and torch.compile for GPU acceleration.

Key Features:
- Single-step compilation (compiles step function, not full trajectory)
- TensorDict-based state and observations
- UnificationEngineVectorized for graph-safe unification
- Memory pruning with state hashing
- Negative sampling support for training
- Auto-reset for continuous training loops

Architecture:
    Instead of compiling full 20-step trajectories (~40k graph nodes, slow compile),
    we compile only the single-step transition (~2k nodes, fast compile).
    A Python loop orchestrates the trajectory.

Usage:
    # Create environment
    env = Env_vec(
        vec_engine=UnificationEngineVectorized(...),
        batch_size=500,
        train_queries=train_data,
        valid_queries=valid_data,
        compile=True,
    )
    
    # Training mode
    env.train()
    obs, state = env.reset()
    
    # Step with auto-reset
    for _ in range(num_steps):
        actions = policy(obs)
        obs, state = env.step_and_reset(state, actions)
        rewards = state['step_rewards']
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Any, Optional, Tuple, Callable
from tensordict import TensorDict

from unification import UnificationEngineVectorized


# ============================================================================
# Raw Tensor Observation Type 
# ============================================================================
EnvObs = TensorDict
EnvState = TensorDict


# ============================================================================
# Compiled Evaluation Environment
# ============================================================================

class EnvVec:
    """
    Vectorized Knowledge Graph Reasoning Environment.
    
    Supports both training (with negative sampling) and evaluation modes.
    Optionally compiles step functions with torch.compile for GPU acceleration.
    
    Key Methods:
        - train() / eval(): Switch modes
        - reset(): Initialize episodes
        - step(): Take single step
        - step_and_reset(): Step + auto-reset (for training loops)
        - compile(): Enable torch.compile acceleration
    """
    
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
        queries: Optional[Tensor] = None, # Backward compat/Initial query pool
        train_queries: Optional[Tensor] = None,
        valid_queries: Optional[Tensor] = None,
        sample_deterministic_per_env: bool = False,
        # Compilation settings
        compile: bool = False,
        compile_mode: str = 'reduce-overhead',
        compile_fullgraph: bool = True,
        # New training params
        sampler: Optional[Any] = None,
        negative_ratio: float = 1.0,
        order: bool = False,  # False=Random (default), True=Round-Robin
        metrics: Optional[Any] = None,
        corruption_scheme: Tuple[str, ...] = ('head', 'tail'),
        reward_type: int = 0,
        train_neg_ratio: float = 1.0, # kept for backward compat if needed, but we use negative_ratio
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.end_proof_action = end_proof_action
        self.memory_pruning = memory_pruning
        self.max_history_size = max_depth + 1
        self.sample_deterministic_per_env = sample_deterministic_per_env
        self._compiled = compile
        
        # Training / Sampling config
        self.sampler = sampler
        self.default_order = order # Default order for train mode
        self.order = order
        self.default_negative_ratio = float(negative_ratio)
        self.negative_ratio = float(negative_ratio)
        self.corruption_scheme = corruption_scheme
        self.reward_type = reward_type
        self.metrics = metrics
        
        # Store query sets
        self.train_queries = None
        if train_queries is not None:
            self.train_queries = train_queries.to(self.device)
        elif queries is not None:
             # Fallback if only 'queries' arg provided
            self.train_queries = queries.to(self.device)
            
        self.valid_queries = None
        if valid_queries is not None:
            self.valid_queries = valid_queries.to(self.device)
        
        # Negative sampling state - MOVED TO EnvState
        if self.negative_ratio > 0:
            self.rejection_weight = 1 / self.negative_ratio
        else:
            self.rejection_weight = 1.0

        
        # Hash computation constants
        self._hash_mix_const = 0x9E3779B97F4A7C15
        self._hash_mask63 = (1 << 63) - 1
        
        # Vectorized unification engine
        self.engine = vec_engine
        
        # Key indices from vectorized engine
        self.padding_idx = vec_engine.padding_idx
        self.true_pred_idx = vec_engine.true_pred_idx
        self.false_pred_idx = vec_engine.false_pred_idx
        self.end_pred_idx = vec_engine.end_pred_idx
        
        # Runtime variable start  
        if runtime_var_start_index is not None:
            self.runtime_var_start_index = runtime_var_start_index
        else:
            self.runtime_var_start_index = vec_engine.constant_no + 1
        
        # Pack base for hash computation
        if hasattr(vec_engine, 'pack_base'):
             self._hash_pack_base = vec_engine.pack_base
        else:
             self._hash_pack_base = vec_engine.constant_no + 1001
        
        # Pre-build end action tensor
        # Pre-build end action tensor
        self.end_state = None
        if self.end_pred_idx is not None and self.end_pred_idx >= 0:
            # Create [padding_atoms, 3] tensor filled with padding_idx
            self.end_state = torch.full((padding_atoms, 3), self.padding_idx, dtype=torch.long, device=self.device)
            # Set first atom to (Endf, pad, pad)
            self.end_state[0, 0] = self.end_pred_idx
        
        # Pre-allocated static tensors for CUDA graph stability
        self._positions_S = torch.arange(padding_states, device=self.device).unsqueeze(0)
        self._batch_idx_B = None
        
        # Query storage for training
        # Query storage and pointers
        self._query_pool = None
        self._per_env_ptrs = None
        
        # Initialize in train mode if possible
        if self.train_queries is not None:
             self.train()
        elif self.valid_queries is not None:
             self.eval() # Fallback
        else:
             # Empty init - waiting for set_queries or train/eval call 
             self._query_pool = None
             self._per_env_ptrs = None
        self._current_state: Optional[EnvState] = None
        self._current_obs: Optional[EnvObs] = None
        
        # =====================================================================
        # Compile functions if requested
        # =====================================================================
        if compile:
            self.compile(compile_mode, compile_fullgraph)
        else:
            # Eager mode: functions are used directly
            self._step_fn = self._step_core
            self._step_and_reset_fn = self._step_and_reset_core
    
    # =========================================================================
    # 1. Public API
    # =========================================================================
    
    def compile(
        self,
        mode: str = 'reduce-overhead',
        fullgraph: bool = True,
    ) -> None:
        """
        Compile environment for faster execution (GPU only).
        
        This compiles the single-step transition function ($step_{impl}$) into a CUDA graph.
        The trajectory loop remains in Python, calling this compiled function.
        
        Args:
           mode: torch.compile mode (default: 'reduce-overhead' for max speed).
           fullgraph: capture full graph without python fallbacks (recommended).
        """
        self._compiled = True
        
        # import torch._inductor.config as inductor_config
        # inductor_config.triton.cudagraphs = True
        
        # Compile standard step
        self._step_fn = torch.compile(
            self._step_core,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=False,
        )
        
        # Compile step+reset fused kernel
        self._step_and_reset_fn = torch.compile(
            self._step_and_reset_core,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=False,
        )
    
    def train(self) -> None:
        """
        Switch to training mode.
        
        - Uses train_queries
        - Enables negative sampling (negative_ratio > 0)
        - Uses configured ordering (default: Random)
        """
        if self.train_queries is None:
             raise ValueError("No train_queries provided at initialization")
        
        self._set_queries_internal(self.train_queries)
        self.negative_ratio = self.default_negative_ratio
        self.order = self.default_order 
        # Rejection weight handled dynamically
        if self.negative_ratio > 0:
             self.rejection_weight = 1 / self.negative_ratio
        else:
             self.rejection_weight = 1.0

    def eval(self, queries: Optional[Tensor] = None) -> None:
        """
        Switch to evaluation mode.
        
        - Uses valid_queries (or provided queries)
        - Disables negative sampling (negative_ratio = 0)
        - Forces sequential ordering (order = True) for deterministic evaluation
        """
        q_pool = queries if queries is not None else self.valid_queries
        
        if q_pool is None:
             raise ValueError("No valid_queries provided and no queries passed to eval()")
             
        self._set_queries_internal(q_pool)
        self.negative_ratio = 0.0
        self.order = True
        self.rejection_weight = 1.0

    def set_eval_dataset(self, queries: Tensor) -> None:
        """
        Setup environment for evaluation on a specific custom dataset.
        Equivalent to eval(queries) but resets internal pointers.
        """
        self.eval(queries)
        # Reset pointers to start ensures we start from query 0
        self._per_env_ptrs = torch.arange(self.batch_size, device=self.device)

    def reset(self) -> Tuple[EnvObs, EnvState]:
        """
        Initialize environment state.
        
        Selects initial queries based on current mode (train/eval),
        samples initial negatives, and computes initial state/observation.
        
        Returns:
            (observation, state)
        """
        if self._query_pool is None:
            raise RuntimeError("Must call train() or eval() or set_queries() before reset()")
        
        device = self.device
        B = self.batch_size
        pool_size = self._query_pool.shape[0]
        
        # 1. Select Queries
        if self.order:
            # Sequential (Round-Robin)
            query_indices = self._per_env_ptrs % pool_size
        else:
            # Random Sampling
            query_indices = torch.randint(0, pool_size, (B,), device=device)
            
        init_queries = self._query_pool[query_indices] # [B, 3]
        init_labels = torch.ones(B, dtype=torch.long, device=device)
        
        # 2. Sample Negatives (if training)
        # Start counters at 0
        current_counters = torch.zeros(B, dtype=torch.long, device=device)
        reset_mask = torch.ones(B, dtype=torch.bool, device=device)
        
        init_queries, init_labels, new_counters = self.sample_negatives(
            init_queries, init_labels, reset_mask, current_counters
        )
        
        # 3. Update Pointers (Ordered mode only)
        if self.order:
            self._per_env_ptrs = (self._per_env_ptrs + 1) % pool_size
        
        # 4. Create Initial State
        state = self._reset_from_queries(init_queries, init_labels)
        
        # Inject mutable tracking variables
        state['per_env_ptrs'] = self._per_env_ptrs.clone()
        state['neg_counters'] = new_counters
        
        # 5. Create Observation
        obs = self._state_to_obs(state)
        
        self._current_state = state
        self._current_obs = obs
        
        return obs, state

    def step(self, state: EnvState, actions: Tensor) -> Tuple[EnvObs, EnvState]:
        """
        Take one environment step.
        
        Args:
            state: Current environment state
            actions: [B] Actions to take
            
        Returns:
            (next_observation, next_state)
            
        Note: The 'next_state' contains rewards/dones for this step.
        """
        # Delegate to compiled core function
        return self._step_fn(state, actions)

    def step_and_reset(self, state: EnvState, actions: Tensor) -> Tuple[EnvObs, EnvState]:
        """
        Take step and immediately reset done environments (Auto-Reset).
        
        This is the primary method for training loops.
        It fuses step() and reset() logic for maximum throughput.
        
        Args:
            state: Current state
            actions: Actions
            
        Returns:
            (next_observation, next_state) 
            Note: next_observation may be from a NEW episode if reset occurred.
        """
        # Delegate to compiled fused function
        return self._step_and_reset_fn(state, actions)




    
    def sample_negatives(
        self,
        batch_q: Tensor,
        batch_labels: Tensor,
        reset_mask: Tensor,
        counters: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Vectorized negative sampling with simplified masking logic.
        
        Handles both 2D [B, 3] and 3D [B, A, 3] inputs.
        Corrupts samples based on counters and reset_mask.
        """
        if self.negative_ratio <= 0:
            return batch_q, batch_labels, counters

        # 1. Determine which envs need sampling
        ratio = int(round(float(self.negative_ratio)))
        if ratio <= 0: return batch_q, batch_labels, counters
        
        cycle = ratio + 1
        needs_sample = (counters % cycle) != 0
        active_sample_mask = reset_mask & needs_sample
        
        # 2. Unify Input Shape (View as 2D for processing)
        # We work with [B, 3] atoms. If input is [B, A, 3], we index [:, 0, :].
        is_3d = batch_q.dim() == 3
        # Reference to the atoms we might modify (view or slice)
        # Note: We need a clone to modify safely without affecting original if it was passed by ref
        # But we return new tensors anyway.
        new_q = batch_q.clone()
        
        # Extract the atoms to potentially corrupt (always [B, 3])
        if is_3d:
            current_atoms = new_q[:, 0, :]
        else:
            current_atoms = new_q

        # 3. Perform Corruption (Parity vs Vectorized)
        if self.sample_deterministic_per_env:
            # --- Parity Mode (Loop) ---
            # Explicit loop required for exact RNG match with BatchedEnv/SB3
            num_active = active_sample_mask.sum().item()
            if num_active > 0:
                active_indices = torch.nonzero(active_sample_mask).squeeze(-1)
                for i in active_indices:
                    atom_to_corrupt = current_atoms[i:i+1] # [1, 3]
                    neg = self.sampler.corrupt(atom_to_corrupt, num_negatives=1, device=self.device)
                    if neg.dim() == 3: neg = neg[:, 0, :]
                    
                    if is_3d:
                        new_q[i, 0, :] = neg
                    else:
                        new_q[i] = neg
        else:
            # --- Vectorized Mode (Graph Safe) ---
            # 1. Sanitise: Ensure sampler sees valid input everywhere (replace masked with valid)
            safe_atoms = torch.where(
                active_sample_mask.unsqueeze(-1),
                current_atoms,
                torch.zeros_like(current_atoms) 
            )
            
            # 2. Corrupt full batch
            corrupted_all = self.sampler.corrupt(safe_atoms, num_negatives=1, device=self.device)
            if corrupted_all.dim() == 3: corrupted_all = corrupted_all[:, 0, :]
            
            # 3. Mask Apply: Only update active indices
            mask_exp = active_sample_mask.unsqueeze(-1).expand_as(corrupted_all)
            update_atoms = torch.where(mask_exp, corrupted_all, current_atoms)
            
            if is_3d:
                new_q[:, 0, :] = update_atoms
            else:
                new_q = update_atoms

        # 4. Update Labels and Counters
        new_labels = torch.where(active_sample_mask, torch.zeros_like(batch_labels), batch_labels)
        
        next_counters = (counters + 1) % cycle
        new_counters = torch.where(reset_mask, next_counters, counters)
        
        return new_q, new_labels, new_counters

    def _get_done_reward(
        self,
        current_states: Tensor,
        current_labels: Tensor,
        current_depths: Tensor,
        n: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute rewards and termination flags.
        Adapted from BatchedEnv._get_done_reward pure functional style.
        """
        device = self.device
        pad = self.padding_idx
        
        # current_states: [n, A, 3]
        non_pad = current_states[:, :, 0] != pad
        preds = current_states[:, :, 0]

        # Optimization: Avoid large intermediate tensors for all_true/any_false/is_end
        # Compute conditions directly into 'terminated' and 'success'
        
        # 1. Compute 'is_end' condition (if enabled)
        is_end = None
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
            
        if is_end is not None:
            # all_true must be false if is_end is true
            all_true = all_true & ~is_end
            
        # 3. Compute 'any_false' (failure condition)
        # We can implement this as part of 'terminated' logic to save a tensor?
        # terminated = all_true | any_false
        
        # Start with all_true as base for terminated
        terminated = all_true.clone()
        
        if self.false_pred_idx is not None:
             # Add any_false cases
             # (preds == false) . any()
             terminated = terminated | (preds == self.false_pred_idx).any(dim=1)
             
        if is_end is not None:
            # is_end implies terminated (failure)
            terminated = terminated | is_end
            
        # 4. Truncation
        depth_exceeded = current_depths >= self.max_depth
        truncated = depth_exceeded & ~terminated
        
        # Done
        done = terminated | truncated # | depth_exceeded (implied)
        
        # 5. Success
        success = all_true
        is_success = success

        # 6. Rewards
        # Initialize rewards with zeros (or efficient fill if all same)
        rewards = torch.zeros(n, dtype=torch.float32, device=device)
        
        # Vectorized reward logic
        # Optimize by checking conditions hierarchically
        labels = current_labels
        
        if self.reward_type == 0:
            # 1.0 if done & success & (label==1)
            reward_mask = done & success & (labels == 1)
            rewards = torch.where(reward_mask, torch.tensor(1.0, device=device), rewards)
            
        elif self.reward_type == 1:
            # success & label=1 -> 1.0
            # success & label=0 -> -1.0
            # Pre-calc done & success
            done_success = done & success
            rewards = torch.where(done_success & (labels == 1), torch.tensor(1.0, device=device), rewards)
            rewards = torch.where(done_success & (labels == 0), torch.tensor(-1.0, device=device), rewards)
            
        elif self.reward_type == 2:
            # success & label=1 -> 1.0
            # fail & label=0 -> 1.0 (but fail means ~success & done)
            done_pos = done & (labels==1)
            done_neg = done & (labels==0)
            
            rewards = torch.where(done_pos & success, torch.tensor(1.0, device=device), rewards)
            rewards = torch.where(done_neg & ~success, torch.tensor(1.0, device=device), rewards)
            
        elif self.reward_type == 3:
            pos = labels == 1
            neg = labels == 0
            # TP: done & success & pos
            # FN: done & ~success & pos
            # FP: done & success & neg
            # TN: done & ~success & neg
            
            # Use combined conditions
            rewards = torch.where(done & success & pos, torch.tensor(1.0, device=device), rewards)
            rewards = torch.where(done & ~success & pos, torch.tensor(-0.5, device=device), rewards)
            rewards = torch.where(done & success & neg, torch.tensor(-1.5, device=device), rewards)
            rewards = torch.where(done & ~success & neg, torch.tensor(1.0, device=device), rewards)
            
        elif self.reward_type == 4:
            pos = labels == 1
            neg = labels == 0
            
            rewards = torch.where(done & success & pos, torch.tensor(1.0, device=device), rewards)
            rewards = torch.where(done & ~success & pos, torch.tensor(-1.0, device=device), rewards)
            rewards = torch.where(done & success & neg, torch.tensor(-1.0, device=device), rewards)
            rewards = torch.where(done & ~success & neg, torch.tensor(self.rejection_weight, device=device), rewards)
        
        return rewards, terminated, truncated, is_success
    
    def _reset_from_queries(
        self, 
        queries: Tensor, 
        labels: Optional[Tensor] = None,
        active_mask: Tensor = None
    ) -> EnvState:
        """
        Create initial state from queries (pure function, no mutation).
        
        Args:
            queries: [B, 3] or [B, A, 3] Query tensor
            labels:  [B] Optional labels. If None, assumes 1.
            active_mask: [B] Optional bool mask - if provided, only compute derived
                         states for True entries. This dramatically speeds up reset
                         when only a few envs need it.
            
        Returns:
            EnvState containing all mutable state as immutable tensors
        """
        device = self.device
        
        # Pad if 2D
        if queries.ndim == 2:
            B = queries.shape[0]
            padded = torch.full((B, self.padding_atoms, 3), self.padding_idx,
                               dtype=torch.long, device=device)
            padded[:, 0, :] = queries.to(device)
            queries = padded
        else:
            queries = queries.to(device)
            B = queries.shape[0]
            
        if labels is None:
            labels = torch.ones(B, dtype=torch.long, device=device)
        else:
            labels = labels.to(device)
        
        # Initialize state tensors
        current_states = queries.clone()
        original_queries = queries.clone()
        current_labels = labels.clone()
        depths = torch.zeros(B, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        success = torch.zeros(B, dtype=torch.bool, device=device)
        next_var_indices = torch.full(
            (B,), self.runtime_var_start_index,
            dtype=torch.long, device=device
        )
        
        # Initialize history buffer with HASHES (not atoms)
        # Pre-allocate static buffer: [B, max_history_size]
        history_hashes = torch.zeros(
            (B, self.max_history_size),
            dtype=torch.long, device=device
        )
        # Compute hash of initial query and store at position 0
        initial_hash = self._compute_state_hash64(current_states)  # [B]
        history_hashes[:, 0] = initial_hash
        history_count = torch.ones(B, dtype=torch.long, device=device)
        
        # Compute initial derived states (with active_mask for efficiency)
        derived, counts, new_var_indices = self._compute_derived(
            current_states, next_var_indices, original_queries,
            history_hashes, history_count,
            active_mask=active_mask,  # Pass through for optimization
        )
        
        # Initialize per-env pointers if NOT provided in state (assumes 0)
        # Note: They will be overwritten in reset() or step_and_reset() if managed externally
        # But for valid state, we init them.
        per_env_ptrs = torch.zeros(B, dtype=torch.long, device=device)
        step_rewards = torch.zeros(B, dtype=torch.float32, device=device)
        step_dones = torch.zeros(B, dtype=torch.bool, device=device)
        neg_counters = torch.zeros(B, dtype=torch.long, device=device)
        
        return TensorDict({
            "current_states": current_states,
            "derived_states": derived,
            "derived_counts": counts,
            "original_queries": original_queries,
            "next_var_indices": new_var_indices,
            "depths": depths,
            "done": done,
            "success": success,
            "current_labels": current_labels,
            "history_hashes": history_hashes,
            "history_count": history_count,
            "step_rewards": step_rewards,
            "step_dones": step_dones,
            "per_env_ptrs": per_env_ptrs,
            "neg_counters": neg_counters,
        }, batch_size=[B], device=device)
    

    def _set_queries_internal(self, queries: Tensor) -> None:
        """Internal helper to set query pool and initialize pointers."""
        self._query_pool = queries.to(self.device)
        self._per_env_ptrs = torch.arange(self.batch_size, device=self.device)

    def _state_to_obs(self, state: EnvState) -> EnvObs:
        """Convert state to observation (TensorDict)."""
        B = state['current_states'].shape[0]
        action_mask = self._positions_S < state['derived_counts'].unsqueeze(1)
        return TensorDict({
            "sub_index": state['current_states'].unsqueeze(1),
            "derived_sub_indices": state['derived_states'],
            "action_mask": action_mask,
        }, batch_size=[B], device=self.device)

    # =========================================================================
    # 2. Internal Core (Compilable)
    # =========================================================================

    def _step_core(
        self, state: EnvState, actions: Tensor
    ) -> Tuple[EnvObs, EnvState]:
        """
        Internal pure-functional implementation of step().
        This is what actually gets compiled by torch.compile.
        """
        n = actions.shape[0]
        device = self.device
        
        # =================================================================
        # OPTIMIZATION (Issue 4): Mask all computation for done queries
        # =================================================================
        # For done envs: skip all computation, preserve existing state
        # Use torch.where throughout for compile compatibility (no branching)
        was_done = state['done']
        active = ~was_done  # [n] - envs that need computation
        
        # Get selected next states (only meaningful for active envs)
        batch_idx = torch.arange(n, device=device)
        next_states = state['derived_states'][batch_idx, actions]
        
        # Update current states for active envs only
        new_current = torch.where(
            active.view(n, 1, 1),
            next_states,
            state['current_states']
        )
        new_depths = torch.where(
            active,
            state['depths'] + 1,
            state['depths']
        )
        
        # Check termination (results only matter for active envs)
        # Use _get_done_reward for consistent logic
        rewards, terminated, truncated, is_success = self._get_done_reward(
            new_current, state['current_labels'], new_depths, n
        )
        
        # Mask termination checks with active - only active envs can newly terminate
        # _get_done_reward returns raw checks on state, we must mask them
        newly_done = active & (terminated | truncated)
        new_done = was_done | newly_done
        # Accumulate success (once success always success in episode context until reset)
        new_success = state['success'] | (active & is_success)
        
        # Zero out rewards for already done envs or inactive
        rewards = torch.where(active, rewards, torch.zeros_like(rewards))
        
        # Update history only for active envs
        
        # Update history only for active envs
        write_pos = state['history_count'].clamp(max=self.max_history_size - 1)
        new_state_hash = self._compute_state_hash64(new_current)  # [n]
        
        # Scatter: write hash to history at write_pos for active envs only
        update_val = torch.where(active, new_state_hash, state['history_hashes'][batch_idx, write_pos])
        new_history_hashes = state['history_hashes'].clone()
        new_history_hashes.scatter_(1, write_pos.unsqueeze(1), update_val.unsqueeze(1))
        
        # Increment count for active envs only
        new_history_count = torch.where(
            active,
            (state['history_count'] + 1).clamp(max=self.max_history_size),
            state['history_count']
        )
        
        # =================================================================
        # Compute derived states - skip for envs that are now done
        # =================================================================
        # still_active = envs that will need derived states for next step
        still_active = ~new_done
        
        # Compute derived with active_mask - done envs get padding input
        new_derived, new_counts, new_var = self._compute_derived(
            new_current, state['next_var_indices'], state['original_queries'],
            new_history_hashes, new_history_count,
            active_mask=still_active,
        )
        # Preserve old derived states for done envs
        new_derived = torch.where(
            still_active.view(n, 1, 1, 1),
            new_derived,
            state['derived_states']
        )
        new_counts = torch.where(still_active, new_counts, state['derived_counts'])
        new_var = torch.where(still_active, new_var, state['next_var_indices'])
        
        # Create new state
        # Create new state
        new_state = TensorDict({
            "current_states": new_current,
            "derived_states": new_derived,
            "derived_counts": new_counts,
            "original_queries": state['original_queries'],
            "next_var_indices": new_var,
            "depths": new_depths,
            "done": new_done,
            "success": new_success,
            "current_labels": state['current_labels'],
            "history_hashes": new_history_hashes,
            "history_count": new_history_count,
            "step_rewards": rewards,
            "step_dones": newly_done,
            "per_env_ptrs": state['per_env_ptrs'], # Preserve pointers
            "neg_counters": state['neg_counters'], # Preserve counters
        }, batch_size=[n], device=device)
        
        # Create observation
        action_mask = self._positions_S < new_counts.unsqueeze(1)
        
        obs = TensorDict({
            "sub_index": new_current.unsqueeze(1),
            "derived_sub_indices": new_derived,
            "action_mask": action_mask,
        }, batch_size=[n], device=device)
        
        return obs, new_state
    
    def _step_and_reset_core(
        self,
        state: EnvState,
        actions: Tensor,
        query_pool: Tensor,
        per_env_ptrs: Tensor,
    ) -> Tuple[EnvObs, EnvState]:
        """
        Internal: Fused step + reset in one compiled graph.
        
        This avoids Python-level orchestration between step and reset,
        keeping all operations in a single CUDA graph for maximum performance.
        
        Returns:
            (new_obs, new_state)
        """
        # First execute the step
        new_obs_step, next_state = self._step_core(state, actions)
        new_state = next_state
        rewards = next_state['step_rewards']
        
        # Detect newly done environments
        # (calculated in step_core as step_dones)
        newly_done = next_state['step_dones']
        
        # === Vectorized reset using masked operations ===
        device = self.device
        n = state['current_states'].shape[0]
        reset_mask = newly_done
        
        # Select Queries logic
        pool_size = query_pool.shape[0]
        
        if self.order:
            # Ordered
            safe_indices = per_env_ptrs % pool_size
            next_ptrs = (per_env_ptrs + 1) % pool_size
        else:
            # Random
            # Generate random indices for ALL envs (simplest static shape)
            # Or mix?
            safe_indices = torch.randint(0, pool_size, (n,), device=device)
            # next_ptrs irrelevant but keep stable
            next_ptrs = per_env_ptrs 
            
        candidate_queries = query_pool[safe_indices]  # [B, 3]
        
        # Apply reset mask to pointers
        new_ptrs = torch.where(reset_mask, next_ptrs, per_env_ptrs)
        
        # Create padding query [3]
        padding_atom = torch.full((3,), self.padding_idx, dtype=torch.long, device=device)
        
        # Mask: if NOT reset, use padding (fast to process)
        reset_mask_3 = reset_mask.unsqueeze(-1).expand(-1, 3)
        queries_for_reset = torch.where(reset_mask_3, candidate_queries, padding_atom)
        
        # Negative Sampling
        # reset_mask indicates which envs are resetting
        # We need labels for candidate_queries (default 1)
        candidate_labels = torch.ones(n, dtype=torch.long, device=device)
        
        # Call sample_negatives (updates queries, labels, and counters for reset envs)
        # Pass current counters from state
        current_counters = state['neg_counters']
        queries_for_reset, labels_for_reset, new_counters = self.sample_negatives(
            queries_for_reset, candidate_labels, reset_mask, current_counters
        )
        
        # Compute initial state for reset queries
        reset_state = self._reset_from_queries(
            queries_for_reset, labels_for_reset, active_mask=reset_mask
        )
        
        # Mix states: where(done, reset_state, new_state)
        reset_mask_A3 = reset_mask.view(-1, 1, 1).expand(-1, self.padding_atoms, 3)
        reset_mask_SA3 = reset_mask.view(-1, 1, 1, 1).expand(-1, self.padding_states, self.padding_atoms, 3)
        reset_mask_H = reset_mask.view(-1, 1).expand(-1, self.max_history_size)
        
        mixed_state = TensorDict({
            "current_states": torch.where(reset_mask_A3, reset_state['current_states'], new_state['current_states']),
            "derived_states": torch.where(reset_mask_SA3, reset_state['derived_states'], new_state['derived_states']),
            "derived_counts": torch.where(reset_mask, reset_state['derived_counts'], new_state['derived_counts']),
            "original_queries": torch.where(reset_mask_A3, reset_state['original_queries'], new_state['original_queries']),
            "next_var_indices": torch.where(reset_mask, reset_state['next_var_indices'], new_state['next_var_indices']),
            "depths": torch.where(reset_mask, reset_state['depths'], new_state['depths']),
            "done": torch.where(reset_mask, reset_state['done'], new_state['done']),
            "success": torch.where(reset_mask, reset_state['success'], new_state['success']),
            "current_labels": torch.where(reset_mask, reset_state['current_labels'], new_state['current_labels']),
            "history_hashes": torch.where(reset_mask_H, reset_state['history_hashes'], new_state['history_hashes']),
            "history_count": torch.where(reset_mask, reset_state['history_count'], new_state['history_count']),
            "step_rewards": rewards, # Keep original step rewards
            "step_dones": newly_done, # Keep original step dones
            "per_env_ptrs": new_ptrs, # Updated pointers
            "neg_counters": new_counters, # Updated counters (masked inside sample_negatives)
        }, batch_size=[n], device=device)
        
        # Update observation
        action_mask = self._positions_S < mixed_state['derived_counts'].unsqueeze(1)
        new_obs = TensorDict({
            "sub_index": mixed_state['current_states'].unsqueeze(1),
            "derived_sub_indices": mixed_state['derived_states'],
            "action_mask": action_mask,
        }, batch_size=[n], device=device)
        
        return new_obs, mixed_state

    def step(
        self,
        state: EnvState,
        actions: Tensor,
    ) -> Tuple[EnvObs, EnvState]:
        """
        Execute a step in the environment.
        
        Args:
            state: Current EnvState
            actions: [B] Action indices
            
        Returns:
            (obs, new_state)
        """
        if self._compiled:
            torch.compiler.cudagraph_mark_step_begin()
            result = self._step_fn(state, actions)
            obs, new_state = result
            return (
                self._clone_obs(obs),
                self._clone_state(new_state),
            )
        else:
            return self._step_fn(state, actions)
    
    def step_and_reset(
        self,
        state: EnvState,
        actions: Tensor,
    ) -> Tuple[EnvObs, EnvState]:
        """
        Execute step and handle resets (high-performance training API).
        
        Args:
            state: Current EnvState
            actions: [B] Action indices
            
        Returns:
            (new_obs, new_state)
        """
        query_pool = self._query_pool
        per_env_ptrs = self._per_env_ptrs
        
        if query_pool is None:
            # No query pool - just step without reset
            return self.step(state, actions)
        
        if self._compiled:
            torch.compiler.cudagraph_mark_step_begin()
            new_obs, new_state = self._step_and_reset_fn(
                state, actions, query_pool, per_env_ptrs
            )
            # Update pointers from state
            self._per_env_ptrs = new_state['per_env_ptrs'].clone()
            
            return (
                self._clone_obs(new_obs),
                self._clone_state(new_state),
            )
        else:
            new_obs, new_state = self._step_and_reset_fn(
                state, actions, query_pool, per_env_ptrs
            )
            self._per_env_ptrs = new_state['per_env_ptrs']
            return new_obs, new_state

    def _compute_state_hash64(self, states: Tensor) -> Tensor:
        """
        Compute order-independent 64-bit hash for states.
        Matches memory.py's _state_hash64 implementation.
        
        Args:
            states: [B, A, 3] or [B, K, A, 3] tensor of states
            
        Returns:
            Hash tensor of shape [B] or [B, K]
        """
        pad = self.padding_idx
        mix = self._hash_mix_const
        mask63 = self._hash_mask63
        
        if states.dim() == 4:
            # [B, K, A, 3] - batch of derived states
            B, K, A, D = states.shape
            states_flat = states.view(B * K, A, D)
            hashes_flat = self._compute_state_hash64(states_flat)
            return hashes_flat.view(B, K)
        
        # states: [B, A, 3]
        B, A, D = states.shape
        s = states.long()
        
        # Validity mask - exclude padding and terminal predicates
        preds = s[:, :, 0]  # [B, A]
        valid = preds != pad
        
        # Filter terminal predicates (matching memory.py behavior)
        if self.true_pred_idx is not None:
            valid = valid & (preds != self.true_pred_idx)
        if self.false_pred_idx is not None:
            valid = valid & (preds != self.false_pred_idx)
        if self.end_pred_idx is not None:
            valid = valid & (preds != self.end_pred_idx)
        
        # Pack atoms: pred * base^2 + arg0 * base + arg1
        # CRITICAL: Use same pack_base as BloomFilter for identical hash values
        base = self._hash_pack_base
        packed = ((s[:, :, 0] * base + s[:, :, 1]) * base + s[:, :, 2]) & mask63  # [B, A]
        
        # Mix for better distribution (golden ratio constant)
        mixed = (packed * mix) & mask63
        
        # Sum valid atoms - order independent
        h = torch.where(valid, mixed, torch.zeros_like(mixed)).sum(dim=1) & mask63  # [B]
        
        return h
    
    def _prune_visited_states(
        self,
        derived_states: Tensor,     # [B, K, A, 3]
        derived_counts: Tensor,     # [B]
        history_hashes: Tensor,     # [B, H] - 64-bit hashes
        history_count: Tensor,      # [B]
    ) -> Tuple[Tensor, Tensor]:
        """
        Remove derived states that match any visited state hash (cycle prevention).
        Uses GPU broadcasting - no Python loops.
        
        Now uses full state hashing (order-independent) instead of first-atom comparison.
        This matches original env.py memory pruning behavior.
        
        Returns:
            pruned_states: [B, K, A, 3] with visited states zeroed out
            pruned_counts: [B] new counts after pruning
        """
        B, K, A, _ = derived_states.shape
        H = history_hashes.shape[1]
        pad = self.padding_idx
        device = self.device
        
        # Compute hash of each derived state: [B, K]
        derived_hashes = self._compute_state_hash64(derived_states)  # [B, K]
        
        # Broadcast compare: [B, K, 1] vs [B, 1, H] -> [B, K, H]
        derived_exp = derived_hashes.unsqueeze(2)  # [B, K, 1]
        history_exp = history_hashes.unsqueeze(1)   # [B, 1, H]
        
        # Match if hashes are equal
        matches = (derived_exp == history_exp)  # [B, K, H]
        
        # Create valid history mask: only compare against valid history entries
        history_valid = torch.arange(H, device=device).unsqueeze(0) < history_count.unsqueeze(1)  # [B, H]
        history_valid_exp = history_valid.unsqueeze(1)  # [B, 1, H]
        
        # State is visited if it matches ANY valid history entry
        is_visited = ((matches & history_valid_exp).sum(dim=-1) > 0)  # [B, K]
        
        # Create valid derived mask
        derived_valid = torch.arange(K, device=device).unsqueeze(0) < derived_counts.unsqueeze(1)  # [B, K]
        
        # States to keep: valid AND not visited
        keep_mask = derived_valid & ~is_visited  # [B, K]
        
        # Zero out visited states using torch.where
        pruned_states = torch.where(
            keep_mask.unsqueeze(-1).unsqueeze(-1),
            derived_states,
            torch.full_like(derived_states, pad)
        )
        
        # Recount: sum of keep_mask
        pruned_counts = keep_mask.sum(dim=-1)  # [B]
        
        # Compact valid states to front using vectorized scatter
        target_pos = keep_mask.long().cumsum(dim=1) - 1  # [B, K], 0-indexed
        target_pos = torch.where(keep_mask, target_pos, torch.full_like(target_pos, K - 1))  # Invalid -> end
        
        # Initialize output with padding
        compacted = torch.full_like(derived_states, pad)
        
        # Use scatter with expanded index: [B, K] -> [B, K, A, 3]
        target_pos_exp = target_pos.unsqueeze(-1).unsqueeze(-1).expand_as(derived_states)
        
        # Mask the source: only scatter kept entries
        src_data = torch.where(
            keep_mask.unsqueeze(-1).unsqueeze(-1),
            derived_states,
            torch.zeros_like(derived_states)
        )
        
        # Scatter: compacted[b, target_pos[b,k], a, d] = src_data[b, k, a, d]
        compacted.scatter_(1, target_pos_exp, src_data)
        
        return compacted, pruned_counts
    
    def _compute_derived(
        self,
        current_states: Tensor,
        next_var_indices: Tensor,
        original_queries: Tensor,
        history_hashes: Tensor = None,
        history_count: Tensor = None,
        active_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute derived states (pure function, no mutation).
        
        IMPORTANT: States exceeding padding_atoms are REJECTED, never truncated.
        Truncation would change the semantics of the state.
        This matches original _postprocess behavior at line 1344-1347.
        
        Args:
            current_states: [n, A, 3] Current states
            next_var_indices: [n] Next variable indices
            original_queries: [n, A, 3] Original queries for exclusion
            history_hashes: [n, H] State history hashes for cycle detection
            history_count: [n] Valid entries in history
            active_mask: [n] Optional mask - if provided, only compute for True entries.
                         For False entries, returns zeros (caller should use old values).
        """
        n = current_states.shape[0]
        pad = self.padding_idx
        
        # =================================================================
        # OPTIMIZATION (Issue 4): Skip unification for inactive queries
        # =================================================================
        # The unification engine (get_derived_states_compiled) is expensive.
        # For done queries, we can replace their current_states with padding
        # so the engine does minimal work (returns single FALSE state).
        if active_mask is not None:
            # Replace inactive queries with padding state
            # This makes the unification engine process them trivially
            padding_state = torch.full_like(current_states, pad)
            current_states = torch.where(
                active_mask.view(n, 1, 1),
                current_states,
                padding_state
            )
            # Also zero out next_var_indices for inactive (though not critical)
            next_var_indices = torch.where(
                active_mask,
                next_var_indices,
                torch.zeros_like(next_var_indices)
            )
        
        excluded = original_queries[:, 0:1, :]  # [n, 1, 3]
        derived, counts, new_var = self.engine.get_derived_states_compiled(
            current_states, next_var_indices, excluded
        )
        
        K, M = derived.shape[1], derived.shape[2]
        
        # Step 1: Compute validity masks matching original _postprocess logic
        # a) Within count mask
        within_count = torch.arange(K, device=self.device).unsqueeze(0) < counts.unsqueeze(1)  # [n, K]
        
        # b) Atom budget check - CRITICAL: reject (not truncate) states exceeding padding_atoms
        # This matches original env.py _postprocess lines 1339-1345
        valid_atom = derived[:, :, :, 0] != pad  # [n, K, M]
        atom_counts = valid_atom.sum(dim=2)  # [n, K] - count of non-padding atoms per derived state
        within_atom_budget = atom_counts <= self.padding_atoms  # [n, K]
        
        # c) State not empty check - use sum instead of .any() for compile compatibility
        state_nonempty = valid_atom.sum(dim=2) > 0  # [n, K]
        
        # Combined validity: within count, within budget, and not empty
        base_valid = within_count & within_atom_budget & state_nonempty  # [n, K]
        
        # Zero out invalid entries (beyond count OR exceeding atom budget)
        derived = torch.where(
            base_valid.unsqueeze(-1).unsqueeze(-1),
            derived,
            torch.full_like(derived, pad)
        )
        
        # Recount valid entries after atom budget rejection
        new_counts = base_valid.sum(dim=1)  # [n]
        
        # Step 1b: Compact valid states to front using scatter (compile-friendly)
        # This replaces the boolean indexing approach
        # Strategy: Use cumsum to compute target positions, then scatter
        target_pos = torch.cumsum(base_valid.long(), dim=1) - 1  # [n, K]
        target_pos = torch.where(
            base_valid,
            target_pos.clamp(min=0, max=K - 1),
            torch.full_like(target_pos, K - 1)  # Invalid entries go to last position (will be overwritten)
        )
        
        # Allocate output
        compact = torch.full_like(derived, pad)
        
        # Expand target_pos for scatter: [n, K] -> [n, K, M, 3]
        target_pos_exp = target_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3)
        
        # Mask source: only scatter valid entries
        src_data = torch.where(
            base_valid.unsqueeze(-1).unsqueeze(-1),
            derived,
            torch.zeros_like(derived)
        )
        
        # Scatter: compact[b, target_pos[b,k], m, d] = src_data[b, k, m, d]
        compact.scatter_(1, target_pos_exp, src_data)
        derived = compact
        
        # Step 2: Inject FALSE if no valid derivations remain
        # This matches original _postprocess: needs_false = new_counts == 0
        needs_false = new_counts == 0
        if self.false_pred_idx is not None:
            # Create FALSE state for rows that need it
            false_state = torch.full((K, M, 3), pad, device=self.device, dtype=torch.long)
            false_state[0, 0, 0] = self.false_pred_idx
            
            # Replace entire derived tensor for rows needing FALSE
            derived = torch.where(
                needs_false.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                false_state.unsqueeze(0).expand(n, -1, -1, -1),
                derived
            )
            new_counts = torch.where(needs_false, torch.ones_like(new_counts), new_counts)
        
        # Step 3: Normalize atoms dimension to padding_atoms
        # Padding UP if M < padding_atoms
        if M < self.padding_atoms:
            pad_a = torch.full(
                (n, K, self.padding_atoms - M, 3),
                pad, dtype=torch.long, device=self.device
            )
            derived = torch.cat([derived, pad_a], dim=2)
        elif M > self.padding_atoms:
            # Slice to padding_atoms for dimension consistency
            # This is SAFE because states > padding_atoms are REJECTED above (zeroed out)
            # We're only removing the extra padding dimensions, not actual atom data
            derived = derived[:, :, :self.padding_atoms, :]
        
        # Step 4: Pad states dimension
        if K < self.padding_states:
            pad_s = torch.full(
                (n, self.padding_states - K, self.padding_atoms, 3),
                pad, dtype=torch.long, device=self.device
            )
            derived = torch.cat([derived, pad_s], dim=1)
        elif K > self.padding_states:
            derived = derived[:, :self.padding_states]
            new_counts = new_counts.clamp(max=self.padding_states)
        
        # Step 5: Memory pruning - remove visited states (cycle detection)
        if self.memory_pruning and history_hashes is not None and history_count is not None:
            derived, new_counts = self._prune_visited_states(
                derived, new_counts, history_hashes, history_count
            )
            
            # Re-inject FALSE if pruning eliminated all derivations
            needs_false_after = new_counts == 0
            if self.false_pred_idx is not None:
                false_state_full = torch.full(
                    (self.padding_states, self.padding_atoms, 3), 
                    pad, device=self.device, dtype=torch.long
                )
                false_state_full[0, 0, 0] = self.false_pred_idx
                
                derived = torch.where(
                    needs_false_after.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                    false_state_full.unsqueeze(0).expand(n, -1, -1, -1),
                    derived
                )
                new_counts = torch.where(needs_false_after, torch.ones_like(new_counts), new_counts)
        
        # Step 6: Add end action
        if self.end_proof_action and self.end_state is not None:
            # CRITICAL: Cap counts to padding_states - 1 BEFORE adding Endf
            # This matches tensor env.py behavior (lines 893-896):
            # max_for_endf = self.padding_states - 1
            # derived_counts_subset = torch.clamp(derived_counts_subset, max=max_for_endf)
            # This ensures there's always room for Endf by truncating excess derived states
            max_for_endf = self.padding_states - 1
            new_counts = torch.clamp(new_counts, max=max_for_endf)
            
            derived, new_counts = self._add_end_action(
                current_states, derived, new_counts
            )
        
        return derived, new_counts, new_var
    
    def _add_end_action(
        self,
        current_states: Tensor,
        states: Tensor,
        counts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Add END action - pure function version.
        
        IMPORTANT: This must check the DERIVED states for terminal predicates,
        not the current state. If any derived state has TRUE/FALSE, don't add Endf.
        This matches the original _add_end_action behavior.
        """
        n = states.shape[0]
        S = states.shape[1]  # padding_states
        A = states.shape[2]  # padding_atoms
        
        # Check DERIVED states for terminal predicates (not current state!)
        # This matches original: first_preds = states[:, :, 0, 0]  # [A, K]
        derived_first_preds = states[:, :, 0, 0]  # [n, S]
        
        # Create valid mask for derived states
        valid_mask = torch.arange(S, device=self.device).unsqueeze(0) < counts.unsqueeze(1)  # [n, S]
        
        # Check for terminal predicates among valid derived states
        is_true_derived = torch.zeros_like(valid_mask)
        is_false_derived = torch.zeros_like(valid_mask)
        
        if self.true_pred_idx is not None:
            is_true_derived = (derived_first_preds == self.true_pred_idx)
        if self.false_pred_idx is not None:
            is_false_derived = (derived_first_preds == self.false_pred_idx)
        
        is_terminal_derived = is_true_derived | is_false_derived
        
        # has_terminal_outcome: any valid derived state has True or False predicate
        # Use sum instead of any for compile-friendliness
        has_terminal_outcome = ((is_terminal_derived & valid_mask).sum(dim=1) > 0) & (counts > 0)
        
        # Can add Endf only if no terminal outcome and room available
        can_end = ~has_terminal_outcome & (counts < self.padding_states)
        
        # Use scatter to place end_state at position counts[i] for each batch
        new_states = states.clone()
        
        # Create one-hot mask for position counts[i]
        positions = torch.arange(S, device=self.device).unsqueeze(0)  # [1, S]
        at_end_pos = (positions == counts.unsqueeze(1))  # [n, S]
        
        # Apply end state at end position where can_end is True
        should_place = can_end.unsqueeze(1) & at_end_pos  # [n, S]
        
        # Expand end_state to match: [n, S, A, 3]
        end_expanded = self.end_state.unsqueeze(0).unsqueeze(0).expand(n, S, -1, -1)
        
        # Apply using where
        new_states = torch.where(
            should_place.unsqueeze(-1).unsqueeze(-1),
            end_expanded,
            new_states
        )
        
        new_counts = torch.where(can_end, counts + 1, counts)
        return new_states, new_counts

    
    def _clone_state(self, state: EnvState) -> EnvState:
        """Clone state tensors to break CUDA graph aliasing."""
        return state.clone()
    
    def _clone_obs(self, obs: EnvObs) -> EnvObs:
        """Clone obs tensors to break CUDA graph aliasing."""
        return obs.clone()
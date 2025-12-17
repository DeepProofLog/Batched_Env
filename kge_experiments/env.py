"""
Optimized Evaluation Environment with Single-Step Compilation.

This module provides a compilation-friendly evaluation environment using:
- Single-step compilation (fast compile, Python loop for trajectory)
- NamedTuple instead of TensorDict for observations
- UnificationEngineVectorized for graph-safe unification
- Memory pruning with full state hashing

Key Design:
    Instead of compiling the full 20-step trajectory (~40k nodes, slow),
    this compiles only ONE policy+step transition (~2k nodes, fast).
    A Python loop orchestrates the trajectory.

Usage:
    from env_optimized import EvalEnvOptimized, create_compiled_step_fn
    
    env = EvalEnvOptimized(
        vec_engine=UnificationEngineVectorized(...),
        batch_size=500,
    )
    
    # Compile single step
    compiled_step = create_compiled_step_fn(env, policy_fn)
    
    # Evaluate trajectory
    log_probs, success, lengths, rewards = env.evaluate_trajectory(
        queries, policy_fn, compiled_step_fn=compiled_step
    )
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Any, Optional, Tuple, NamedTuple, Callable

from unification import UnificationEngineVectorized


# ============================================================================
# Raw Tensor Observation Type (No TensorDict)
# ============================================================================

class EnvObs(NamedTuple):
    """Observation type for compiled evaluation environment."""
    sub_index: Tensor           # [B, 1, A, 3] Current state
    derived_sub_indices: Tensor # [B, S, A, 3] Successor states
    action_mask: Tensor         # [B, S] Valid action mask


class EnvState(NamedTuple):
    """Immutable state for functional evaluation loop.
    
    This enables full trajectory compilation by avoiding mutable self.* attributes.
    """
    current_states: Tensor      # [B, A, 3] Current proof states
    derived_states: Tensor      # [B, S, A, 3] Successor states
    derived_counts: Tensor      # [B] Number of valid successors
    original_queries: Tensor    # [B, A, 3] Original queries (for exclusion)
    next_var_indices: Tensor    # [B] Next variable index per env
    depths: Tensor              # [B] Current depth per env
    done: Tensor                # [B] Whether env is done
    success: Tensor             # [B] Whether proof succeeded
    # Static history buffer for memory pruning (cycle detection) - stores STATE HASHES
    history_hashes: Tensor      # [B, H] 64-bit hashes of visited states (order-independent)
    history_count: Tensor       # [B] Number of valid history entries


class EnvStepFunctionalOutput(NamedTuple):
    """Output from functional step (includes new state)."""
    state: EnvState            # New state after step
    obs: EnvObs                # Observation
    rewards: Tensor             # [B] Rewards



# ============================================================================
# Compiled Evaluation Environment
# ============================================================================

class Env_vec:
    """
    Optimized evaluation environment with optional compilation.
    
    Args:
        compile: If True, compile step functions for GPU acceleration
        compile_mode: torch.compile mode ('reduce-overhead', 'default', etc.)
    
    Public API:
        reset() -> (obs, state): Initialize environment
        step(state, actions) -> StepResult: Execute one step
        step_and_reset(state, actions) -> (state, obs, rewards, dones): Step + auto-reset
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
        queries: Optional[Tensor] = None,
        sample_deterministic_per_env: bool = False,
        # Compilation settings
        compile: bool = False,
        compile_mode: str = 'reduce-overhead',
        compile_fullgraph: bool = True,
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
        self.end_state = None
        if self.end_pred_idx is not None and self.end_pred_idx >= 0:
            end_state = torch.full((padding_atoms, 3), self.padding_idx,
                                   dtype=torch.long, device=self.device)
            end_state[0, 0] = self.end_pred_idx
            end_state[0, 1] = self.padding_idx
            end_state[0, 2] = self.padding_idx
            self.end_state = end_state
        
        # Pre-allocated static tensors for CUDA graph stability
        self._positions_S = torch.arange(padding_states, device=self.device).unsqueeze(0)
        self._batch_idx_B = None
        
        # Query storage for training
        if queries is not None:
            self._query_pool = queries.to(self.device)
            self._per_env_ptrs = torch.arange(batch_size, device=self.device)
        else:
            self._query_pool = None
            self._per_env_ptrs = None
        self._current_state: Optional[EnvState] = None
        self._current_obs: Optional[EnvObs] = None
        
        # =====================================================================
        # Compile functions if requested
        # =====================================================================
        if compile:
            self._setup_compilation(compile_mode, compile_fullgraph)
        else:
            # Eager mode: functions are used directly
            self._step_fn = self._step_core
            self._step_and_reset_fn = self._step_and_reset_core
    
    def _setup_compilation(self, mode: str, fullgraph: bool) -> None:
        """Internal: Setup compiled versions of step functions."""
        import os
        import torch._inductor.config as inductor_config
        
        torch.set_float32_matmul_precision('high')
        os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'
        os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING'] = '0'
        os.environ['TORCHINDUCTOR_FREEZING'] = '0'
        inductor_config.compile_threads = 4
        
        print(f"Compiling env.step (mode='{mode}', fullgraph={fullgraph})...")
        
        self._step_fn = torch.compile(
            self._step_core,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=False,
        )
        
        self._step_and_reset_fn = torch.compile(
            self._step_and_reset_core,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=False,
        )
    
    def compile(
        self,
        mode: str = 'reduce-overhead',
        fullgraph: bool = True,
    ) -> None:
        """
        Compile environment for faster execution.
        
        Alternative to passing compile=True at construction time.
        Useful for deferred compilation or changing compile settings.
        """
        self._compiled = True
        self._setup_compilation(mode, fullgraph)


    def set_queries(self, queries: Tensor) -> None:
        """
        Set the query pool for training with round-robin cycling.
        
        This mirrors BatchedEnv's approach of storing queries internally.
        Call reset() after this to initialize the environment.
        
        Args:
            queries: [N, 3] Query tensor (pool of training queries)
        """
        self._query_pool = queries.to(self.device)
        # Initialize per-env pointers: env i starts pointing at query i
        # This matches BatchedEnv's _per_env_train_ptrs initialization
        self._per_env_ptrs = torch.arange(self.batch_size, device=self.device)
    
    def reset_from_queries(self, queries: Tensor, active_mask: Tensor = None) -> EnvState:
        """
        Create initial state from queries (pure function, no mutation).
        
        Args:
            queries: [B, 3] or [B, A, 3] Query tensor
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
        
        # Initialize state tensors
        current_states = queries.clone()
        original_queries = queries.clone()
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
        derived, counts, new_var_indices = self._compute_derived_functional(
            current_states, next_var_indices, original_queries,
            history_hashes, history_count,
            active_mask=active_mask,  # Pass through for optimization
        )
        
        return EnvState(
            current_states=current_states,
            derived_states=derived,
            derived_counts=counts,
            original_queries=original_queries,
            next_var_indices=new_var_indices,
            depths=depths,
            done=done,
            success=success,
            history_hashes=history_hashes,
            history_count=history_count,
        )
    
    def reset(self) -> Tuple[EnvObs, EnvState]:
        """
        Reset environment using stored query pool with round-robin cycling.
        
        Each environment gets the query at its pointer position.
        Pointers are NOT advanced here - they're advanced in step_with_policy
        when an environment finishes (newly_done).
        
        Returns:
            Tuple of (EnvObs, EnvState)
        """
        if self._query_pool is None:
            raise RuntimeError("Must call set_queries() before reset()")
        
        pool_size = self._query_pool.shape[0]
        
        # Get queries for each env using current pointers (modulo pool size)
        query_indices = self._per_env_ptrs % pool_size
        init_queries = self._query_pool[query_indices]  # [batch_size, 3]
        
        # Advance pointers for next reset (the first reset after init picks next query)
        self._per_env_ptrs = (self._per_env_ptrs + 1) % pool_size

        
        # Initialize state
        state = self.reset_from_queries(init_queries)
        
        # Create observation
        action_mask = self._positions_S < state.derived_counts.unsqueeze(1)
        obs = EnvObs(
            sub_index=state.current_states.unsqueeze(1),
            derived_sub_indices=state.derived_states,
            action_mask=action_mask,
        )
        
        self._current_state = state
        self._current_obs = obs
        
        return obs, state
    


    def _step_core(
        self, state: EnvState, actions: Tensor
    ) -> EnvStepFunctionalOutput:
        """
        Internal: Execute one step - pure function, returns new state.
        
        Use step() as the public API.
        
        Args:
            state: Current EnvState
            actions: [B] Action indices
            
        Returns:
            EnvStepFunctionalOutput with (new_state, obs, rewards)
        """
        n = actions.shape[0]
        device = self.device
        
        # =================================================================
        # OPTIMIZATION (Issue 4): Mask all computation for done queries
        # =================================================================
        # For done envs: skip all computation, preserve existing state
        # Use torch.where throughout for compile compatibility (no branching)
        was_done = state.done
        active = ~was_done  # [n] - envs that need computation
        
        # Get selected next states (only meaningful for active envs)
        batch_idx = torch.arange(n, device=device)
        next_states = state.derived_states[batch_idx, actions]
        
        # Update current states for active envs only
        new_current = torch.where(
            active.view(n, 1, 1),
            next_states,
            state.current_states
        )
        new_depths = torch.where(
            active,
            state.depths + 1,
            state.depths
        )
        
        # Check termination (results only matter for active envs)
        first_pred = next_states[:, 0, 0]
        
        is_true = (first_pred == self.true_pred_idx) if self.true_pred_idx else \
                  torch.zeros(n, dtype=torch.bool, device=device)
        is_false = (first_pred == self.false_pred_idx) if self.false_pred_idx else \
                   torch.zeros(n, dtype=torch.bool, device=device)
        is_end = (first_pred == self.end_pred_idx) if self.end_pred_idx else \
                 torch.zeros(n, dtype=torch.bool, device=device)
        is_depth_limit = (new_depths >= self.max_depth)
        
        # Mask termination checks with active - only active envs can newly terminate
        newly_done = active & (is_true | is_false | is_end | is_depth_limit)
        new_done = was_done | newly_done
        new_success = state.success | (active & is_true)
        
        # Rewards (only for active envs that found TRUE)
        rewards = torch.zeros(n, device=device)
        rewards = torch.where(active & is_true, torch.ones_like(rewards), rewards)
        
        # Update history only for active envs
        write_pos = state.history_count.clamp(max=self.max_history_size - 1)
        new_state_hash = self._compute_state_hash64(new_current)  # [n]
        
        # Scatter: write hash to history at write_pos for active envs only
        update_val = torch.where(active, new_state_hash, state.history_hashes[batch_idx, write_pos])
        new_history_hashes = state.history_hashes.clone()
        new_history_hashes.scatter_(1, write_pos.unsqueeze(1), update_val.unsqueeze(1))
        
        # Increment count for active envs only
        new_history_count = torch.where(
            active,
            (state.history_count + 1).clamp(max=self.max_history_size),
            state.history_count
        )
        
        # =================================================================
        # Compute derived states - skip for envs that are now done
        # =================================================================
        # still_active = envs that will need derived states for next step
        still_active = ~new_done
        
        # Compute derived with active_mask - done envs get padding input
        new_derived, new_counts, new_var = self._compute_derived_functional(
            new_current, state.next_var_indices, state.original_queries,
            new_history_hashes, new_history_count,
            active_mask=still_active,
        )
        # Preserve old derived states for done envs
        new_derived = torch.where(
            still_active.view(n, 1, 1, 1),
            new_derived,
            state.derived_states
        )
        new_counts = torch.where(still_active, new_counts, state.derived_counts)
        new_var = torch.where(still_active, new_var, state.next_var_indices)
        
        # Create new state
        new_state = EnvState(
            current_states=new_current,
            derived_states=new_derived,
            derived_counts=new_counts,
            original_queries=state.original_queries,
            next_var_indices=new_var,
            depths=new_depths,
            done=new_done,
            success=new_success,
            history_hashes=new_history_hashes,
            history_count=new_history_count,
        )
        
        # Create observation
        action_mask = self._positions_S < new_counts.unsqueeze(1)
        
        obs = EnvObs(
            sub_index=new_current.unsqueeze(1),
            derived_sub_indices=new_derived,
            action_mask=action_mask,
        )
        
        return EnvStepFunctionalOutput(state=new_state, obs=obs, rewards=rewards)
    
    def _step_and_reset_core(
        self,
        state: EnvState,
        actions: Tensor,
        query_pool: Tensor,
        per_env_ptrs: Tensor,
    ) -> Tuple[EnvState, EnvObs, Tensor, Tensor, Tensor]:
        """
        Internal: Fused step + reset in one compiled graph.
        
        This avoids Python-level orchestration between step and reset,
        keeping all operations in a single CUDA graph for maximum performance.
        
        Returns:
            (new_state, new_obs, rewards, newly_done, new_ptrs)
        """
        # First execute the step
        step_result = self._step_core(state, actions)
        new_state = step_result.state
        rewards = step_result.rewards
        
        # Detect newly done environments
        was_done = state.done
        newly_done = new_state.done & ~was_done
        
        # === Vectorized reset using masked operations ===
        device = self.device
        n = state.current_states.shape[0]
        reset_mask = newly_done
        
        # Get queries for reset using CURRENT pointer (before advancing)
        pool_size = query_pool.shape[0]
        safe_indices = per_env_ptrs % pool_size
        candidate_queries = query_pool[safe_indices]  # [B, 3]
        
        # Compute next pointers for envs that reset
        next_ptrs = (per_env_ptrs + 1) % pool_size
        new_ptrs = torch.where(reset_mask, next_ptrs, per_env_ptrs)
        
        # Create padding query [3]
        padding_atom = torch.full((3,), self.padding_idx, dtype=torch.long, device=device)
        
        # Mask: if NOT reset, use padding (fast to process)
        reset_mask_3 = reset_mask.unsqueeze(-1).expand(-1, 3)
        queries_for_reset = torch.where(reset_mask_3, candidate_queries, padding_atom)
        
        # Compute initial state for reset queries (with active_mask to skip unification for non-reset envs)
        reset_state = self.reset_from_queries(queries_for_reset, active_mask=reset_mask)
        
        # Mix states: where(done, reset_state, new_state)
        reset_mask_A3 = reset_mask.view(-1, 1, 1).expand(-1, self.padding_atoms, 3)
        reset_mask_SA3 = reset_mask.view(-1, 1, 1, 1).expand(-1, self.padding_states, self.padding_atoms, 3)
        reset_mask_H = reset_mask.view(-1, 1).expand(-1, self.max_history_size)
        
        mixed_state = EnvState(
            current_states=torch.where(reset_mask_A3, reset_state.current_states, new_state.current_states),
            derived_states=torch.where(reset_mask_SA3, reset_state.derived_states, new_state.derived_states),
            derived_counts=torch.where(reset_mask, reset_state.derived_counts, new_state.derived_counts),
            original_queries=torch.where(reset_mask_A3, reset_state.original_queries, new_state.original_queries),
            next_var_indices=torch.where(reset_mask, reset_state.next_var_indices, new_state.next_var_indices),
            depths=torch.where(reset_mask, reset_state.depths, new_state.depths),
            done=torch.where(reset_mask, reset_state.done, new_state.done),
            success=torch.where(reset_mask, reset_state.success, new_state.success),
            history_hashes=torch.where(reset_mask_H, reset_state.history_hashes, new_state.history_hashes),
            history_count=torch.where(reset_mask, reset_state.history_count, new_state.history_count),
        )
        
        # Update observation
        action_mask = self._positions_S < mixed_state.derived_counts.unsqueeze(1)
        new_obs = EnvObs(
            sub_index=mixed_state.current_states.unsqueeze(1),
            derived_sub_indices=mixed_state.derived_states,
            action_mask=action_mask,
        )
        
        return mixed_state, new_obs, rewards, newly_done, new_ptrs

    def step(
        self,
        state: EnvState,
        actions: Tensor,
    ) -> EnvStepFunctionalOutput:
        """
        Execute a step in the environment.
        
        Args:
            state: Current EnvState
            actions: [B] Action indices
            
        Returns:
            EnvStepFunctionalOutput with (new_state, obs, rewards)
        """
        if self._compiled:
            torch.compiler.cudagraph_mark_step_begin()
            result = self._step_fn(state, actions)
            return EnvStepFunctionalOutput(
                state=self._clone_state(result.state),
                obs=self._clone_obs(result.obs),
                rewards=result.rewards.clone(),
            )
        else:
            return self._step_fn(state, actions)
    
    def step_and_reset(
        self,
        state: EnvState,
        actions: Tensor,
    ) -> Tuple[EnvState, EnvObs, Tensor, Tensor]:
        """
        Execute step and handle resets (high-performance training API).
        
        Args:
            state: Current EnvState
            actions: [B] Action indices
            
        Returns:
            (new_state, new_obs, rewards, newly_done)
        """
        query_pool = self._query_pool
        per_env_ptrs = self._per_env_ptrs
        
        if query_pool is None:
            # No query pool - just step without reset
            result = self.step(state, actions)
            dones = result.state.done & ~state.done
            return result.state, result.obs, result.rewards, dones
        
        if self._compiled:
            torch.compiler.cudagraph_mark_step_begin()
            new_state, new_obs, rewards, newly_done, new_ptrs = self._step_and_reset_fn(
                state, actions, query_pool, per_env_ptrs
            )
            self._per_env_ptrs = new_ptrs.clone()
            return (
                self._clone_state(new_state),
                self._clone_obs(new_obs),
                rewards.clone(),
                newly_done.clone(),
            )
        else:
            new_state, new_obs, rewards, newly_done, new_ptrs = self._step_and_reset_fn(
                state, actions, query_pool, per_env_ptrs
            )
            self._per_env_ptrs = new_ptrs
            return new_state, new_obs, rewards, newly_done

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
    
    def _compute_derived_functional(
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
            
            derived, new_counts = self._add_end_action_functional(
                current_states, derived, new_counts
            )
        
        return derived, new_counts, new_var
    
    def _add_end_action_functional(
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
        return EnvState(
            current_states=state.current_states.clone(),
            derived_states=state.derived_states.clone(),
            derived_counts=state.derived_counts.clone(),
            original_queries=state.original_queries.clone(),
            next_var_indices=state.next_var_indices.clone(),
            depths=state.depths.clone(),
            done=state.done.clone(),
            success=state.success.clone(),
            history_hashes=state.history_hashes.clone(),
            history_count=state.history_count.clone(),
        )
    
    def _clone_obs(self, obs: EnvObs) -> EnvObs:
        """Clone obs tensors to break CUDA graph aliasing."""
        return EnvObs(
            sub_index=obs.sub_index.clone(),
            derived_sub_indices=obs.derived_sub_indices.clone(),
            action_mask=obs.action_mask.clone(),
        )
    

EvalEnvOptimized = Env_vec

"""
Compiled Evaluation Environment for torch.compile() compatibility.

This module provides a fully compilable version of EvalOnlyEnv that:
- Uses NamedTuple instead of TensorDict for observations
- Uses UnificationEngineVectorized for graph-safe unification
- Replaces all loops with vectorized operations
- Supports torch.compile(fullgraph=True)

Usage:
    from env_eval_compiled import EvalOnlyEnvCompiled, EvalObs
    
    env = EvalOnlyEnvCompiled(
        vec_engine=UnificationEngineVectorized(...),
        batch_size=500,
    )
    
    obs = env.reset_batch(n_envs=100)
    # obs is a NamedTuple, not TensorDict
    
    # Compile the step function
    compiled_step = torch.compile(env.step_compiled, mode='reduce-overhead')
    obs, rewards, dones, success = compiled_step(actions)
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Any, Optional, Tuple, NamedTuple

from unification_vectorized import UnificationEngineVectorized


# ============================================================================
# Raw Tensor Observation Type (No TensorDict)
# ============================================================================

class EvalObs(NamedTuple):
    """Observation type for compiled evaluation environment."""
    sub_index: Tensor           # [B, 1, A, 3] Current state
    derived_sub_indices: Tensor # [B, S, A, 3] Successor states
    action_mask: Tensor         # [B, S] Valid action mask


class EvalStepOutput(NamedTuple):
    """Step output for compiled evaluation environment."""
    obs: EvalObs
    rewards: Tensor   # [B]
    dones: Tensor     # [B]  
    success: Tensor   # [B]


class EvalState(NamedTuple):
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


class EvalStepFunctionalOutput(NamedTuple):
    """Output from functional step (includes new state)."""
    state: EvalState            # New state after step
    obs: EvalObs                # Observation
    rewards: Tensor             # [B] Rewards



# ============================================================================
# Compiled Evaluation Environment
# ============================================================================

class EvalOnlyEnvCompiled:
    """
    Fully compilable evaluation environment.
    
    This environment is designed for use with torch.compile(). Key differences
    from EvalOnlyEnv:
    - Returns EvalObs (NamedTuple) instead of TensorDict
    - Uses UnificationEngineVectorized for graph-safe unification
    - All operations are vectorized (no Python loops)
    - Supports torch.compile(fullgraph=True)
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
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.end_proof_action = end_proof_action
        
        # Vectorized unification engine
        self.engine = vec_engine
        
        # Key indices from base engine
        base = vec_engine.engine
        im = getattr(base, 'index_manager', None)
        
        self.padding_idx = vec_engine.padding_idx
        self.true_pred_idx = im.predicate_str2idx.get('True') if im else None
        self.false_pred_idx = im.predicate_str2idx.get('False') if im else None
        self.end_pred_idx = im.predicate_str2idx.get('Endf') if im else None
        
        # Runtime variable start  
        if runtime_var_start_index is not None:
            self.runtime_var_start_index = runtime_var_start_index
        elif im is not None:
            self.runtime_var_start_index = getattr(im, 'constant_no', 0) + 1
        else:
            self.runtime_var_start_index = 1000
        
        # Pre-build end action tensor
        self.end_state = None
        if self.end_pred_idx is not None and self.end_pred_idx >= 0:
            end_state = torch.full((padding_atoms, 3), self.padding_idx,
                                   dtype=torch.long, device=self.device)
            end_state[0, 0] = self.end_pred_idx
            end_state[0, 1] = self.padding_idx
            end_state[0, 2] = self.padding_idx
            self.end_state = end_state  # [A, 3]
        
        # State buffers (pre-allocated on first use)
        self._current_states = None      # [B, A, 3]
        self._original_queries = None    # [B, A, 3]
        self._derived_states = None      # [B, S, A, 3]
        self._derived_counts = None      # [B]
        self._depths = None              # [B]
        self._next_var_indices = None    # [B]
        self._done = None                # [B]
        self._success = None             # [B]
        
        # Dataset handling
        self._queries_dataset = None     # [M, A, 3]
        self._query_ptr = 0
    
    def set_queries(self, queries: Tensor) -> None:
        """Set queries to evaluate."""
        if queries.ndim == 2:
            M = queries.shape[0]
            padded = torch.full((M, self.padding_atoms, 3), self.padding_idx,
                               dtype=torch.long, device=self.device)
            padded[:, 0, :] = queries.to(self.device)
            queries = padded
        else:
            queries = queries.to(self.device)
        
        self._queries_dataset = queries
        self._query_ptr = 0
    
    def reset_batch(self, n_envs: Optional[int] = None) -> Optional[EvalObs]:
        """Reset a batch with the next n_envs queries (raw tensor output)."""
        n = n_envs or self.batch_size
        n = min(n, len(self._queries_dataset) - self._query_ptr)
        
        if n <= 0:
            return None
        
        # Get next batch
        queries = self._queries_dataset[self._query_ptr:self._query_ptr + n]
        self._query_ptr += n
        
        # Initialize buffers
        self._current_states = queries.clone()
        self._original_queries = queries.clone()
        self._depths = torch.zeros(n, dtype=torch.long, device=self.device)
        self._done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._next_var_indices = torch.full(
            (n,), self.runtime_var_start_index,
            dtype=torch.long, device=self.device
        )
        
        # Compute initial derived states
        self._compute_derived_states_compiled()
        
        return self._make_obs_raw()
    
    def step_compiled(self, actions: Tensor) -> EvalStepOutput:
        """
        Execute one step - designed for torch.compile().
        
        Args:
            actions: [n] Action indices
            
        Returns:
            EvalStepOutput with (obs, rewards, dones, success)
        """
        n = len(actions)
        
        # Get selected next states: [n, A, 3]
        batch_idx = torch.arange(n, device=self.device)
        next_states = self._derived_states[batch_idx, actions]
        
        # Update current states for non-done envs using where
        self._current_states = torch.where(
            self._done.view(n, 1, 1),
            self._current_states,
            next_states
        )
        self._depths = torch.where(
            self._done,
            self._depths,
            self._depths + 1
        )
        
        # Check termination
        first_pred = next_states[:, 0, 0]
        
        is_true = (first_pred == self.true_pred_idx) if self.true_pred_idx is not None \
                  else torch.zeros(n, dtype=torch.bool, device=self.device)
        is_false = (first_pred == self.false_pred_idx) if self.false_pred_idx is not None \
                   else torch.zeros(n, dtype=torch.bool, device=self.device)
        is_end = (first_pred == self.end_pred_idx) if self.end_pred_idx is not None \
                 else torch.zeros(n, dtype=torch.bool, device=self.device)
        is_depth_limit = (self._depths >= self.max_depth)
        
        # Update done/success - remember previous done state for rewards
        was_done = self._done.clone()
        newly_done = is_true | is_false | is_end | is_depth_limit
        self._done = self._done | newly_done
        self._success = self._success | is_true
        
        # Rewards: give reward if is_true and wasn't already done
        rewards = torch.zeros(n, device=self.device)
        rewards = torch.where(is_true & ~was_done, torch.ones_like(rewards), rewards)
        
        # Compute derived states for next step
        # We compute for all to maintain fixed shapes (compilation-friendly)
        self._compute_derived_states_compiled()
        
        return EvalStepOutput(
            obs=self._make_obs_raw(),
            rewards=rewards,
            dones=self._done.clone(),
            success=self._success.clone(),
        )
    
    def _compute_derived_states_compiled(self) -> None:
        """Compute successor states using vectorized engine (compilation-safe)."""
        n = self._current_states.shape[0]
        
        # Call vectorized engine
        excluded = self._original_queries[:, 0:1, :]  # [n, 1, 3]
        
        derived, counts, new_var_indices = self.engine.get_derived_states_compiled(
            self._current_states,
            self._next_var_indices,
            excluded,
        )
        
        self._next_var_indices = new_var_indices
        
        # Output shape is [n, K_max, M_max, 3]
        K = derived.shape[1]
        M = derived.shape[2]
        
        # Handle shapes - pad/truncate to padding_atoms and padding_states
        # Atoms: M -> padding_atoms
        if M < self.padding_atoms:
            pad_a = torch.full(
                (n, K, self.padding_atoms - M, 3),
                self.padding_idx, dtype=torch.long, device=self.device
            )
            derived = torch.cat([derived, pad_a], dim=2)
        elif M > self.padding_atoms:
            derived = derived[:, :, :self.padding_atoms, :]
        
        # States: K -> padding_states
        if K < self.padding_states:
            pad_s = torch.full(
                (n, self.padding_states - K, self.padding_atoms, 3),
                self.padding_idx, dtype=torch.long, device=self.device
            )
            derived = torch.cat([derived, pad_s], dim=1)
        elif K > self.padding_states:
            derived = derived[:, :self.padding_states]
            counts = counts.clamp(max=self.padding_states)
        
        # Add end action (vectorized - no loops)
        if self.end_proof_action and self.end_state is not None:
            derived, counts = self._add_end_action_vectorized(derived, counts)
        
        self._derived_states = derived
        self._derived_counts = counts
    
    def _add_end_action_vectorized(
        self, states: Tensor, counts: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Add END action - fully vectorized, no loops."""
        n = states.shape[0]
        
        # Check if current state is terminal
        first_pred = self._current_states[:, 0, 0]
        is_terminal = torch.zeros(n, dtype=torch.bool, device=self.device)
        
        if self.true_pred_idx is not None:
            is_terminal = is_terminal | (first_pred == self.true_pred_idx)
        if self.false_pred_idx is not None:
            is_terminal = is_terminal | (first_pred == self.false_pred_idx)
        
        # Can add end action?
        can_add = (~is_terminal) & (counts < self.padding_states)
        
        # Prepare end state for all batches: [n, A, 3]
        end_state_expanded = self.end_state.unsqueeze(0).expand(n, -1, -1)
        
        # Determine slots (counts = where to insert)
        # Use scatter to place end state at position counts[i]
        # But we need the slot index, which varies per batch element
        
        # Create a mask for valid insertion positions
        # slot_idx = counts (0-indexed position to insert)
        slot_idx = counts.clamp(max=self.padding_states - 1)
        
        # Use advanced indexing with where to avoid loop
        batch_idx = torch.arange(n, device=self.device)
        
        # Clone to avoid in-place modification that might break grads (though we're no_grad)
        states = states.clone()
        
        # Write end state where can_add is True
        # Using where: for each batch element, if can_add, set states[b, slot_idx[b]] = end_state
        # This requires scatter-like behavior
        
        # Alternative: create a full update tensor and use where
        # This is simpler and compilation-friendly
        for_update = states[batch_idx, slot_idx]  # Current values at insert positions
        
        # Create the updated values
        new_vals = torch.where(
            can_add.view(n, 1, 1),
            end_state_expanded,
            for_update
        )
        
        # Write back
        states[batch_idx, slot_idx] = new_vals
        
        # Update counts
        counts = torch.where(can_add, counts + 1, counts)
        
        return states, counts
    
    def _make_obs_raw(self) -> EvalObs:
        """Create observation as NamedTuple (compilation-friendly)."""
        n = self._current_states.shape[0]
        
        # Action mask: vectorized (no loop)
        positions = torch.arange(self.padding_states, device=self.device).unsqueeze(0)
        action_mask = positions < self._derived_counts.unsqueeze(1)
        
        return EvalObs(
            sub_index=self._current_states.unsqueeze(1),
            derived_sub_indices=self._derived_states,
            action_mask=action_mask,
        )
    
    @property
    def remaining_queries(self) -> int:
        """Queries remaining to process."""
        if self._queries_dataset is None:
            return 0
        return len(self._queries_dataset) - self._query_ptr
    
    def reset_pointer(self) -> None:
        """Reset query pointer."""
        self._query_ptr = 0
    
    # =========================================================================
    # Functional API for Full Trajectory Compilation
    # =========================================================================
    
    def init_state_from_queries(self, queries: Tensor) -> EvalState:
        """
        Create initial state from queries (pure function, no mutation).
        
        Args:
            queries: [B, 3] or [B, A, 3] Query tensor
            
        Returns:
            EvalState containing all mutable state as immutable tensors
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
        
        # Compute initial derived states
        derived, counts, new_var_indices = self._compute_derived_functional(
            current_states, next_var_indices, original_queries
        )
        
        return EvalState(
            current_states=current_states,
            derived_states=derived,
            derived_counts=counts,
            original_queries=original_queries,
            next_var_indices=new_var_indices,
            depths=depths,
            done=done,
            success=success,
        )
    
    def _compute_derived_functional(
        self,
        current_states: Tensor,
        next_var_indices: Tensor,
        original_queries: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute derived states (pure function, no mutation)."""
        n = current_states.shape[0]
        
        excluded = original_queries[:, 0:1, :]  # [n, 1, 3]
        derived, counts, new_var = self.engine.get_derived_states_compiled(
            current_states, next_var_indices, excluded
        )
        
        K, M = derived.shape[1], derived.shape[2]
        
        # Pad atoms
        if M < self.padding_atoms:
            pad_a = torch.full(
                (n, K, self.padding_atoms - M, 3),
                self.padding_idx, dtype=torch.long, device=self.device
            )
            derived = torch.cat([derived, pad_a], dim=2)
        elif M > self.padding_atoms:
            derived = derived[:, :, :self.padding_atoms, :]
        
        # Pad states
        if K < self.padding_states:
            pad_s = torch.full(
                (n, self.padding_states - K, self.padding_atoms, 3),
                self.padding_idx, dtype=torch.long, device=self.device
            )
            derived = torch.cat([derived, pad_s], dim=1)
        elif K > self.padding_states:
            derived = derived[:, :self.padding_states]
            counts = counts.clamp(max=self.padding_states)
        
        # Add end action
        if self.end_proof_action and self.end_state is not None:
            derived, counts = self._add_end_action_functional(
                current_states, derived, counts
            )
        
        return derived, counts, new_var
    
    def _add_end_action_functional(
        self,
        current_states: Tensor,
        states: Tensor,
        counts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Add END action - pure function version."""
        n = states.shape[0]
        S = states.shape[1]  # padding_states
        A = states.shape[2]  # padding_atoms
        
        first_pred = current_states[:, 0, 0]
        is_terminal = torch.zeros(n, dtype=torch.bool, device=self.device)
        
        if self.true_pred_idx is not None:
            is_terminal = is_terminal | (first_pred == self.true_pred_idx)
        if self.false_pred_idx is not None:
            is_terminal = is_terminal | (first_pred == self.false_pred_idx)
        
        can_end = ~is_terminal & (counts < self.padding_states)
        
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
    
    def step_functional(
        self, state: EvalState, actions: Tensor
    ) -> EvalStepFunctionalOutput:
        """
        Execute one step - pure function, returns new state.
        
        Args:
            state: Current EvalState
            actions: [B] Action indices
            
        Returns:
            EvalStepFunctionalOutput with (new_state, obs, rewards)
        """
        n = actions.shape[0]
        device = self.device
        
        # Get selected next states
        batch_idx = torch.arange(n, device=device)
        next_states = state.derived_states[batch_idx, actions]
        
        # Update current states for non-done envs
        new_current = torch.where(
            state.done.view(n, 1, 1),
            state.current_states,
            next_states
        )
        new_depths = torch.where(
            state.done,
            state.depths,
            state.depths + 1
        )
        
        # Check termination
        first_pred = next_states[:, 0, 0]
        
        is_true = (first_pred == self.true_pred_idx) if self.true_pred_idx else \
                  torch.zeros(n, dtype=torch.bool, device=device)
        is_false = (first_pred == self.false_pred_idx) if self.false_pred_idx else \
                   torch.zeros(n, dtype=torch.bool, device=device)
        is_end = (first_pred == self.end_pred_idx) if self.end_pred_idx else \
                 torch.zeros(n, dtype=torch.bool, device=device)
        is_depth_limit = (new_depths >= self.max_depth)
        
        # Update done/success - clone to avoid CUDA graph aliasing
        was_done = state.done.clone()
        newly_done = is_true | is_false | is_end | is_depth_limit
        new_done = was_done | newly_done
        new_success = state.success.clone() | is_true
        
        # Rewards
        rewards = torch.zeros(n, device=device)
        rewards = torch.where(is_true & ~was_done, torch.ones_like(rewards), rewards)
        
        # Compute derived states for next step
        new_derived, new_counts, new_var = self._compute_derived_functional(
            new_current, state.next_var_indices, state.original_queries
        )
        
        # Create new state
        new_state = EvalState(
            current_states=new_current,
            derived_states=new_derived,
            derived_counts=new_counts,
            original_queries=state.original_queries,
            next_var_indices=new_var,
            depths=new_depths,
            done=new_done,
            success=new_success,
        )
        
        # Create observation
        positions = torch.arange(self.padding_states, device=device).unsqueeze(0)
        action_mask = positions < new_counts.unsqueeze(1)
        
        obs = EvalObs(
            sub_index=new_current.unsqueeze(1),
            derived_sub_indices=new_derived,
            action_mask=action_mask,
        )
        
        return EvalStepFunctionalOutput(state=new_state, obs=obs, rewards=rewards)
    
    def evaluate_trajectory_compiled(
        self,
        queries: Tensor,
        policy_fn: callable,
        max_steps: int = 20,
        deterministic: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Run full evaluation trajectory in a single compiled graph.
        
        This method runs max_steps iterations without Python control flow,
        enabling full trajectory compilation with torch.compile().
        
        Args:
            queries: [B, 3] Query triples
            policy_fn: Callable(obs) -> logits [B, S]
            max_steps: Maximum trajectory length (fixed, always runs all steps)
            deterministic: Use argmax for action selection
            
        Returns:
            log_probs: [B] Accumulated log probs per query
            success: [B] Whether proof succeeded
            lengths: [B] Trajectory lengths
            rewards: [B] Accumulated rewards
        """
        device = self.device
        
        # Initialize state
        state = self.init_state_from_queries(queries)
        B = state.current_states.shape[0]
        
        # Pre-allocate trajectory accumulators
        total_log_probs = torch.zeros(B, device=device)
        total_rewards = torch.zeros(B, device=device)
        
        # Create initial observation
        positions = torch.arange(self.padding_states, device=device).unsqueeze(0)
        action_mask = positions < state.derived_counts.unsqueeze(1)
        
        obs = EvalObs(
            sub_index=state.current_states.unsqueeze(1),
            derived_sub_indices=state.derived_states,
            action_mask=action_mask,
        )
        
        # Unrolled loop - always runs max_steps iterations
        # Each step is masked by done state
        for step_idx in range(max_steps):
            # Get policy output
            logits = policy_fn(obs)  # [B, S]
            
            # Mask invalid actions
            masked_logits = torch.where(
                obs.action_mask,
                logits,
                torch.full_like(logits, float('-inf'))
            )
            
            # Select actions (masked by done)
            if deterministic:
                actions = masked_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(masked_logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)
            
            # Get log probs
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Accumulate log probs for non-done envs
            active = ~state.done
            total_log_probs = torch.where(
                active,
                total_log_probs + selected_log_probs,
                total_log_probs
            )
            
            # Step environment
            result = self.step_functional(state, actions)
            
            # Accumulate rewards
            total_rewards = total_rewards + result.rewards
            
            # Update state and obs for next iteration
            state = result.state
            obs = result.obs
        
        # Return final results
        return total_log_probs, state.success, state.depths, total_rewards


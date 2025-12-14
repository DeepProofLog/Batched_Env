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
    # Static history buffer for memory pruning (cycle detection) - stores STATE HASHES
    history_hashes: Tensor      # [B, H] 64-bit hashes of visited states (order-independent)
    history_count: Tensor       # [B] Number of valid history entries


class EvalStepFunctionalOutput(NamedTuple):
    """Output from functional step (includes new state)."""
    state: EvalState            # New state after step
    obs: EvalObs                # Observation
    rewards: Tensor             # [B] Rewards



# ============================================================================
# Compiled Evaluation Environment
# ============================================================================

class EvalEnvOptimized:
    """
    Optimized evaluation environment with single-step compilation.
    
    This environment uses torch.compile() for the single-step transition
    (policy + env step) rather than the full trajectory. This provides:
    - Fast compilation: ~5-10s vs ~22-24s for full trajectory
    - Same correctness: Python loop doesn't affect results
    - Good runtime: Minimal Python overhead
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
        memory_pruning: bool = True,  # Enable cycle detection
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.end_proof_action = end_proof_action
        self.memory_pruning = memory_pruning
        self.max_history_size = max_depth + 1  # +1 for initial query
        
        # Hash computation constants (matching memory.py)
        self._hash_mix_const = 0x9E3779B97F4A7C15  # Golden ratio mixing constant
        self._hash_mask63 = (1 << 63) - 1
        
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
        
        # ====================================================================
        # Pre-allocated static tensors for CUDA graph stability
        # These tensors are created once and reused across calls to avoid
        # memory allocation during compiled execution.
        # ====================================================================
        
        # Index tensors - these are constant and can be safely reused
        self._positions_S = torch.arange(padding_states, device=self.device).unsqueeze(0)  # [1, S]
        self._batch_idx_B = None  # Lazy allocated when batch size is known
        
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
        
        # Compute initial derived states (with or without memory pruning)
        derived, counts, new_var_indices = self._compute_derived_functional(
            current_states, next_var_indices, original_queries,
            history_hashes, history_count
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
            history_hashes=history_hashes,
            history_count=history_count,
        )
    
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
        # Use base large enough to avoid collisions (similar to memory.py)
        base = 2 ** 20  # Supports vocab up to 1M
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute derived states (pure function, no mutation).
        
        IMPORTANT: States exceeding padding_atoms are REJECTED, never truncated.
        Truncation would change the semantics of the state.
        This matches original _postprocess behavior at line 1344-1347.
        """
        n = current_states.shape[0]
        pad = self.padding_idx
        
        excluded = original_queries[:, 0:1, :]  # [n, 1, 3]
        derived, counts, new_var = self.engine.get_derived_states_compiled(
            current_states, next_var_indices, excluded
        )
        
        K, M = derived.shape[1], derived.shape[2]
        
        # Valid mask based on count only 
        # Note: Atom budget rejection is NOT needed because M_max is now aligned with original
        # engine's max_atoms_out = padding_atoms + max_rule_body_size. States are implicitly
        # truncated at slicing step below, matching original behavior.
        within_count = torch.arange(K, device=self.device).unsqueeze(0) < counts.unsqueeze(1)  # [n, K]
        
        # Zero out beyond-count entries
        derived = torch.where(
            within_count.unsqueeze(-1).unsqueeze(-1),
            derived,
            torch.full_like(derived, pad)
        )
        
        # Count is unchanged since we only removed beyond-count entries
        new_counts = counts.clone()
        
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
        
        # Update history: append new current state HASH for non-done envs
        # This must happen BEFORE computing derived states so pruning uses updated history
        new_history_hashes = state.history_hashes.clone()
        new_history_count = state.history_count.clone()
        
        # Only update history for envs that are NOT done
        active = ~was_done  # envs that were active before this step
        
        # Write position is current count (clamped to max-1 for safety)
        write_pos = state.history_count.clamp(max=self.max_history_size - 1)
        
        # Compute hash of new current state
        new_state_hash = self._compute_state_hash64(new_current)  # [n]
        
        # Scatter: write hash to history at write_pos for active envs
        batch_idx = torch.arange(n, device=device)
        update_val = torch.where(active, new_state_hash, new_history_hashes[batch_idx, write_pos])
        new_history_hashes[batch_idx, write_pos] = update_val
        
        # Increment count for active envs (clamped to max_history_size)
        new_history_count = torch.where(
            active,
            (state.history_count + 1).clamp(max=self.max_history_size),
            state.history_count
        )
        
        # Compute derived states for next step (with updated history for pruning)
        new_derived, new_counts, new_var = self._compute_derived_functional(
            new_current, state.next_var_indices, state.original_queries,
            new_history_hashes, new_history_count
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
            history_hashes=new_history_hashes,
            history_count=new_history_count,
        )
        
        # Create observation - use pre-allocated positions tensor
        action_mask = self._positions_S < new_counts.unsqueeze(1)
        
        obs = EvalObs(
            sub_index=new_current.unsqueeze(1),
            derived_sub_indices=new_derived,
            action_mask=action_mask,
        )
        
        return EvalStepFunctionalOutput(state=new_state, obs=obs, rewards=rewards)
    
    def step_with_policy_functional(
        self,
        state: EvalState,
        obs: EvalObs,
        policy_fn: Callable[[EvalObs], Tensor],
        deterministic: bool = True,
    ) -> Tuple[EvalState, EvalObs, Tensor, Tensor]:
        """
        Single transition: policy forward + action selection + environment step.
        
        This is the core function to be compiled - combines policy and step into
        one graph for efficient CUDA execution.
        
        Args:
            state: Current EvalState
            obs: Current observation
            policy_fn: Function that returns logits [B, S]
            deterministic: Use argmax for action selection
            
        Returns:
            new_state: Updated EvalState
            new_obs: New observation
            selected_log_probs: [B] Log probs of selected actions
            rewards: [B] Rewards from this step
        """
        # Get policy output
        logits = policy_fn(obs)  # [B, S]
        
        # Mask invalid actions
        masked_logits = torch.where(
            obs.action_mask,
            logits,
            torch.full_like(logits, float('-inf'))
        )
        
        # Select actions
        if deterministic:
            actions = masked_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(masked_logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
        
        # Get log probs
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Mask log probs for done environments
        active = ~state.done
        selected_log_probs = torch.where(
            active,
            selected_log_probs,
            torch.zeros_like(selected_log_probs)
        )
        
        # Step environment
        result: EvalStepFunctionalOutput = self.step_functional(state, actions)
        
        return result.state, result.obs, selected_log_probs, result.rewards
    
    def evaluate_trajectory(
        self,
        queries: Tensor,
        policy_fn: Callable[[EvalObs], Tensor],
        max_steps: int = 20,
        deterministic: bool = True,
        compiled_step_fn: Optional[Callable] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Run evaluation with single-step compilation and Python loop.
        
        This method:
        1. Initializes state from queries
        2. Creates initial observation
        3. Loops max_steps times in Python
        4. Each iteration calls the compiled single-step function
        5. Accumulates log_probs, rewards, success, lengths
        
        Args:
            queries: [B, 3] Query triples
            policy_fn: Callable(obs) -> logits [B, S]
            max_steps: Maximum trajectory length
            deterministic: Use argmax for action selection
            compiled_step_fn: Pre-compiled step function (for efficiency)
            
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
        
        # Pre-allocate accumulators
        total_log_probs = torch.zeros(B, device=device)
        total_rewards = torch.zeros(B, device=device)
        
        # Create initial observation
        action_mask = self._positions_S < state.derived_counts.unsqueeze(1)
        obs = EvalObs(
            sub_index=state.current_states.unsqueeze(1),
            derived_sub_indices=state.derived_states,
            action_mask=action_mask,
        )
        
        # Use provided compiled function or default
        step_fn = compiled_step_fn or (
            lambda s, o: self.step_with_policy_functional(s, o, policy_fn, deterministic)
        )
        
        # Python loop over transitions
        for step_idx in range(max_steps):
            # Execute single compiled step
            state, obs, step_log_probs, rewards = step_fn(state, obs)
            
            # Accumulate
            total_log_probs = total_log_probs + step_log_probs
            total_rewards = total_rewards + rewards
        
        return total_log_probs, state.success, state.depths, total_rewards


def create_compiled_step_fn(
    env: EvalEnvOptimized,
    policy_fn: Callable[[EvalObs], Tensor],
    deterministic: bool = True,
    compile_mode: str = 'default',
    fullgraph: bool = True,
) -> Callable:
    """
    Create a compiled single-step function.
    
    This compiles only the step_with_policy_functional method, which is much
    smaller than the full trajectory graph (~2k nodes vs ~40k nodes).
    
    Args:
        env: EvalEnvOptimized environment
        policy_fn: Policy function that returns logits
        deterministic: Use argmax for action selection
        compile_mode: torch.compile mode ('default', 'reduce-overhead', etc.)
        fullgraph: If True, require fullgraph compilation
        
    Returns:
        Compiled function that takes (state, obs) and returns (new_state, new_obs, log_probs, rewards)
    """
    import os
    import torch._inductor.config as inductor_config
    
    # Set TF32 precision for better performance
    torch.set_float32_matmul_precision('high')
    
    # Speed up compilation
    os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'
    os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING'] = '0'
    os.environ['TORCHINDUCTOR_FREEZING'] = '0'
    inductor_config.compile_threads = 4
    
    def step_fn(state: EvalState, obs: EvalObs):
        return env.step_with_policy_functional(state, obs, policy_fn, deterministic)
    
    print(f"Compiling single-step with mode='{compile_mode}', fullgraph={fullgraph}...")
    
    compiled_fn = torch.compile(
        step_fn,
        mode=compile_mode,
        fullgraph=fullgraph,
        dynamic=False,
    )
    
    return compiled_fn

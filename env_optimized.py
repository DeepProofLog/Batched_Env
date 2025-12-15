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
        
        # Pack base for hash computation - MUST match BloomFilter._pack_base
        # BloomFilter uses total_vocab_size + 1 where total_vocab_size = constant_no + runtime_vars
        # This ensures identical hash values between BloomFilter and _compute_state_hash64
        if im is not None:
            total_vocab_size = im.constant_no + 1000  # Same as env.py creates BloomFilter with
            self._hash_pack_base = total_vocab_size + 1
        else:
            self._hash_pack_base = 2 ** 20  # Fallback to large value
        
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
        
        # Compilation state
        self._compiled = False
        self._policy_logits_fn = None
        self._policy_value_fn = None
        self._compiled_step_fn = None
        self._compile_deterministic = True
    
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
            
        Note: For CUDA graph compatibility (reduce-overhead mode), outputs are
        cloned OUTSIDE this function in step_with_policy().
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
        
        # Update done/success
        was_done = state.done
        newly_done = is_true | is_false | is_end | is_depth_limit
        new_done = was_done | newly_done
        new_success = state.success | is_true
        
        # Rewards
        rewards = torch.zeros(n, device=device)
        rewards = torch.where(is_true & ~was_done, torch.ones_like(rewards), rewards)
        
        # Update history: append new current state HASH for non-done envs
        # This must happen BEFORE computing derived states so pruning uses updated history
        new_history_hashes = state.history_hashes
        new_history_count = state.history_count.clone()
        
        # Only update history for envs that are NOT done
        active = ~was_done  # envs that were active before this step
        
        # Write position is current count (clamped to max-1 for safety)
        write_pos = state.history_count.clamp(max=self.max_history_size - 1)
        
        # Compute hash of new current state
        new_state_hash = self._compute_state_hash64(new_current)  # [n]
        
        # Scatter: write hash to history at write_pos for active envs
        # Use scatter_ with explicit clone for CUDA graph compatibility (reduce-overhead mode)
        batch_idx = torch.arange(n, device=device)
        update_val = torch.where(active, new_state_hash, new_history_hashes[batch_idx, write_pos])
        
        # Clone before scatter to avoid CUDA graph tensor aliasing
        new_history_hashes = new_history_hashes.clone()
        new_history_hashes.scatter_(1, write_pos.unsqueeze(1), update_val.unsqueeze(1))
        
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
        
        # Create new state - no clones needed here since step_with_policy clones
        # outputs OUTSIDE the compiled function for CUDA graph compatibility
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
        
        # Create observation
        action_mask = self._positions_S < new_counts.unsqueeze(1)
        
        obs = EvalObs(
            sub_index=new_current.unsqueeze(1),
            derived_sub_indices=new_derived,
            action_mask=action_mask,
        )
        
        return EvalStepFunctionalOutput(state=new_state, obs=obs, rewards=rewards)
    
    def step_and_maybe_reset_functional(
        self,
        state: EvalState,
        actions: Tensor,
        query_pool: Tensor,
        per_env_ptrs: Tensor,
    ) -> Tuple[EvalStepFunctionalOutput, Tensor]:
        """
        Execute one step and reset done environments with next query from pool.
        
        This matches tensor env's step_and_maybe_reset behavior by cycling
        through queries in round-robin fashion per environment.
        
        Args:
            state: Current EvalState
            actions: [B] Action indices
            query_pool: [N, 3] Pool of queries to cycle through
            per_env_ptrs: [B] Current pointer per environment into query_pool
            
        Returns:
            Tuple of (EvalStepFunctionalOutput, updated_per_env_ptrs)
        """
        # First execute the step
        step_result = self.step_functional(state, actions)
        new_state = step_result.state
        new_obs = step_result.obs
        rewards = step_result.rewards
        
        # Check for newly done environments (weren't done before, now done)
        was_done = state.done
        newly_done = new_state.done & ~was_done
        
        # If any environments are newly done, reset them
        if newly_done.any():
            n = actions.shape[0]
            device = self.device
            num_queries = query_pool.shape[0]
            
            # Get which query each done env should reset to
            done_indices = torch.where(newly_done)[0]
            
            # Advance pointers for done envs and get next query
            new_ptrs = per_env_ptrs.clone()
            reset_query_indices = new_ptrs[done_indices] % num_queries
            new_ptrs[done_indices] = (new_ptrs[done_indices] + 1) % num_queries
            
            # Get reset queries and reinitialize each done env
            for i, idx in enumerate(done_indices):
                idx_i = int(idx)
                query_idx = int(reset_query_indices[i])
                reset_query = query_pool[query_idx:query_idx+1]  # [1, 3]
                
                reset_state = self.init_state_from_queries(reset_query)
                
                # Update state components for this env
                new_current_u = new_state.current_states.clone()
                new_current_u[idx_i] = reset_state.current_states[0]
                
                new_derived_u = new_state.derived_states.clone()
                new_derived_u[idx_i] = reset_state.derived_states[0]
                
                new_counts_u = new_state.derived_counts.clone()
                new_counts_u[idx_i] = reset_state.derived_counts[0]
                
                new_depths_u = new_state.depths.clone()
                new_depths_u[idx_i] = 0
                
                new_done_u = new_state.done.clone()
                new_done_u[idx_i] = False
                
                new_success_u = new_state.success.clone()
                new_success_u[idx_i] = False
                
                new_var_u = new_state.next_var_indices.clone()
                new_var_u[idx_i] = reset_state.next_var_indices[0]
                
                new_hist_u = new_state.history_hashes.clone()
                new_hist_u[idx_i] = reset_state.history_hashes[0]
                
                new_hcnt_u = new_state.history_count.clone()
                new_hcnt_u[idx_i] = reset_state.history_count[0]
                
                # Update original queries for this env
                new_orig_u = new_state.original_queries.clone()
                new_orig_u[idx_i] = reset_state.original_queries[0]
                
                new_state = EvalState(
                    current_states=new_current_u,
                    derived_states=new_derived_u,
                    derived_counts=new_counts_u,
                    original_queries=new_orig_u,
                    next_var_indices=new_var_u,
                    depths=new_depths_u,
                    done=new_done_u,
                    success=new_success_u,
                    history_hashes=new_hist_u,
                    history_count=new_hcnt_u,
                )
                
                # Update observation for this env
                new_obs_sub = new_obs.sub_index.clone()
                new_obs_sub[idx_i] = reset_state.current_states[0].unsqueeze(0)
                
                new_obs_derived = new_obs.derived_sub_indices.clone()
                new_obs_derived[idx_i] = reset_state.derived_states[0]
                
                new_obs_mask = new_obs.action_mask.clone()
                new_obs_mask[idx_i] = torch.arange(self.padding_states, device=device) < reset_state.derived_counts[0]
                
                new_obs = EvalObs(
                    sub_index=new_obs_sub,
                    derived_sub_indices=new_obs_derived,
                    action_mask=new_obs_mask,
                )
            
            per_env_ptrs = new_ptrs
        
        result = EvalStepFunctionalOutput(state=new_state, obs=new_obs, rewards=rewards)
        return result, per_env_ptrs, newly_done
    

    
    # =========================================================================
    # Compilation API
    # =========================================================================
    
    def compile(
        self,
        policy: 'nn.Module',
        deterministic: bool = True,
        mode: str = 'default',
        fullgraph: bool = True,
        include_value: bool = True,
    ) -> None:
        """
        Compile the step function with the given policy.
        
        After calling this, step_with_policy() uses the compiled version.
        
        Note: For parity testing with parity_mode=True in the vectorized engine,
        use compile=False (eager mode) since parity_mode uses dynamic shapes.
        For production with parity_mode=False, fullgraph=True works.
        
        Args:
            policy: ActorCriticPolicy or similar with mlp_extractor
            deterministic: Default deterministic mode (can be overridden in step_with_policy)
            mode: torch.compile mode ('default', 'reduce-overhead', etc.)
            fullgraph: If True, require fullgraph compilation
            include_value: If True, also compile value function for training mode
        """
        from model import create_policy_logits_fn, create_policy_value_fn
        
        # Store policy logits function
        self._policy_logits_fn = create_policy_logits_fn(policy)
        
        # Store policy value function for training mode
        if include_value:
            self._policy_value_fn = create_policy_value_fn(policy)
        else:
            self._policy_value_fn = None
        
        self._compile_deterministic = deterministic
        
        # Compile settings
        import os
        import torch._inductor.config as inductor_config
        
        torch.set_float32_matmul_precision('high')
        os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'
        os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING'] = '0'
        os.environ['TORCHINDUCTOR_FREEZING'] = '0'
        inductor_config.compile_threads = 4
        
        print(f"Compiling step_with_policy (mode='{mode}', fullgraph={fullgraph})...")
        
        # Compile the implementation function directly
        self._compiled_step_fn = torch.compile(
            self._step_with_policy_impl,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=False,
        )
        self._compiled = True
    
    def _step_with_policy_impl(
        self,
        state: EvalState,
        obs: EvalObs,
        query_pool: Tensor,
        per_env_ptrs: Tensor,
        deterministic: bool,
        eval_mode: bool,
        eval_done_mask: Tensor,
    ) -> Tuple[EvalState, EvalObs, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Unified step with policy - works for both training and evaluation.
        
        Args:
            state: Current EvalState
            obs: Current observation
            query_pool: [N, 3] Pool of queries to cycle through
            per_env_ptrs: [B] Current pointer per environment into query_pool
            deterministic: Use argmax (True) or sample (False)
            eval_mode: If True, stop processing finished slots
            eval_done_mask: [B] bool mask of slots that finished ALL their queries
            
        Returns:
            new_state, new_obs, actions, log_probs, values, rewards, dones, per_env_ptrs, eval_done_mask
        """
        device = self.device
        n = state.current_states.shape[0]
        
        # Get policy output
        logits = self._policy_logits_fn(obs)  # [B, S]
        
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
        
        # Get values for rollout collection (needed for PPO)
        # Values are computed only if we have a value function and not in eval_mode
        if self._policy_value_fn is not None and not eval_mode:
            values = self._policy_value_fn(obs)  # [B]
            # Mask values for done environments
            values = torch.where(active, values, torch.zeros_like(values))
        else:
            values = torch.zeros(n, device=device)
        
        # Execute step and maybe reset
        step_result = self.step_functional(state, actions)
        new_state = step_result.state
        new_obs = step_result.obs
        rewards = step_result.rewards
        
        # Check for newly done environments
        was_done = state.done
        newly_done = new_state.done & ~was_done
        
        # Handle resets for done environments
        num_queries = query_pool.shape[0]
        new_ptrs = per_env_ptrs.clone()
        
        # For eval_mode: update eval_done_mask when an env finishes its last query
        new_eval_done_mask = eval_done_mask.clone() if eval_done_mask is not None else \
                             torch.zeros(n, dtype=torch.bool, device=device)
        
        # Vectorized Reset Logic
        # We perform resets only in training mode (eval_mode=False) and when query_pool is provided.
        # To maintain static graph (fullgraph=True), we avoid python loops and use masked operations.
        new_ptrs = per_env_ptrs
        if (not eval_mode) and (query_pool is not None) and (query_pool.numel() > 0):
            # 1. Identify envs needing reset
            reset_mask = newly_done  # [B] bool (already 1D)
            
            # 2. Get queries for reset using CURRENT pointer (before advancing)
            # Tensor env pattern: use ptr, THEN advance
            pool_size = query_pool.shape[0]
            safe_indices = per_env_ptrs % pool_size
            candidate_queries = query_pool[safe_indices]  # [B, 3]
            
            # 3. Compute next pointers for envs that reset
            next_ptrs = (per_env_ptrs + 1) % pool_size
            
            # 4. Select appropriate next pointer (only update if done)
            new_ptrs = torch.where(reset_mask, next_ptrs, per_env_ptrs)
            
            # Create padding query [3]
            padding_atom = torch.full((3,), self.padding_idx, dtype=torch.long, device=device)
            
            # Mask: if NOT reset, use padding
            reset_mask_3 = reset_mask.unsqueeze(-1).expand(-1, 3)  # [B, 3]
            queries_for_reset_calc = torch.where(reset_mask_3, candidate_queries, padding_atom)
            
            # 5. Compute initial state for reset queries
            # This runs unification on [B] queries. Active envs process "padding", which is fast (0 derived).
            reset_state = self.init_state_from_queries(queries_for_reset_calc)
            
            # 6. Mix states: where(done, reset_state, new_state)
            # Helper masks for different tensor shapes
            reset_mask_A3 = reset_mask.view(-1, 1, 1).expand(-1, self.padding_atoms, 3)  # [B, A, 3]
            reset_mask_SA3 = reset_mask.view(-1, 1, 1, 1).expand(-1, self.padding_states, self.padding_atoms, 3)  # [B, S, A, 3]
            reset_mask_H = reset_mask.view(-1, 1).expand(-1, self.max_history_size)  # [B, H]
            
            new_current_states = torch.where(reset_mask_A3, reset_state.current_states, new_state.current_states)
            new_derived_states = torch.where(reset_mask_SA3, reset_state.derived_states, new_state.derived_states)
            new_derived_counts = torch.where(reset_mask, reset_state.derived_counts, new_state.derived_counts)
            new_original_queries = torch.where(reset_mask_A3, reset_state.original_queries, new_state.original_queries)
            new_next_var_indices = torch.where(reset_mask, reset_state.next_var_indices, new_state.next_var_indices)
            new_depths = torch.where(reset_mask, reset_state.depths, new_state.depths)
            new_done = torch.where(reset_mask, reset_state.done, new_state.done)
            new_success = torch.where(reset_mask, reset_state.success, new_state.success)
            new_history_hashes = torch.where(reset_mask_H, reset_state.history_hashes, new_state.history_hashes)
            new_history_count = torch.where(reset_mask, reset_state.history_count, new_state.history_count)
            
            new_state = EvalState(
                current_states=new_current_states,
                derived_states=new_derived_states,
                derived_counts=new_derived_counts,
                original_queries=new_original_queries,
                next_var_indices=new_next_var_indices,
                depths=new_depths,
                done=new_done,
                success=new_success,
                history_hashes=new_history_hashes,
                history_count=new_history_count,
            )
            
            # 7. Update Observation (must match new_state)
            action_mask = self._positions_S < new_state.derived_counts.unsqueeze(1)
            new_obs = EvalObs(
                sub_index=new_state.current_states.unsqueeze(1),
                derived_sub_indices=new_state.derived_states,
                action_mask=action_mask,
            )
        
        return new_state, new_obs, actions, selected_log_probs, values, rewards, newly_done, new_ptrs, new_eval_done_mask
    
    def step_with_policy(
        self,
        state: EvalState,
        obs: EvalObs,
        query_pool: Tensor,
        per_env_ptrs: Tensor,
        deterministic: bool = None,
        eval_mode: bool = False,
        eval_done_mask: Tensor = None,
    ) -> Tuple[EvalState, EvalObs, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Unified step function for both training and evaluation.
        
        Must call compile() first to set the policy.
        
        Args:
            state: Current EvalState
            obs: Current observation  
            query_pool: [N, 3] Pool of queries
            per_env_ptrs: [B] Pointer per env
            deterministic: True=argmax, False=sample (default: use compile setting)
            eval_mode: If True, stop slots that finish all queries
            eval_done_mask: [B] Optional mask for eval mode
            
        Returns:
            new_state, new_obs, actions, log_probs, values, rewards, dones, per_env_ptrs, eval_done_mask
        """
        if not self._compiled and self._policy_logits_fn is None:
            raise RuntimeError("Must call env.compile(policy) before step_with_policy()")
        
        if deterministic is None:
            deterministic = self._compile_deterministic
        
        if eval_done_mask is None:
            eval_done_mask = torch.zeros(state.current_states.shape[0], 
                                          dtype=torch.bool, device=self.device)
        
        if self._compiled and self._compiled_step_fn is not None:
            # Mark step begin for CUDA graphs (reduce-overhead mode)
            # This signals that prior iteration tensors can be freed
            torch.compiler.cudagraph_mark_step_begin()
            result = self._compiled_step_fn(
                state, obs, query_pool, per_env_ptrs,
                deterministic, eval_mode, eval_done_mask
            )
            # Clone all output tensors OUTSIDE the compiled graph
            # This breaks aliasing with CUDA graph internal buffers
            new_state, new_obs, actions, log_probs, values, rewards, dones, new_ptrs, new_eval_done_mask = result
            
            # Clone state tensors
            cloned_state = EvalState(
                current_states=new_state.current_states.clone(),
                derived_states=new_state.derived_states.clone(),
                derived_counts=new_state.derived_counts.clone(),
                original_queries=new_state.original_queries.clone(),
                next_var_indices=new_state.next_var_indices.clone(),
                depths=new_state.depths.clone(),
                done=new_state.done.clone(),
                success=new_state.success.clone(),
                history_hashes=new_state.history_hashes.clone(),
                history_count=new_state.history_count.clone(),
            )
            
            # Clone obs tensors
            cloned_obs = EvalObs(
                sub_index=new_obs.sub_index.clone(),
                derived_sub_indices=new_obs.derived_sub_indices.clone(),
                action_mask=new_obs.action_mask.clone(),
            )
            
            return (
                cloned_state, cloned_obs,
                actions.clone(), log_probs.clone(), values.clone(),
                rewards.clone(), dones.clone(), new_ptrs.clone(), new_eval_done_mask.clone()
            )
        else:
            # Eager mode
            return self._step_with_policy_impl(
                state, obs, query_pool, per_env_ptrs,
                deterministic, eval_mode, eval_done_mask
            )

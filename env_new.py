"""
Modified BatchedEnv that mirrors SB3's unary handling behavior.

We override `_apply_skip_unary_to_current_state` so that unary collapse does
not remove successor branches using the memory backend. This matches SB3's
LogicEnv, which follows unary chains without pruning their descendants.
"""

import torch
from env import BatchedEnv as _BatchedEnv


class BatchedEnvSB3(_BatchedEnv):
    """Batched Env variant with SB3-style skip-unary handling."""

    def _apply_skip_unary_to_current_state(self, env_indices: torch.Tensor):
        if env_indices.numel() == 0:
            return

        if not (self.skip_unary_actions and self.use_exact_memory):
            return super()._apply_skip_unary_to_current_state(env_indices)

        device = self._device
        pad = self.padding_idx
        max_iters = self.max_skip_unary_iters
        active_envs = env_indices.clone()
        hit_limit = False

        for iter_count in range(max_iters):
            if active_envs.numel() == 0:
                break

            terminal_mask = self.unification_engine.is_terminal_state(
                self.current_queries[active_envs]
            )
            active_envs = active_envs[~terminal_mask]

            if active_envs.numel() == 0:
                break

            current_states = self.current_queries.index_select(0, active_envs)
            next_vars = self.next_var_indices.index_select(0, active_envs)
            excluded = self.original_queries.index_select(0, active_envs).unsqueeze(1)

            # CRITICAL FIX: Add current state to memory BEFORE getting derived states
            # This matches SB3 behavior in _skip_unary_chain (sb3/env.py lines 556-558)
            if self.memory_pruning:
                self.memory_backend.add_current(active_envs, self.current_queries)

            derived_batch, derived_counts, updated_vars = self.unification_engine.get_derived_states(
                current_states=current_states,
                next_var_indices=next_vars,
                excluded_queries=excluded,
                verbose=0,
            )

            self.next_var_indices.index_copy_(0, active_envs, updated_vars)

            # CRITICAL FIX: Apply memory pruning to derived states
            # This matches SB3 behavior in _skip_unary_chain (sb3/env.py lines 563-571)
            if self.memory_pruning:
                A, K, M, D = derived_batch.shape
                
                # Check membership for all derived states
                visited = self.memory_backend.membership(derived_batch, active_envs)  # [A, K]
                
                # Get first predicate to identify terminals
                first_preds = derived_batch[:, :, 0, 0]  # [A, K]
                is_terminal_state = self.unification_engine.is_terminal_pred(first_preds)  # [A, K]
                
                # Only prune non-terminal visited states (terminals are protected)
                prune_mask = visited & ~is_terminal_state  # [A, K]
                keep_mask = ~prune_mask  # [A, K]
                
                # Apply pruning by removing visited non-terminal states
                for i in range(A):
                    valid_range = torch.arange(K, device=device) < derived_counts[i]
                    kept_in_range = keep_mask[i] & valid_range
                    new_count = kept_in_range.sum().item()
                    
                    if new_count < derived_counts[i].item():
                        # Compact: move kept states to front
                        kept_indices = torch.arange(K, device=device)[kept_in_range]
                        if kept_indices.numel() > 0:
                            derived_batch[i, :kept_indices.numel()] = derived_batch[i, kept_indices]
                            # Zero out the rest
                            if kept_indices.numel() < K:
                                derived_batch[i, kept_indices.numel():] = pad
                        else:
                            # All pruned, zero out
                            derived_batch[i] = pad
                        
                        derived_counts[i] = new_count

            # Determine unary rows (single non-terminal child)
            is_single = derived_counts == 1
            first_preds = torch.full((active_envs.shape[0],), pad, dtype=torch.long, device=device)
            mask_single = is_single
            if mask_single.any():
                first_preds[mask_single] = derived_batch[mask_single, 0, 0, 0]
            is_child_terminal = self.unification_engine.is_terminal_pred(first_preds)
            is_unary_nonterminal = is_single & ~is_child_terminal

            if not is_unary_nonterminal.any():
                break

            unary_envs = active_envs[is_unary_nonterminal]
            unary_local_idx = torch.arange(active_envs.shape[0], device=device)[is_unary_nonterminal]
            promoted = derived_batch[unary_local_idx, 0]

            valid_atom = promoted[:, :, 0] != pad
            atom_counts = valid_atom.sum(dim=1)
            within_budget = atom_counts <= self.padding_atoms
            if not within_budget.all():
                if not within_budget.any():
                    break
                unary_envs = unary_envs[within_budget]
                unary_local_idx = unary_local_idx[within_budget]
                promoted = promoted[within_budget]

            if promoted.shape[1] < self.padding_atoms:
                pad_cols = self.padding_atoms - promoted.shape[1]
                pad_tail = torch.full(
                    (promoted.shape[0], pad_cols, promoted.shape[2]),
                    pad,
                    dtype=promoted.dtype,
                    device=device,
                )
                promoted = torch.cat([promoted, pad_tail], dim=1)
            elif promoted.shape[1] > self.padding_atoms:
                promoted = promoted[:, : self.padding_atoms]

            self.current_queries.index_copy_(0, unary_envs, promoted)
            active_envs = unary_envs
        else:
            hit_limit = True

        if hit_limit and active_envs.numel() > 0:
            if self.false_pred_idx is None:
                raise RuntimeError("False predicate index is undefined; cannot inject False() state.")
            false_state = torch.full(
                (self.padding_atoms, self.max_arity + 1),
                pad,
                dtype=self.current_queries.dtype,
                device=device,
            )
            false_state[0, 0] = self.false_pred_idx
            false_expanded = false_state.unsqueeze(0).expand(active_envs.shape[0], -1, -1)
            self.current_queries.index_copy_(0, active_envs, false_expanded)

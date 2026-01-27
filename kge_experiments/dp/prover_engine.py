"""
Engine-based DP Prover using PPO's UnificationEngineVectorized.

This prover leverages the highly optimized unification engine from PPO
for batched proof expansion. The engine is already optimized for CUDA
and torch.compile.

Key optimizations:
    - Uses UnificationEngineVectorized.get_derived_states_compiled()
    - Batched state expansion (all queries processed together)
    - Fixed tensor shapes for compilation
    - Iterative BFS with vectorized operations
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict
import torch
from torch import Tensor

from kge_experiments.unification import UnificationEngineVectorized, GPUFactIndex


class DPProverEngine:
    """
    Engine-based prover using PPO's UnificationEngineVectorized.

    Uses the same optimized kernels as PPO training for proof search.
    """

    def __init__(
        self,
        engine: UnificationEngineVectorized,
        max_depth: int = 6,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
    ):
        self.engine = engine
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.device = device or engine.device

        # Extract from engine
        self.padding_idx = engine.padding_idx
        self.true_pred_idx = engine.true_pred_idx
        self.false_pred_idx = engine.false_pred_idx
        self.constant_no = engine.constant_no
        self.pack_base = getattr(engine, 'pack_base', engine.constant_no + 1001)

        # Use engine's fact index
        self.fact_index = engine.fact_index

        # Get dimensions from engine
        self.padding_atoms = engine.padding_atoms_limit
        self.padding_states = engine.K_max

    @torch.no_grad()
    def prove_batch(
        self,
        goals: Tensor,
        max_depth: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prove a batch of goals using the engine's state expansion.

        Args:
            goals: [B, 3] ground queries (pred, arg0, arg1)
            max_depth: Override default

        Returns:
            proven: [B] bool
            depths: [B] int (-1 if not proven)
        """
        B = goals.shape[0]
        device = self.device
        depth_limit = max_depth or self.max_depth
        A = self.padding_atoms

        if B == 0:
            return (
                torch.zeros(0, dtype=torch.bool, device=device),
                torch.full((0,), -1, dtype=torch.long, device=device),
            )

        # Initialize results
        proven = torch.zeros(B, dtype=torch.bool, device=device)
        depths = torch.full((B,), -1, dtype=torch.long, device=device)

        # Check direct facts first
        is_fact = self.fact_index.contains(goals)
        proven = proven | is_fact
        depths = torch.where(is_fact, torch.zeros_like(depths), depths)

        # Process remaining in batches
        remaining_mask = ~proven
        if not remaining_mask.any():
            return proven, depths

        remaining_idx = remaining_mask.nonzero(as_tuple=True)[0]

        # Process in batches for memory efficiency
        for batch_start in range(0, remaining_idx.shape[0], self.batch_size):
            batch_end = min(batch_start + self.batch_size, remaining_idx.shape[0])
            batch_idx = remaining_idx[batch_start:batch_end]
            batch_goals = goals[batch_idx]

            batch_proven, batch_depths = self._prove_batch_bfs(batch_goals, depth_limit)

            proven[batch_idx] = batch_proven
            depths[batch_idx] = batch_depths

        return proven, depths

    def _prove_batch_bfs(self, goals: Tensor, max_depth: int) -> Tuple[Tensor, Tensor]:
        """
        BFS proof search using engine for state expansion.

        Explores MULTIPLE derivation paths per goal using a frontier.
        Uses chunked processing to avoid OOM.
        """
        B = goals.shape[0]
        device = self.device
        A = self.padding_atoms
        max_frontier = 10  # Max states to track per goal (reduced for memory)
        max_expand_batch = 64  # Max states to expand at once

        proven = torch.zeros(B, dtype=torch.bool, device=device)
        depths = torch.full((B,), -1, dtype=torch.long, device=device)

        # Frontier: [B, max_frontier, A, 3] - multiple states per goal
        frontier = torch.full((B, max_frontier, A, 3), self.padding_idx, dtype=torch.long, device=device)
        frontier[:, 0, 0, :] = goals  # Initial state is just the goal

        # Frontier counts
        frontier_counts = torch.ones(B, dtype=torch.long, device=device)

        # Next vars per frontier state
        frontier_vars = torch.full((B, max_frontier), self.constant_no + 1, dtype=torch.long, device=device)

        # Track active goals
        active = torch.ones(B, dtype=torch.bool, device=device)

        for depth in range(1, max_depth + 1):
            if not active.any():
                break

            active_idx = active.nonzero(as_tuple=True)[0]

            # Collect frontier states
            all_states = []
            all_vars = []
            all_goal_idx = []

            for goal_idx in active_idx.tolist():
                n_states = min(frontier_counts[goal_idx].item(), max_frontier)
                for j in range(n_states):
                    state = frontier[goal_idx, j]
                    if state[0, 0] == self.padding_idx:
                        continue
                    all_states.append(state)
                    all_vars.append(frontier_vars[goal_idx, j])
                    all_goal_idx.append(goal_idx)

            if not all_states:
                break

            # Process in chunks to avoid OOM
            new_frontiers = {i: [] for i in active_idx.tolist()}
            M_total = len(all_states)

            for chunk_start in range(0, M_total, max_expand_batch):
                chunk_end = min(chunk_start + max_expand_batch, M_total)

                batch_states = torch.stack(all_states[chunk_start:chunk_end])
                batch_vars = torch.stack(all_vars[chunk_start:chunk_end])
                chunk_goal_idx = all_goal_idx[chunk_start:chunk_end]

                # Expand using engine
                derived, counts, new_vars = self.engine.get_derived_states_compiled(
                    batch_states, batch_vars, excluded_queries=None
                )

                K = derived.shape[1]
                first_preds = derived[:, :, 0, 0]
                is_true = first_preds == self.true_pred_idx

                for m in range(batch_states.shape[0]):
                    goal_idx = chunk_goal_idx[m]

                    if proven[goal_idx]:
                        continue

                    if is_true[m].any():
                        proven[goal_idx] = True
                        depths[goal_idx] = depth
                        continue

                    # Collect non-terminal derived states (limit per state)
                    count = 0
                    for k in range(K):
                        if count >= 3:  # Limit states per expansion
                            break
                        fp = derived[m, k, 0, 0].item()
                        if fp in (self.true_pred_idx, self.false_pred_idx, self.padding_idx):
                            continue
                        if len(new_frontiers[goal_idx]) < max_frontier:
                            new_frontiers[goal_idx].append((derived[m, k].clone(), new_vars[m].clone()))
                            count += 1

                # Clear GPU memory
                del derived, counts, new_vars, batch_states, batch_vars
                torch.cuda.empty_cache()

            # Update frontiers
            for goal_idx in active_idx.tolist():
                if proven[goal_idx]:
                    active[goal_idx] = False
                    continue

                new_states = new_frontiers[goal_idx]
                if not new_states:
                    active[goal_idx] = False
                    continue

                frontier[goal_idx].fill_(self.padding_idx)
                frontier_vars[goal_idx].fill_(self.constant_no + 1)

                for j, (state, var) in enumerate(new_states[:max_frontier]):
                    M_state = min(A, state.shape[0])
                    frontier[goal_idx, j, :M_state, :] = state[:M_state]
                    frontier_vars[goal_idx, j] = var

                frontier_counts[goal_idx] = min(len(new_states), max_frontier)

        return proven, depths

    @classmethod
    def from_index_manager(
        cls,
        im,
        max_depth: int = 6,
        batch_size: int = 256,
        padding_states: int = 120,
        padding_atoms: int = 10,
        **kwargs,
    ) -> "DPProverEngine":
        """Create from IndexManager."""
        engine = UnificationEngineVectorized.from_index_manager(
            im,
            padding_states=padding_states,
            padding_atoms=padding_atoms,
        )

        return cls(
            engine=engine,
            max_depth=max_depth,
            batch_size=batch_size,
            device=im.device,
            **kwargs,
        )

    def clear_cache(self) -> None:
        pass

    def cache_stats(self) -> dict:
        return {"engine": "UnificationEngineVectorized"}


class DPProverEngineCompiled(DPProverEngine):
    """
    Compiled version with torch.compile for CUDA graph support.
    """

    def __init__(self, *args, compile_mode: str = "reduce-overhead", **kwargs):
        super().__init__(*args, **kwargs)

        # Compile the expansion step
        self._compiled_expand = torch.compile(
            self._expand_step,
            mode=compile_mode,
            fullgraph=False,  # Allow graph breaks for flexibility
        )

    def _expand_step(
        self,
        states: Tensor,
        next_vars: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Single expansion step - compilable."""
        return self.engine.get_derived_states_compiled(
            states, next_vars, excluded_queries=None
        )

    def _prove_batch_bfs(self, goals: Tensor, max_depth: int) -> Tuple[Tensor, Tensor]:
        """BFS with compiled expansion."""
        B = goals.shape[0]
        device = self.device
        A = self.padding_atoms

        proven = torch.zeros(B, dtype=torch.bool, device=device)
        depths = torch.full((B,), -1, dtype=torch.long, device=device)

        states = torch.full((B, A, 3), self.padding_idx, dtype=torch.long, device=device)
        states[:, 0, :] = goals

        active = torch.ones(B, dtype=torch.bool, device=device)
        next_vars = torch.full((B,), self.constant_no + 1, dtype=torch.long, device=device)

        for depth in range(1, max_depth + 1):
            if not active.any():
                break

            active_idx = active.nonzero(as_tuple=True)[0]
            active_states = states[active_idx]
            active_vars = next_vars[active_idx]

            # Use compiled expansion
            try:
                derived, counts, new_vars = self._compiled_expand(active_states, active_vars)
            except Exception:
                # Fall back to non-compiled
                derived, counts, new_vars = self.engine.get_derived_states_compiled(
                    active_states, active_vars, excluded_queries=None
                )

            K = derived.shape[1]
            first_preds = derived[:, :, 0, 0]
            is_true = first_preds == self.true_pred_idx
            any_proven = is_true.any(dim=1)

            proven[active_idx[any_proven]] = True
            depths[active_idx[any_proven]] = depth
            active[active_idx] = active[active_idx] & ~any_proven

            if not active.any():
                break

            # Update states for next iteration
            active_idx = active.nonzero(as_tuple=True)[0]
            N = active_idx.shape[0]

            for i in range(N):
                glob_idx = active_idx[i]

                for k in range(K):
                    first_pred = derived[i, k, 0, 0].item()
                    if first_pred in (self.true_pred_idx, self.false_pred_idx, self.padding_idx):
                        continue

                    M = min(A, derived.shape[2])
                    states[glob_idx, :M, :] = derived[i, k, :M, :]
                    next_vars[glob_idx] = new_vars[i]
                    break

        return proven, depths

    @classmethod
    def from_index_manager(cls, im, compile_mode: str = "reduce-overhead", **kwargs):
        """Create compiled prover from IndexManager."""
        engine = UnificationEngineVectorized.from_index_manager(
            im,
            padding_states=kwargs.get('padding_states', 120),
            padding_atoms=kwargs.get('padding_atoms', 10),
        )

        return cls(
            engine=engine,
            compile_mode=compile_mode,
            device=im.device,
            **kwargs,
        )

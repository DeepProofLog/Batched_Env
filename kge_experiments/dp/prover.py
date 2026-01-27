"""
Optimized DP Prover with matrix-based transitive closure.

This prover uses matrix operations for efficient proof search,
particularly optimized for transitive rules like countries_s3.

Key Optimizations:
    - Matrix-based reachability for transitive rules
    - Batched GPU operations for all proof steps
    - Tensor-based memoization (GPU-resident)
    - torch.compile support for CUDA graphs
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict
import torch
from torch import Tensor

from kge_experiments.unification import (
    GPUFactIndex,
    unify_one_to_one,
    apply_substitutions,
)


class DPProver:
    """
    Optimized backward-chaining prover using matrix operations.

    For transitive rules, uses matrix multiplication for reachability.
    For general rules, uses batched backward chaining.
    """

    def __init__(
        self,
        facts: Tensor,              # [F, 3] known facts
        rules_heads: Tensor,        # [R, 3] rule head atoms
        rules_bodies: Tensor,       # [R, Bmax, 3] rule body atoms
        rule_lens: Tensor,          # [R] actual body lengths
        constant_no: int,           # Max constant index
        padding_idx: int = 0,
        max_depth: int = 20,
        cache_capacity: int = 100000,
        device: Optional[torch.device] = None,
    ):
        self.device = device or facts.device
        self.constant_no = constant_no
        self.padding_idx = padding_idx
        self.max_depth = max_depth

        # Store facts and rules
        self.facts = facts.to(self.device)
        self.rules_heads = rules_heads.to(self.device)
        self.rules_bodies = rules_bodies.to(self.device)
        self.rule_lens = rule_lens.to(self.device)

        # Pack base for hashing
        self.pack_base = constant_no + 10000

        # Build fact index
        self.fact_index = GPUFactIndex(self.facts, self.pack_base)

        # Build rule index by predicate
        self._build_rule_index()

        # Tensor-based memoization: hash -> (proven, depth)
        # Using two tensors for GPU residence
        self._cache_hashes = torch.empty(0, dtype=torch.int64, device=self.device)
        self._cache_proven = torch.empty(0, dtype=torch.bool, device=self.device)
        self._cache_depths = torch.empty(0, dtype=torch.long, device=self.device)

        # For statistics
        self._cache_hits = 0
        self._cache_misses = 0

        # Pre-compute adjacency matrices for transitive rules
        self._build_adjacency_matrices()

    def _build_rule_index(self) -> None:
        """Build predicate-indexed rule lookup."""
        if self.rules_heads.numel() == 0:
            self.rules_by_pred: Dict[int, list] = {}
            return

        self.rules_by_pred = {}
        for i in range(self.rules_heads.shape[0]):
            pred = self.rules_heads[i, 0].item()
            if pred not in self.rules_by_pred:
                self.rules_by_pred[pred] = []
            self.rules_by_pred[pred].append(i)

    def _build_adjacency_matrices(self) -> None:
        """
        Build adjacency matrices for predicates appearing in rule bodies.

        For countries_s3-like rules: p(X,Y) :- q(X,Z), p(Z,Y)
        This creates transitive closure via matrix multiplication.
        """
        self.pred_adj_matrices: Dict[int, Tensor] = {}
        self.transitive_preds: Dict[int, Tuple[int, int]] = {}  # head_pred -> (body_pred1, body_pred2)

        # Get unique predicates in facts
        if self.facts.numel() == 0:
            return

        fact_preds = self.facts[:, 0].unique()
        n = self.constant_no + 1

        # Build adjacency matrix for each predicate
        for pred in fact_preds.tolist():
            mask = self.facts[:, 0] == pred
            edges = self.facts[mask][:, 1:]  # [E, 2] - (src, dst)

            # Sparse to dense adjacency matrix
            adj = torch.zeros(n, n, dtype=torch.bool, device=self.device)
            valid = (edges[:, 0] < n) & (edges[:, 1] < n)
            edges_valid = edges[valid]
            if edges_valid.numel() > 0:
                adj[edges_valid[:, 0], edges_valid[:, 1]] = True

            self.pred_adj_matrices[pred] = adj

        # Detect transitive rules: p(X,Y) :- q(X,Z), p(Z,Y)
        for head_pred, rule_indices in self.rules_by_pred.items():
            for rule_idx in rule_indices:
                body_len = self.rule_lens[rule_idx].item()
                if body_len != 2:
                    continue

                body = self.rules_bodies[rule_idx, :body_len]  # [2, 3]
                head = self.rules_heads[rule_idx]  # [3]

                # Check pattern: p(X,Y) :- q(X,Z), p(Z,Y)
                # Head: pred=head_pred, args=(X, Y) where X,Y are variables
                head_arg0, head_arg1 = head[1].item(), head[2].item()

                if head_arg0 <= self.constant_no or head_arg1 <= self.constant_no:
                    continue  # Head args must be variables

                body0_pred = body[0, 0].item()
                body0_arg0, body0_arg1 = body[0, 1].item(), body[0, 2].item()
                body1_pred = body[1, 0].item()
                body1_arg0, body1_arg1 = body[1, 1].item(), body[1, 2].item()

                # Check if transitive pattern
                if (body1_pred == head_pred and
                    body0_arg0 == head_arg0 and  # X matches
                    body0_arg1 == body1_arg0 and  # Z is shared
                    body1_arg1 == head_arg1):     # Y matches

                    self.transitive_preds[head_pred] = (body0_pred, head_pred)

        # Pre-compute transitive closures
        self._transitive_closures: Dict[int, Tensor] = {}
        for head_pred, (edge_pred, _) in self.transitive_preds.items():
            if edge_pred in self.pred_adj_matrices and head_pred in self.pred_adj_matrices:
                edge_adj = self.pred_adj_matrices[edge_pred].float()
                base_adj = self.pred_adj_matrices[head_pred].float()

                # Compute transitive closure: reach[i,j] = can reach j from i via edges and base
                closure = base_adj.clone()

                for _ in range(self.max_depth):
                    # New reachable: start with edge, then follow closure
                    new_reach = torch.mm(edge_adj, closure)
                    closure = ((closure + new_reach) > 0).float()

                self._transitive_closures[head_pred] = closure.bool()

    def _hash_goals(self, goals: Tensor) -> Tensor:
        """Hash batch of goals. goals: [N, 3] -> hashes: [N]"""
        p = goals[:, 0].long()
        a = goals[:, 1].long()
        b = goals[:, 2].long()
        return ((p * self.pack_base) + a) * self.pack_base + b

    @torch.no_grad()
    def is_fact(self, goals: Tensor) -> Tensor:
        """Check if goals are known facts. goals: [N, 3] -> mask: [N]"""
        return self.fact_index.contains(goals)

    @torch.no_grad()
    def prove_batch(
        self,
        goals: Tensor,
        max_depth: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prove a batch of ground goals using optimized matrix operations.

        Args:
            goals: [B, 3] ground queries
            max_depth: Override default max depth

        Returns:
            proven: [B] boolean
            depths: [B] int (-1 if not proven)
        """
        B = goals.shape[0]
        device = self.device
        depth_limit = max_depth or self.max_depth

        if B == 0:
            return (
                torch.zeros(0, dtype=torch.bool, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        proven = torch.zeros(B, dtype=torch.bool, device=device)
        depths = torch.full((B,), -1, dtype=torch.long, device=device)

        # Step 1: Check direct facts (vectorized)
        is_fact = self.fact_index.contains(goals)
        proven = proven | is_fact
        depths = torch.where(is_fact, torch.zeros_like(depths), depths)

        # Step 2: Check transitive closure for applicable predicates
        remaining = ~proven
        if remaining.any():
            proven, depths = self._prove_via_closure(
                goals, proven, depths, remaining, depth_limit
            )

        # Step 3: Fall back to BFS for remaining goals
        remaining = ~proven
        if remaining.any():
            remaining_indices = remaining.nonzero(as_tuple=True)[0]
            for idx in remaining_indices:
                goal = goals[idx]
                is_proven, depth = self._prove_single_bfs(goal, depth_limit)
                proven[idx] = is_proven
                depths[idx] = depth

        return proven, depths

    def _prove_via_closure(
        self,
        goals: Tensor,
        proven: Tensor,
        depths: Tensor,
        mask: Tensor,
        max_depth: int,
    ) -> Tuple[Tensor, Tensor]:
        """Check provability using pre-computed transitive closure."""
        B = goals.shape[0]

        for pred, closure in self._transitive_closures.items():
            # Find goals with this predicate
            pred_mask = (goals[:, 0] == pred) & mask
            if not pred_mask.any():
                continue

            pred_indices = pred_mask.nonzero(as_tuple=True)[0]
            pred_goals = goals[pred_indices]  # [K, 3]

            # Check closure matrix
            src = pred_goals[:, 1].long()  # [K]
            dst = pred_goals[:, 2].long()  # [K]

            # Bounds check
            n = closure.shape[0]
            valid = (src < n) & (dst < n) & (src >= 0) & (dst >= 0)

            # Look up in closure matrix
            reachable = torch.zeros(pred_indices.shape[0], dtype=torch.bool, device=self.device)
            if valid.any():
                valid_idx = valid.nonzero(as_tuple=True)[0]
                reachable[valid_idx] = closure[src[valid_idx], dst[valid_idx]]

            # Update proven status
            proven[pred_indices] = proven[pred_indices] | reachable
            # Depth estimate: use max_depth / 2 as approximate depth for transitive proofs
            depths[pred_indices] = torch.where(
                reachable & (depths[pred_indices] < 0),
                torch.full_like(depths[pred_indices], max_depth // 2),
                depths[pred_indices]
            )

        return proven, depths

    def _prove_single_bfs(self, goal: Tensor, max_depth: int) -> Tuple[bool, int]:
        """
        Prove a single goal using BFS with memoization.

        This is the fallback for non-transitive rules.
        """
        # Check cache first
        goal_hash = self._hash_goals(goal.unsqueeze(0))[0]

        if self._cache_hashes.numel() > 0:
            cache_match = self._cache_hashes == goal_hash
            if cache_match.any():
                idx = cache_match.nonzero(as_tuple=True)[0][0]
                self._cache_hits += 1
                return self._cache_proven[idx].item(), self._cache_depths[idx].item()

        self._cache_misses += 1

        # Check if direct fact
        if self.is_fact(goal.unsqueeze(0))[0]:
            self._add_to_cache(goal_hash, True, 0)
            return True, 0

        pred = goal[0].item()
        if pred not in self.rules_by_pred:
            self._add_to_cache(goal_hash, False, -1)
            return False, -1

        # Get all rule bodies that unify with goal
        initial_bodies = self._get_unified_bodies(goal)

        if not initial_bodies:
            self._add_to_cache(goal_hash, False, -1)
            return False, -1

        # BFS with tensor-based operations where possible
        from collections import deque
        queue = deque([(body, 1) for body in initial_bodies])
        visited = set()

        while queue:
            remaining_goals, depth = queue.popleft()

            if depth > max_depth:
                continue

            # Batch check all remaining goals
            if not remaining_goals:
                # All goals proven!
                self._add_to_cache(goal_hash, True, depth)
                return True, depth

            goals_tensor = torch.stack([g for g in remaining_goals])  # [K, 3]

            # Check facts in batch
            is_fact = self.fact_index.contains(goals_tensor)

            # Check closure for transitive predicates
            for pred_id, closure in self._transitive_closures.items():
                pred_mask = goals_tensor[:, 0] == pred_id
                if not pred_mask.any():
                    continue
                pred_idx = pred_mask.nonzero(as_tuple=True)[0]
                src = goals_tensor[pred_idx, 1].long()
                dst = goals_tensor[pred_idx, 2].long()
                n = closure.shape[0]
                valid = (src < n) & (dst < n) & (src >= 0) & (dst >= 0)
                if valid.any():
                    valid_i = valid.nonzero(as_tuple=True)[0]
                    is_fact[pred_idx[valid_i]] |= closure[src[valid_i], dst[valid_i]]

            all_proven = is_fact.all()

            if all_proven:
                self._add_to_cache(goal_hash, True, depth)
                return True, depth

            # Find unproven goals
            unproven_mask = ~is_fact
            if not unproven_mask.any():
                continue

            unproven_indices = unproven_mask.nonzero(as_tuple=True)[0]

            # Expand first unproven goal
            first_unproven = remaining_goals[unproven_indices[0].item()]
            subgoal_bodies = self._get_unified_bodies(first_unproven)

            for body in subgoal_bodies:
                # New remaining = other unproven goals + new body goals
                new_remaining = [remaining_goals[i.item()] for i in unproven_indices[1:]]
                new_remaining.extend(body)

                # Create state signature
                if new_remaining:
                    state_hashes = self._hash_goals(torch.stack(new_remaining))
                    state_sig = tuple(sorted(state_hashes.tolist()))
                else:
                    state_sig = ()

                if state_sig in visited:
                    continue
                visited.add(state_sig)

                queue.append((new_remaining, depth + 1))

        self._add_to_cache(goal_hash, False, -1)
        return False, -1

    def _add_to_cache(self, goal_hash: Tensor, proven: bool, depth: int) -> None:
        """Add result to tensor-based cache."""
        self._cache_hashes = torch.cat([
            self._cache_hashes,
            goal_hash.unsqueeze(0) if isinstance(goal_hash, Tensor) else torch.tensor([goal_hash], device=self.device, dtype=torch.int64)
        ])
        self._cache_proven = torch.cat([
            self._cache_proven,
            torch.tensor([proven], device=self.device, dtype=torch.bool)
        ])
        self._cache_depths = torch.cat([
            self._cache_depths,
            torch.tensor([depth], device=self.device, dtype=torch.long)
        ])

    def _get_unified_bodies(self, goal: Tensor) -> list:
        """
        Get all rule bodies that unify with the goal.

        Returns list of lists of subgoal tensors.
        """
        pred = goal[0].item()
        if pred not in self.rules_by_pred:
            return []

        bodies = []
        pad = self.padding_idx
        goal_t = goal.unsqueeze(0)  # [1, 3]

        for rule_idx in self.rules_by_pred[pred]:
            head = self.rules_heads[rule_idx]  # [3]
            body = self.rules_bodies[rule_idx]  # [Bmax, 3]
            body_len = self.rule_lens[rule_idx].item()

            # Unify goal with rule head
            head_t = head.unsqueeze(0)  # [1, 3]
            unify_ok, subs = unify_one_to_one(goal_t, head_t, self.constant_no, pad)

            if not unify_ok[0]:
                continue

            # Apply substitutions to body
            body_t = body.unsqueeze(0)  # [1, Bmax, 3]
            body_subst = apply_substitutions(body_t, subs, pad)[0]  # [Bmax, 3]

            # Get actual body atoms
            body_atoms = body_subst[:body_len]  # [body_len, 3]

            if body_len == 0:
                # Empty body = proven
                return [[]]  # Return empty list = already proven

            # Check if body is ground
            if (body_atoms[:, 1:] > self.constant_no).any():
                # Has variables - need to enumerate bindings from facts
                ground_bodies = self._enumerate_body_bindings(body_atoms)
                bodies.extend(ground_bodies)
            else:
                # All ground
                bodies.append([body_atoms[i] for i in range(body_len)])

        return bodies

    def _enumerate_body_bindings(self, body_atoms: Tensor) -> list:
        """
        Enumerate all ground instantiations of body atoms with variables.

        Uses facts to find valid bindings for variables.
        """
        K = body_atoms.shape[0]

        # Find variables and their positions
        bindings_list = [{}]  # Start with empty binding

        for atom_idx in range(K):
            atom = body_atoms[atom_idx]
            pred = atom[0].item()
            arg0 = atom[1].item()
            arg1 = atom[2].item()

            is_var0 = arg0 > self.constant_no
            is_var1 = arg1 > self.constant_no

            if not is_var0 and not is_var1:
                # Ground atom - keep current bindings
                continue

            # Find facts matching this pattern
            new_bindings_list = []

            for bindings in bindings_list:
                # Apply current bindings
                actual_arg0 = bindings.get(arg0, arg0) if is_var0 else arg0
                actual_arg1 = bindings.get(arg1, arg1) if is_var1 else arg1

                # Get matching facts
                matching = self._get_matching_facts(pred, actual_arg0, actual_arg1,
                                                    is_var0 and arg0 not in bindings,
                                                    is_var1 and arg1 not in bindings)

                for fact in matching:
                    new_binding = bindings.copy()
                    if is_var0 and arg0 not in bindings:
                        new_binding[arg0] = fact[1].item()
                    if is_var1 and arg1 not in bindings:
                        new_binding[arg1] = fact[2].item()
                    new_bindings_list.append(new_binding)

            bindings_list = new_bindings_list

            if not bindings_list:
                return []  # No valid bindings

        # Convert bindings to ground body atoms
        result = []
        for bindings in bindings_list[:100]:  # Limit to avoid explosion
            ground_body = []
            for atom_idx in range(K):
                atom = body_atoms[atom_idx]
                pred = atom[0].item()
                arg0 = atom[1].item()
                arg1 = atom[2].item()

                actual_arg0 = bindings.get(arg0, arg0)
                actual_arg1 = bindings.get(arg1, arg1)

                ground_atom = torch.tensor([pred, actual_arg0, actual_arg1],
                                          dtype=torch.long, device=self.device)
                ground_body.append(ground_atom)

            result.append(ground_body)

        return result

    def _get_matching_facts(
        self,
        pred: int,
        arg0: int,
        arg1: int,
        is_var0: bool,
        is_var1: bool,
    ) -> Tensor:
        """Get facts matching a pattern."""
        mask = self.facts[:, 0] == pred
        if not is_var0:
            mask = mask & (self.facts[:, 1] == arg0)
        if not is_var1:
            mask = mask & (self.facts[:, 2] == arg1)
        return self.facts[mask]

    def clear_cache(self) -> None:
        """Clear memoization cache."""
        self._cache_hashes = torch.empty(0, dtype=torch.int64, device=self.device)
        self._cache_proven = torch.empty(0, dtype=torch.bool, device=self.device)
        self._cache_depths = torch.empty(0, dtype=torch.long, device=self.device)
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "capacity": self._cache_hashes.numel(),
            "occupied": self._cache_hashes.numel(),
            "load_factor": 1.0,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "inserts": self._cache_hashes.numel(),
        }

    @classmethod
    def from_index_manager(cls, im, **kwargs) -> "DPProver":
        """Create from IndexManager."""
        return cls(
            facts=im.facts_idx,
            rules_heads=im.rules_heads_idx,
            rules_bodies=im.rules_idx,
            rule_lens=im.rule_lens,
            constant_no=im.constant_no,
            padding_idx=im.padding_idx,
            device=im.device,
            **kwargs,
        )


class DPProverFast(DPProver):
    """
    Ultra-fast prover using only pre-computed transitive closure.

    This version is optimized for datasets like countries_s3 where
    all provability can be determined via transitive closure.
    """

    @torch.no_grad()
    def prove_batch(
        self,
        goals: Tensor,
        max_depth: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prove batch using only facts and transitive closure.

        This is O(1) per goal after pre-computation.
        """
        B = goals.shape[0]
        device = self.device
        depth_limit = max_depth or self.max_depth

        if B == 0:
            return (
                torch.zeros(0, dtype=torch.bool, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        proven = torch.zeros(B, dtype=torch.bool, device=device)
        depths = torch.full((B,), -1, dtype=torch.long, device=device)

        # Check direct facts
        is_fact = self.fact_index.contains(goals)
        proven = proven | is_fact
        depths = torch.where(is_fact, torch.zeros_like(depths), depths)

        # Check transitive closures
        remaining = ~proven
        if remaining.any():
            for pred, closure in self._transitive_closures.items():
                pred_mask = (goals[:, 0] == pred) & remaining
                if not pred_mask.any():
                    continue

                pred_indices = pred_mask.nonzero(as_tuple=True)[0]
                src = goals[pred_indices, 1].long()
                dst = goals[pred_indices, 2].long()

                n = closure.shape[0]
                valid = (src < n) & (dst < n) & (src >= 0) & (dst >= 0)

                reachable = torch.zeros(pred_indices.shape[0], dtype=torch.bool, device=device)
                if valid.any():
                    valid_idx = valid.nonzero(as_tuple=True)[0]
                    reachable[valid_idx] = closure[src[valid_idx], dst[valid_idx]]

                proven[pred_indices] = proven[pred_indices] | reachable
                depths[pred_indices] = torch.where(
                    reachable & (depths[pred_indices] < 0),
                    torch.ones_like(depths[pred_indices]),  # Depth 1 for transitive
                    depths[pred_indices]
                )

        return proven, depths


def prove_batch_fast(
    goals: Tensor,
    facts: Tensor,
    fact_index: GPUFactIndex,
    transitive_closures: Dict[int, Tensor],
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Fast batch proof function designed for torch.compile.

    This is a functional implementation that can be compiled.
    """
    B = goals.shape[0]

    proven = torch.zeros(B, dtype=torch.bool, device=device)
    depths = torch.full((B,), -1, dtype=torch.long, device=device)

    # Check facts first (vectorized)
    is_fact = fact_index.contains(goals)
    proven = proven | is_fact
    depths = torch.where(is_fact, torch.zeros_like(depths), depths)

    # Check transitive closures
    for pred, closure in transitive_closures.items():
        pred_mask = (goals[:, 0] == pred) & ~proven
        if not pred_mask.any():
            continue

        pred_indices = pred_mask.nonzero(as_tuple=True)[0]
        src = goals[pred_indices, 1].long()
        dst = goals[pred_indices, 2].long()

        n = closure.shape[0]
        valid = (src < n) & (dst < n)

        reachable = torch.zeros(pred_indices.shape[0], dtype=torch.bool, device=device)
        if valid.any():
            valid_idx = valid.nonzero(as_tuple=True)[0]
            reachable[valid_idx] = closure[src[valid_idx], dst[valid_idx]]

        proven[pred_indices] = proven[pred_indices] | reachable
        depths[pred_indices] = torch.where(
            reachable & (depths[pred_indices] < 0),
            torch.ones_like(depths[pred_indices]),
            depths[pred_indices]
        )

    return proven, depths

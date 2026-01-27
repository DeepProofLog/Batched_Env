"""
Highly Optimized DP Prover with Parallel BFS, Rule Compilation, and Incremental Closure.

Optimizations:
    1. Parallel BFS: Process multiple queries simultaneously with vectorized frontier
    2. Rule Compilation: Pre-compile rule bodies into GPU lookup tables
    3. Incremental Closure: Pre-compute predicate compositions using sparse tensors
    4. Depth Limit: Hard limit of 6 for efficient search

Designed for complex rule sets like the family dataset.
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, List, Set
import torch
from torch import Tensor

from kge_experiments.unification import GPUFactIndex


class DPProverOptimized:
    """
    Highly optimized prover using parallel BFS and pre-computed closures.

    Key features:
    - Sparse predicate adjacency for memory efficiency
    - Pre-computed rule compositions (limited iterations)
    - Parallel BFS across all queries
    - Vectorized fact/rule matching via hash tables
    """

    def __init__(
        self,
        facts: Tensor,              # [F, 3] known facts
        rules_heads: Tensor,        # [R, 3] rule head atoms
        rules_bodies: Tensor,       # [R, Bmax, 3] rule body atoms
        rule_lens: Tensor,          # [R] actual body lengths
        constant_no: int,
        padding_idx: int = 0,
        max_depth: int = 6,
        max_closure_iters: int = 3,  # Limit closure iterations
        max_pairs_per_pred: int = 100000,  # Limit pairs to avoid OOM
        device: Optional[torch.device] = None,
    ):
        self.device = device or facts.device
        self.constant_no = constant_no
        self.padding_idx = padding_idx
        self.max_depth = max_depth
        self.max_closure_iters = max_closure_iters
        self.max_pairs_per_pred = max_pairs_per_pred

        # Store tensors
        self.facts = facts.to(self.device)
        self.rules_heads = rules_heads.to(self.device)
        self.rules_bodies = rules_bodies.to(self.device)
        self.rule_lens = rule_lens.to(self.device)

        # Pack base for hashing
        self.pack_base = constant_no + 10000

        # Build fact index
        self.fact_index = GPUFactIndex(self.facts, self.pack_base)

        # Build optimized indices
        self._build_predicate_index()
        self._build_rule_index()
        self._compile_rules()
        self._compute_incremental_closures()

        # Statistics
        self._stats = {"cache_hits": 0, "cache_misses": 0}

    def _build_predicate_index(self) -> None:
        """Build hash-based predicate index for O(1) fact lookup."""
        self.pred_facts: Dict[int, Tensor] = {}  # pred -> [E, 2] edges
        self.pred_fact_hashes: Dict[int, Tensor] = {}  # pred -> sorted hashes

        if self.facts.numel() == 0:
            return

        # Group facts by predicate
        preds = self.facts[:, 0].unique()

        for pred in preds.tolist():
            mask = self.facts[:, 0] == pred
            edges = self.facts[mask][:, 1:]  # [E, 2]
            self.pred_facts[pred] = edges

            # Create hash for O(1) lookup
            hashes = edges[:, 0].long() * self.pack_base + edges[:, 1].long()
            self.pred_fact_hashes[pred] = hashes.sort()[0]

    def _build_rule_index(self) -> None:
        """Build predicate-indexed rule lookup."""
        self.rules_by_pred: Dict[int, List[int]] = {}

        if self.rules_heads.numel() == 0:
            return

        for i in range(self.rules_heads.shape[0]):
            pred = self.rules_heads[i, 0].item()
            if pred not in self.rules_by_pred:
                self.rules_by_pred[pred] = []
            self.rules_by_pred[pred].append(i)

    def _compile_rules(self) -> None:
        """
        Compile rules into efficient lookup structures.

        Identifies composition patterns: p(X,Y) :- q(X,Z), r(Z,Y)
        """
        self.compiled_rules: Dict[int, List[dict]] = {}
        self.composition_rules: List[Tuple[int, int, int]] = []  # (head, pred1, pred2)

        for head_pred, rule_indices in self.rules_by_pred.items():
            compiled = []

            for rule_idx in rule_indices:
                head = self.rules_heads[rule_idx]
                body = self.rules_bodies[rule_idx]
                body_len = self.rule_lens[rule_idx].item()

                if body_len == 0:
                    compiled.append({"type": "fact", "rule_idx": rule_idx})
                    continue

                body_atoms = body[:body_len]
                body_preds = body_atoms[:, 0].tolist()

                # Check for composition pattern
                if body_len == 2:
                    pattern = self._check_composition_pattern(head, body_atoms)
                    if pattern is not None:
                        pred1, pred2 = pattern
                        compiled.append({
                            "type": "composition",
                            "rule_idx": rule_idx,
                            "pred1": pred1,
                            "pred2": pred2,
                        })
                        self.composition_rules.append((head_pred, pred1, pred2))
                        continue

                compiled.append({
                    "type": "general",
                    "rule_idx": rule_idx,
                    "body_preds": body_preds,
                    "body_len": body_len,
                })

            self.compiled_rules[head_pred] = compiled

    def _check_composition_pattern(self, head: Tensor, body: Tensor) -> Optional[Tuple[int, int]]:
        """Check if rule is composition pattern, return (pred1, pred2) or None."""
        head_arg0, head_arg1 = head[1].item(), head[2].item()
        b0_pred, b0_arg0, b0_arg1 = body[0, 0].item(), body[0, 1].item(), body[0, 2].item()
        b1_pred, b1_arg0, b1_arg1 = body[1, 0].item(), body[1, 1].item(), body[1, 2].item()

        # Pattern: p(X,Y) :- q(X,Z), r(Z,Y) where Z is shared variable
        if (b0_arg0 == head_arg0 and b0_arg1 == b1_arg0 and
            b1_arg1 == head_arg1 and b0_arg1 > self.constant_no):
            return (b0_pred, b1_pred)

        # Reversed: p(X,Y) :- q(Z,X), r(Y,Z)
        if (b0_arg1 == head_arg0 and b0_arg0 == b1_arg1 and
            b1_arg0 == head_arg1 and b0_arg0 > self.constant_no):
            return (b0_pred, b1_pred)

        return None

    def _compute_incremental_closures(self) -> None:
        """
        Pre-compute predicate compositions using sparse operations.

        Uses hash-based join for memory efficiency on large graphs.
        """
        # Start with direct facts
        self.closure_hashes: Dict[int, Tensor] = {}

        for pred, hashes in self.pred_fact_hashes.items():
            self.closure_hashes[pred] = hashes.clone()

        # Iteratively compute compositions
        for iteration in range(self.max_closure_iters):
            changed = False

            for head_pred, pred1, pred2 in self.composition_rules:
                if pred1 not in self.pred_facts or pred2 not in self.closure_hashes:
                    continue

                # Compose pred1 edges with current closure of pred2
                new_pairs = self._sparse_compose(pred1, pred2)

                if new_pairs.numel() == 0:
                    continue

                # Compute hashes
                new_hashes = new_pairs[:, 0] * self.pack_base + new_pairs[:, 1]

                # Merge with existing closure
                if head_pred in self.closure_hashes:
                    combined = torch.cat([self.closure_hashes[head_pred], new_hashes])
                    unique = combined.unique()

                    if unique.shape[0] > self.closure_hashes[head_pred].shape[0]:
                        changed = True

                    # Limit size
                    if unique.shape[0] > self.max_pairs_per_pred:
                        unique = unique[:self.max_pairs_per_pred]

                    self.closure_hashes[head_pred] = unique.sort()[0]
                else:
                    self.closure_hashes[head_pred] = new_hashes.unique().sort()[0]
                    changed = True

            if not changed:
                break

    def _sparse_compose(self, pred1: int, pred2: int) -> Tensor:
        """
        Compute sparse composition: {(x,y) : ∃z. pred1(x,z) ∧ pred2(z,y)}

        Uses hash-based join for efficiency.
        """
        edges1 = self.pred_facts[pred1]  # [E1, 2]: (x, z)

        if pred2 not in self.closure_hashes:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)

        # Get closure pairs for pred2
        hashes2 = self.closure_hashes[pred2]

        # For each unique z in edges1, find matching (z, y) in pred2
        z_values = edges1[:, 1].unique()

        result_pairs = []
        batch_size = 1000  # Process in batches

        for i in range(0, z_values.shape[0], batch_size):
            z_batch = z_values[i:i+batch_size]

            for z in z_batch:
                z_val = z.item()

                # Find all x where pred1(x, z)
                x_mask = edges1[:, 1] == z_val
                x_vals = edges1[x_mask, 0]

                if x_vals.numel() == 0:
                    continue

                # Find all y where pred2(z, y) - using hash lookup
                # Hash pattern: z * pack_base + y
                z_prefix = z_val * self.pack_base

                # Binary search for range [z*pack_base, (z+1)*pack_base)
                lo = torch.searchsorted(hashes2, z_prefix)
                hi = torch.searchsorted(hashes2, z_prefix + self.pack_base)

                if lo >= hi:
                    continue

                y_hashes = hashes2[lo:hi]
                y_vals = y_hashes - z_prefix  # Extract y from hash

                # Create all (x, y) pairs
                if x_vals.numel() * y_vals.numel() > 10000:
                    # Limit to avoid explosion
                    x_vals = x_vals[:100]
                    y_vals = y_vals[:100]

                x_exp = x_vals.unsqueeze(1).expand(-1, y_vals.shape[0])
                y_exp = y_vals.unsqueeze(0).expand(x_vals.shape[0], -1)

                pairs = torch.stack([x_exp.flatten(), y_exp.flatten()], dim=1)
                result_pairs.append(pairs)

                if sum(p.shape[0] for p in result_pairs) > self.max_pairs_per_pred:
                    break

        if result_pairs:
            combined = torch.cat(result_pairs, dim=0)
            # Remove duplicates
            hashes = combined[:, 0] * self.pack_base + combined[:, 1]
            unique_hashes = hashes.unique()
            if unique_hashes.shape[0] > self.max_pairs_per_pred:
                unique_hashes = unique_hashes[:self.max_pairs_per_pred]
            return torch.stack([unique_hashes // self.pack_base, unique_hashes % self.pack_base], dim=1)

        return torch.empty((0, 2), dtype=torch.long, device=self.device)

    def _check_in_closure(self, pred: int, arg0: Tensor, arg1: Tensor) -> Tensor:
        """Check if (arg0, arg1) pairs are in the closure for pred."""
        B = arg0.shape[0]
        result = torch.zeros(B, dtype=torch.bool, device=self.device)

        if pred not in self.closure_hashes:
            return result

        hashes = self.closure_hashes[pred]
        query_hashes = arg0.long() * self.pack_base + arg1.long()

        # Binary search
        idx = torch.searchsorted(hashes, query_hashes)
        valid = idx < hashes.shape[0]
        result[valid] = hashes[idx[valid]] == query_hashes[valid]

        return result

    @torch.no_grad()
    def prove_batch(
        self,
        goals: Tensor,
        max_depth: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prove a batch of goals using parallel BFS with pre-computed closures.

        Args:
            goals: [B, 3] ground queries (pred, arg0, arg1)
            max_depth: Override default (default: 6)

        Returns:
            proven: [B] bool
            depths: [B] int (-1 if not proven)
        """
        B = goals.shape[0]
        device = self.device
        depth_limit = max_depth or self.max_depth

        if B == 0:
            return (
                torch.zeros(0, dtype=torch.bool, device=device),
                torch.full((0,), -1, dtype=torch.long, device=device),
            )

        proven = torch.zeros(B, dtype=torch.bool, device=device)
        depths = torch.full((B,), -1, dtype=torch.long, device=device)

        # Step 1: Check direct facts (vectorized)
        is_fact = self.fact_index.contains(goals)
        proven = proven | is_fact
        depths = torch.where(is_fact, torch.zeros_like(depths), depths)

        # Step 2: Check pre-computed closures (vectorized)
        remaining = ~proven
        if remaining.any():
            for pred in self.closure_hashes.keys():
                pred_mask = (goals[:, 0] == pred) & remaining
                if not pred_mask.any():
                    continue

                idx = pred_mask.nonzero(as_tuple=True)[0]
                in_closure = self._check_in_closure(pred, goals[idx, 1], goals[idx, 2])

                proven[idx] = proven[idx] | in_closure
                depths[idx] = torch.where(
                    in_closure & (depths[idx] < 0),
                    torch.ones_like(depths[idx]),
                    depths[idx]
                )

        # Step 3: Parallel BFS for remaining goals
        remaining = ~proven
        if remaining.any():
            proven, depths = self._parallel_bfs(goals, proven, depths, remaining, depth_limit)

        return proven, depths

    def _parallel_bfs(
        self,
        goals: Tensor,
        proven: Tensor,
        depths: Tensor,
        active_mask: Tensor,
        max_depth: int,
    ) -> Tuple[Tensor, Tensor]:
        """Parallel BFS for goals not resolved by closures."""
        B = goals.shape[0]
        device = self.device

        active_idx = active_mask.nonzero(as_tuple=True)[0]
        if active_idx.numel() == 0:
            return proven, depths

        active_goals = goals[active_idx]
        N = active_goals.shape[0]

        # Initialize per-goal state
        # Each goal has a frontier of subgoal lists to explore
        local_proven = torch.zeros(N, dtype=torch.bool, device=device)
        local_depths = torch.full((N,), -1, dtype=torch.long, device=device)

        # Process in parallel batches
        # Frontier: for each goal, list of (subgoals, depth)
        frontiers: List[List[Tuple[List[Tensor], int]]] = [
            [([active_goals[i]], 1)] for i in range(N)
        ]

        for _ in range(max_depth * 10):  # Max iterations
            any_work = False

            # Collect all frontier items across goals
            all_subgoals = []
            all_indices = []
            all_depths_list = []
            all_frontier_idx = []

            for i in range(N):
                if local_proven[i] or not frontiers[i]:
                    continue

                item = frontiers[i].pop(0)
                subgoals, d = item

                if d > max_depth:
                    continue

                if not subgoals:
                    local_proven[i] = True
                    local_depths[i] = d - 1
                    continue

                all_subgoals.append(torch.stack(subgoals))
                all_indices.append(i)
                all_depths_list.append(d)
                all_frontier_idx.append(len(all_subgoals) - 1)
                any_work = True

            if not any_work:
                break

            # Batch process all subgoal sets
            for batch_idx, (subgoals, goal_idx, d) in enumerate(zip(all_subgoals, all_indices, all_depths_list)):
                if local_proven[goal_idx]:
                    continue

                # Check which subgoals are provable
                sg_proven = self.fact_index.contains(subgoals)

                # Also check closures
                for pred in self.closure_hashes.keys():
                    pred_mask = subgoals[:, 0] == pred
                    if pred_mask.any():
                        pidx = pred_mask.nonzero(as_tuple=True)[0]
                        in_closure = self._check_in_closure(pred, subgoals[pidx, 1], subgoals[pidx, 2])
                        sg_proven[pidx] = sg_proven[pidx] | in_closure

                if sg_proven.all():
                    local_proven[goal_idx] = True
                    local_depths[goal_idx] = d
                    continue

                # Expand first unproven subgoal
                unproven_idx = (~sg_proven).nonzero(as_tuple=True)[0]
                if unproven_idx.numel() == 0:
                    continue

                first_unproven = subgoals[unproven_idx[0]]
                pred = first_unproven[0].item()

                if pred not in self.rules_by_pred:
                    continue

                # Apply each applicable rule
                remaining = [subgoals[j] for j in unproven_idx[1:].tolist()]

                for rule_idx in self.rules_by_pred[pred][:5]:  # Limit rules
                    new_subgoals = self._apply_rule(first_unproven, rule_idx)
                    if new_subgoals is None:
                        continue

                    combined = remaining + new_subgoals
                    if len(combined) <= 10:  # Limit subgoal count
                        frontiers[goal_idx].append((combined, d + 1))

                # Limit frontier size
                frontiers[goal_idx] = frontiers[goal_idx][:50]

        # Update global results
        proven[active_idx] = proven[active_idx] | local_proven
        depths[active_idx] = torch.where(
            local_proven & (depths[active_idx] < 0),
            local_depths,
            depths[active_idx]
        )

        return proven, depths

    def _apply_rule(self, goal: Tensor, rule_idx: int) -> Optional[List[Tensor]]:
        """Apply rule to goal, return new subgoals or None."""
        head = self.rules_heads[rule_idx]
        body = self.rules_bodies[rule_idx]
        body_len = self.rule_lens[rule_idx].item()

        if goal[0] != head[0]:
            return None

        # Build substitution
        subs = {}
        for i in range(1, 3):
            h_arg = head[i].item()
            g_arg = goal[i].item()

            if h_arg > self.constant_no:
                if h_arg in subs and subs[h_arg] != g_arg:
                    return None
                subs[h_arg] = g_arg
            elif h_arg != g_arg:
                return None

        if body_len == 0:
            return []

        # Apply substitution to body
        new_subgoals = []
        for j in range(body_len):
            b_atom = body[j]
            new_atom = torch.zeros(3, dtype=torch.long, device=self.device)
            new_atom[0] = b_atom[0]

            has_unbound = False
            for k in range(1, 3):
                arg = b_atom[k].item()
                if arg > self.constant_no:
                    if arg in subs:
                        new_atom[k] = subs[arg]
                    else:
                        has_unbound = True
                        new_atom[k] = arg
                else:
                    new_atom[k] = arg

            if has_unbound:
                # Enumerate from facts (limited)
                ground_atoms = self._enumerate_bindings(new_atom, subs)
                if ground_atoms is None or len(ground_atoms) == 0:
                    return None
                new_subgoals.extend(ground_atoms[:10])
            else:
                new_subgoals.append(new_atom)

        return new_subgoals[:10]  # Limit

    def _enumerate_bindings(self, atom: Tensor, subs: dict) -> Optional[List[Tensor]]:
        """Enumerate ground instantiations from facts."""
        pred = atom[0].item()

        if pred not in self.pred_facts:
            return None

        facts = self.pred_facts[pred]
        mask = torch.ones(facts.shape[0], dtype=torch.bool, device=self.device)

        for i, arg in enumerate([atom[1].item(), atom[2].item()]):
            if arg <= self.constant_no:
                mask = mask & (facts[:, i] == arg)
            elif arg in subs:
                mask = mask & (facts[:, i] == subs[arg])

        matching = facts[mask]
        if matching.numel() == 0:
            return None

        matching = matching[:20]  # Limit
        result = []
        for i in range(matching.shape[0]):
            new_atom = torch.zeros(3, dtype=torch.long, device=self.device)
            new_atom[0] = pred
            new_atom[1] = matching[i, 0]
            new_atom[2] = matching[i, 1]
            result.append(new_atom)

        return result

    def clear_cache(self) -> None:
        self._stats = {"cache_hits": 0, "cache_misses": 0}

    def cache_stats(self) -> dict:
        total_pairs = sum(h.shape[0] for h in self.closure_hashes.values())
        return {
            "compositions": len(self.closure_hashes),
            "total_pairs": total_pairs,
            "composition_rules": len(self.composition_rules),
            **self._stats,
        }

    @classmethod
    def from_index_manager(cls, im, **kwargs) -> "DPProverOptimized":
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

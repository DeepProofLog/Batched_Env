"""
Fast BFS Prover optimized for complex rule sets.

This prover focuses on fast parallel BFS without expensive pre-computation.
Suitable for datasets like family with many rules.

Key optimizations:
    1. Hash-based fact lookup (O(1))
    2. Parallel BFS with limited branching
    3. No expensive closure pre-computation
    4. Depth limit of 6
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, List
import torch
from torch import Tensor

from kge_experiments.unification import GPUFactIndex


class DPProverFastBFS:
    """
    Fast BFS prover without expensive pre-computation.

    Uses hash-based lookups and limited branching for speed.
    """

    def __init__(
        self,
        facts: Tensor,
        rules_heads: Tensor,
        rules_bodies: Tensor,
        rule_lens: Tensor,
        constant_no: int,
        padding_idx: int = 0,
        max_depth: int = 6,
        max_branches: int = 100,
        max_subgoals: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.device = device or facts.device
        self.constant_no = constant_no
        self.padding_idx = padding_idx
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.max_subgoals = max_subgoals

        self.facts = facts.to(self.device)
        self.rules_heads = rules_heads.to(self.device)
        self.rules_bodies = rules_bodies.to(self.device)
        self.rule_lens = rule_lens.to(self.device)

        self.pack_base = constant_no + 10000

        # Fast fact lookup
        self.fact_index = GPUFactIndex(self.facts, self.pack_base)

        # Build indices
        self._build_pred_fact_index()
        self._build_rule_index()

    def _build_pred_fact_index(self) -> None:
        """Build per-predicate fact indices for fast lookup."""
        self.pred_facts: Dict[int, Tensor] = {}
        self.pred_fact_hashes: Dict[int, Tensor] = {}

        if self.facts.numel() == 0:
            return

        for pred in self.facts[:, 0].unique().tolist():
            mask = self.facts[:, 0] == pred
            edges = self.facts[mask][:, 1:]
            self.pred_facts[pred] = edges

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

    def _check_fact(self, pred: int, arg0: int, arg1: int) -> bool:
        """O(log n) fact check."""
        if pred not in self.pred_fact_hashes:
            return False

        hashes = self.pred_fact_hashes[pred]
        query_hash = arg0 * self.pack_base + arg1

        idx = torch.searchsorted(hashes, torch.tensor([query_hash], device=self.device))
        if idx >= hashes.shape[0]:
            return False
        return hashes[idx].item() == query_hash

    def _check_facts_batch(self, goals: Tensor) -> Tensor:
        """Batch fact check."""
        return self.fact_index.contains(goals)

    @torch.no_grad()
    def prove_batch(
        self,
        goals: Tensor,
        max_depth: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prove batch using parallel BFS.

        Args:
            goals: [B, 3] queries
            max_depth: Override default

        Returns:
            proven: [B] bool
            depths: [B] int
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

        # Check facts first
        is_fact = self._check_facts_batch(goals)
        proven = proven | is_fact
        depths = torch.where(is_fact, torch.zeros_like(depths), depths)

        # BFS for remaining
        remaining = ~proven
        if remaining.any():
            for i in remaining.nonzero(as_tuple=True)[0].tolist():
                is_proven, depth = self._prove_single_bfs(goals[i], depth_limit)
                proven[i] = is_proven
                depths[i] = depth

        return proven, depths

    def _prove_single_bfs(self, goal: Tensor, max_depth: int) -> Tuple[bool, int]:
        """BFS proof search for single goal."""
        pred = goal[0].item()
        arg0 = goal[1].item()
        arg1 = goal[2].item()

        # Check fact
        if self._check_fact(pred, arg0, arg1):
            return True, 0

        if pred not in self.rules_by_pred:
            return False, -1

        # BFS queue: [(subgoals_list, depth)]
        from collections import deque
        queue = deque()

        # Initialize with rule expansions
        for rule_idx in self.rules_by_pred[pred]:
            new_subgoals = self._apply_rule(goal, rule_idx)
            if new_subgoals is not None:
                if len(new_subgoals) == 0:
                    return True, 1  # Empty body = proven
                queue.append((new_subgoals, 1))

        visited = set()
        visited.add(self._hash_subgoals([goal]))

        while queue:
            subgoals, depth = queue.popleft()

            if depth > max_depth:
                continue

            # Check all subgoals
            all_proven = True
            unproven = []

            for sg in subgoals:
                p, a0, a1 = sg[0].item(), sg[1].item(), sg[2].item()
                if self._check_fact(p, a0, a1):
                    continue
                all_proven = False
                unproven.append(sg)

            if all_proven:
                return True, depth

            if not unproven:
                continue

            # Expand first unproven
            first = unproven[0]
            first_pred = first[0].item()

            if first_pred not in self.rules_by_pred:
                continue

            remaining = unproven[1:]

            for rule_idx in self.rules_by_pred[first_pred][:5]:  # Limit rules
                new_subgoals = self._apply_rule(first, rule_idx)
                if new_subgoals is None:
                    continue

                combined = remaining + new_subgoals
                if len(combined) > self.max_subgoals:
                    continue

                state_hash = self._hash_subgoals(combined)
                if state_hash in visited:
                    continue
                visited.add(state_hash)

                if len(visited) > self.max_branches:
                    break

                queue.append((combined, depth + 1))

        return False, -1

    def _hash_subgoals(self, subgoals: List[Tensor]) -> int:
        """Hash a list of subgoals for visited check."""
        if not subgoals:
            return 0

        hashes = []
        for sg in subgoals:
            h = sg[0].item() * self.pack_base * self.pack_base
            h += sg[1].item() * self.pack_base
            h += sg[2].item()
            hashes.append(h)

        return hash(tuple(sorted(hashes)))

    def _apply_rule(self, goal: Tensor, rule_idx: int) -> Optional[List[Tensor]]:
        """Apply rule to goal."""
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

        # Apply subs to body
        new_subgoals = []
        for j in range(body_len):
            b = body[j]
            new_atom = torch.zeros(3, dtype=torch.long, device=self.device)
            new_atom[0] = b[0]

            has_unbound = False
            for k in range(1, 3):
                arg = b[k].item()
                if arg > self.constant_no:
                    if arg in subs:
                        new_atom[k] = subs[arg]
                    else:
                        has_unbound = True
                        new_atom[k] = arg
                else:
                    new_atom[k] = arg

            if has_unbound:
                ground = self._enumerate_bindings(new_atom, subs)
                if ground is None:
                    return None
                new_subgoals.extend(ground[:5])
            else:
                new_subgoals.append(new_atom)

        return new_subgoals[:self.max_subgoals]

    def _enumerate_bindings(self, atom: Tensor, subs: dict) -> Optional[List[Tensor]]:
        """Enumerate bindings from facts."""
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

        result = []
        for i in range(min(10, matching.shape[0])):
            new_atom = torch.zeros(3, dtype=torch.long, device=self.device)
            new_atom[0] = pred
            new_atom[1] = matching[i, 0]
            new_atom[2] = matching[i, 1]
            result.append(new_atom)

        return result

    def clear_cache(self) -> None:
        pass

    def cache_stats(self) -> dict:
        return {"predicates": len(self.pred_facts), "rules": len(self.rules_by_pred)}

    @classmethod
    def from_index_manager(cls, im, **kwargs) -> "DPProverFastBFS":
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

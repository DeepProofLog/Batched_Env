"""
Multiprocessing BFS Prover for parallel query processing.

Uses Python multiprocessing to parallelize BFS across CPU cores.
This can significantly speed up evaluation when proving many queries.
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, List
import torch
from torch import Tensor
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from kge_experiments.unification import GPUFactIndex


def _prove_single_worker(args):
    """Worker function for multiprocessing."""
    (goal_data, facts_data, rules_heads_data, rules_bodies_data, rule_lens_data,
     constant_no, padding_idx, pack_base, pred_facts_keys, pred_facts_data,
     rules_by_pred, max_depth, max_branches, max_subgoals) = args

    goal = torch.tensor(goal_data, dtype=torch.long)
    facts = torch.tensor(facts_data, dtype=torch.long) if facts_data is not None else None
    rules_heads = torch.tensor(rules_heads_data, dtype=torch.long)
    rules_bodies = torch.tensor(rules_bodies_data, dtype=torch.long)
    rule_lens = torch.tensor(rule_lens_data, dtype=torch.long)

    # Reconstruct pred_facts from serialized data
    pred_facts = {}
    pred_fact_hashes = {}
    for pred, edges_data in zip(pred_facts_keys, pred_facts_data):
        edges = torch.tensor(edges_data, dtype=torch.long)
        pred_facts[pred] = edges
        hashes = edges[:, 0].long() * pack_base + edges[:, 1].long()
        pred_fact_hashes[pred] = hashes.sort()[0]

    # Run BFS
    return _prove_single_bfs_cpu(
        goal, pred_facts, pred_fact_hashes, rules_heads, rules_bodies, rule_lens,
        constant_no, padding_idx, pack_base, rules_by_pred,
        max_depth, max_branches, max_subgoals
    )


def _check_fact_cpu(pred: int, arg0: int, arg1: int, pred_fact_hashes: Dict[int, Tensor], pack_base: int,
                    excluded: Tuple[int, int, int] = None) -> bool:
    """CPU fact check, optionally excluding a specific triple."""
    # Check if this is the excluded query
    if excluded is not None and (pred, arg0, arg1) == excluded:
        return False

    if pred not in pred_fact_hashes:
        return False
    hashes = pred_fact_hashes[pred]
    query_hash = arg0 * pack_base + arg1
    idx = torch.searchsorted(hashes, torch.tensor([query_hash]))
    if idx >= hashes.shape[0]:
        return False
    return hashes[idx].item() == query_hash


def _prove_single_bfs_cpu(
    goal: Tensor,
    pred_facts: Dict[int, Tensor],
    pred_fact_hashes: Dict[int, Tensor],
    rules_heads: Tensor,
    rules_bodies: Tensor,
    rule_lens: Tensor,
    constant_no: int,
    padding_idx: int,
    pack_base: int,
    rules_by_pred: Dict[int, List[int]],
    max_depth: int,
    max_branches: int,
    max_subgoals: int
) -> Tuple[bool, int]:
    """BFS proof search on CPU.

    IMPORTANT: The goal itself is EXCLUDED from fact lookups.
    This ensures we test if the goal can be DERIVED via rules,
    not just looked up as an existing fact.
    """
    from collections import deque

    pred = goal[0].item()
    arg0 = goal[1].item()
    arg1 = goal[2].item()

    # The goal itself is excluded from fact checks
    excluded = (pred, arg0, arg1)

    # NOTE: We do NOT check if goal is a direct fact.
    # The goal must be derived via rules, not found as an existing fact.

    if pred not in rules_by_pred:
        return False, -1

    # BFS
    queue = deque()
    visited = set()

    # Initial expansion
    for rule_idx in rules_by_pred[pred]:
        new_subgoals = _apply_rule_cpu(
            goal, rule_idx, rules_heads, rules_bodies, rule_lens,
            constant_no, padding_idx, pred_facts
        )
        if new_subgoals is None:
            continue
        if len(new_subgoals) == 0:
            return True, 1

        state_hash = _hash_state_cpu(new_subgoals, pack_base)
        if state_hash not in visited:
            visited.add(state_hash)
            queue.append((new_subgoals, 1))

    while queue:
        if len(visited) > max_branches:
            break

        subgoals, depth = queue.popleft()

        if depth > max_depth:
            continue

        # Check subgoals (excluding the original goal from fact lookups)
        all_facts = True
        unproven = []

        for sg in subgoals:
            p = sg[0].item()
            a0 = sg[1].item()
            a1 = sg[2].item()
            if _check_fact_cpu(p, a0, a1, pred_fact_hashes, pack_base, excluded=excluded):
                continue
            all_facts = False
            unproven.append(sg)

        if all_facts:
            return True, depth

        if not unproven:
            continue

        # Expand first unproven
        first = unproven[0]
        first_pred = first[0].item()

        if first_pred not in rules_by_pred:
            continue

        remaining = unproven[1:]

        for rule_idx in rules_by_pred[first_pred][:5]:
            new_sgs = _apply_rule_cpu(
                first, rule_idx, rules_heads, rules_bodies, rule_lens,
                constant_no, padding_idx, pred_facts
            )
            if new_sgs is None:
                continue

            combined = remaining + new_sgs
            if len(combined) > max_subgoals:
                continue

            state_hash = _hash_state_cpu(combined, pack_base)
            if state_hash in visited:
                continue

            if len(visited) >= max_branches:
                break

            visited.add(state_hash)
            queue.append((combined, depth + 1))

    return False, -1


def _hash_state_cpu(subgoals: List[Tensor], pack_base: int) -> int:
    """Hash subgoals for visited check."""
    if not subgoals:
        return 0
    hashes = []
    for sg in subgoals:
        h = sg[0].item() * pack_base * pack_base
        h += sg[1].item() * pack_base
        h += sg[2].item()
        hashes.append(h)
    return hash(tuple(sorted(hashes)))


def _apply_rule_cpu(
    goal: Tensor,
    rule_idx: int,
    rules_heads: Tensor,
    rules_bodies: Tensor,
    rule_lens: Tensor,
    constant_no: int,
    padding_idx: int,
    pred_facts: Dict[int, Tensor]
) -> Optional[List[Tensor]]:
    """Apply rule on CPU with proper shared variable handling."""
    head = rules_heads[rule_idx]
    body = rules_bodies[rule_idx]
    body_len = rule_lens[rule_idx].item()

    if goal[0] != head[0]:
        return None

    # Build substitution from unifying goal with head
    subs = {}
    for i in range(1, 3):
        h_arg = head[i].item()
        g_arg = goal[i].item()

        if h_arg > constant_no:
            if h_arg in subs and subs[h_arg] != g_arg:
                return None
            subs[h_arg] = g_arg
        elif h_arg != g_arg:
            return None

    if body_len == 0:
        return []

    # Collect body atoms and identify unbound variables
    body_atoms = []
    unbound_vars = set()
    for j in range(body_len):
        b = body[j]
        atom = (b[0].item(), b[1].item(), b[2].item())
        body_atoms.append(atom)
        for arg in atom[1:]:
            if arg > constant_no and arg not in subs:
                unbound_vars.add(arg)

    # If no unbound variables, just apply subs and return
    if not unbound_vars:
        new_subgoals = []
        for pred, arg1, arg2 in body_atoms:
            new_atom = torch.zeros(3, dtype=torch.long)
            new_atom[0] = pred
            new_atom[1] = subs.get(arg1, arg1) if arg1 > constant_no else arg1
            new_atom[2] = subs.get(arg2, arg2) if arg2 > constant_no else arg2
            new_subgoals.append(new_atom)
        return new_subgoals

    # Find valid bindings for unbound variables
    valid_bindings = _find_valid_bindings_cpu(body_atoms, subs, unbound_vars, constant_no, pred_facts)

    if not valid_bindings:
        return None

    # Use first valid binding
    binding = valid_bindings[0]
    full_subs = {**subs, **binding}

    # Apply full substitution to all body atoms
    new_subgoals = []
    for pred, arg1, arg2 in body_atoms:
        new_atom = torch.zeros(3, dtype=torch.long)
        new_atom[0] = pred
        new_atom[1] = full_subs.get(arg1, arg1) if arg1 > constant_no else arg1
        new_atom[2] = full_subs.get(arg2, arg2) if arg2 > constant_no else arg2
        new_subgoals.append(new_atom)

    return new_subgoals[:8]


def _find_valid_bindings_cpu(
    body_atoms: List[Tuple[int, int, int]],
    subs: dict,
    unbound_vars: set,
    constant_no: int,
    pred_facts: Dict[int, Tensor]
) -> List[dict]:
    """Find bindings for unbound variables that satisfy all body atoms."""
    # Find atoms that constrain unbound variables
    constraining_atoms = []
    for pred, arg1, arg2 in body_atoms:
        has_unbound = (arg1 in unbound_vars) or (arg2 in unbound_vars)
        if has_unbound:
            constraining_atoms.append((pred, arg1, arg2))

    if not constraining_atoms:
        return [{}]

    # Start with possible bindings from first constraining atom
    first_atom = constraining_atoms[0]
    pred, arg1, arg2 = first_atom

    if pred not in pred_facts:
        return []

    facts = pred_facts[pred]

    # Pre-filter facts by known substitutions
    mask = torch.ones(facts.shape[0], dtype=torch.bool)

    if arg1 > constant_no and arg1 in subs:
        mask = mask & (facts[:, 0] == subs[arg1])
    elif arg1 <= constant_no:
        mask = mask & (facts[:, 0] == arg1)

    if arg2 > constant_no and arg2 in subs:
        mask = mask & (facts[:, 1] == subs[arg2])
    elif arg2 <= constant_no:
        mask = mask & (facts[:, 1] == arg2)

    filtered_facts = facts[mask]
    if filtered_facts.numel() == 0:
        return []

    # Extract bindings from filtered facts
    candidates = []
    for i in range(min(50, filtered_facts.shape[0])):
        binding = {}
        f_arg1 = filtered_facts[i, 0].item()
        f_arg2 = filtered_facts[i, 1].item()

        if arg1 in unbound_vars:
            binding[arg1] = f_arg1
        if arg2 in unbound_vars:
            if arg2 in binding and binding[arg2] != f_arg2:
                continue
            binding[arg2] = f_arg2

        candidates.append(binding)

    # Filter candidates against remaining constraining atoms
    valid_bindings = []
    for binding in candidates:
        full_subs = {**subs, **binding}
        all_satisfied = True

        for pred, arg1, arg2 in constraining_atoms[1:]:
            resolved_arg1 = full_subs.get(arg1, arg1) if arg1 > constant_no else arg1
            resolved_arg2 = full_subs.get(arg2, arg2) if arg2 > constant_no else arg2

            if resolved_arg1 > constant_no or resolved_arg2 > constant_no:
                if pred not in pred_facts:
                    all_satisfied = False
                    break
                continue

            # Check if this ground atom is a fact
            if pred not in pred_facts:
                all_satisfied = False
                break
            p_facts = pred_facts[pred]
            query_hash = resolved_arg1 * (constant_no + 10000) + resolved_arg2
            fact_hashes = p_facts[:, 0] * (constant_no + 10000) + p_facts[:, 1]
            if not (fact_hashes == query_hash).any():
                all_satisfied = False
                break

        if all_satisfied:
            valid_bindings.append(binding)
            if len(valid_bindings) >= 5:
                break

    return valid_bindings


class DPProverMultiProc:
    """
    Multiprocessing BFS prover.

    Uses ProcessPoolExecutor for parallel query processing.
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
        max_branches: int = 50,
        max_subgoals: int = 8,
        n_workers: int = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device('cpu')
        self.constant_no = constant_no
        self.padding_idx = padding_idx
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.max_subgoals = max_subgoals
        self.n_workers = n_workers or mp.cpu_count()

        # Store tensors on CPU for worker access
        self.facts = facts.cpu()
        self.rules_heads = rules_heads.cpu()
        self.rules_bodies = rules_bodies.cpu()
        self.rule_lens = rule_lens.cpu()

        self.pack_base = constant_no + 10000

        # Build indices
        self._build_indices()

        # Prepare serialized data for workers
        self._prepare_worker_data()

    def _build_indices(self) -> None:
        """Build predicate-indexed lookups."""
        self.pred_facts: Dict[int, Tensor] = {}

        if self.facts.numel() > 0:
            for pred in self.facts[:, 0].unique().tolist():
                mask = self.facts[:, 0] == pred
                edges = self.facts[mask][:, 1:]
                self.pred_facts[pred] = edges

        self.rules_by_pred: Dict[int, List[int]] = {}
        if self.rules_heads.numel() > 0:
            for i in range(self.rules_heads.shape[0]):
                pred = self.rules_heads[i, 0].item()
                if pred not in self.rules_by_pred:
                    self.rules_by_pred[pred] = []
                self.rules_by_pred[pred].append(i)

    def _prepare_worker_data(self) -> None:
        """Prepare serialized data for workers."""
        self._facts_data = self.facts.tolist() if self.facts.numel() > 0 else None
        self._rules_heads_data = self.rules_heads.tolist()
        self._rules_bodies_data = self.rules_bodies.tolist()
        self._rule_lens_data = self.rule_lens.tolist()

        self._pred_facts_keys = list(self.pred_facts.keys())
        self._pred_facts_data = [self.pred_facts[k].tolist() for k in self._pred_facts_keys]

    @torch.no_grad()
    def prove_batch(
        self,
        goals: Tensor,
        max_depth: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prove batch using multiprocessing.
        """
        B = goals.shape[0]
        device = self.device
        depth_limit = max_depth or self.max_depth

        if B == 0:
            return (
                torch.zeros(0, dtype=torch.bool, device=device),
                torch.full((0,), -1, dtype=torch.long, device=device),
            )

        # Prepare arguments for workers
        goals_cpu = goals.cpu()

        args_list = [
            (goals_cpu[i].tolist(),
             self._facts_data,
             self._rules_heads_data,
             self._rules_bodies_data,
             self._rule_lens_data,
             self.constant_no,
             self.padding_idx,
             self.pack_base,
             self._pred_facts_keys,
             self._pred_facts_data,
             self.rules_by_pred,
             depth_limit,
             self.max_branches,
             self.max_subgoals)
            for i in range(B)
        ]

        # Process in parallel
        proven_list = []
        depths_list = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(_prove_single_worker, args): i for i, args in enumerate(args_list)}
            results = [None] * B

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = (False, -1)

        for is_proven, depth in results:
            proven_list.append(is_proven)
            depths_list.append(depth)

        proven = torch.tensor(proven_list, dtype=torch.bool, device=device)
        depths = torch.tensor(depths_list, dtype=torch.long, device=device)

        return proven, depths

    def clear_cache(self) -> None:
        pass

    def cache_stats(self) -> dict:
        return {
            "predicates": len(self.pred_facts),
            "rules": len(self.rules_by_pred),
            "workers": self.n_workers,
        }

    @classmethod
    def from_index_manager(cls, im, **kwargs) -> "DPProverMultiProc":
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

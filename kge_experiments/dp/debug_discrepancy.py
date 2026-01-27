#!/usr/bin/env python
"""
Debug script to investigate queries that PPO proves but DP doesn't.
"""

from __future__ import annotations
import sys
from pathlib import Path

import torch

# Set up paths
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
kge_root = str(Path(__file__).parent.parent)
if kge_root not in sys.path:
    sys.path.insert(0, kge_root)

from data_handler import DataHandler
from index_manager import IndexManager
from kge_experiments.dp.prover_parallel_bfs import DPProverParallelBFS


def load_depth_file(path: Path) -> dict:
    """Load depth file and return dict of (pred, arg0, arg1) -> depth.

    Format: predicate(arg0,arg1) depth
    Example: aunt(1369,1287) 2
    """
    import re
    depths = {}
    pattern = re.compile(r'(\w+)\((\w+),(\w+)\)\s+(-?\d+)')

    with open(path) as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                pred, arg0, arg1, depth = match.groups()
                depths[(pred, arg0, arg1)] = int(depth)
    return depths


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = str(Path(__file__).parent.parent / "data")

    # Load family dataset
    print("Loading family dataset...")
    dh = DataHandler(
        dataset_name="family",
        base_path=base_path,
        facts_file="train.txt",
        filter_queries_by_rules=True,
    )

    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=4096,
        max_arity=dh.max_arity,
        padding_atoms=10,
        device=device,
        rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)

    # Load depth file
    depth_path = Path(__file__).parent.parent / "data" / "family" / "train_depths.txt"
    print(f"Loading depth file: {depth_path}")
    ppo_depths = load_depth_file(depth_path)

    # Get PPO-provable queries (depth >= 0)
    ppo_provable = {k for k, v in ppo_depths.items() if v >= 0}
    print(f"PPO-provable queries: {len(ppo_provable)}")

    # Create prover with very high limits
    prover = DPProverParallelBFS.from_index_manager(
        im,
        max_depth=10,
        max_branches=1000,
        max_subgoals=30,
    )

    # Test all PPO-provable queries
    print("\nTesting all PPO-provable queries with high-limit DP prover...")

    ppo_only = []  # PPO proves but DP doesn't

    for i, (pred_str, arg0_str, arg1_str) in enumerate(ppo_provable):
        # Convert to indices
        pred_idx = im.predicate_str2idx.get(pred_str)
        arg0_idx = im.constant_str2idx.get(arg0_str)
        arg1_idx = im.constant_str2idx.get(arg1_str)

        if pred_idx is None or arg0_idx is None or arg1_idx is None:
            continue

        goal = torch.tensor([[pred_idx, arg0_idx, arg1_idx]], dtype=torch.long, device=device)
        proven, depths = prover.prove_batch(goal, max_depth=10)

        if not proven[0].item():
            ppo_only.append({
                "query": (pred_str, arg0_str, arg1_str),
                "indices": (pred_idx, arg0_idx, arg1_idx),
                "ppo_depth": ppo_depths[(pred_str, arg0_str, arg1_str)],
            })

        if (i + 1) % 1000 == 0:
            print(f"  Tested {i+1}/{len(ppo_provable)}, found {len(ppo_only)} discrepancies")

    print(f"\n=== RESULTS ===")
    print(f"Total PPO-provable: {len(ppo_provable)}")
    print(f"DP fails to prove: {len(ppo_only)}")

    if ppo_only:
        print(f"\n=== DISCREPANT QUERIES ===")
        for item in ppo_only[:20]:  # Show first 20
            q = item["query"]
            print(f"  {q[0]}({q[1]}, {q[2]}) - PPO depth: {item['ppo_depth']}")

        # Deep dive into first discrepant query
        if ppo_only:
            print(f"\n=== DEEP DIVE: First discrepant query ===")
            item = ppo_only[0]
            q = item["query"]
            pred_str, arg0_str, arg1_str = q
            pred_idx, arg0_idx, arg1_idx = item["indices"]

            print(f"Query: {pred_str}({arg0_str}, {arg1_str})")
            print(f"Indices: pred={pred_idx}, arg0={arg0_idx}, arg1={arg1_idx}")
            print(f"PPO depth: {item['ppo_depth']}")

            # Check if it's a direct fact
            is_fact = prover._check_fact(pred_idx, arg0_idx, arg1_idx)
            print(f"Is direct fact (no exclusion): {is_fact}")

            # Check rules for this predicate
            rules = prover.rules_by_pred.get(pred_idx, [])
            print(f"Rules for predicate {pred_str}: {len(rules)}")

            def idx_to_pred(idx):
                return im.idx2predicate[idx] if 0 <= idx < len(im.idx2predicate) else f"P{idx}"

            def idx_to_const(idx):
                if 0 <= idx < len(im.idx2constant):
                    return im.idx2constant[idx]
                elif idx > im.constant_no:
                    return f"V{idx}"
                return f"?{idx}"

            for rule_idx in rules[:10]:
                head = prover.rules_heads[rule_idx]
                body = prover.rules_bodies[rule_idx]
                body_len = prover.rule_lens[rule_idx].item()

                head_str = f"{idx_to_pred(head[0].item())}({idx_to_const(head[1].item())}, {idx_to_const(head[2].item())})"

                body_strs = []
                for j in range(body_len):
                    b = body[j]
                    p = idx_to_pred(b[0].item())
                    a0 = idx_to_const(b[1].item())
                    a1 = idx_to_const(b[2].item())
                    body_strs.append(f"{p}({a0}, {a1})")

                print(f"  Rule {rule_idx}: {head_str} :- {', '.join(body_strs) if body_strs else 'true'}")

            # Try manual BFS trace
            print(f"\n=== Manual BFS Trace ===")
            goal = torch.tensor([pred_idx, arg0_idx, arg1_idx], dtype=torch.long, device=device)
            trace_bfs(prover, goal, im, max_depth=5)


def trace_bfs(prover, goal, im, max_depth=3):
    """Manually trace BFS to see why proof fails."""
    from collections import deque

    def idx_to_pred(idx):
        return im.idx2predicate[idx] if 0 <= idx < len(im.idx2predicate) else f"P{idx}"

    def idx_to_const(idx):
        if 0 <= idx < len(im.idx2constant):
            return im.idx2constant[idx]
        elif idx > im.constant_no:
            return f"V{idx}"
        return f"?{idx}"

    pred = goal[0].item()
    arg0 = goal[1].item()
    arg1 = goal[2].item()
    excluded = (pred, arg0, arg1)

    print(f"Goal: {idx_to_pred(pred)}({idx_to_const(arg0)}, {idx_to_const(arg1)})")
    print(f"Excluded from fact checks: {excluded}")

    if pred not in prover.rules_by_pred:
        print("No rules for this predicate!")
        return

    queue = deque()
    visited = set()

    # Initial expansion
    print(f"\nInitial rule applications:")
    for rule_idx in prover.rules_by_pred[pred]:
        new_subgoals = prover._apply_rule(goal, rule_idx)
        if new_subgoals is None:
            continue
        if len(new_subgoals) == 0:
            print(f"  Rule {rule_idx}: Empty body -> PROVEN at depth 1!")
            return

        subgoal_strs = []
        for sg in new_subgoals:
            p = idx_to_pred(sg[0].item())
            a0 = idx_to_const(sg[1].item())
            a1 = idx_to_const(sg[2].item())
            subgoal_strs.append(f"{p}({a0}, {a1})")

        state_hash = prover._hash_state(new_subgoals)
        if state_hash not in visited:
            visited.add(state_hash)
            queue.append((new_subgoals, 1, rule_idx))
            print(f"  Rule {rule_idx}: {' & '.join(subgoal_strs)}")

    print(f"\nBFS exploration (max_depth={max_depth}):")
    depth_counts = {}
    proven_at_depth = {}

    while queue:
        subgoals, depth, parent_rule = queue.popleft()

        if depth > max_depth:
            continue

        depth_counts[depth] = depth_counts.get(depth, 0) + 1

        # Check if all subgoals are facts
        all_facts = True
        unproven = []

        for sg in subgoals:
            p, a0, a1 = sg[0].item(), sg[1].item(), sg[2].item()
            if prover._check_fact(p, a0, a1, excluded=excluded):
                continue
            all_facts = False
            unproven.append(sg)

        if all_facts:
            print(f"  PROVEN at depth {depth}!")
            return

        if not unproven:
            continue

        # Expand first unproven
        first = unproven[0]
        first_pred = first[0].item()

        if first_pred not in prover.rules_by_pred:
            continue

        remaining = unproven[1:]

        for rule_idx in prover.rules_by_pred[first_pred][:10]:  # Limit for tracing
            new_sgs = prover._apply_rule(first, rule_idx)
            if new_sgs is None:
                continue

            combined = remaining + new_sgs
            if len(combined) > prover.max_subgoals:
                continue

            state_hash = prover._hash_state(combined)
            if state_hash in visited:
                continue

            if len(visited) >= 100:  # Low limit for tracing
                break

            visited.add(state_hash)
            queue.append((combined, depth + 1, rule_idx))

    print(f"  BFS exhausted without proof")
    print(f"  States explored per depth: {depth_counts}")
    print(f"  Total states visited: {len(visited)}")


if __name__ == "__main__":
    main()

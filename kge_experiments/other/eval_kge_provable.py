#!/usr/bin/env python3
"""Evaluate KGE model performance on provable vs non-provable queries.

This script evaluates KGE models on train/valid/test sets and segments
results by provability (based on depth files).

Usage:
    python eval_kge_provable.py --dataset family --split test
    python eval_kge_provable.py --dataset wn18rr --split all
    python eval_kge_provable.py --all  # Run on all datasets with depth files
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import numpy as np

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kge_module.kge_trainer.data_utils import load_dataset_split, load_triples, load_triples_with_mappings, TripleExample
from kge_module.kge_trainer.model_torch import build_model


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_filter_maps(
    *triple_collections: List[Tuple[int, int, int]]
) -> Tuple[Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int], Set[int]]]:
    """Collect triples into lookup tables for filtered head and tail corruption."""
    head_filter: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    tail_filter: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

    for triples in triple_collections:
        if not triples:
            continue
        for h, r, t in triples:
            head_filter[(r, t)].add(h)
            tail_filter[(h, r)].add(t)

    return dict(head_filter), dict(tail_filter)


@dataclass
class EvalConfig:
    """Configuration for KGE evaluation."""
    dataset: str = "family"
    data_root: str = str(Path(__file__).resolve().parent / "data")
    checkpoint_dir: str = str(Path(__file__).resolve().parent / "kge_module" / "kge_trainer" / "models")
    run_signature: Optional[str] = None
    splits: List[str] = field(default_factory=lambda: ["train", "valid", "test"])
    device: str = "cuda"
    chunk_size: int = 2048
    rank_mode: str = "realistic"
    seed: int = 42
    verbose: bool = True
    sampled_eval: bool = True  # Use sampled evaluation (100 negatives per query)
    sampled_negatives: int = 100  # Number of negatives for sampled evaluation


@dataclass
class SplitResults:
    """Aggregated results for a dataset split."""
    split_name: str
    total_queries: int
    provable_queries: int
    non_provable_queries: int
    overall_mrr: float
    overall_hits1: float
    overall_hits3: float
    overall_hits10: float
    provable_mrr: float
    provable_hits1: float
    provable_hits3: float
    provable_hits10: float
    non_provable_mrr: float
    non_provable_hits1: float
    non_provable_hits3: float
    non_provable_hits10: float
    mrr_by_depth: Dict[int, float]
    count_by_depth: Dict[int, int]


def parse_depth_file(depth_file: str) -> Dict[str, int]:
    """Parse depth file into query -> depth mapping.

    Format: "predicate(arg1,arg2) depth"
    Returns: {"predicate(arg1,arg2)": depth}
    """
    depth_map = {}
    if not os.path.exists(depth_file):
        return depth_map

    with open(depth_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on last space to handle complex predicates
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                query, depth = parts
                try:
                    depth_map[query.strip()] = int(depth)
                except ValueError:
                    continue
    return depth_map


def triple_to_query_str(head: str, relation: str, tail: str) -> str:
    """Convert triple to query string format matching depth files."""
    return f"{relation}({head},{tail})"


def find_checkpoint(checkpoint_dir: str, dataset: str) -> Optional[str]:
    """Find the latest checkpoint for a dataset."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        return None

    prefix = f"torch_{dataset}_"
    candidates = [p for p in checkpoint_path.iterdir() if p.is_dir() and p.name.startswith(prefix)]

    if not candidates:
        return None

    # Sort by modification time, return latest
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.name


def load_model_and_mappings(
    checkpoint_dir: str,
    run_signature: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, int], Dict[str, int], dict]:
    """Load KGE model and entity/relation mappings."""
    model_dir = os.path.join(checkpoint_dir, run_signature)

    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load mappings
    with open(os.path.join(model_dir, "entity2id.json"), "r") as f:
        entity2id = json.load(f)
    with open(os.path.join(model_dir, "relation2id.json"), "r") as f:
        relation2id = json.load(f)

    # Build model
    model = build_model(
        config.get("model", "RotatE"),
        config["num_entities"],
        config["num_relations"],
        dim=config.get("dim") or config.get("entity_dim"),
        gamma=config.get("gamma", 12.0),
        p_norm=config.get("p", 1),
        relation_dim=config.get("relation_dim"),
        dropout=config.get("dropout", 0.0),
    )

    # Load weights
    weights_path = os.path.join(model_dir, "weights.pth")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model, entity2id, relation2id, config


def rank_query_sampled(
    model: torch.nn.Module,
    head_idx: int,
    relation_idx: int,
    tail_idx: int,
    num_entities: int,
    head_filter: Set[int],
    tail_filter: Set[int],
    device: torch.device,
    n_negatives: int,
    rank_mode: str,
    rng: random.Random,
) -> Tuple[float, float]:
    """Compute head and tail ranks using sampled evaluation (N negatives)."""

    # Head prediction: sample N negative heads
    excluded_heads = head_filter | {head_idx}
    candidate_heads = [i for i in range(num_entities) if i not in excluded_heads]
    if len(candidate_heads) < n_negatives:
        neg_heads = candidate_heads
    else:
        neg_heads = rng.sample(candidate_heads, n_negatives)

    # Score positive + negatives
    all_heads = [head_idx] + neg_heads
    h_tensor = torch.tensor(all_heads, dtype=torch.long, device=device)
    r_tensor = torch.full((len(all_heads),), relation_idx, dtype=torch.long, device=device)
    t_tensor = torch.full((len(all_heads),), tail_idx, dtype=torch.long, device=device)
    scores = model.score_triples(h_tensor, r_tensor, t_tensor)

    target_score = scores[0]
    neg_scores = scores[1:]
    greater = float((neg_scores > target_score).sum().item())
    equal = float((neg_scores == target_score).sum().item())

    if rank_mode == "optimistic":
        head_rank = float(greater + 1.0)
    elif rank_mode == "pessimistic":
        head_rank = float(greater + 1.0 + equal)
    else:  # realistic
        head_rank = float(greater + 1.0 + 0.5 * equal)

    # Tail prediction: sample N negative tails
    excluded_tails = tail_filter | {tail_idx}
    candidate_tails = [i for i in range(num_entities) if i not in excluded_tails]
    if len(candidate_tails) < n_negatives:
        neg_tails = candidate_tails
    else:
        neg_tails = rng.sample(candidate_tails, n_negatives)

    # Score positive + negatives
    all_tails = [tail_idx] + neg_tails
    h_tensor = torch.full((len(all_tails),), head_idx, dtype=torch.long, device=device)
    r_tensor = torch.full((len(all_tails),), relation_idx, dtype=torch.long, device=device)
    t_tensor = torch.tensor(all_tails, dtype=torch.long, device=device)
    scores = model.score_triples(h_tensor, r_tensor, t_tensor)

    target_score = scores[0]
    neg_scores = scores[1:]
    greater = float((neg_scores > target_score).sum().item())
    equal = float((neg_scores == target_score).sum().item())

    if rank_mode == "optimistic":
        tail_rank = float(greater + 1.0)
    elif rank_mode == "pessimistic":
        tail_rank = float(greater + 1.0 + equal)
    else:  # realistic
        tail_rank = float(greater + 1.0 + 0.5 * equal)

    return head_rank, tail_rank


def rank_query_full(
    model: torch.nn.Module,
    head_idx: int,
    relation_idx: int,
    tail_idx: int,
    num_entities: int,
    head_filter: Set[int],
    tail_filter: Set[int],
    device: torch.device,
    chunk_size: int,
    rank_mode: str,
    all_entities: torch.Tensor,
    scores_buffer: torch.Tensor,
) -> Tuple[float, float]:
    """Compute head and tail ranks using full exhaustive ranking."""

    # Head prediction: rank head against all entities given (?, r, t)
    for chunk in torch.split(all_entities, chunk_size):
        size = chunk.size(0)
        h_chunk = chunk
        r_chunk = torch.full((size,), relation_idx, dtype=torch.long, device=device)
        t_chunk = torch.full((size,), tail_idx, dtype=torch.long, device=device)
        scores_buffer[chunk] = model.score_triples(h_chunk, r_chunk, t_chunk)

    # Filter: mask other correct heads
    if head_filter:
        mask = [idx for idx in head_filter if idx != head_idx]
        if mask:
            mask_tensor = torch.tensor(mask, dtype=torch.long, device=device)
            scores_buffer[mask_tensor] = float("-inf")

    target_score = scores_buffer[head_idx]
    greater = float((scores_buffer > target_score).sum().item())
    equal = float((scores_buffer == target_score).sum().item())

    if rank_mode == "optimistic":
        head_rank = float(greater + 1.0)
    elif rank_mode == "pessimistic":
        head_rank = float(greater + equal)
    else:  # realistic
        head_rank = float(greater + 1.0 + 0.5 * max(0.0, equal - 1.0))

    # Tail prediction: rank tail against all entities given (h, r, ?)
    for chunk in torch.split(all_entities, chunk_size):
        size = chunk.size(0)
        h_chunk = torch.full((size,), head_idx, dtype=torch.long, device=device)
        r_chunk = torch.full((size,), relation_idx, dtype=torch.long, device=device)
        t_chunk = chunk
        scores_buffer[chunk] = model.score_triples(h_chunk, r_chunk, t_chunk)

    # Filter: mask other correct tails
    if tail_filter:
        mask = [idx for idx in tail_filter if idx != tail_idx]
        if mask:
            mask_tensor = torch.tensor(mask, dtype=torch.long, device=device)
            scores_buffer[mask_tensor] = float("-inf")

    target_score = scores_buffer[tail_idx]
    greater = float((scores_buffer > target_score).sum().item())
    equal = float((scores_buffer == target_score).sum().item())

    if rank_mode == "optimistic":
        tail_rank = float(greater + 1.0)
    elif rank_mode == "pessimistic":
        tail_rank = float(greater + equal)
    else:  # realistic
        tail_rank = float(greater + 1.0 + 0.5 * max(0.0, equal - 1.0))

    return head_rank, tail_rank


def evaluate_split(
    model: torch.nn.Module,
    split_triples: List[Tuple[int, int, int]],
    split_raw: List[TripleExample],
    depth_map: Dict[str, int],
    num_entities: int,
    head_filter: Dict[Tuple[int, int], Set[int]],
    tail_filter: Dict[Tuple[int, int], Set[int]],
    device: torch.device,
    config: EvalConfig,
    split_name: str,
) -> SplitResults:
    """Evaluate a split and return results segmented by provability."""

    provable_rrs = []
    non_provable_rrs = []
    all_rrs = []
    depth_rrs = defaultdict(list)

    provable_hits1, non_provable_hits1, all_hits1 = [], [], []
    provable_hits3, non_provable_hits3, all_hits3 = [], [], []
    provable_hits10, non_provable_hits10, all_hits10 = [], [], []

    eval_mode = "sampled" if config.sampled_eval else "full"
    print(f"\nEvaluating {split_name} split ({len(split_triples)} triples, {eval_mode} ranking)...")
    start_time = time.time()

    # Pre-allocate tensors for full ranking
    all_entities = torch.arange(num_entities, device=device)
    scores_buffer = torch.empty(num_entities, dtype=torch.float32, device=device)

    # RNG for sampled evaluation
    rng = random.Random(config.seed)

    with torch.no_grad():
        for i, ((h_idx, r_idx, t_idx), raw_triple) in enumerate(zip(split_triples, split_raw)):
            if config.verbose and (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(split_triples) - i - 1) / rate if rate > 0 else 0
                print(f"  Progress: {i+1}/{len(split_triples)} ({rate:.1f} triples/s, ETA: {eta:.0f}s)", end="\r")

            # Get query string for depth lookup
            query_str = triple_to_query_str(raw_triple.head, raw_triple.relation, raw_triple.tail)
            depth = depth_map.get(query_str, -1)  # Default to -1 if not in depth file
            is_provable = depth >= 1

            # Get filters for this query
            hf = head_filter.get((r_idx, t_idx), set())
            tf = tail_filter.get((h_idx, r_idx), set())

            # Compute ranks
            if config.sampled_eval:
                head_rank, tail_rank = rank_query_sampled(
                    model, h_idx, r_idx, t_idx,
                    num_entities, hf, tf,
                    device, config.sampled_negatives, config.rank_mode, rng
                )
            else:
                head_rank, tail_rank = rank_query_full(
                    model, h_idx, r_idx, t_idx,
                    num_entities, hf, tf,
                    device, config.chunk_size, config.rank_mode,
                    all_entities, scores_buffer
                )

            # Compute metrics for both head and tail
            for rank in [head_rank, tail_rank]:
                rr = 1.0 / rank
                h1 = 1.0 if rank <= 1 else 0.0
                h3 = 1.0 if rank <= 3 else 0.0
                h10 = 1.0 if rank <= 10 else 0.0

                all_rrs.append(rr)
                all_hits1.append(h1)
                all_hits3.append(h3)
                all_hits10.append(h10)

                if is_provable:
                    provable_rrs.append(rr)
                    provable_hits1.append(h1)
                    provable_hits3.append(h3)
                    provable_hits10.append(h10)
                else:
                    non_provable_rrs.append(rr)
                    non_provable_hits1.append(h1)
                    non_provable_hits3.append(h3)
                    non_provable_hits10.append(h10)

                depth_rrs[depth].append(rr)

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f}s ({len(split_triples)/elapsed:.1f} triples/s)")

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    # Aggregate by depth
    mrr_by_depth = {d: safe_mean(rrs) for d, rrs in depth_rrs.items()}
    count_by_depth = {d: len(rrs) // 2 for d, rrs in depth_rrs.items()}  # Divide by 2 (head+tail)

    return SplitResults(
        split_name=split_name,
        total_queries=len(split_triples),
        provable_queries=len(provable_rrs) // 2,
        non_provable_queries=len(non_provable_rrs) // 2,
        overall_mrr=safe_mean(all_rrs),
        overall_hits1=safe_mean(all_hits1),
        overall_hits3=safe_mean(all_hits3),
        overall_hits10=safe_mean(all_hits10),
        provable_mrr=safe_mean(provable_rrs),
        provable_hits1=safe_mean(provable_hits1),
        provable_hits3=safe_mean(provable_hits3),
        provable_hits10=safe_mean(provable_hits10),
        non_provable_mrr=safe_mean(non_provable_rrs),
        non_provable_hits1=safe_mean(non_provable_hits1),
        non_provable_hits3=safe_mean(non_provable_hits3),
        non_provable_hits10=safe_mean(non_provable_hits10),
        mrr_by_depth=mrr_by_depth,
        count_by_depth=count_by_depth,
    )


def evaluate_dataset(config: EvalConfig) -> Dict[str, SplitResults]:
    """Evaluate a dataset on all specified splits."""
    set_seed(config.seed)

    # Find checkpoint
    run_signature = config.run_signature
    if not run_signature:
        run_signature = find_checkpoint(config.checkpoint_dir, config.dataset)
        if not run_signature:
            print(f"No checkpoint found for {config.dataset}")
            return {}

    print(f"\n{'='*60}")
    print(f"Dataset: {config.dataset}")
    print(f"Checkpoint: {run_signature}")
    print(f"{'='*60}")

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and mappings
    model, entity2id, relation2id, model_config = load_model_and_mappings(
        config.checkpoint_dir, run_signature, device
    )
    num_entities = len(entity2id)
    print(f"Model: {model_config.get('model', 'Unknown')}")
    print(f"Entities: {num_entities}, Relations: {len(relation2id)}")

    # Load training triples for filter construction
    train_path = load_dataset_split(config.data_root, config.dataset, "train.txt")
    train_triples_raw = load_triples(train_path)
    train_triples = []
    for t in train_triples_raw:
        h = entity2id.get(t.head)
        r = relation2id.get(t.relation)
        ta = entity2id.get(t.tail)
        if h is not None and r is not None and ta is not None:
            train_triples.append((h, r, ta))

    results = {}

    for split_name in config.splits:
        # Load depth file
        depth_file = os.path.join(config.data_root, config.dataset, f"{split_name}_depths.txt")
        depth_map = parse_depth_file(depth_file)

        if not depth_map:
            print(f"\nWarning: No depth file found for {split_name} split at {depth_file}")
            print(f"  All queries will be marked as non-provable (depth=-1)")
        else:
            provable_count = sum(1 for d in depth_map.values() if d >= 1)
            print(f"\nDepth file loaded: {len(depth_map)} queries, {provable_count} provable")

        # Load split triples
        split_path = load_dataset_split(config.data_root, config.dataset, f"{split_name}.txt")
        split_triples_raw = load_triples(split_path)
        split_triples = []
        valid_raw = []

        for t in split_triples_raw:
            h = entity2id.get(t.head)
            r = relation2id.get(t.relation)
            ta = entity2id.get(t.tail)
            if h is not None and r is not None and ta is not None:
                split_triples.append((h, r, ta))
                valid_raw.append(t)

        if not split_triples:
            print(f"No valid triples in {split_name} split")
            continue

        # Build filters (for filtered evaluation)
        head_filter, tail_filter = build_filter_maps(train_triples, split_triples)

        # Evaluate
        split_results = evaluate_split(
            model, split_triples, valid_raw, depth_map,
            num_entities, head_filter, tail_filter,
            device, config, split_name
        )

        results[split_name] = split_results

        # Print summary
        print(f"\n{split_name.upper()} Results:")
        print(f"  Total queries: {split_results.total_queries}")
        print(f"  Provable: {split_results.provable_queries}, Non-provable: {split_results.non_provable_queries}")
        print(f"  Overall MRR: {split_results.overall_mrr:.4f}")
        print(f"  Provable MRR: {split_results.provable_mrr:.4f}")
        print(f"  Non-provable MRR: {split_results.non_provable_mrr:.4f}")
        print(f"  MRR by depth: {dict(sorted(split_results.mrr_by_depth.items()))}")

    return results


def generate_markdown_report(
    all_results: Dict[str, Dict[str, SplitResults]],
    output_path: str,
) -> None:
    """Generate a markdown report from evaluation results."""

    lines = [
        "# KGE Performance Report: Provable vs Non-Provable Queries",
        "",
        "This report analyzes KGE (Knowledge Graph Embedding) model performance on queries",
        "segmented by provability. A query is **provable** if the RL agent can derive it",
        "through logical reasoning (depth >= 1), and **non-provable** otherwise (depth = -1).",
        "",
        "Understanding KGE performance on these subsets helps identify the potential for",
        "RL to improve link prediction through hybrid approaches.",
        "",
        "## Summary",
        "",
    ]

    # Summary table
    lines.extend([
        "| Dataset | Split | Total | Provable | Non-Provable | Overall MRR | Provable MRR | Non-Provable MRR | Delta |",
        "|---------|-------|-------|----------|--------------|-------------|--------------|------------------|-------|",
    ])

    for dataset, split_results in all_results.items():
        for split_name, results in split_results.items():
            delta = results.provable_mrr - results.non_provable_mrr
            lines.append(
                f"| {dataset} | {split_name} | {results.total_queries} | "
                f"{results.provable_queries} | {results.non_provable_queries} | "
                f"{results.overall_mrr:.4f} | {results.provable_mrr:.4f} | "
                f"{results.non_provable_mrr:.4f} | {delta:+.4f} |"
            )

    lines.extend(["", "## Detailed Results", ""])

    # Detailed results per dataset
    for dataset, split_results in all_results.items():
        lines.extend([f"### {dataset}", ""])

        for split_name, results in split_results.items():
            lines.extend([
                f"#### {split_name.capitalize()} Split",
                "",
                f"- **Total queries**: {results.total_queries}",
                f"- **Provable queries**: {results.provable_queries} ({100*results.provable_queries/max(1,results.total_queries):.1f}%)",
                f"- **Non-provable queries**: {results.non_provable_queries} ({100*results.non_provable_queries/max(1,results.total_queries):.1f}%)",
                "",
                "**Metrics Comparison:**",
                "",
                "| Metric | Overall | Provable | Non-Provable |",
                "|--------|---------|----------|--------------|",
                f"| MRR | {results.overall_mrr:.4f} | {results.provable_mrr:.4f} | {results.non_provable_mrr:.4f} |",
                f"| Hits@1 | {results.overall_hits1:.4f} | {results.provable_hits1:.4f} | {results.non_provable_hits1:.4f} |",
                f"| Hits@3 | {results.overall_hits3:.4f} | {results.provable_hits3:.4f} | {results.non_provable_hits3:.4f} |",
                f"| Hits@10 | {results.overall_hits10:.4f} | {results.provable_hits10:.4f} | {results.non_provable_hits10:.4f} |",
                "",
            ])

            # MRR by depth
            if results.mrr_by_depth:
                lines.extend([
                    "**MRR by Proof Depth:**",
                    "",
                    "| Depth | Count | MRR |",
                    "|-------|-------|-----|",
                ])
                for depth in sorted(results.mrr_by_depth.keys()):
                    count = results.count_by_depth.get(depth, 0)
                    mrr = results.mrr_by_depth[depth]
                    depth_label = str(depth) if depth >= 0 else "Non-provable"
                    lines.append(f"| {depth_label} | {count} | {mrr:.4f} |")
                lines.append("")

    # Analysis section
    lines.extend([
        "## Analysis",
        "",
        "### Key Observations",
        "",
    ])

    # Compute insights
    insights = []
    for dataset, split_results in all_results.items():
        for split_name, results in split_results.items():
            if results.provable_queries > 0 and results.non_provable_queries > 0:
                delta = results.provable_mrr - results.non_provable_mrr
                if delta > 0.05:
                    insights.append(
                        f"- **{dataset}/{split_name}**: KGE performs {delta:.4f} MRR better on "
                        f"provable queries, suggesting these queries have more learnable patterns."
                    )
                elif delta < -0.05:
                    insights.append(
                        f"- **{dataset}/{split_name}**: KGE performs {-delta:.4f} MRR better on "
                        f"non-provable queries, indicating RL could help with the harder provable cases."
                    )
                else:
                    insights.append(
                        f"- **{dataset}/{split_name}**: KGE performance is similar on both subsets (delta={delta:.4f})."
                    )

    if insights:
        lines.extend(insights)
    else:
        lines.append("- Insufficient data for comparative analysis.")

    lines.extend([
        "",
        "### Implications for RL Hybrid Approach",
        "",
        "The RL agent can prove queries where `depth >= 1`. The potential improvement from RL",
        "depends on:",
        "",
        "1. **Provable query proportion**: Higher proportion means more queries can benefit from RL.",
        "2. **KGE performance gap**: If KGE underperforms on provable queries, RL can complement.",
        "3. **Depth distribution**: Shallow proofs (depth 1-2) are easier to find than deep ones.",
        "",
        "For hybrid scoring, consider:",
        "- Weighting RL proofs based on proof depth (shallower = more confident)",
        "- Using KGE as fallback for non-provable queries",
        "- Adjusting `kge_eval_rl_weight` based on dataset characteristics",
        "",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate KGE on provable vs non-provable queries")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., family, wn18rr)")
    parser.add_argument("--all", action="store_true", help="Run on all datasets with depth files")
    parser.add_argument("--splits", type=str, default="train,valid,test", help="Comma-separated splits")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size for evaluation")
    parser.add_argument("--output", type=str, default="kge_perf.md", help="Output report path")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Custom checkpoint directory")
    parser.add_argument("--run_signature", type=str, default=None, help="Specific checkpoint to use")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    # Determine datasets to evaluate
    if args.all:
        # Datasets with both depth files and checkpoints
        datasets = ["family", "nations", "wn18rr", "umls", "pharmkg_full", "fb15k237"]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Specify --dataset or --all")

    # Base configuration
    base_data_root = str(Path(__file__).resolve().parent / "data")
    base_checkpoint_dir = str(Path(__file__).resolve().parent / "kge_module" / "kge_trainer" / "models")

    if args.checkpoint_dir:
        base_checkpoint_dir = args.checkpoint_dir

    # Evaluate all datasets
    all_results = {}
    for dataset in datasets:
        config = EvalConfig(
            dataset=dataset,
            data_root=base_data_root,
            checkpoint_dir=base_checkpoint_dir,
            run_signature=args.run_signature,
            splits=args.splits.split(","),
            device=args.device,
            chunk_size=args.chunk_size,
            verbose=args.verbose,
        )

        results = evaluate_dataset(config)
        if results:
            all_results[dataset] = results

    # Generate report
    if all_results:
        output_path = os.path.join(
            Path(__file__).resolve().parent.parent,
            args.output
        )
        generate_markdown_report(all_results, output_path)

        # Also save raw results as JSON
        json_path = output_path.replace(".md", ".json")
        json_results = {}
        for dataset, split_results in all_results.items():
            json_results[dataset] = {}
            for split_name, results in split_results.items():
                json_results[dataset][split_name] = {
                    "total_queries": results.total_queries,
                    "provable_queries": results.provable_queries,
                    "non_provable_queries": results.non_provable_queries,
                    "overall_mrr": results.overall_mrr,
                    "overall_hits1": results.overall_hits1,
                    "overall_hits3": results.overall_hits3,
                    "overall_hits10": results.overall_hits10,
                    "provable_mrr": results.provable_mrr,
                    "provable_hits1": results.provable_hits1,
                    "provable_hits3": results.provable_hits3,
                    "provable_hits10": results.provable_hits10,
                    "non_provable_mrr": results.non_provable_mrr,
                    "non_provable_hits1": results.non_provable_hits1,
                    "non_provable_hits3": results.non_provable_hits3,
                    "non_provable_hits10": results.non_provable_hits10,
                    "mrr_by_depth": {str(k): v for k, v in results.mrr_by_depth.items()},
                    "count_by_depth": {str(k): v for k, v in results.count_by_depth.items()},
                }

        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Raw results saved to: {json_path}")
    else:
        print("No results to report.")


if __name__ == "__main__":
    main()

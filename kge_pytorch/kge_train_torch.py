
#!/usr/bin/env python3
"""Train a KGE model (RotatE/ComplEx/TuckER) on project datasets or generic triples.

Enhancements in this version:
- Optional reciprocal relations (add inverse triples) via --use_reciprocal
- Optional self-adversarial negative sampling via --adv_temp (>0 enables)
- Weight decay, gradient clipping
- Linear warmup + cosine LR scheduler (per-step), configurable via --warmup_ratio
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Make repo root importable if running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_utils import load_dataset_split, load_triples, load_triples_with_mappings
from kge_model_torch import build_model


class TripleDataset(Dataset):
    """Simple tensor wrapper around integer triples."""

    def __init__(self, triples: List[Tuple[int, int, int]]):
        self.triples = torch.tensor(triples, dtype=torch.long)

    def __len__(self) -> int:
        return self.triples.size(0)

    def __getitem__(self, idx: int) -> Tensor:
        return self.triples[idx]


def compute_bernoulli_probs(
    triples: List[Tuple[int, int, int]], num_relations: int
) -> torch.Tensor:
    counts_head = [dict() for _ in range(num_relations)]
    counts_tail = [dict() for _ in range(num_relations)]
    for h, r, t in triples:
        counts_head[r][h] = counts_head[r].get(h, 0) + 1
        counts_tail[r][t] = counts_tail[r].get(t, 0) + 1
    tph = torch.zeros(num_relations, dtype=torch.float)
    hpt = torch.zeros(num_relations, dtype=torch.float)
    for ridx in range(num_relations):
        if counts_head[ridx]:
            tph[ridx] = sum(counts_head[ridx].values()) / max(1, len(counts_head[ridx]))
        if counts_tail[ridx]:
            hpt[ridx] = sum(counts_tail[ridx].values()) / max(1, len(counts_tail[ridx]))
    denom = tph + hpt
    probs = torch.where(denom > 0, tph / denom, torch.full_like(denom, 0.5))
    return probs.clamp(0.05, 0.95)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    save_dir: str
    train_path: str | None = None
    dataset: str | None = None
    data_root: str = "./data"
    train_split: str = "train.txt"
    valid_path: str | None = None
    test_path: str | None = None
    valid_split: str = "valid.txt"
    test_split: str = "test.txt"
    model: str = "RotatE"
    dim: int = 1024
    gamma: float = 12.0
    p: int = 1
    relation_dim: int | None = None
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 4096
    neg_ratio: int = 1
    epochs: int = 5
    num_workers: int = 2
    amp: bool = False
    compile: bool = False
    cpu: bool = False
    seed: int = 3
    eval_chunk_size: int = 2048
    eval_limit: int = 0
    report_train_mrr: bool = True
    # New knobs
    use_reciprocal: bool = False
    adv_temp: float = 0.0             # 0 = disabled (use BCE); >0 enables self-adversarial
    weight_decay: float = 1e-6
    grad_clip: float = 2.0
    warmup_ratio: float = 0.1         # fraction of total steps
    scheduler: str = "cosine"         # "cosine" or "none"


@dataclass
class TrainArtifacts:
    entity2id: dict
    relation2id: dict
    config_path: str
    weights_path: str
    metrics: Optional[Dict[str, float]] = None


def resolve_train_path(cfg: TrainConfig) -> str:
    if cfg.train_path:
        return cfg.train_path
    if cfg.dataset:
        return load_dataset_split(cfg.data_root, cfg.dataset, cfg.train_split)
    raise ValueError("Provide either train_path or dataset")


def resolve_split_path(cfg: TrainConfig, split_name: str) -> Optional[str]:
    """Resolve an optional dataset split path for validation or test data."""
    path_value = getattr(cfg, f"{split_name}_path", None)
    if path_value:
        if not os.path.isfile(path_value):
            raise FileNotFoundError(f"Provided {split_name} path '{path_value}' does not exist")
        return path_value

    dataset_name = cfg.dataset
    split_filename = getattr(cfg, f"{split_name}_split", None)
    if dataset_name and split_filename:
        try:
            return load_dataset_split(cfg.data_root, dataset_name, split_filename)
        except FileNotFoundError as err:
            print(f"Warning: {split_name} split not found ({err}); continuing without it.")
            return None
    return None


def encode_split_triples(
    path: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    split_name: str,
) -> List[Tuple[int, int, int]]:
    """Map triples from a split onto training identifiers, ensuring no OOV tokens."""
    raw_triples = load_triples(path)
    encoded: List[Tuple[int, int, int]] = []
    missing_entities: Set[str] = set()
    missing_relations: Set[str] = set()

    for example in raw_triples:
        h = entity2id.get(example.head)
        r = relation2id.get(example.relation)
        t = entity2id.get(example.tail)

        if h is None:
            missing_entities.add(example.head)
        if t is None:
            missing_entities.add(example.tail)
        if r is None:
            missing_relations.add(example.relation)
        if h is None or r is None or t is None:
            continue
        encoded.append((h, r, t))

    if missing_entities or missing_relations:
        details: List[str] = []
        if missing_entities:
            samples = sorted(missing_entities)
            preview = ", ".join(samples[:5])
            suffix = "..." if len(samples) > 5 else ""
            details.append(f"entities {{{preview}{suffix}}}")
        if missing_relations:
            samples = sorted(missing_relations)
            preview = ", ".join(samples[:5])
            suffix = "..." if len(samples) > 5 else ""
            details.append(f"relations {{{preview}{suffix}}}")
        raise ValueError(
            f"{split_name} split contains identifiers unseen in training data: {', '.join(details)}"
        )

    return encoded


def build_filter_maps(
    *triple_collections: Sequence[Tuple[int, int, int]]
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


def _rank_candidates(
    model: torch.nn.Module,
    all_entities: torch.Tensor,
    scores_buffer: torch.Tensor,
    relation: int,
    anchor_entity: int,
    true_entity: int,
    filtered_candidates: Set[int],
    chunk_size: int,
    device: torch.device,
    predict_head: bool,
) -> int:
    """Return the 1-indexed rank of the true entity under filtered evaluation."""
    for i,chunk in enumerate(torch.split(all_entities, chunk_size)):
        size = chunk.size(0)
        if predict_head:
            h_chunk = chunk
            r_chunk = torch.full((size,), relation, dtype=torch.long, device=device)
            t_chunk = torch.full((size,), anchor_entity, dtype=torch.long, device=device)
        else:
            h_chunk = torch.full((size,), anchor_entity, dtype=torch.long, device=device)
            r_chunk = torch.full((size,), relation, dtype=torch.long, device=device)
            t_chunk = chunk
        scores_buffer[chunk] = model.score_triples(h_chunk, r_chunk, t_chunk)

    if filtered_candidates:
        mask = [idx for idx in filtered_candidates if idx != true_entity]
        if mask:
            mask_tensor = torch.tensor(mask, dtype=torch.long, device=device)
            scores_buffer[mask_tensor] = float("-inf")

    target_score = scores_buffer[true_entity]
    if not torch.isfinite(target_score):
        raise RuntimeError("True triple masked during evaluation; check filtering logic.")

    greater = (scores_buffer > target_score).sum().item()
    return int(greater + 1)


def evaluate_mrr(
    model: torch.nn.Module,
    triples: Sequence[Tuple[int, int, int]],
    num_entities: int,
    head_filter: Dict[Tuple[int, int], Set[int]],
    tail_filter: Dict[Tuple[int, int], Set[int]],
    device: torch.device,
    chunk_size: int,
    verbose: bool = False,
) -> float:
    """Compute filtered MRR for a split using exhaustive negative corruption."""
    if not triples:
        return float("nan")

    chunk_size = max(1, min(chunk_size, num_entities))
    training_mode = model.training
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.get_default_dtype()
    model.eval()

    with torch.no_grad():
        all_entities = torch.arange(num_entities, device=device)
        scores_buffer = torch.empty(num_entities, dtype=dtype, device=device)
        total_rr = 0.0
        count = 0
        for i,(h, r, t) in enumerate(triples):
            print(f"Evaluating triple {i+1}/{len(triples)}", end="\r") if verbose else None
            head_rank = _rank_candidates(
                model,
                all_entities,
                scores_buffer,
                relation=r,
                anchor_entity=t,
                true_entity=h,
                filtered_candidates=head_filter.get((r, t), set()),
                chunk_size=chunk_size,
                device=device,
                predict_head=True,
            )
            tail_rank = _rank_candidates(
                model,
                all_entities,
                scores_buffer,
                relation=r,
                anchor_entity=h,
                true_entity=t,
                filtered_candidates=tail_filter.get((h, r), set()),
                chunk_size=chunk_size,
                device=device,
                predict_head=False,
            )
            total_rr += (1.0 / head_rank) + (1.0 / tail_rank)
            count += 2

    if training_mode:
        model.train()

    return total_rr / max(1, count)


def add_reciprocal_triples(
    triples: List[Tuple[int, int, int]],
    relation2id: Dict[str, int],
    inv_suffix: str = "__inv",
) -> Tuple[List[Tuple[int, int, int]], Dict[str, int], int]:
    """Augment training triples with (t, r_inv, h) and expand relation2id accordingly.

    We ensure r_inv id == r + num_relations for consistency with saved config.
    """
    num_relations = len(relation2id)
    # Build inverse mapping: name -> id
    id2rel = {idx: name for name, idx in relation2id.items()}
    # Expand relation2id deterministically
    expanded = dict(relation2id)
    for ridx in range(num_relations):
        inv_name = f"{id2rel[ridx]}{inv_suffix}"
        expanded[inv_name] = ridx + num_relations
    # Create inverse triples with ids offset by +num_relations
    inv = [(t, r + num_relations, h) for (h, r, t) in triples]
    augmented = triples + inv
    return augmented, expanded, num_relations * 2


def train_model(cfg: TrainConfig) -> TrainArtifacts:
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")

    train_file = resolve_train_path(cfg)
    print(f"Loading triples from {train_file} ...")
    triples, e2id, r2id = load_triples_with_mappings(train_file)
    num_entities, num_relations = len(e2id), len(r2id)
    if not triples:
        raise ValueError("No triples found for training")
    print(
        f"#entities={num_entities:,}, #relations={num_relations:,}, #train triples={len(triples):,}"
    )

    # Prepare validation/test splits BEFORE augmentation to keep filtering keys aligned with original relations
    valid_triples: List[Tuple[int, int, int]] = []
    valid_path = resolve_split_path(cfg, "valid")
    if valid_path:
        print(f"Loading validation triples from {valid_path} ...")
        valid_triples = encode_split_triples(valid_path, e2id, r2id, "validation")
        print(f"#valid triples={len(valid_triples):,}")

    test_triples: List[Tuple[int, int, int]] = []
    test_path = resolve_split_path(cfg, "test")
    if test_path:
        print(f"Loading test triples from {test_path} ...")
        test_triples = encode_split_triples(test_path, e2id, r2id, "test")
        print(f"#test triples={len(test_triples):,}")

    # Build filters from original-relation triples
    head_filter, tail_filter = build_filter_maps(triples, valid_triples, test_triples)

    # Optionally add reciprocal training data
    if cfg.use_reciprocal:
        triples, r2id, num_relations = add_reciprocal_triples(triples, r2id, inv_suffix="__inv")
        print(f"Reciprocal relations enabled: new #relations={num_relations:,}, #train triples={len(triples):,}")

    bern_probs = compute_bernoulli_probs(triples, num_relations)

    dataset = TripleDataset(triples)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(
        cfg.model,
        num_entities,
        num_relations,
        dim=cfg.dim,
        gamma=cfg.gamma,
        p_norm=cfg.p,
        relation_dim=cfg.relation_dim,
        dropout=cfg.dropout,
    )
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

    scheduler = None
    if cfg.scheduler != "none":
        total_steps = max(1, cfg.epochs * len(dataloader))
        warmup_steps = int(cfg.warmup_ratio * total_steps)

        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def loss_fn(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        """Either BCE or self-adversarial negative sampling (adv_temp>0)."""
        pos_loss = -torch.nn.functional.logsigmoid(pos_scores).mean()
        neg_scores = neg_scores.view(-1, cfg.neg_ratio)  # [B, N]
        if cfg.adv_temp and cfg.adv_temp > 0:
            with torch.no_grad():
                w = torch.softmax(cfg.adv_temp * neg_scores, dim=1)
            neg_loss = (w * (-torch.nn.functional.logsigmoid(-neg_scores))).sum(dim=1).mean()
        else:
            neg_loss = -torch.nn.functional.logsigmoid(-neg_scores).mean()
        return pos_loss + neg_loss

    @torch.no_grad()
    def sample_negatives(batch: Tensor) -> Tensor:
        rels = batch[:, 1]
        probs = bern_probs.to(rels.device)[rels]
        corrupt_head = torch.bernoulli(probs).bool()
        expanded = batch.repeat_interleave(cfg.neg_ratio, dim=0)
        corrupt_head = corrupt_head.repeat_interleave(cfg.neg_ratio).to(expanded.device)
        rand_entities = torch.randint(0, num_entities, size=(expanded.size(0),), device=expanded.device)
        expanded[corrupt_head, 0] = rand_entities[corrupt_head]
        expanded[~corrupt_head, 2] = rand_entities[~corrupt_head]
        return expanded

    set_seed(cfg.seed)
    model.train()

    global_step = 0
    epoch_durations: List[float] = []
    train_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        running = 0.0
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            negatives = sample_negatives(batch)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg.amp):
                pos_scores, neg_scores = model(batch, negatives)
                loss = loss_fn(pos_scores, neg_scores)

            # Backward + optional grad clip
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if hasattr(model, "project_entity_modulus_"):
                model.project_entity_modulus_()
            running += loss.item()
            global_step += 1
        epoch_time = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_time)
        print(f"Epoch {epoch:03d} | loss={running/len(dataloader):.4f} | time={epoch_time:.2f}s")

    total_train_time = time.perf_counter() - train_start
    if epoch_durations:
        avg_epoch = sum(epoch_durations) / len(epoch_durations)
        print(
            f"Training time | epochs={len(epoch_durations)} | total={total_train_time:.2f}s | avg_per_epoch={avg_epoch:.2f}s"
        )

    # Compute filtered MRR for available splits.
    metrics: Dict[str, float] = {}
    metric_logs: List[str] = []
    eval_limit = cfg.eval_limit if cfg.eval_limit > 0 else None

    def maybe_limit(name: str, triples_list: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        if not triples_list:
            return []
        if eval_limit and len(triples_list) > eval_limit:
            print(f"Evaluation on {name}: limiting to first {eval_limit} of {len(triples_list)} triples")
            return triples_list[:eval_limit]
        return triples_list

    if cfg.report_train_mrr:
        train_eval_triples = maybe_limit("train", triples if not cfg.use_reciprocal else [t for t in triples if t[1] < num_relations // (2 if cfg.use_reciprocal else 1)])
        if train_eval_triples:
            train_mrr = evaluate_mrr(
                model,
                train_eval_triples,
                num_entities,
                head_filter,
                tail_filter,
                device,
                cfg.eval_chunk_size,
                verbose=True,
            )
            if not math.isnan(train_mrr):
                metrics["train_mrr"] = float(train_mrr)
                metric_logs.append(f"train={train_mrr:.4f}")

    if valid_triples:
        valid_eval_triples = maybe_limit("validation", valid_triples)
        if valid_eval_triples:
            print('Computing validation MRR ...')
            valid_mrr = evaluate_mrr(
                model,
                valid_eval_triples,
                num_entities,
                head_filter,
                tail_filter,
                device,
                cfg.eval_chunk_size,
                verbose=True,
            )
            if not math.isnan(valid_mrr):
                metrics["valid_mrr"] = float(valid_mrr)
                metric_logs.append(f"valid={valid_mrr:.4f}")

    if test_triples:
        print('Computing test MRR ...')
        test_mrr = evaluate_mrr(
            model,
            test_triples,
            num_entities,
            head_filter,
            tail_filter,
            device,
            cfg.eval_chunk_size,
            verbose=True,
        )
        if not math.isnan(test_mrr):
            metrics["test_mrr"] = float(test_mrr)
            metric_logs.append(f"test(all_neg)={test_mrr:.4f}")

    if metric_logs:
        print("Evaluation | " + " | ".join(metric_logs))

    os.makedirs(cfg.save_dir, exist_ok=True)
    weights_path = os.path.join(cfg.save_dir, "weights.pth")
    torch.save(model.state_dict(), weights_path)
    config_path = os.path.join(cfg.save_dir, "config.json")
    config_payload = {
        "model": cfg.model,
        "dim": cfg.dim,
        "num_entities": num_entities,
        "num_relations": num_relations,
    }
    model_name = cfg.model.lower()
    if model_name == "rotate":
        config_payload.update({"gamma": cfg.gamma, "p": cfg.p})
    elif model_name == "tucker":
        config_payload.update({
            "entity_dim": cfg.dim,
            "relation_dim": cfg.relation_dim or cfg.dim,
            "dropout": cfg.dropout,
        })
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)
    with open(os.path.join(cfg.save_dir, "entity2id.json"), "w", encoding="utf-8") as handle:
        json.dump(e2id, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(cfg.save_dir, "relation2id.json"), "w", encoding="utf-8") as handle:
        json.dump(r2id, handle, indent=2, ensure_ascii=False)

    print(f"Saved model to {cfg.save_dir}")
    return TrainArtifacts(
        entity2id=e2id,
        relation2id=r2id,
        config_path=config_path,
        weights_path=weights_path,
        metrics=metrics or None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="./kge_pytorch/models", help="Directory to save weights and metadata")
    parser.add_argument("--train", dest="train_path", help="Path to triples file (TSV/CSV/prolog)")
    parser.add_argument("--valid", dest="valid_path", help="Path to validation triples file (TSV/CSV/prolog)")
    parser.add_argument("--test", dest="test_path", help="Path to test triples file (TSV/CSV/prolog)")
    parser.add_argument("--dataset", default='family', help="Dataset name under data_root to load train.txt from")
    parser.add_argument("--data_root", default="./data", help="Root directory containing dataset folders")
    parser.add_argument("--train_split", default="train.txt", help="Filename for the training split when using dataset")
    parser.add_argument("--valid_split", default="valid.txt", help="Filename for the validation split when using dataset")
    parser.add_argument("--test_split", default="test.txt", help="Filename for the test split when using dataset")
    parser.add_argument("--model", type=str, default="RotatE", choices=["RotatE", "ComplEx", "TuckER"], help="KGE model architecture")
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dimension for entities")
    parser.add_argument("--relation_dim", type=int, default=None, help="Relation embedding dimension (TuckER only; defaults to dim)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for embeddings (TuckER)")
    parser.add_argument("--gamma", type=float, default=12.0)
    parser.add_argument("--p", type=int, choices=[1, 2], default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--neg_ratio", type=int, default=1, help="Number of negatives per positive")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use CUDA AMP")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--eval_chunk_size", type=int, default=2048, help="Number of entity candidates scored per chunk during evaluation")
    parser.add_argument("--eval_limit", type=int, default=0, help="Limit triples per split when computing metrics (0 means all)")
    parser.add_argument("--no_train_mrr", action="store_true", help="Do not compute MRR on training data")
    # New arguments
    parser.add_argument("--use_reciprocal", action="store_true", help="Train with reciprocal relations (adds inverse triples)")
    parser.add_argument("--adv_temp", type=float, default=0.0, help="Self-adversarial temperature (0 disables)")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=2.0, help="Gradient clipping max-norm (0 disables)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Linear warmup ratio for LR schedule")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"], help="LR schedule type")
    return parser


def main(argv: List[str] | None = None) -> TrainArtifacts:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = TrainConfig(
        save_dir=args.save_dir,
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        dataset=args.dataset,
        data_root=args.data_root,
        train_split=args.train_split,
        valid_split=args.valid_split,
        test_split=args.test_split,
        model=args.model,
        dim=args.dim,
        gamma=args.gamma,
        p=args.p,
        relation_dim=args.relation_dim,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        neg_ratio=args.neg_ratio,
        epochs=args.epochs,
        num_workers=args.num_workers,
        amp=args.amp,
        compile=args.compile,
        cpu=args.cpu,
        seed=args.seed,
        eval_chunk_size=args.eval_chunk_size,
        eval_limit=args.eval_limit,
        report_train_mrr=not args.no_train_mrr,
        # New fields
        use_reciprocal=args.use_reciprocal,
        adv_temp=args.adv_temp,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        scheduler=args.scheduler,
    )
    return train_model(cfg)


if __name__ == "__main__":
    main()

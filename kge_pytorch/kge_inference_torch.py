#!/usr/bin/env python3
"""Batch-score triples with a trained KGE model."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch

from kge_pytorch.data_utils import detect_triple_format, load_dataset_split, load_triples
from kge_pytorch.kge_model_torch import build_model


@dataclass
class PredictConfig:
    model_dir: str
    output_path: str
    input_path: Optional[str] = None
    dataset: Optional[str] = None
    data_root: str = "./data"
    input_split: str = "valid.txt"
    batch_size: int = 65536
    amp: bool = False
    cpu: bool = False


@dataclass
class PredictArtifacts:
    output_path: str
    num_scored: int
    skipped: int


def resolve_input_path(cfg: PredictConfig) -> str:
    if cfg.input_path:
        return cfg.input_path
    if cfg.dataset:
        return load_dataset_split(cfg.data_root, cfg.dataset, cfg.input_split)
    raise ValueError("Provide either input_path or dataset")


def load_model(model_dir: str, device: torch.device) -> Tuple[torch.nn.Module, dict, dict, dict]:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    with open(os.path.join(model_dir, "entity2id.json"), "r", encoding="utf-8") as handle:
        entity2id = json.load(handle)
    with open(os.path.join(model_dir, "relation2id.json"), "r", encoding="utf-8") as handle:
        relation2id = json.load(handle)

    model_name = cfg.get("model", "RotatE")
    model = build_model(
        model_name,
        cfg["num_entities"],
        cfg["num_relations"],
        dim=cfg.get("dim") or cfg.get("entity_dim"),
        gamma=cfg.get("gamma", 12.0),
        p_norm=cfg.get("p", 1),
        relation_dim=cfg.get("relation_dim"),
        dropout=cfg.get("dropout", 0.0),
    )
    state = torch.load(os.path.join(model_dir, "weights.pth"), map_location="cpu")
    
    # Handle models saved with torch.compile() wrapper (_orig_mod. prefix)
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    
    # Test that the model works with a dummy forward pass
    with torch.no_grad():
        try:
            dummy_h = torch.zeros(1, dtype=torch.long, device=device)
            dummy_r = torch.zeros(1, dtype=torch.long, device=device)
            dummy_t = torch.zeros(1, dtype=torch.long, device=device)
            _ = model.score_triples(dummy_h, dummy_r, dummy_t)
        except Exception as e:
            print(f"Warning: Model test forward pass failed: {e}")
    
    return model, entity2id, relation2id, cfg


def _filter_known(
    triples: Iterable[Tuple[str, str, str]],
    entity2id: dict,
    relation2id: dict,
) -> Tuple[torch.Tensor, List[int]]:
    ids = []
    keep = []
    for idx, (h, r, t) in enumerate(triples):
        if h in entity2id and r in relation2id and t in entity2id:
            ids.append((entity2id[h], relation2id[r], entity2id[t]))
            keep.append(idx)
    if not ids:
        return torch.empty(0, 3, dtype=torch.long), keep
    return torch.tensor(ids, dtype=torch.long), keep


def _write_scores(
    path: str,
    triples: List[Tuple[str, str, str]],
    scores: List[float],
    input_format: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if input_format in {"tsv", "csv"}:
        delimiter = "\t" if input_format == "tsv" else ","
        header = ["head", "relation", "tail", "score"]
        rows = [list(triple) + [f"{score:.6f}"] for triple, score in zip(triples, scores)]
    else:
        delimiter = ","
        header = ["fact", "score"]
        rows = [
            [f"{rel}({head},{tail})", f"{score:.6f}"]
            for (head, rel, tail), score in zip(triples, scores)
        ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(header)
        writer.writerows(rows)


def predict(cfg: PredictConfig) -> PredictArtifacts:
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    input_file = resolve_input_path(cfg)
    triples_raw = load_triples(input_file)
    if not triples_raw:
        raise ValueError("No triples found to score")

    first_line = ""
    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                first_line = line
                break
    input_format = detect_triple_format(first_line)

    model, entity2id, relation2id, _ = load_model(cfg.model_dir, device)
    triples_strings = [(t.head, t.relation, t.tail) for t in triples_raw]
    triples_ids, keep_idx = _filter_known(triples_strings, entity2id, relation2id)
    triples_kept = [triples_strings[i] for i in keep_idx]
    skipped = len(triples_strings) - len(triples_kept)
    if not triples_kept:
        print("[warn] No triples matched the model vocabulary; nothing to score")
        _write_scores(cfg.output_path, [], [], input_format)
        return PredictArtifacts(output_path=cfg.output_path, num_scored=0, skipped=skipped)

    scores = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
        for start in range(0, triples_ids.size(0), cfg.batch_size):
            batch = triples_ids[start : start + cfg.batch_size].to(device)
            batch_scores = model.score_triples(batch[:, 0], batch[:, 1], batch[:, 2]).float().cpu()
            scores.append(batch_scores)
    if scores:
        flat_scores = torch.cat(scores, dim=0).tolist()
    else:
        flat_scores = []

    _write_scores(cfg.output_path, triples_kept, flat_scores, input_format)
    print(f"Wrote {len(flat_scores)} scored triples to {cfg.output_path}")
    if skipped:
        print(f"[warn] Skipped {skipped} triples due to unknown entities/relations")
    return PredictArtifacts(output_path=cfg.output_path, num_scored=len(flat_scores), skipped=skipped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--dataset", help="Dataset name under data_root")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--input_split", default="valid.txt")
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> PredictArtifacts:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = PredictConfig(
        model_dir=args.model_dir,
        output_path=args.output_path,
        input_path=args.input_path,
        dataset=args.dataset,
        data_root=args.data_root,
        input_split=args.input_split,
        batch_size=args.batch_size,
        amp=args.amp,
        cpu=args.cpu,
    )
    return predict(cfg)


if __name__ == "__main__":
    main()

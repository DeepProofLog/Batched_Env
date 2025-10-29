#!/usr/bin/env python3
"""Batch-score triples with a trained PyKEEN model."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import numpy as np
from pykeen.triples import TriplesFactory

# Suppress the harmless PyKEEN IterableDataset warning about batch counts
import warnings
warnings.filterwarnings('ignore', message='.*Length of IterableDataset.*', category=UserWarning)


@dataclass
class PredictConfig:
    model_dir: str
    output_path: str
    input_path: Optional[str] = None
    batch_size: int = 2048
    cpu: bool = False


@dataclass
class PredictArtifacts:
    output_path: str
    num_scored: int
    skipped: int


def load_triples_from_file(path: str) -> List[Tuple[str, str, str]]:
    """Load triples from file. Supports both TSV and Prolog-style formats."""
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check if it's Prolog-style format: relation(entity1,entity2).
            if '(' in line and ')' in line and line.endswith('.'):
                # Parse Prolog format: relation(entity1,entity2).
                try:
                    rel_start = line.index('(')
                    rel_end = line.rindex(')')
                    relation = line[:rel_start].strip()
                    args = line[rel_start+1:rel_end].strip()
                    entities = [e.strip() for e in args.split(',')]
                    if len(entities) == 2:
                        head, tail = entities
                        triples.append((head, relation, tail))
                    else:
                        print(f"Warning: Skipping malformed line: {line}")
                except Exception as e:
                    print(f"Warning: Could not parse line '{line}': {e}")
            else:
                # Try TSV/CSV format
                parts = line.replace('\t', ' ').replace(',', ' ').split()
                if len(parts) >= 3:
                    triples.append((parts[0], parts[1], parts[2]))
                else:
                    print(f"Warning: Skipping line with insufficient columns: {line}")
    
    return triples


def load_model(model_dir: str, device: torch.device):
    """Load a PyKEEN model from directory."""
    # Load model using PyKEEN's model loading
    model_path = os.path.join(model_dir, "trained_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Workaround for torchvision compatibility issues
    # Disable torchvision imports in PyKEEN by mocking the vision module
    import sys
    import types
    
    # Create a dummy torchvision module to prevent import errors
    if 'torchvision' not in sys.modules:
        sys.modules['torchvision'] = types.ModuleType('torchvision')
        sys.modules['torchvision.models'] = types.ModuleType('torchvision.models')
        sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')
    
    # Now import pykeen and load model
    import pykeen
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    # Get entity and relation mappings from the triples factory
    training_path = os.path.join(model_dir, "training_triples")
    if os.path.exists(training_path):
        # Check if it's a directory (newer PyKEEN format)
        if os.path.isdir(training_path):
            # Load from TSV files
            import gzip
            entity_tsv = os.path.join(training_path, "entity_to_id.tsv.gz")
            relation_tsv = os.path.join(training_path, "relation_to_id.tsv.gz")
            
            entity_to_id = {}
            with gzip.open(entity_tsv, 'rt') as f:
                next(f)  # Skip header
                for line in f:
                    idx, entity = line.strip().split('\t')
                    entity_to_id[entity] = int(idx)
            
            relation_to_id = {}
            with gzip.open(relation_tsv, 'rt') as f:
                next(f)  # Skip header
                for line in f:
                    idx, relation = line.strip().split('\t')
                    relation_to_id[relation] = int(idx)
        else:
            # Old format - single file
            training_factory = torch.load(training_path, weights_only=False)
            entity_to_id = training_factory.entity_to_id
            relation_to_id = training_factory.relation_to_id
    else:
        # Try loading from JSON files if available
        entity_path = os.path.join(model_dir, "entity_to_id.json")
        relation_path = os.path.join(model_dir, "relation_to_id.json")
        
        if os.path.exists(entity_path) and os.path.exists(relation_path):
            with open(entity_path, "r") as f:
                entity_to_id = json.load(f)
            with open(relation_path, "r") as f:
                relation_to_id = json.load(f)
        else:
            raise FileNotFoundError("Could not find entity/relation mappings")
    
    return model, entity_to_id, relation_to_id


def _filter_known(
    triples: List[Tuple[str, str, str]],
    entity_to_id: dict,
    relation_to_id: dict,
) -> Tuple[torch.Tensor, List[int]]:
    """Filter triples to only those with known entities/relations."""
    ids = []
    keep = []
    for idx, (h, r, t) in enumerate(triples):
        if h in entity_to_id and r in relation_to_id and t in entity_to_id:
            ids.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
            keep.append(idx)
    if not ids:
        return torch.empty(0, 3, dtype=torch.long), keep
    return torch.tensor(ids, dtype=torch.long), keep


def _write_scores(
    path: str,
    triples: List[Tuple[str, str, str]],
    scores: List[float],
) -> None:
    """Write scores to output file in functional format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for (head, rel, tail), score in zip(triples, scores):
            handle.write(f"{rel}({head},{tail})\t{score:.6f}\n")


def predict(cfg: PredictConfig) -> PredictArtifacts:
    """Run inference on triples and return scores."""
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    
    if not cfg.input_path:
        raise ValueError("input_path is required")
    
    triples_raw = load_triples_from_file(cfg.input_path)
    if not triples_raw:
        raise ValueError("No triples found to score")
    
    model, entity_to_id, relation_to_id = load_model(cfg.model_dir, device)
    triples_ids, keep_idx = _filter_known(triples_raw, entity_to_id, relation_to_id)
    triples_kept = [triples_raw[i] for i in keep_idx]
    skipped = len(triples_raw) - len(triples_kept)
    
    if not triples_kept:
        print("[warn] No triples matched the model vocabulary; nothing to score")
        _write_scores(cfg.output_path, [], [])
        return PredictArtifacts(output_path=cfg.output_path, num_scored=0, skipped=skipped)
    
    scores = []
    with torch.no_grad():
        for start in range(0, triples_ids.size(0), cfg.batch_size):
            batch = triples_ids[start : start + cfg.batch_size].to(device)
            # PyKEEN models expect batch of shape (batch_size, 3) with columns [head, relation, tail]
            batch_scores = model.score_hrt(batch).float().cpu()
            # Ensure scores are 1D
            if batch_scores.dim() > 1:
                batch_scores = batch_scores.squeeze()
            scores.append(batch_scores)
    
    if scores:
        flat_scores = torch.cat(scores, dim=0)
        # Ensure it's 1D and convert to list
        if flat_scores.dim() > 1:
            flat_scores = flat_scores.squeeze()
        flat_scores = flat_scores.tolist()
        # Handle case where single value becomes a scalar
        if not isinstance(flat_scores, list):
            flat_scores = [flat_scores]
    else:
        flat_scores = []
    
    _write_scores(cfg.output_path, triples_kept, flat_scores)
    print(f"Wrote {len(flat_scores)} scored triples to {cfg.output_path}")
    if skipped:
        print(f"[warn] Skipped {skipped} triples due to unknown entities/relations")
    return PredictArtifacts(output_path=cfg.output_path, num_scored=len(flat_scores), skipped=skipped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Directory containing trained PyKEEN model")
    parser.add_argument("--output", dest="output_path", required=True, help="Output file path for scores")
    parser.add_argument("--input", dest="input_path", required=True, help="Input file with triples to score")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for inference")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    return parser


def main(argv: List[str] | None = None) -> PredictArtifacts:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = PredictConfig(
        model_dir=args.model_dir,
        output_path=args.output_path,
        input_path=args.input_path,
        batch_size=args.batch_size,
        cpu=args.cpu,
    )
    return predict(cfg)


if __name__ == "__main__":
    main()

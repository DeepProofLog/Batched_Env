#!/usr/bin/env python3
"""
Top-K scorer for KGE models with vectorized scoring and resume support.
total number of predictions = num_relations * 2 (roles) * num_entities * k
    e.g. for wn18rr  11*2*40943*5 = ~4.5 million predictions
Usage:
    python generate_kge_topk.py --dataset wn18rr --k 4 --max-anchors 75000

KEY OPTIONS:
- --max-anchors N   : Stop after N total anchors (75000 with k=4 gives ~300k scores)
- --anchor-batch N  : Process N anchors per GPU batch (higher = faster, more memory)
- --fresh           : Ignore saved progress and start over

Output format:
    predicate(head,tail) score rank
"""

import argparse
import gc
import json
import os
import sys
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from kge_module import KGEInference, find_latest_run, normalize_backend

# Compute paths relative to script location for portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "..", "kge_trainer", "models")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "files")

# Best KGE checkpoints per dataset (from runner_kge.py)
BEST_CHECKPOINTS = {
    "wn18rr": "torch_wn18rr_RotatE_1024_20260107_125531_s42",
    "family": "torch_family_RotatE_1024_20260107_124531_s42",
    "fb15k237": "torch_fb15k237_TuckER_512_20260111_002222_s42",
    "pharmkg_full": "torch_pharmkg_full_ComplEx_1024_20260111_054518_s42",
    "umls": "torch_umls_ComplEx_1024_20260110_223751_s42",
    "nations": "torch_nations_TuckER_512_20260110_224506_s42",
}


def infer_run_signature(dataset_name: str, override: Optional[str], checkpoint_dir: str, backend: str) -> str:
    """Infer the run signature from checkpoint directory or use override."""
    if override:
        return override
    # Use best checkpoint if available
    if dataset_name in BEST_CHECKPOINTS:
        return BEST_CHECKPOINTS[dataset_name]
    # Fallback to finding latest run
    backend_norm = normalize_backend(backend)
    prefix = f"{'torch' if backend_norm == 'pytorch' else 'pykeen'}_{dataset_name}_"
    run_signature = find_latest_run(checkpoint_dir, prefix=prefix)
    if run_signature is None:
        raise ValueError(f"No checkpoint found for dataset '{dataset_name}' under {checkpoint_dir}.")
    return run_signature


def load_existing_facts(dataset_name: str, data_path: str) -> set:
    """Load facts from train/valid/test for filtering."""
    files = [
        os.path.join(data_path, dataset_name, "train.txt"),
        os.path.join(data_path, dataset_name, "valid.txt"),
        os.path.join(data_path, dataset_name, "test.txt"),
    ]
    facts = set()
    for p in files:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    facts.add(line.rstrip("."))
    print(f"[info] Loaded {len(facts)} existing facts to filter.")
    return facts


def resolve_output_path(path: Optional[str], dataset_name: str, k: int) -> str:
    if path:
        return path
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    return os.path.join(DEFAULT_OUTPUT_DIR, f"kge_top{k}_{dataset_name}_facts.txt")


def display_prediction_estimate(num_entities: int, num_relations: int, k: int, dataset_name: str) -> None:
    """Display expected number of predictions and file size before starting."""
    num_roles = 2  # head and tail
    total_predictions = num_relations * num_roles * num_entities * k

    # Estimate file size (avg ~60 bytes per line)
    avg_line_size = 60
    estimated_bytes = total_predictions * avg_line_size
    estimated_mb = estimated_bytes / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"PREDICTION ESTIMATE for {dataset_name.upper()} (k={k})")
    print(f"{'='*60}")
    print(f"  Entities:      {num_entities:>10,}")
    print(f"  Relations:     {num_relations:>10}")
    print(f"  Roles:         {num_roles:>10} (head, tail)")
    print(f"  Top-k:         {k:>10}")
    print(f"  {'-'*58}")
    print(f"  Total preds:   {total_predictions:>10,} ({num_relations} × {num_roles} × {num_entities:,} × {k})")
    print(f"  File size:     ~{estimated_mb:>9.1f} MB ({estimated_bytes:,} bytes)")
    print(f"{'='*60}\n")


# --------- Resume helpers ----------
def progress_path_for(output_path: str) -> str:
    return output_path + ".progress.json"


def load_progress(ppath: str) -> Optional[dict]:
    if not os.path.exists(ppath):
        return None
    try:
        with open(ppath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_progress(ppath: str, state: dict) -> None:
    tmp = ppath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, ppath)


def start_or_resume(output_path: str, fresh: bool):
    """Returns (file_handle, progress_state or None, resume_active: bool)."""
    ppath = progress_path_for(output_path)
    if fresh:
        if os.path.exists(ppath):
            os.remove(ppath)
        return open(output_path, "w", encoding="utf-8"), None, False

    state = load_progress(ppath)
    if state and not state.get("finished", False) and os.path.exists(output_path):
        fh = open(output_path, "r+", encoding="utf-8")
        safe_off = min(max(0, int(state.get("file_offset", 0) or 0)), fh.seek(0, os.SEEK_END))
        fh.seek(safe_off)
        fh.truncate()
        print(f"[resume] From predicate={state.get('predicate_idx')} role={state.get('role_idx')} anchor={state.get('anchor_idx')}")
        return fh, state, True

    return open(output_path, "w", encoding="utf-8"), None, False


# --------- Batched vectorized scoring ----------
def get_topk_batch(
    model: torch.nn.Module,
    entity2id: dict,
    id2entity: List[str],  # Pre-computed list where id2entity[idx] = entity_name
    anchors: List[str],
    r_idx: int,
    k: int,
    role: str,
    device: torch.device,
    entity_chunk_size: int = 0,
) -> List[List[Tuple[str, float]]]:
    """
    Get top-k for a batch of anchors using vectorized scoring.
    Returns list of [(entity, score), ...] for each anchor.
    """
    num_entities = len(id2entity)

    # Filter valid anchors
    valid_indices = [i for i, a in enumerate(anchors) if a in entity2id]
    if not valid_indices:
        return [[] for _ in anchors]

    valid_anchors = [anchors[i] for i in valid_indices]
    anchor_ids = torch.tensor([entity2id[a] for a in valid_anchors], device=device)
    r_tensor = torch.tensor(r_idx, device=device)

    results = [[] for _ in anchors]

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        # Score all entities for all anchors at once
        # Check if model supports entity_chunk_size parameter
        import inspect
        sig = inspect.signature(model.score_all_tails_batch)
        supports_chunking = 'entity_chunk_size' in sig.parameters

        if role == "head":
            if supports_chunking:
                scores = model.score_all_tails_batch(anchor_ids, r_tensor, entity_chunk_size=entity_chunk_size)
            else:
                scores = model.score_all_tails_batch(anchor_ids, r_tensor)
        else:
            if supports_chunking:
                scores = model.score_all_heads_batch(r_tensor, anchor_ids, entity_chunk_size=entity_chunk_size)
            else:
                scores = model.score_all_heads_batch(r_tensor, anchor_ids)

        scores = torch.sigmoid(scores.float())

        # Vectorized self-prediction exclusion
        batch_indices = torch.arange(len(anchor_ids), device=device)
        scores[batch_indices, anchor_ids] = float('-inf')

        # Get top-k for each anchor
        top_scores, top_indices = torch.topk(scores, min(k, num_entities), dim=1)

        # Move to CPU once for faster iteration
        top_indices_cpu = top_indices.cpu().numpy()
        top_scores_cpu = top_scores.cpu().numpy()

        # Convert to results
        for i, orig_idx in enumerate(valid_indices):
            results[orig_idx] = [
                (id2entity[top_indices_cpu[i, j]], float(top_scores_cpu[i, j]))
                for j in range(top_indices_cpu.shape[1])
            ]

    return results


def main():
    parser = argparse.ArgumentParser(description="KGE Top-K scorer with vectorized scoring.")
    parser.add_argument("--dataset", default="countries_s3", help="Dataset name.")
    parser.add_argument("--data-path", default="./data", help="Base path with dataset folders.")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, help="KGE checkpoints directory.")
    parser.add_argument("--run-signature", default=None, help="Checkpoint run signature (optional).")
    parser.add_argument("--output", default=None, help="Output file.")
    parser.add_argument("--k", type=int, default=5, help="Top-K predictions per anchor.")
    parser.add_argument("--device", default="cuda:all", help="Device for inference.")
    parser.add_argument("--backend", default="torch", choices=["torch", "pykeen"])
    parser.add_argument("--anchor-batch", type=int, default=64, help="Anchors per GPU batch (8-16 for RotatE, higher for DistMult/ComplEx).")
    parser.add_argument("--clear-cache-every", type=int, default=1000, help="Clear GPU cache every N anchors.")
    parser.add_argument("--filter-existing", action="store_true", default=True)
    parser.add_argument("--fresh", action="store_true", help="Ignore saved progress.")
    parser.add_argument("--max-anchors", type=int, default=0, help="Stop after N anchors (0=unlimited).")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (only beneficial for 10k+ anchors due to warmup).")
    parser.add_argument("--entity-chunk", type=int, default=8192, help="Chunk entities for memory efficiency (0=auto, try 8192 for large batches).")
    args = parser.parse_args()

    run_signature = infer_run_signature(args.dataset, args.run_signature, args.checkpoint_dir, args.backend)
    print(f"[info] Using run signature: {run_signature}")

    output_path = resolve_output_path(args.output, args.dataset, args.k)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ppath = progress_path_for(output_path)

    existing_facts = load_existing_facts(args.dataset, args.data_path) if args.filter_existing else set()

    # Initialize engine and load model
    engine = KGEInference(
        dataset_name=args.dataset,
        base_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        run_signature=run_signature,
        backend=args.backend,
        runtime_cache_max_entries=0,
        persist_runtime_scores=False,
    )

    # Trigger model loading
    _ = engine.get_topk_tails("dummy", "dummy", 1)

    model = engine.model
    entity2id = engine.entity2id
    relation2id = engine.relation2id
    device = engine.device

    entities = sorted(entity2id.keys())
    relations = sorted(relation2id.keys())
    # Pre-compute id2entity list for fast lookup (avoid dict creation per batch)
    id2entity = [""] * len(entity2id)
    for name, idx in entity2id.items():
        id2entity[idx] = name
    print(f"[info] Loaded {len(entities)} entities and {len(relations)} relations")

    # Display prediction estimate
    display_prediction_estimate(len(entities), len(relations), args.k, args.dataset)

    # Check if model has batch methods, if not fall back to single
    has_batch_methods = hasattr(model, 'score_all_tails_batch')

    # Optional torch.compile for batch methods
    if args.compile and has_batch_methods and device.type == "cuda":
        print("[info] Compiling batch methods with torch.compile...")
        model.score_all_tails_batch = torch.compile(model.score_all_tails_batch, mode="reduce-overhead")
        model.score_all_heads_batch = torch.compile(model.score_all_heads_batch, mode="reduce-overhead")

    fh, state, resuming = start_or_resume(output_path, fresh=args.fresh)

    pred_start = int(state.get("predicate_idx", 0) or 0) if resuming else 0
    role_start = int(state.get("role_idx", 0) or 0) if resuming else 0
    anchor_start = (int(state.get("anchor_idx", -1) or -1) + 1) if resuming else 0

    written = 0
    total_anchors = 0
    anchors_since_clear = 0
    max_reached = False

    try:
        for p_idx, pred_name in enumerate(relations):
            if max_reached or p_idx < pred_start:
                continue

            print(f"[info] Predicate {p_idx+1}/{len(relations)}: {pred_name}")

            for r_idx, role in enumerate(["head", "tail"]):
                if max_reached:
                    break
                if p_idx == pred_start and r_idx < role_start:
                    continue

                print(f"  role={role}")
                start_idx = anchor_start if (p_idx == pred_start and r_idx == role_start) else 0

                # Process in batches
                for batch_start in range(start_idx, len(entities), args.anchor_batch):
                    if args.max_anchors > 0 and total_anchors >= args.max_anchors:
                        print(f"\n[info] Reached max-anchors limit ({args.max_anchors}).")
                        max_reached = True
                        break

                    batch_end = min(batch_start + args.anchor_batch, len(entities))
                    anchor_batch = entities[batch_start:batch_end]

                    # Limit batch if max_anchors would be exceeded
                    remaining = args.max_anchors - total_anchors if args.max_anchors > 0 else len(anchor_batch)
                    anchor_batch = anchor_batch[:remaining]

                    print(f"    anchors {batch_start}-{batch_start + len(anchor_batch) - 1}/{len(entities)}", end="\r")

                    if has_batch_methods:
                        batch_results = get_topk_batch(
                            model, entity2id, id2entity,
                            anchor_batch, relation2id[pred_name], args.k, role, device,
                            entity_chunk_size=args.entity_chunk
                        )
                    else:
                        # Fallback to single-anchor method
                        batch_results = []
                        for anchor in anchor_batch:
                            if role == "head":
                                topk = engine.get_topk_tails(anchor, pred_name, args.k)
                            else:
                                topk = engine.get_topk_heads(pred_name, anchor, args.k)
                            batch_results.append(topk)

                    # Write results
                    for i, anchor in enumerate(anchor_batch):
                        for rank, (entity, score) in enumerate(batch_results[i], start=1):
                            if role == "head":
                                fact = f"{pred_name}({anchor},{entity})"
                            else:
                                fact = f"{pred_name}({entity},{anchor})"

                            if args.filter_existing and fact in existing_facts:
                                continue
                            fh.write(f"{fact} {score:.6f} {rank}\n")
                            written += 1

                    total_anchors += len(anchor_batch)
                    anchors_since_clear += len(anchor_batch)

                    if anchors_since_clear >= args.clear_cache_every:
                        torch.cuda.empty_cache()
                        gc.collect()
                        anchors_since_clear = 0

                    # Save progress
                    fh.flush()
                    save_progress(ppath, {
                        "dataset": args.dataset, "k": args.k,
                        "predicate_idx": p_idx, "role_idx": r_idx,
                        "anchor_idx": batch_start + len(anchor_batch) - 1,
                        "file_offset": fh.tell(), "finished": False,
                    })

                # Reset resume state after first role
                if resuming and p_idx == pred_start and r_idx == role_start:
                    anchor_start = 0

        # Mark finished
        fh.flush()
        save_progress(ppath, {"finished": True, "file_offset": fh.tell()})

    finally:
        fh.close()

    print(f"\n[done] Wrote {written} predictions to {output_path}")


if __name__ == "__main__":
    main()

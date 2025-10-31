#!/usr/bin/env python3
"""
Top-K scorer for KGE models with simple resume support and simple batching.

You choose ONE batching axis and ONE batch size:
- --batch-mode candidates  : process ONE anchor at a time and split its candidates
                             into chunks of --batch-size (recommended for 16GB VRAM).
- --batch-mode anchors     : process --batch-size anchors at once, scoring ALL candidates
                             for each anchor in a single call (memory heavy).

Resume behavior (simple & robust):
- We write a small progress JSON next to the output file: <output>.progress.json
- After every anchor (or anchor-batch) is written, we save:
    {predicate_idx, role_idx, anchor_idx, file_offset, finished}
- On start, if progress + output exist and finished == False, we:
    - open output in r+, truncate to file_offset (so no duplicates), and
    - continue from the next anchor.
- Use --fresh to ignore progress and overwrite output.

Assumes `KGEInference` exposes:
- .predict_batch(list_of_triples) -> List[float]
- .data_handler.predicates -> iterable of predicate objects with:
    - predicate.name
    - predicate.arity == 2
    - predicate.domains[0].constants (head constants)
    - predicate.domains[1].constants (tail constants)

Output format:
    predicate(head,tail) score rank
"""

import argparse
import gc
import heapq
import json
import os
import sys
import warnings
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Dict

# Suppress TensorFlow warnings (PTX compilation, etc.)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logging (0=all, 1=info, 2=warning, 3=error only)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TF Python logging
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', message='.*PTX.*')
warnings.filterwarnings('ignore', message='.*ptxas.*')

# Allow imports from project root (adjust if needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kge_inference import KGEInference  # your engine

# Optional GPU stacks
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# Default run signatures for datasets
DEFAULT_RUN_SIGNATURES: Dict[str, str] = {
    "family": "kinship_family-backward_0_1-no_reasoner-rotate-True-256-256-4-rules.txt",
    "wn18rr": "wn18rr-backward_0_1-no_reasoner-rotate-True-256-256-1-rules.txt",
    "countries_s3": "countries_s3-backward_0_1-no_reasoner-rotate-True-256-256-128-rules.txt",
}


T = TypeVar("T")


def infer_run_signature(dataset_name: str, override: Optional[str]) -> str:
    """Infer the run signature from dataset name or use override."""
    if override:
        return override
    if dataset_name in DEFAULT_RUN_SIGNATURES:
        return DEFAULT_RUN_SIGNATURES[dataset_name]
    raise ValueError(
        f"No default run signature for dataset '{dataset_name}'. Please provide --run-signature."
    )


# --------- small helpers ----------
def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    batch: List[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_existing_facts(dataset_name: str, data_path: str) -> set:
    """Load facts from train/valid/test for filtering (optional)."""
    files = [
        os.path.join(data_path, dataset_name, "train.txt"),
        os.path.join(data_path, dataset_name, "valid.txt"),
        os.path.join(data_path, dataset_name, "test.txt"),
    ]
    s = set()
    for p in files:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                s.add(line.rstrip("."))
    print(f"[info] Loaded {len(s)} existing facts to filter.")
    return s


def score_triples(
    engine: KGEInference,
    triples: Sequence[Tuple[str, str, str]],
    precision: str = "fp16",
) -> List[float]:
    """Single call to engine with optional mixed precision for PyTorch."""
    if TORCH_AVAILABLE:
        with torch.inference_mode():
            if torch.cuda.is_available():
                if precision == "fp16":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        return engine.predict_batch(triples)
                elif precision == "bf16":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        return engine.predict_batch(triples)
            # fp32 or CPU
            return engine.predict_batch(triples)
    else:
        # TF or other backends; precision handled by backend if configured
        return engine.predict_batch(triples)


def resolve_output_path(path: Optional[str], dataset_name: str, k: int) -> str:
    if path:
        return path
    os.makedirs("./top_k_scores", exist_ok=True)
    return os.path.join("./top_k_scores","files", f"kge_top{k}_{dataset_name}_facts.txt")


def maybe_setup_tf(precision: str):
    """Enable TF memory growth and (optional) mixed precision."""
    if not TF_AVAILABLE:
        return
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[info] TF: enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"[warn] TF memory growth not set: {e}")
    # Mixed precision
    try:
        from tensorflow.keras import mixed_precision
        if precision == "fp16":
            mixed_precision.set_global_policy("mixed_float16")
            print("[info] TF precision: mixed_float16")
        elif precision == "bf16":
            mixed_precision.set_global_policy("mixed_bfloat16")
            print("[info] TF precision: mixed_bfloat16")
        else:
            print("[info] TF precision: fp32")
    except Exception:
        pass


def clear_all_caches():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if TF_AVAILABLE:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
    gc.collect()


# --------- resume helpers ----------
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
    """
    Returns (file_handle, progress_state or None, resume_active: bool).
    If resuming, file is opened r+, truncated to file_offset.
    If fresh, new file is opened in 'w' and progress cleared.
    """
    ppath = progress_path_for(output_path)
    if fresh:
        if os.path.exists(ppath):
            os.remove(ppath)
        fh = open(output_path, "w", encoding="utf-8")
        return fh, None, False

    state = load_progress(ppath)
    if state and not state.get("finished", False) and os.path.exists(output_path):
        fh = open(output_path, "r+", encoding="utf-8")
        # truncate to last safe offset to avoid duplicates
        safe_off = int(state.get("file_offset", 0) or 0)
        fh.seek(0, os.SEEK_END)
        file_size = fh.tell()
        safe_off = min(max(0, safe_off), file_size)
        fh.seek(safe_off)
        fh.truncate()
        fh.seek(safe_off)
        print(f"[resume] From predicate={state.get('predicate_idx')} role_idx={state.get('role_idx')} anchor_idx={state.get('anchor_idx')}")
        return fh, state, True

    # No valid progress: start fresh
    fh = open(output_path, "w", encoding="utf-8")
    return fh, None, False


# --------- core: top-k per one anchor (batch-mode = candidates) ----------
def topk_for_anchor_candidates_mode(
    engine: KGEInference,
    predicate_name: str,
    role: str,  # "head" or "tail"
    anchor: str,
    candidates: Sequence[str],
    k: int,
    candidate_batch_size: int,
    precision: str,
) -> List[Tuple[Tuple[str, str, str], float]]:
    """Process ONE anchor; split candidates into chunks of size candidate_batch_size."""
    heap: List[Tuple[float, Tuple[str, str, str]]] = []

    for cand_batch in batched(candidates, candidate_batch_size):
        if role == "head":
            triples = [(predicate_name, anchor, c) for c in cand_batch if c != anchor]
        else:
            triples = [(predicate_name, c, anchor) for c in cand_batch if c != anchor]

        if not triples:
            continue

        scores = score_triples(engine, triples, precision=precision)

        for t, s in zip(triples, scores):
            if len(heap) < k:
                heapq.heappush(heap, (s, t))
            elif s > heap[0][0]:
                heapq.heapreplace(heap, (s, t))

    heap.sort(key=lambda x: x[0], reverse=True)
    return [(t, s) for s, t in heap]


# --------- core: top-k for a batch of anchors (batch-mode = anchors) ----------
def topk_for_anchor_batch_anchors_mode(
    engine: KGEInference,
    predicate_name: str,
    role: str,  # "head" or "tail"
    anchor_batch: Sequence[str],
    candidates: Sequence[str],
    k: int,
    precision: str,
) -> Dict[str, List[Tuple[Tuple[str, str, str], float]]]:
    """
    Process a BATCH of anchors at once, scoring ALL candidates for each anchor in a single call.

    WARNING: Memory heavy. Total scored triples ~= len(anchor_batch) * len(candidates)
    """
    triples: List[Tuple[str, str, str]] = []
    anchor_to_indices: Dict[str, List[int]] = {a: [] for a in anchor_batch}

    for a in anchor_batch:
        if role == "head":
            for c in candidates:
                if c == a:
                    continue
                anchor_to_indices[a].append(len(triples))
                triples.append((predicate_name, a, c))
        else:
            for c in candidates:
                if c == a:
                    continue
                anchor_to_indices[a].append(len(triples))
                triples.append((predicate_name, c, a))

    if not triples:
        return {a: [] for a in anchor_batch}

    scores = score_triples(engine, triples, precision=precision)

    out: Dict[str, List[Tuple[Tuple[str, str, str], float]]] = {}
    for a in anchor_batch:
        idxs = anchor_to_indices[a]
        heap: List[Tuple[float, Tuple[str, str, str]]] = []
        for idx in idxs:
            t = triples[idx]
            s = scores[idx]
            if len(heap) < k:
                heapq.heappush(heap, (s, t))
            elif s > heap[0][0]:
                heapq.heapreplace(heap, (s, t))
        heap.sort(key=lambda x: x[0], reverse=True)
        out[a] = [(t, s) for s, t in heap]
    return out


# --------- main ----------
def main():
    parser = argparse.ArgumentParser(description="KGE Top-K scorer with simple resume + simple batching.")
    parser.add_argument("--dataset", default="wn18rr", help="Dataset name.")
    parser.add_argument("--data-path", default="./data", help="Base path with dataset folders.")
    parser.add_argument("--checkpoint-dir", default="./../checkpoints/", help="KGE checkpoints directory.")
    parser.add_argument("--run-signature", default=None, help="Checkpoint run signature (optional).")
    parser.add_argument("--output", default=None, help="Output file (defaults to ./top_k_scores/...).")
    parser.add_argument("--k", type=int, default=5, help="Top-K predictions per (predicate, role, anchor).")
    parser.add_argument("--device", default="cuda:0", help="Device for inference: 'cuda:0' or 'cpu'.")
    parser.add_argument("--backend", default="tf", choices=["tf", "torch", "pykeen"], help="Backend used by KGEInference.")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"], help="Inference precision hint.")
    parser.add_argument("--batch-mode", default="anchors", choices=["candidates", "anchors"],
                        help="Batch along 'anchors' (recommended) or 'anchors' (memory heavy).")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="If batch-mode=candidates => candidates per microbatch (per anchor). "
                             "If batch-mode=anchors => number of anchors to score at once.")
    parser.add_argument("--clear-cache-every-constants", type=int, default=100  ,
                        help="Clear GPU caches every N processed anchors.")
    parser.add_argument("--filter-existing", action="store_true", default=True,
                        help="Filter out facts present in train/valid/test.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any saved progress and overwrite output.")
    args = parser.parse_args()

    # Optional allocator tweak for PyTorch
    if TORCH_AVAILABLE:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    if args.backend == "tf":
        maybe_setup_tf(args.precision)
    else:
        print(f"[info] Backend: {args.backend} (precision handled by engine or torch autocast).")

    # Infer run signature if not provided
    run_signature = infer_run_signature(args.dataset, args.run_signature)
    print(f"[info] Using run signature: {run_signature}")

    output_path = resolve_output_path(args.output, args.dataset, args.k)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ppath = progress_path_for(output_path)

    existing_facts = load_existing_facts(args.dataset, args.data_path) if args.filter_existing else set()

    engine = KGEInference(
        dataset_name=args.dataset,
        base_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        run_signature=run_signature,
        backend=args.backend,
        runtime_cache_max_entries=0,
        persist_runtime_scores=False,
    )

    data = engine.data_handler

    # resume or start fresh
    fh, state, resuming = start_or_resume(output_path, fresh=args.fresh)

    # establish resume cursors
    if resuming:
        pred_start = int(state.get("predicate_idx", 0) or 0)
        role_start = int(state.get("role_idx", 0) or 0)
        anchor_start = int(state.get("anchor_idx", -1) or -1) + 1  # next one after the last completed
    else:
        pred_start, role_start, anchor_start = 0, 0, 0

    written = 0
    anchors_since_clear = 0

    try:
        predicates = list(data.predicates)

        for p_idx, predicate in enumerate(predicates):
            if getattr(predicate, "arity", 2) != 2:
                continue
            if p_idx < pred_start:
                continue

            head_consts = list(predicate.domains[0].constants)
            tail_consts = list(predicate.domains[1].constants)
            # deterministic role order: 0=head, 1=tail
            roles = [
                ("head", head_consts, tail_consts),  # role_idx 0
                ("tail", tail_consts, head_consts),  # role_idx 1
            ]

            print(f"[info] Predicate {p_idx+1}: {predicate.name}")

            for r_idx, (role_name, anchors, candidates) in enumerate(roles):
                print(f"# predicate={predicate.name} ({p_idx+1}/{len(predicates)}). role={role_name}.\n")
                if not anchors or not candidates:
                    continue
                if p_idx == pred_start and r_idx < role_start:
                    continue

                start_anchor_idx = anchor_start if (p_idx == pred_start and r_idx == role_start) else 0

                if args.batch_mode == "candidates":
                    # One anchor at a time; split candidates into --batch-size
                    for j, anchor in enumerate(anchors[start_anchor_idx:], start=start_anchor_idx):
                        print(f"Scoring constant {j}/{len(anchors)}: {anchor}", end="\r")
                        topk = topk_for_anchor_candidates_mode(
                            engine=engine,
                            predicate_name=predicate.name,
                            role=role_name,
                            anchor=anchor,
                            candidates=candidates,
                            k=args.k,
                            candidate_batch_size=args.batch_size,
                            precision=args.precision,
                        )
                        for rank, (triple, score) in enumerate(topk, start=1):
                            fact = f"{triple[0]}({triple[1]},{triple[2]})"
                            if args.filter_existing and fact in existing_facts:
                                continue
                            fh.write(f"{fact} {score:.6f} {rank}\n")
                            written += 1

                        anchors_since_clear += 1
                        if anchors_since_clear >= args.clear_cache_every_constants:
                            clear_all_caches()
                            anchors_since_clear = 0

                        # save progress after each anchor
                        fh.flush()
                        current = {
                            "dataset": args.dataset,
                            "k": args.k,
                            "predicate_idx": p_idx,
                            "role_idx": r_idx,
                            "anchor_idx": j,               # last finished
                            "file_offset": fh.tell(),
                            "finished": False,
                        }
                        save_progress(ppath, current)

                        if (j + 1 - start_anchor_idx) % 50 == 0:
                            print(f"  [progress] {predicate.name}/{role_name}: {j+1}/{len(anchors)} anchors...")

                else:
                    # args.batch_mode == "anchors"
                    # Process --batch-size anchors at once; score ALL candidates for each.
                    # Convert anchors slice first (so resume starts at correct place)
                    anchors_slice = anchors[start_anchor_idx:]
                    for batch_idx, anchor_batch in enumerate(batched(anchors_slice, args.batch_size)):
                        print(f"Scoring anchors {start_anchor_idx + batch_idx * args.batch_size}"
                              f" - {start_anchor_idx + batch_idx * args.batch_size + len(anchor_batch) - 1}"
                              f" / {len(anchors)}", end="\r")
                        batch_topk = topk_for_anchor_batch_anchors_mode(
                            engine=engine,
                            predicate_name=predicate.name,
                            role=role_name,
                            anchor_batch=anchor_batch,
                            candidates=candidates,
                            k=args.k,
                            precision=args.precision,
                        )
                        # write all anchors in the batch
                        for anchor in anchor_batch:
                            topk = batch_topk.get(anchor, [])
                            for rank, (triple, score) in enumerate(topk, start=1):
                                fact = f"{triple[0]}({triple[1]},{triple[2]})"
                                if args.filter_existing and fact in existing_facts:
                                    continue
                                fh.write(f"{fact} {score:.6f} {rank}\n")
                                written += 1

                        anchors_since_clear += len(anchor_batch)
                        if anchors_since_clear >= args.clear_cache_every_constants:
                            clear_all_caches()
                            anchors_since_clear = 0

                        # save progress after this batch (anchor_idx points to last finished)
                        last_anchor_idx = start_anchor_idx + (batch_idx + 1) * len(anchor_batch) - 1
                        fh.flush()
                        current = {
                            "dataset": args.dataset,
                            "k": args.k,
                            "predicate_idx": p_idx,
                            "role_idx": r_idx,
                            "anchor_idx": last_anchor_idx,
                            "file_offset": fh.tell(),
                            "finished": False,
                        }
                        save_progress(ppath, current)

                        if (batch_idx + 1) % 10 == 0:
                            done = min(start_anchor_idx + (batch_idx + 1) * len(anchor_batch), len(anchors))
                            # print(f"  [progress] {predicate.name}/{role_name}: {done}/{len(anchors)} anchors...")

                # after finishing this role, reset resume cursors
                if resuming and p_idx == pred_start and r_idx == role_start:
                    resuming = False  # back to normal flow
                    anchor_start = 0

        # mark finished
        fh.flush()
        final = {
            "dataset": args.dataset,
            "k": args.k,
            "predicate_idx": -1,
            "role_idx": -1,
            "anchor_idx": -1,
            "file_offset": fh.tell(),
            "finished": True,
        }
        save_progress(ppath, final)

    finally:
        fh.close()

    print(f"[done] Wrote {written} predictions to {output_path}")


if __name__ == "__main__":
    main()

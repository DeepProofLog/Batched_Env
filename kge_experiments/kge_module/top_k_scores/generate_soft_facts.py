#!/usr/bin/env python3
"""
Optimized soft facts generator for KGE models.

This is an optimized rewrite of generate_kge_topk.py with the following improvements:
- Multi-GPU process parallelism (one worker per GPU)
- Async I/O with background writer thread
- Vectorized fact filtering using tensor/set operations
- Removed inspect.signature from hot path
- Auto-tuned batch sizing based on GPU memory
- Binary output format (parquet) support
- Reduced progress save frequency

Usage:
    python generate_soft_facts.py --dataset wn18rr --k 4 --max-anchors 75000
    python generate_soft_facts.py --dataset fb15k237 --k 5 --output-format parquet

Output format:
    Text: predicate(head,tail) score rank
    Parquet: columns [predicate, head, tail, score, rank]
"""

import argparse
import gc
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from kge_module import KGEInference, find_latest_run, normalize_backend

# Paths relative to script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "..", "kge_trainer", "models")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "files")

# Best checkpoints per dataset
BEST_CHECKPOINTS = {
    "wn18rr": "torch_wn18rr_RotatE_1024_20260107_125531_s42",
    "family": "torch_family_RotatE_1024_20260107_124531_s42",
    "fb15k237": "torch_fb15k237_TuckER_512_20260111_002222_s42",
    "pharmkg_full": "torch_pharmkg_full_ComplEx_1024_20260111_054518_s42",
    "umls": "torch_umls_ComplEx_1024_20260110_223751_s42",
    "nations": "torch_nations_TuckER_512_20260110_224506_s42",
}

# Dataset-specific k overrides (to control output size for large datasets)
DATASET_K_OVERRIDES = {
    "fb15k237": 1,  # Large dataset (14k entities, 237 relations) - k=1 gives ~328 MB
}

# Global maximum k (upper limit for all datasets)
MAX_K = 10

# Maximum target file size in MB (will auto-reduce k if exceeded)
MAX_FILE_SIZE_MB = 512  # 0.5 GB


@dataclass
class SoftFactsConfig:
    """Configuration for soft facts generation."""
    dataset: str
    data_path: str
    checkpoint_dir: str
    run_signature: str
    k: int
    output_path: str
    output_format: str  # "text" or "parquet"
    backend: str
    anchor_batch: int
    entity_chunk: int
    max_anchors: int
    filter_existing: bool
    num_gpus: int
    progress_save_every: int
    use_compile: bool
    use_cuda_graphs: bool
    write_buffer_size: int


def infer_run_signature(dataset_name: str, override: Optional[str],
                        checkpoint_dir: str, backend: str) -> str:
    """Infer run signature from checkpoint directory or use override."""
    if override:
        return override
    if dataset_name in BEST_CHECKPOINTS:
        return BEST_CHECKPOINTS[dataset_name]
    backend_norm = normalize_backend(backend)
    prefix = f"{'torch' if backend_norm == 'pytorch' else 'pykeen'}_{dataset_name}_"
    run_signature = find_latest_run(checkpoint_dir, prefix=prefix)
    if run_signature is None:
        raise ValueError(f"No checkpoint found for dataset '{dataset_name}' under {checkpoint_dir}.")
    return run_signature


def load_existing_facts_as_set(dataset_name: str, data_path: str) -> Set[str]:
    """Load existing facts as a set of 'pred(head,tail)' strings."""
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
    return facts


def load_existing_facts_as_tuples(
    dataset_name: str,
    data_path: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int]
) -> FrozenSet[Tuple[int, int, int]]:
    """Load existing facts as frozen set of (h_idx, r_idx, t_idx) for fast filtering."""
    facts_str = load_existing_facts_as_set(dataset_name, data_path)
    tuples = set()

    for fact in facts_str:
        # Parse "pred(head,tail)" format
        try:
            paren_idx = fact.index("(")
            pred = fact[:paren_idx]
            args = fact[paren_idx+1:-1].split(",")
            if len(args) == 2:
                head, tail = args[0].strip(), args[1].strip()
                if head in entity2id and tail in entity2id and pred in relation2id:
                    tuples.add((entity2id[head], relation2id[pred], entity2id[tail]))
        except (ValueError, IndexError):
            continue

    return frozenset(tuples)


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def auto_tune_batch_size(
    model: torch.nn.Module,
    num_entities: int,
    device: torch.device,
    target_memory_fraction: float = 0.5,
    embedding_dim: int = 1024,
) -> int:
    """Auto-tune anchor batch size based on available GPU memory.

    Conservative estimate for RotatE/ComplEx models which have higher
    memory requirements due to complex-valued embeddings.
    """
    if device.type != "cuda":
        return 32

    try:
        torch.cuda.reset_peak_memory_stats(device)
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory

        # Get free memory (more accurate than total)
        free_memory = torch.cuda.mem_get_info(device)[0]
        available_memory = min(free_memory, total_memory * target_memory_fraction)

        # For RotatE/ComplEx: intermediate tensors are [batch, num_entities, dim]
        # Memory per batch = batch * entities * dim * sizeof(float32) * overhead
        # RotatE uses 2 components (re, im) and has 2x intermediate tensors
        model_name = model.__class__.__name__
        if "RotatE" in model_name:
            # RotatE has high memory: [batch, entities, dim] for re and im
            bytes_per_element = 4  # float32
            # Peak memory: batch * entities * dim * 4 * 2 (re/im) * 2 (diff tensors)
            overhead = 16 * bytes_per_element  # Very conservative for RotatE
        elif "ComplEx" in model_name:
            # ComplEx is more efficient (uses matmul)
            overhead = 8 * 4
        else:
            # TuckER, DistMult, TransE are relatively efficient
            overhead = 4 * 4

        # Calculate max batch size
        memory_per_batch_elem = num_entities * overhead
        max_batch = int(available_memory / memory_per_batch_elem)

        # Additional safety: For very large entity sets, be extra conservative
        if num_entities > 30000:
            max_batch = max_batch // 2

        # Clamp to reasonable range (powers of 2 for efficiency)
        result = max(8, min(max_batch, 256))

        # Round down to nearest power of 2
        power = 1
        while power * 2 <= result:
            power *= 2

        return power

    except Exception:
        # Fallback to conservative default
        return 32 if num_entities > 10000 else 64


class DirectWriter:
    """Simple buffered writer - faster than async for most cases."""

    def __init__(self, output_path: str, buffer_size: int = 100000):
        self.output_path = output_path
        # Large buffer (4MB) for efficient writes
        self.fh = open(output_path, "w", encoding="utf-8", buffering=4*1024*1024)
        self.written_count = 0

    def write_batch(self, chunks: List[str]):
        """Write chunks directly."""
        for chunk in chunks:
            if chunk:
                self.fh.write(chunk)
                self.written_count += chunk.count("\n")

    def flush_and_close(self):
        """Flush and close."""
        self.fh.flush()
        self.fh.close()


class AsyncWriter:
    """Background thread for async file writes (legacy, slower than DirectWriter)."""

    def __init__(self, output_path: str, buffer_size: int = 100000):
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.queue: queue.Queue = queue.Queue(maxsize=10)
        self.fh = open(output_path, "w", encoding="utf-8", buffering=1024*1024)
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.stop_event = threading.Event()
        self.written_count = 0
        self.thread.start()

    def _writer_loop(self):
        """Background writer loop."""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                chunk = self.queue.get(timeout=0.1)
                if chunk is not None:
                    self.fh.write(chunk)
                    self.written_count += chunk.count("\n")
                self.queue.task_done()
            except queue.Empty:
                continue

    def write_batch(self, lines: List[str]):
        """Queue a batch of lines for writing."""
        if lines:
            chunk = "".join(lines)
            self.queue.put(chunk)

    def flush_and_close(self):
        """Wait for all writes and close."""
        self.queue.join()
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        self.fh.flush()
        self.fh.close()

    def get_file_offset(self) -> int:
        """Get current file position (approximate)."""
        return self.fh.tell()


class ParquetWriter:
    """Writer for parquet output format."""

    def __init__(self, output_path: str, buffer_size: int = 100000):
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.predicates: List[str] = []
        self.heads: List[str] = []
        self.tails: List[str] = []
        self.scores: List[float] = []
        self.ranks: List[int] = []
        self.written_count = 0

    def write_batch(self, records: List[Tuple[str, str, str, float, int]]):
        """Add records to buffer."""
        for pred, head, tail, score, rank in records:
            self.predicates.append(pred)
            self.heads.append(head)
            self.tails.append(tail)
            self.scores.append(score)
            self.ranks.append(rank)

        if len(self.predicates) >= self.buffer_size:
            self._flush_to_file()

    def _flush_to_file(self):
        """Flush buffer to parquet file."""
        if not self.predicates:
            return

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            # Fallback to numpy archive
            np.savez_compressed(
                self.output_path.replace(".parquet", ".npz"),
                predicates=np.array(self.predicates, dtype=object),
                heads=np.array(self.heads, dtype=object),
                tails=np.array(self.tails, dtype=object),
                scores=np.array(self.scores, dtype=np.float32),
                ranks=np.array(self.ranks, dtype=np.int32),
            )
            self.written_count += len(self.predicates)
            self._clear_buffer()
            return

        table = pa.table({
            "predicate": self.predicates,
            "head": self.heads,
            "tail": self.tails,
            "score": self.scores,
            "rank": self.ranks,
        })

        if os.path.exists(self.output_path):
            existing = pq.read_table(self.output_path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, self.output_path, compression="snappy")
        self.written_count += len(self.predicates)
        self._clear_buffer()

    def _clear_buffer(self):
        """Clear buffer."""
        self.predicates.clear()
        self.heads.clear()
        self.tails.clear()
        self.scores.clear()
        self.ranks.clear()

    def flush_and_close(self):
        """Flush remaining buffer."""
        self._flush_to_file()


class CUDAGraphScorer:
    """CUDA Graph-accelerated scorer for repeated batch scoring."""

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        num_entities: int,
        k: int,
        device: torch.device,
        supports_chunking: bool,
        entity_chunk_size: int = 0,
    ):
        self.model = model
        self.batch_size = batch_size
        self.num_entities = num_entities
        self.k = k
        self.device = device
        self.supports_chunking = supports_chunking
        self.entity_chunk_size = entity_chunk_size

        # Static input tensors (will be copied into)
        self.static_anchors = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.static_r = torch.zeros(1, dtype=torch.long, device=device)

        # Pre-computed batch indices for masking
        self.batch_idx = torch.arange(batch_size, device=device)

        # Output tensors
        self.static_top_indices = torch.zeros(batch_size, k, dtype=torch.long, device=device)
        self.static_top_scores = torch.zeros(batch_size, k, dtype=torch.float32, device=device)

        # Graphs for head and tail roles
        self.graph_head = None
        self.graph_tail = None
        self._warmup_done = False

    def _score_fn(self, role: str) -> torch.Tensor:
        """Scoring function that avoids in-place ops for CUDA graph compatibility."""
        if role == "head":
            scores = self.model.score_all_tails_batch(self.static_anchors, self.static_r[0])
        else:
            scores = self.model.score_all_heads_batch(self.static_r[0], self.static_anchors)

        scores = torch.sigmoid(scores.float())

        # Create mask for self-predictions (graph-compatible, no in-place ops)
        # Use scatter to create the mask - set self-predictions to very low score
        mask = torch.ones_like(scores)
        mask = mask.scatter(1, self.static_anchors.unsqueeze(1), 0.0)
        # Apply mask: set self-predictions to -1e9 (large negative, not -inf to avoid NaN issues)
        scores = scores * mask + (1 - mask) * (-1e9)

        return scores

    def _capture_graph(self, role: str):
        """Capture CUDA graph for the given role."""
        # Warmup runs (required before capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for _ in range(3):  # Warmup iterations
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    scores = self._score_fn(role)
                    top_scores, top_indices = torch.topk(scores, self.k, dim=1)
                    # Copy to static tensors (this is what we'll capture)
                    self.static_top_scores.copy_(top_scores)
                    self.static_top_indices.copy_(top_indices)

        torch.cuda.current_stream().wait_stream(s)

        # Capture the graph - the copy_ operations will be part of the graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                scores = self._score_fn(role)
                top_scores, top_indices = torch.topk(scores, self.k, dim=1)
                # These copy_ operations ensure outputs go to static tensors
                self.static_top_scores.copy_(top_scores)
                self.static_top_indices.copy_(top_indices)

        return g

    def warmup(self):
        """Warmup and capture graphs for both roles."""
        if self._warmup_done:
            return

        print("[info] Capturing CUDA graphs for scoring...")
        self.graph_head = self._capture_graph("head")
        self.graph_tail = self._capture_graph("tail")
        self._warmup_done = True
        print("[info] CUDA graphs captured successfully")

    def score(
        self,
        anchor_ids: torch.Tensor,
        r_idx: int,
        role: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Score using CUDA graph replay."""
        actual_batch = anchor_ids.shape[0]

        if actual_batch != self.batch_size:
            # Fallback to regular scoring for non-standard batch sizes
            return score_batch_vectorized_impl(
                self.model, anchor_ids, r_idx, self.k, role, self.device,
                self.num_entities, self.entity_chunk_size, self.supports_chunking
            )

        # Copy inputs to static tensors
        self.static_anchors.copy_(anchor_ids)
        self.static_r[0] = r_idx

        # Replay the appropriate graph
        if role == "head":
            self.graph_head.replay()
        else:
            self.graph_tail.replay()

        # Return copies of outputs
        return self.static_top_indices.cpu().numpy(), self.static_top_scores.cpu().numpy()


# ============================================================================
# Standalone Scoring Functions (avoid dependency on model methods)
# ============================================================================

def _score_all_tails_rotate(model, h_indices: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
    """
    RotatE tail scoring using torch.cdist - avoids huge intermediate tensors.
    Score = gamma - ||h * r - t||_p where h*r is complex rotation.
    """
    import math

    h_re = model.ent_re(h_indices)
    h_im = model.ent_im(h_indices)

    phase = torch.remainder(model.rel_phase(r_idx), 2 * math.pi)
    c, s = torch.cos(phase), torch.sin(phase)
    hr_re = h_re * c - h_im * s
    hr_im = h_re * s + h_im * c

    # Concatenate real and imaginary parts: [batch, 2*dim]
    hr_concat = torch.cat([hr_re, hr_im], dim=-1)
    t_concat = torch.cat([model.ent_re.weight, model.ent_im.weight], dim=-1)

    # cdist computes pairwise L1 distances efficiently
    dist = torch.cdist(hr_concat.unsqueeze(0), t_concat.unsqueeze(0), p=float(model.p)).squeeze(0)

    return model.gamma - dist


def _score_all_heads_rotate(model, r_idx: torch.Tensor, t_indices: torch.Tensor) -> torch.Tensor:
    """
    RotatE head scoring using torch.cdist.
    Score = gamma - ||h * r - t||_p for all heads h.
    """
    import math

    t_re = model.ent_re(t_indices)
    t_im = model.ent_im(t_indices)

    phase = torch.remainder(model.rel_phase(r_idx), 2 * math.pi)
    c, s = torch.cos(phase), torch.sin(phase)

    # Concatenate for cdist
    t_concat = torch.cat([t_re, t_im], dim=-1)  # [batch, 2*dim]

    # For all heads, compute h*r
    all_h_re = model.ent_re.weight
    all_h_im = model.ent_im.weight
    hr_re = all_h_re * c - all_h_im * s
    hr_im = all_h_re * s + all_h_im * c
    hr_concat = torch.cat([hr_re, hr_im], dim=-1)  # [num_entities, 2*dim]

    # cdist: [batch, num_entities]
    dist = torch.cdist(t_concat.unsqueeze(0), hr_concat.unsqueeze(0), p=float(model.p)).squeeze(0)

    return model.gamma - dist


def _score_all_tails_complex(model, h_indices: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
    """
    ComplEx tail scoring using efficient matmul.
    Score = Re(<h, r, conj(t)>) = sum of component-wise products.
    """
    h_re = model.ent_re(h_indices)  # [batch, dim]
    h_im = model.ent_im(h_indices)  # [batch, dim]
    r_re = model.rel_re(r_idx)  # [dim]
    r_im = model.rel_im(r_idx)  # [dim]
    all_t_re = model.ent_re.weight  # [num_entities, dim]
    all_t_im = model.ent_im.weight  # [num_entities, dim]

    # Compute hr terms: [batch, dim]
    hr_re_re = h_re * r_re
    hr_im_re = h_im * r_re
    hr_re_im = h_re * r_im
    hr_im_im = h_im * r_im

    # [batch, dim] @ [dim, num_entities] -> [batch, num_entities]
    scores = (hr_re_re @ all_t_re.T + hr_im_re @ all_t_im.T +
              hr_re_im @ all_t_im.T - hr_im_im @ all_t_re.T)
    return scores


def _score_all_heads_complex(model, r_idx: torch.Tensor, t_indices: torch.Tensor) -> torch.Tensor:
    """
    ComplEx head scoring using efficient matmul.
    Score = Re(<h, r, conj(t)>) for all heads h.
    """
    t_re = model.ent_re(t_indices)  # [batch, dim]
    t_im = model.ent_im(t_indices)  # [batch, dim]
    r_re = model.rel_re(r_idx)  # [dim]
    r_im = model.rel_im(r_idx)  # [dim]
    all_h_re = model.ent_re.weight  # [num_entities, dim]
    all_h_im = model.ent_im.weight  # [num_entities, dim]

    # Compute rt terms: [batch, dim]
    rt_re_re = r_re * t_re
    rt_re_im = r_re * t_im
    rt_im_im = r_im * t_im
    rt_im_re = r_im * t_re

    # [batch, dim] @ [dim, num_entities] -> [batch, num_entities]
    scores = (rt_re_re @ all_h_re.T + rt_re_im @ all_h_im.T +
              rt_im_im @ all_h_re.T - rt_im_re @ all_h_im.T)
    return scores


def _score_all_tails_tucker(model, h_indices: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
    """
    TuckER tail scoring using efficient matmul.
    Score = W x1 e_h x2 r x3 e_t (tensor decomposition).
    """
    e_h = model.ent(h_indices)  # [batch, d_e]
    e_r = model.rel(r_idx)  # [d_r]
    all_t = model.ent.weight  # [num_entities, d_e]

    # W x2 r -> [d_e, d_e]
    Wr = torch.tensordot(e_r, model.W, dims=([0], [0]))  # [d_e, d_e]

    # (e_h @ Wr) -> [batch, d_e]
    x = e_h @ Wr  # [batch, d_e]

    # Score all tails: [batch, d_e] @ [d_e, num_entities] -> [batch, num_entities]
    return x @ all_t.T


def _score_all_heads_tucker(model, r_idx: torch.Tensor, t_indices: torch.Tensor) -> torch.Tensor:
    """
    TuckER head scoring using efficient matmul.
    Score = W x1 e_h x2 r x3 e_t for all heads h.
    """
    e_t = model.ent(t_indices)  # [batch, d_e]
    e_r = model.rel(r_idx)  # [d_r]
    all_h = model.ent.weight  # [num_entities, d_e]

    # W x2 r -> [d_e, d_e]
    Wr = torch.tensordot(e_r, model.W, dims=([0], [0]))  # [d_e, d_e]

    # (Wr @ e_t.T) -> [d_e, batch]
    y = Wr @ e_t.T  # [d_e, batch]

    # Score all heads: [num_entities, d_e] @ [d_e, batch] -> [num_entities, batch]
    return (all_h @ y).T  # [batch, num_entities]


def _score_all_tails_distmult(model, h_indices: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
    """
    DistMult tail scoring using efficient matmul.
    Score = <h, r, t> (trilinear product).
    """
    h_emb = model.entity_embeddings(h_indices)  # [batch, dim]
    r_emb = model.relation_embeddings(r_idx)  # [dim]
    hr = h_emb * r_emb.unsqueeze(0)  # [batch, dim]
    all_t = model.entity_embeddings.weight  # [num_entities, dim]
    return hr @ all_t.T  # [batch, num_entities]


def _score_all_heads_distmult(model, r_idx: torch.Tensor, t_indices: torch.Tensor) -> torch.Tensor:
    """
    DistMult head scoring using efficient matmul.
    Score = <h, r, t> for all heads h.
    """
    t_emb = model.entity_embeddings(t_indices)  # [batch, dim]
    r_emb = model.relation_embeddings(r_idx)  # [dim]
    rt = t_emb * r_emb.unsqueeze(0)  # [batch, dim]
    all_h = model.entity_embeddings.weight  # [num_entities, dim]
    return rt @ all_h.T  # [batch, num_entities]


def _score_all_tails_transe(model, h_indices: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
    """
    TransE tail scoring.
    Score = -||h + r - t||_p
    """
    h_emb = model.entity_embeddings(h_indices)  # [batch, dim]
    r_emb = model.relation_embeddings(r_idx)  # [dim]
    hr = h_emb + r_emb.unsqueeze(0)  # [batch, dim]
    all_t = model.entity_embeddings.weight  # [num_entities, dim]

    # -||hr - t||_p: [batch, 1, dim] - [1, num_entities, dim] -> [batch, num_entities, dim]
    diff = hr.unsqueeze(1) - all_t.unsqueeze(0)  # [batch, num_entities, dim]
    dist = torch.norm(diff, p=model.p, dim=-1)  # [batch, num_entities]
    return -dist


def _score_all_heads_transe(model, r_idx: torch.Tensor, t_indices: torch.Tensor) -> torch.Tensor:
    """
    TransE head scoring.
    Score = -||h + r - t||_p for all heads h.
    """
    t_emb = model.entity_embeddings(t_indices)  # [batch, dim]
    r_emb = model.relation_embeddings(r_idx)  # [dim]
    rt = t_emb - r_emb.unsqueeze(0)  # [batch, dim] (h = t - r)
    all_h = model.entity_embeddings.weight  # [num_entities, dim]

    # -||h - rt||_p: [1, num_entities, dim] - [batch, 1, dim] -> [batch, num_entities, dim]
    diff = all_h.unsqueeze(0) - rt.unsqueeze(1)  # [batch, num_entities, dim]
    dist = torch.norm(diff, p=model.p, dim=-1)  # [batch, num_entities]
    return -dist


def get_model_scorer(model_name: str) -> Tuple[callable, callable]:
    """
    Get the appropriate scoring functions for a model type.
    Returns (score_all_tails_fn, score_all_heads_fn).
    """
    scorers = {
        "RotatE": (_score_all_tails_rotate, _score_all_heads_rotate),
        "ComplEx": (_score_all_tails_complex, _score_all_heads_complex),
        "TuckER": (_score_all_tails_tucker, _score_all_heads_tucker),
        "DistMult": (_score_all_tails_distmult, _score_all_heads_distmult),
        "TransE": (_score_all_tails_transe, _score_all_heads_transe),
    }
    if model_name not in scorers:
        raise ValueError(f"Unknown model type: {model_name}. Supported: {list(scorers.keys())}")
    return scorers[model_name]


def score_batch_vectorized_impl(
    model: torch.nn.Module,
    anchor_ids: torch.Tensor,
    r_idx: int,
    k: int,
    role: str,
    device: torch.device,
    num_entities: int,
    entity_chunk_size: int = 0,
    supports_chunking: bool = False,
    use_standalone: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core scoring implementation using standalone scoring functions.

    Uses model-specific scoring functions defined in this file rather than
    relying on model methods, making this module self-contained.

    Args:
        model: The KGE model (RotatE, ComplEx, TuckER, etc.)
        anchor_ids: Batch of anchor entity indices [batch]
        r_idx: Relation index
        k: Number of top predictions to return
        role: "head" (anchors are heads, predict tails) or "tail" (anchors are tails, predict heads)
        device: Torch device
        num_entities: Total number of entities
        entity_chunk_size: Unused (kept for compatibility)
        supports_chunking: Unused (kept for compatibility)
        use_standalone: If True, use standalone scoring functions (default)
    """
    r_tensor = torch.tensor(r_idx, device=device)
    model_name = model.__class__.__name__

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        if use_standalone:
            # Use standalone scoring functions (self-contained, no model method dependency)
            score_tails_fn, score_heads_fn = get_model_scorer(model_name)
            if role == "head":
                scores = score_tails_fn(model, anchor_ids, r_tensor)
            else:
                scores = score_heads_fn(model, r_tensor, anchor_ids)
        else:
            # Fallback to model methods (legacy path)
            if role == "head":
                scores = model.score_all_tails_batch(anchor_ids, r_tensor)
            else:
                scores = model.score_all_heads_batch(r_tensor, anchor_ids)

        scores = torch.sigmoid(scores.float())
        batch_indices = torch.arange(len(anchor_ids), device=device)
        scores[batch_indices, anchor_ids] = float('-inf')
        top_scores, top_indices = torch.topk(scores, min(k, num_entities), dim=1)

    return top_indices.cpu().numpy(), top_scores.cpu().numpy()


def score_batch_vectorized(
    model: torch.nn.Module,
    anchor_ids: torch.Tensor,  # [batch]
    r_idx: int,
    k: int,
    role: str,
    device: torch.device,
    id2entity: List[str],
    entity_chunk_size: int = 0,
    supports_chunking: bool = False,
    cuda_graph_scorer: Optional['CUDAGraphScorer'] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized batch scoring using standalone scoring functions.

    Returns (top_indices, top_scores) as numpy arrays with shape [batch, k].
    Uses self-contained scoring functions for all supported model types
    (RotatE, ComplEx, TuckER, DistMult, TransE).
    """
    num_entities = len(id2entity)

    # Use CUDA graph if available and batch size matches
    if cuda_graph_scorer is not None and anchor_ids.shape[0] == cuda_graph_scorer.batch_size:
        return cuda_graph_scorer.score(anchor_ids, r_idx, role)

    # Use standalone scoring functions (self-contained implementation)
    return score_batch_vectorized_impl(
        model, anchor_ids, r_idx, k, role, device,
        num_entities, use_standalone=True
    )


def filter_and_format_results_fast(
    anchor_names: List[str],
    anchor_indices: np.ndarray,  # [batch] - pre-computed anchor indices
    top_indices: np.ndarray,  # [batch, k]
    top_scores: np.ndarray,   # [batch, k]
    pred_name: str,
    role: str,
    id2entity: List[str],
    existing_facts: Optional[FrozenSet[Tuple[int, int, int]]],
    r_idx: int,
) -> str:
    """
    Optimized filter and format - returns single string for bulk write.
    Uses pre-allocated buffer and minimizes Python object creation.
    """
    batch_size, k = top_indices.shape

    # Pre-allocate list with estimated capacity
    lines = []
    lines_append = lines.append  # Local reference for speed

    # Pre-format the predicate prefix
    pred_prefix = f"{pred_name}("

    if existing_facts:
        # With filtering
        for i in range(batch_size):
            anchor = anchor_names[i]
            anchor_idx = anchor_indices[i]

            for j in range(k):
                entity_idx = top_indices[i, j]

                # Build triple for filtering
                if role == "head":
                    h_idx, t_idx = anchor_idx, entity_idx
                else:
                    h_idx, t_idx = entity_idx, anchor_idx

                if (h_idx, r_idx, t_idx) in existing_facts:
                    continue

                entity = id2entity[entity_idx]
                score = top_scores[i, j]

                if role == "head":
                    lines_append(f"{pred_prefix}{anchor},{entity}) {score:.6f} {j+1}\n")
                else:
                    lines_append(f"{pred_prefix}{entity},{anchor}) {score:.6f} {j+1}\n")
    else:
        # Without filtering - faster path
        for i in range(batch_size):
            anchor = anchor_names[i]

            for j in range(k):
                entity = id2entity[top_indices[i, j]]
                score = top_scores[i, j]

                if role == "head":
                    lines_append(f"{pred_prefix}{anchor},{entity}) {score:.6f} {j+1}\n")
                else:
                    lines_append(f"{pred_prefix}{entity},{anchor}) {score:.6f} {j+1}\n")

    return "".join(lines)


def filter_and_format_results(
    anchor_names: List[str],
    top_indices: np.ndarray,  # [batch, k]
    top_scores: np.ndarray,   # [batch, k]
    pred_name: str,
    role: str,
    id2entity: List[str],
    entity2id: Dict[str, int],
    existing_facts: Optional[FrozenSet[Tuple[int, int, int]]],
    r_idx: int,
) -> List[str]:
    """
    Filter existing facts and format results for text output.
    Returns list of formatted lines.
    """
    lines = []
    batch_size, k = top_indices.shape

    for i in range(batch_size):
        anchor = anchor_names[i]
        anchor_idx = entity2id[anchor]

        for j in range(k):
            entity_idx = int(top_indices[i, j])
            score = float(top_scores[i, j])
            entity = id2entity[entity_idx]
            rank = j + 1

            # Build triple indices for filtering
            if role == "head":
                h_idx, t_idx = anchor_idx, entity_idx
            else:
                h_idx, t_idx = entity_idx, anchor_idx

            # Skip if in existing facts
            if existing_facts and (h_idx, r_idx, t_idx) in existing_facts:
                continue

            # Format output
            if role == "head":
                fact = f"{pred_name}({anchor},{entity})"
            else:
                fact = f"{pred_name}({entity},{anchor})"

            lines.append(f"{fact} {score:.6f} {rank}\n")

    return lines


def filter_and_format_parquet(
    anchor_names: List[str],
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    pred_name: str,
    role: str,
    id2entity: List[str],
    entity2id: Dict[str, int],
    existing_facts: Optional[FrozenSet[Tuple[int, int, int]]],
    r_idx: int,
) -> List[Tuple[str, str, str, float, int]]:
    """Filter and format results for parquet output."""
    records = []
    batch_size, k = top_indices.shape

    for i in range(batch_size):
        anchor = anchor_names[i]
        anchor_idx = entity2id[anchor]

        for j in range(k):
            entity_idx = int(top_indices[i, j])
            score = float(top_scores[i, j])
            entity = id2entity[entity_idx]
            rank = j + 1

            if role == "head":
                h_idx, t_idx = anchor_idx, entity_idx
                head, tail = anchor, entity
            else:
                h_idx, t_idx = entity_idx, anchor_idx
                head, tail = entity, anchor

            if existing_facts and (h_idx, r_idx, t_idx) in existing_facts:
                continue

            records.append((pred_name, head, tail, score, rank))

    return records


def process_work_unit(
    config: SoftFactsConfig,
    work_items: List[Tuple[int, int, str, str]],  # [(p_idx, r_idx, pred_name, role), ...]
    gpu_id: int,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    """
    Worker function for multi-GPU processing.
    Processes assigned (predicate, role) pairs on specified GPU.
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Initialize model on this GPU
    engine = KGEInference(
        dataset_name=config.dataset,
        base_path=config.data_path,
        checkpoint_dir=config.checkpoint_dir,
        run_signature=config.run_signature,
        backend=config.backend,
        runtime_cache_max_entries=0,
        persist_runtime_scores=False,
        device=str(device),
    )

    # Trigger model loading
    _ = engine.get_topk_tails("dummy", "dummy", 1)

    model = engine.model
    entity2id = engine.entity2id
    relation2id = engine.relation2id

    entities = sorted(entity2id.keys())
    id2entity = [""] * len(entity2id)
    for name, idx in entity2id.items():
        id2entity[idx] = name

    # Check model capabilities once (not in hot path)
    has_batch_methods = hasattr(model, 'score_all_tails_batch')
    supports_chunking = False
    if has_batch_methods:
        import inspect
        sig = inspect.signature(model.score_all_tails_batch)
        supports_chunking = 'entity_chunk_size' in sig.parameters

    # Load existing facts for filtering
    existing_facts = None
    if config.filter_existing:
        existing_facts = load_existing_facts_as_tuples(
            config.dataset, config.data_path, entity2id, relation2id
        )

    # Optional torch.compile
    if config.use_compile and has_batch_methods and device.type == "cuda":
        model.score_all_tails_batch = torch.compile(model.score_all_tails_batch, mode="reduce-overhead")
        model.score_all_heads_batch = torch.compile(model.score_all_heads_batch, mode="reduce-overhead")

    # Auto-tune batch size if not specified
    anchor_batch = config.anchor_batch
    if anchor_batch <= 0:
        anchor_batch = auto_tune_batch_size(model, len(entities), device)

    total_anchors = 0
    results_buffer = []

    for p_idx, r_idx, pred_name, role in work_items:
        if config.max_anchors > 0 and total_anchors >= config.max_anchors:
            break

        for batch_start in range(0, len(entities), anchor_batch):
            if config.max_anchors > 0 and total_anchors >= config.max_anchors:
                break

            batch_end = min(batch_start + anchor_batch, len(entities))
            anchor_batch_names = entities[batch_start:batch_end]

            # Limit if max_anchors would be exceeded
            if config.max_anchors > 0:
                remaining = config.max_anchors - total_anchors
                anchor_batch_names = anchor_batch_names[:remaining]

            # Filter valid anchors
            valid_anchors = [a for a in anchor_batch_names if a in entity2id]
            if not valid_anchors:
                continue

            anchor_ids = torch.tensor([entity2id[a] for a in valid_anchors], device=device)

            # Score batch
            if has_batch_methods:
                top_indices, top_scores = score_batch_vectorized(
                    model, anchor_ids, relation2id[pred_name], config.k, role, device,
                    id2entity, config.entity_chunk, supports_chunking
                )

                # Format results
                if config.output_format == "parquet":
                    records = filter_and_format_parquet(
                        valid_anchors, top_indices, top_scores, pred_name, role,
                        id2entity, entity2id, existing_facts, relation2id[pred_name]
                    )
                    results_buffer.extend(records)
                else:
                    lines = filter_and_format_results(
                        valid_anchors, top_indices, top_scores, pred_name, role,
                        id2entity, entity2id, existing_facts, relation2id[pred_name]
                    )
                    results_buffer.extend(lines)
            else:
                # Fallback to single-anchor method
                for anchor in valid_anchors:
                    if role == "head":
                        topk = engine.get_topk_tails(anchor, pred_name, config.k)
                    else:
                        topk = engine.get_topk_heads(pred_name, anchor, config.k)

                    for rank, (entity, score) in enumerate(topk, start=1):
                        if role == "head":
                            h, t = anchor, entity
                        else:
                            h, t = entity, anchor

                        if existing_facts:
                            h_idx = entity2id.get(h)
                            t_idx = entity2id.get(t)
                            r_idx_lookup = relation2id[pred_name]
                            if h_idx and t_idx and (h_idx, r_idx_lookup, t_idx) in existing_facts:
                                continue

                        if config.output_format == "parquet":
                            results_buffer.append((pred_name, h, t, score, rank))
                        else:
                            results_buffer.append(f"{pred_name}({h},{t}) {score:.6f} {rank}\n")

            total_anchors += len(valid_anchors)

            # Send progress update
            progress_queue.put((gpu_id, p_idx, role, batch_start + len(valid_anchors), len(entities)))

            # Periodically clear GPU cache
            if total_anchors % 1000 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    # Send results
    result_queue.put((gpu_id, results_buffer))
    progress_queue.put((gpu_id, -1, "done", total_anchors, 0))


def generate_soft_facts_single_gpu(config: SoftFactsConfig) -> int:
    """Single GPU implementation for comparison/fallback."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[info] Using single GPU: {device}")

    engine = KGEInference(
        dataset_name=config.dataset,
        base_path=config.data_path,
        checkpoint_dir=config.checkpoint_dir,
        run_signature=config.run_signature,
        backend=config.backend,
        runtime_cache_max_entries=0,
        persist_runtime_scores=False,
    )

    # Trigger model loading
    _ = engine.get_topk_tails("dummy", "dummy", 1)

    model = engine.model
    entity2id = engine.entity2id
    relation2id = engine.relation2id

    entities = sorted(entity2id.keys())
    relations = sorted(relation2id.keys())
    id2entity = [""] * len(entity2id)
    for name, idx in entity2id.items():
        id2entity[idx] = name

    # Check model capabilities once
    has_batch_methods = hasattr(model, 'score_all_tails_batch')
    supports_chunking = False
    if has_batch_methods:
        import inspect
        sig = inspect.signature(model.score_all_tails_batch)
        supports_chunking = 'entity_chunk_size' in sig.parameters

    # Load existing facts
    existing_facts = None
    if config.filter_existing:
        existing_facts = load_existing_facts_as_tuples(
            config.dataset, config.data_path, entity2id, relation2id
        )
        print(f"[info] Loaded {len(existing_facts)} existing facts for filtering")

    # Optional torch.compile
    if config.use_compile and has_batch_methods and device.type == "cuda":
        print("[info] Compiling batch methods with torch.compile...")
        model.score_all_tails_batch = torch.compile(model.score_all_tails_batch, mode="reduce-overhead")
        model.score_all_heads_batch = torch.compile(model.score_all_heads_batch, mode="reduce-overhead")

    # Optional CUDA graphs
    cuda_graph_scorer = None
    if config.use_cuda_graphs and has_batch_methods and device.type == "cuda":
        # Auto-tune batch size first if needed
        if config.anchor_batch <= 0:
            anchor_batch = auto_tune_batch_size(model, len(entities), device)
        else:
            anchor_batch = config.anchor_batch

        # CUDA graphs need extra memory for capture, use smaller batch
        # For large entity sets, disable CUDA graphs (memory/correctness issues with RotatE)
        graph_batch_size = anchor_batch
        if len(entities) > 20000:
            print("[info] CUDA graphs disabled for large entity sets (>20k entities)")
        elif len(entities) > 10000:
            graph_batch_size = min(64, anchor_batch)

        if len(entities) <= 20000:
            try:
                cuda_graph_scorer = CUDAGraphScorer(
                    model=model,
                    batch_size=graph_batch_size,
                    num_entities=len(entities),
                    k=config.k,
                    device=device,
                    supports_chunking=supports_chunking,
                    entity_chunk_size=config.entity_chunk,
                )
                cuda_graph_scorer.warmup()
                print(f"[info] CUDA graph batch size: {graph_batch_size}")
            except torch.cuda.OutOfMemoryError:
                print("[warn] CUDA graph capture failed due to OOM, disabling CUDA graphs")
                cuda_graph_scorer = None
                torch.cuda.empty_cache()

    # Auto-tune batch size (use graph batch size if CUDA graphs are enabled)
    if cuda_graph_scorer is not None:
        anchor_batch = cuda_graph_scorer.batch_size
    elif config.anchor_batch <= 0:
        anchor_batch = auto_tune_batch_size(model, len(entities), device)
        print(f"[info] Auto-tuned anchor batch size: {anchor_batch}")
    else:
        anchor_batch = config.anchor_batch

    # Setup writer - use DirectWriter for text (faster than async)
    if config.output_format == "parquet":
        writer = ParquetWriter(config.output_path, buffer_size=config.write_buffer_size)
    else:
        writer = DirectWriter(config.output_path, buffer_size=config.write_buffer_size)

    total_anchors = 0
    start_time = time.time()

    try:
        for p_idx, pred_name in enumerate(relations):
            if config.max_anchors > 0 and total_anchors >= config.max_anchors:
                break

            print(f"[info] Predicate {p_idx+1}/{len(relations)}: {pred_name}")

            for role in ["head", "tail"]:
                if config.max_anchors > 0 and total_anchors >= config.max_anchors:
                    break

                for batch_start in range(0, len(entities), anchor_batch):
                    if config.max_anchors > 0 and total_anchors >= config.max_anchors:
                        break

                    batch_end = min(batch_start + anchor_batch, len(entities))
                    anchor_batch_names = entities[batch_start:batch_end]

                    if config.max_anchors > 0:
                        remaining = config.max_anchors - total_anchors
                        anchor_batch_names = anchor_batch_names[:remaining]

                    valid_anchors = [a for a in anchor_batch_names if a in entity2id]
                    if not valid_anchors:
                        continue

                    anchor_ids = torch.tensor([entity2id[a] for a in valid_anchors], device=device)

                    print(f"  {role}: anchors {batch_start}-{batch_start+len(valid_anchors)-1}/{len(entities)}", end="\r")

                    if has_batch_methods:
                        # Score batch with OOM recovery
                        try:
                            top_indices, top_scores = score_batch_vectorized(
                                model, anchor_ids, relation2id[pred_name], config.k, role, device,
                                id2entity, config.entity_chunk, supports_chunking, cuda_graph_scorer
                            )
                        except torch.cuda.OutOfMemoryError:
                            # If OOM, clear cache and reduce batch size for next iteration
                            print(f"\n[warn] OOM with batch {len(anchor_ids)}, clearing cache and reducing batch size")
                            torch.cuda.empty_cache()
                            gc.collect()

                            # Dynamically reduce global batch size
                            anchor_batch = max(8, anchor_batch // 2)
                            print(f"[warn] New batch size: {anchor_batch}")

                            # Retry with smaller batch (no CUDA graph for irregular size)
                            top_indices, top_scores = score_batch_vectorized(
                                model, anchor_ids[:anchor_batch], relation2id[pred_name], config.k, role, device,
                                id2entity, config.entity_chunk, supports_chunking, None
                            )
                            valid_anchors = valid_anchors[:anchor_batch]

                        if config.output_format == "parquet":
                            records = filter_and_format_parquet(
                                valid_anchors, top_indices, top_scores, pred_name, role,
                                id2entity, entity2id, existing_facts, relation2id[pred_name]
                            )
                            writer.write_batch(records)
                        else:
                            # Use fast formatter with pre-computed anchor indices
                            anchor_idx_arr = np.array([entity2id[a] for a in valid_anchors], dtype=np.int64)
                            chunk = filter_and_format_results_fast(
                                valid_anchors, anchor_idx_arr, top_indices, top_scores, pred_name, role,
                                id2entity, existing_facts, relation2id[pred_name]
                            )
                            writer.write_batch([chunk])  # Pass as single-element list

                    total_anchors += len(valid_anchors)

                    if total_anchors % 1000 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

        print()  # Newline after progress

    finally:
        writer.flush_and_close()

    elapsed = time.time() - start_time
    print(f"[done] Generated soft facts in {elapsed:.2f}s ({total_anchors/elapsed:.1f} anchors/sec)")
    print(f"[done] Output: {config.output_path}")

    return writer.written_count


def generate_soft_facts_multi_gpu(config: SoftFactsConfig) -> int:
    """Multi-GPU implementation using process parallelism."""
    available_gpus = get_available_gpus()
    num_gpus = min(config.num_gpus, len(available_gpus)) if config.num_gpus > 0 else len(available_gpus)

    if num_gpus <= 1:
        print("[info] Only 1 GPU available, falling back to single-GPU mode")
        return generate_soft_facts_single_gpu(config)

    print(f"[info] Using {num_gpus} GPUs: {available_gpus[:num_gpus]}")

    # Load mappings to distribute work
    engine = KGEInference(
        dataset_name=config.dataset,
        base_path=config.data_path,
        checkpoint_dir=config.checkpoint_dir,
        run_signature=config.run_signature,
        backend=config.backend,
        runtime_cache_max_entries=0,
        persist_runtime_scores=False,
    )
    _ = engine.get_topk_tails("dummy", "dummy", 1)

    relations = sorted(engine.relation2id.keys())

    # Create work items: (p_idx, r_idx, pred_name, role)
    work_items = []
    for p_idx, pred_name in enumerate(relations):
        r_idx = engine.relation2id[pred_name]
        work_items.append((p_idx, r_idx, pred_name, "head"))
        work_items.append((p_idx, r_idx, pred_name, "tail"))

    # Distribute work across GPUs (round-robin)
    gpu_work = [[] for _ in range(num_gpus)]
    for i, item in enumerate(work_items):
        gpu_work[i % num_gpus].append(item)

    print(f"[info] Distributed {len(work_items)} work units across {num_gpus} GPUs")

    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    progress_queue = mp.Queue()

    # Launch workers
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=process_work_unit,
            args=(config, gpu_work[gpu_id], available_gpus[gpu_id], result_queue, progress_queue)
        )
        p.start()
        processes.append(p)

    # Monitor progress
    start_time = time.time()
    done_gpus = set()

    while len(done_gpus) < num_gpus:
        try:
            gpu_id, p_idx, role, current, total = progress_queue.get(timeout=1.0)
            if p_idx == -1:  # Done signal
                done_gpus.add(gpu_id)
                print(f"[info] GPU {gpu_id} finished ({current} anchors)")
            else:
                print(f"[gpu{gpu_id}] pred={p_idx} role={role} {current}/{total}", end="\r")
        except queue.Empty:
            continue

    # Collect results
    all_results = []
    for _ in range(num_gpus):
        gpu_id, results = result_queue.get()
        all_results.extend(results)

    # Wait for processes
    for p in processes:
        p.join()

    # Write combined results
    if config.output_format == "parquet":
        writer = ParquetWriter(config.output_path, buffer_size=config.write_buffer_size)
        writer.write_batch(all_results)
    else:
        with open(config.output_path, "w", encoding="utf-8", buffering=1024*1024) as fh:
            fh.writelines(all_results)

    elapsed = time.time() - start_time
    print(f"\n[done] Generated {len(all_results)} soft facts in {elapsed:.2f}s")
    print(f"[done] Output: {config.output_path}")

    return len(all_results)


def resolve_output_path(path: Optional[str], dataset: str, k: int, fmt: str) -> str:
    """Resolve output path with appropriate extension."""
    if path:
        return path
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    ext = ".parquet" if fmt == "parquet" else ".txt"
    return os.path.join(DEFAULT_OUTPUT_DIR, f"soft_top{k}_{dataset}_facts{ext}")


def estimate_soft_facts(
    dataset: str,
    data_path: str,
    checkpoint_dir: str,
    run_signature: str,
    backend: str,
    k: int,
    filter_existing: bool,
    max_anchors: int = 0,
) -> Dict[str, any]:
    """
    Estimate the number of soft facts and file size without running generation.

    Returns dict with: num_entities, num_relations, num_existing_facts,
                       estimated_facts, estimated_file_size_mb, estimated_time_sec
    """
    # Load minimal info
    engine = KGEInference(
        dataset_name=dataset,
        base_path=data_path,
        checkpoint_dir=checkpoint_dir,
        run_signature=run_signature,
        backend=backend,
        runtime_cache_max_entries=0,
        persist_runtime_scores=False,
    )
    # Trigger loading
    _ = engine.get_topk_tails("dummy", "dummy", 1)

    num_entities = len(engine.entity2id)
    num_relations = len(engine.relation2id)

    # Count existing facts
    num_existing = 0
    if filter_existing:
        existing = load_existing_facts_as_set(dataset, data_path)
        num_existing = len(existing)

    # Calculate estimates
    # Total anchors = entities * relations * 2 (head and tail roles)
    total_anchors = num_entities * num_relations * 2
    if max_anchors > 0:
        total_anchors = min(total_anchors, max_anchors * num_relations * 2)

    # Each anchor produces k facts (minus filtered existing ones)
    # Estimate ~5-15% overlap with existing facts depending on dataset density
    raw_facts = total_anchors * k
    if filter_existing and num_existing > 0:
        # Estimate overlap rate based on dataset density
        density = num_existing / (num_entities * num_entities * num_relations)
        overlap_rate = min(0.3, density * k)  # Cap at 30%
        estimated_facts = int(raw_facts * (1 - overlap_rate))
    else:
        estimated_facts = raw_facts

    # Estimate file size
    # Text format: "predicate(entity1,entity2) 0.123456 1\n" ~ 50 bytes avg
    # Parquet format: ~20 bytes per record (compressed)
    avg_line_length = 50  # bytes for text
    estimated_size_text_mb = (estimated_facts * avg_line_length) / (1024 * 1024)
    estimated_size_parquet_mb = (estimated_facts * 20) / (1024 * 1024)

    # Estimate time based on ~4000 anchors/sec throughput
    throughput = 4000  # anchors per second (conservative)
    estimated_time_sec = total_anchors / throughput

    return {
        "dataset": dataset,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_existing_facts": num_existing,
        "total_anchors": total_anchors,
        "k": k,
        "estimated_facts": estimated_facts,
        "estimated_size_text_mb": estimated_size_text_mb,
        "estimated_size_parquet_mb": estimated_size_parquet_mb,
        "estimated_time_sec": estimated_time_sec,
    }


def format_size(size_mb: float) -> str:
    """Format size in human-readable format."""
    if size_mb >= 1024:
        return f"{size_mb/1024:.2f} GB"
    elif size_mb >= 1:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb*1024:.1f} KB"


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds >= 3600:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
    elif seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        return f"{seconds:.1f}s"


def print_estimates(estimates: List[Dict], output_format: str):
    """Print estimation table for multiple datasets."""
    print("\n" + "=" * 90)
    print("SOFT FACTS GENERATION ESTIMATES")
    print("=" * 90)

    # Header
    print(f"{'Dataset':<15} {'Entities':>10} {'Relations':>10} {'Anchors':>12} "
          f"{'Est. Facts':>12} {'Est. Size':>12} {'Est. Time':>10}")
    print("-" * 90)

    total_facts = 0
    total_size = 0
    total_time = 0

    for est in estimates:
        size_mb = est["estimated_size_parquet_mb"] if output_format == "parquet" else est["estimated_size_text_mb"]
        print(f"{est['dataset']:<15} {est['num_entities']:>10,} {est['num_relations']:>10} "
              f"{est['total_anchors']:>12,} {est['estimated_facts']:>12,} "
              f"{format_size(size_mb):>12} {format_time(est['estimated_time_sec']):>10}")
        total_facts += est["estimated_facts"]
        total_size += size_mb
        total_time += est["estimated_time_sec"]

    if len(estimates) > 1:
        print("-" * 90)
        print(f"{'TOTAL':<15} {'':<10} {'':<10} {'':<12} "
              f"{total_facts:>12,} {format_size(total_size):>12} {format_time(total_time):>10}")

    print("=" * 90)
    print(f"Note: Estimates assume ~4,000 anchors/sec throughput. Actual may vary.")
    print(f"      File size for {'parquet' if output_format == 'parquet' else 'text'} format.")
    print()


def get_available_datasets(checkpoint_dir: str) -> List[str]:
    """Get list of datasets with available checkpoints."""
    available = []
    for dataset in BEST_CHECKPOINTS.keys():
        sig = BEST_CHECKPOINTS[dataset]
        checkpoint_path = os.path.join(checkpoint_dir, sig)
        if os.path.exists(checkpoint_path):
            available.append(dataset)
    return available


def get_effective_k(dataset: str, requested_k: int) -> Tuple[int, bool, str]:
    """
    Get effective k for a dataset, considering overrides and global max.

    Returns (effective_k, was_overridden, reason).
    """
    effective_k = requested_k
    was_overridden = False
    reason = ""

    # Apply global maximum first
    if effective_k > MAX_K:
        effective_k = MAX_K
        was_overridden = True
        reason = f"global max k={MAX_K}"

    # Apply dataset-specific override (may further reduce)
    if dataset in DATASET_K_OVERRIDES:
        override_k = DATASET_K_OVERRIDES[dataset]
        if effective_k > override_k:
            effective_k = override_k
            was_overridden = True
            reason = f"dataset limit k={override_k}"

    return effective_k, was_overridden, reason


def check_and_adjust_k_for_size(
    estimated_size_mb: float,
    current_k: int,
    dataset: str,
) -> Tuple[int, float]:
    """
    Check if estimated size exceeds limit and suggest reduced k.

    Returns (suggested_k, estimated_size_with_new_k).
    """
    if estimated_size_mb <= MAX_FILE_SIZE_MB:
        return current_k, estimated_size_mb

    # Calculate what k would give us MAX_FILE_SIZE_MB
    ratio = MAX_FILE_SIZE_MB / estimated_size_mb
    suggested_k = max(1, int(current_k * ratio))
    new_estimated_size = estimated_size_mb * (suggested_k / current_k)

    return suggested_k, new_estimated_size


def main():
    parser = argparse.ArgumentParser(
        description="Optimized soft facts generator for KGE models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings
    python generate_soft_facts.py --dataset wn18rr --k 5

    # Multiple datasets (comma-separated)
    python generate_soft_facts.py --dataset wn18rr,family,nations --k 5

    # All available datasets
    python generate_soft_facts.py --dataset all --k 5

    # Estimate only (show expected facts and file sizes)
    python generate_soft_facts.py --dataset all --k 5 --estimate-only

    # List available datasets
    python generate_soft_facts.py --list-datasets

    # Multi-GPU with parquet output
    python generate_soft_facts.py --dataset fb15k237 --k 4 --num-gpus 4 --output-format parquet

    # Limit anchors for quick test
    python generate_soft_facts.py --dataset wn18rr --k 3 --max-anchors 1000

    # Profile mode
    python generate_soft_facts.py --dataset wn18rr --k 5 --profile
        """
    )

    parser.add_argument("--dataset", default="nations,family",
                        help="Dataset name(s): single name, comma-separated list, or 'all'")
    parser.add_argument("--data-path", default="./data", help="Base path with dataset folders")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, help="KGE checkpoints directory")
    parser.add_argument("--run-signature", default=None, help="Checkpoint run signature (for single dataset)")
    parser.add_argument("--output", default=None, help="Output file path (for single dataset)")
    parser.add_argument("--output-format", default="text", choices=["text", "parquet"], help="Output format")
    parser.add_argument("--k", type=int, default=5, help="Top-K predictions per anchor")
    parser.add_argument("--backend", default="torch", choices=["torch", "pykeen"], help="KGE backend")
    parser.add_argument("--anchor-batch", type=int, default=0, help="Anchors per GPU batch (0=auto-tune)")
    parser.add_argument("--entity-chunk", type=int, default=8192, help="Entity chunk size for memory efficiency")
    parser.add_argument("--max-anchors", type=int, default=0, help="Stop after N anchors per dataset (0=unlimited)")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs (0=all available)")
    parser.add_argument("--no-filter", action="store_true", help="Don't filter existing facts")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--cuda-graphs", action="store_true", help="Use CUDA graphs for faster scoring")
    parser.add_argument("--single-gpu", action="store_true", help="Force single-GPU mode")
    parser.add_argument("--write-buffer", type=int, default=100000, help="Write buffer size")
    parser.add_argument("--progress-every", type=int, default=100, help="Save progress every N batches")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Only show estimates without generating facts")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List available datasets with checkpoints")

    args = parser.parse_args()

    # Handle --list-datasets
    if args.list_datasets:
        available = get_available_datasets(args.checkpoint_dir)
        print("\nAvailable datasets with checkpoints:")
        print("-" * 50)
        for ds in sorted(available):
            sig = BEST_CHECKPOINTS.get(ds, "unknown")
            print(f"  {ds:<20} ({sig})")
        print("-" * 50)
        print(f"Total: {len(available)} datasets")
        print(f"\nUse --dataset <name> or --dataset all to process")
        return

    # Parse dataset list
    if args.dataset.lower() == "all":
        datasets = get_available_datasets(args.checkpoint_dir)
        if not datasets:
            print("[error] No datasets found with checkpoints")
            return
        print(f"[info] Processing all {len(datasets)} available datasets: {', '.join(datasets)}")
    else:
        datasets = [d.strip() for d in args.dataset.split(",")]

    # Handle --estimate-only
    if args.estimate_only:
        estimates = []
        for dataset in datasets:
            try:
                # Apply dataset-specific k override
                effective_k, was_overridden, reason = get_effective_k(dataset, args.k)
                if was_overridden:
                    print(f"[info] {dataset}: k={args.k} -> k={effective_k} ({reason})")

                run_sig = infer_run_signature(dataset, None, args.checkpoint_dir, args.backend)
                est = estimate_soft_facts(
                    dataset=dataset,
                    data_path=args.data_path,
                    checkpoint_dir=args.checkpoint_dir,
                    run_signature=run_sig,
                    backend=args.backend,
                    k=effective_k,
                    filter_existing=not args.no_filter,
                    max_anchors=args.max_anchors,
                )

                # Check if size exceeds limit and warn
                size_mb = est["estimated_size_text_mb"] if args.output_format == "text" else est["estimated_size_parquet_mb"]
                if size_mb > MAX_FILE_SIZE_MB:
                    suggested_k, new_size = check_and_adjust_k_for_size(size_mb, effective_k, dataset)
                    print(f"[warn] {dataset}: estimated size {format_size(size_mb)} exceeds {format_size(MAX_FILE_SIZE_MB)} limit")
                    print(f"       Consider using --k {suggested_k} for ~{format_size(new_size)}")

                estimates.append(est)
            except Exception as e:
                print(f"[warn] Could not estimate for {dataset}: {e}")

        if estimates:
            print_estimates(estimates, args.output_format)
        return

    # Process each dataset
    total_start_time = time.time()
    all_results = []

    for i, dataset in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"DATASET {i+1}/{len(datasets)}: {dataset}")
        print(f"{'='*60}")

        try:
            # Apply dataset-specific k override
            effective_k, was_overridden, reason = get_effective_k(dataset, args.k)
            if was_overridden:
                print(f"[info] k={args.k} -> k={effective_k} ({reason})")

            run_signature = infer_run_signature(dataset, args.run_signature if len(datasets) == 1 else None,
                                                args.checkpoint_dir, args.backend)
            print(f"[info] Using run signature: {run_signature}")

            output_path = resolve_output_path(
                args.output if len(datasets) == 1 else None,
                dataset, effective_k, args.output_format
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            config = SoftFactsConfig(
                dataset=dataset,
                data_path=args.data_path,
                checkpoint_dir=args.checkpoint_dir,
                run_signature=run_signature,
                k=effective_k,
                output_path=output_path,
                output_format=args.output_format,
                backend=args.backend,
                anchor_batch=args.anchor_batch,
                entity_chunk=args.entity_chunk,
                max_anchors=args.max_anchors,
                filter_existing=not args.no_filter,
                num_gpus=args.num_gpus,
                progress_save_every=args.progress_every,
                use_compile=args.compile,
                use_cuda_graphs=args.cuda_graphs,
                write_buffer_size=args.write_buffer,
            )

            if args.profile:
                import cProfile
                import pstats
                profiler = cProfile.Profile()
                profiler.enable()

            start_time = time.time()

            if args.single_gpu:
                written = generate_soft_facts_single_gpu(config)
            else:
                written = generate_soft_facts_multi_gpu(config)

            elapsed = time.time() - start_time

            # Get actual file size
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0

            all_results.append({
                "dataset": dataset,
                "facts_written": written,
                "time_sec": elapsed,
                "output_path": output_path,
                "file_size_mb": file_size_mb,
            })

            if args.profile:
                profiler.disable()
                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative')
                print("\n" + "="*60)
                print("PROFILE RESULTS (top 30 by cumulative time)")
                print("="*60)
                stats.print_stats(30)

                profile_path = output_path.replace(".txt", "_profile.txt").replace(".parquet", "_profile.txt")
                with open(profile_path, "w") as f:
                    stats = pstats.Stats(profiler, stream=f)
                    stats.sort_stats('cumulative')
                    stats.print_stats()
                print(f"[profile] Saved to {profile_path}")

            print(f"\n[done] {dataset}: {written:,} facts in {elapsed:.2f}s -> {output_path}")
            print(f"       File size: {format_size(file_size_mb)}")

        except Exception as e:
            print(f"[error] Failed to process {dataset}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "dataset": dataset,
                "facts_written": 0,
                "time_sec": 0,
                "output_path": "",
                "file_size_mb": 0,
                "error": str(e),
            })

    # Print final summary
    total_elapsed = time.time() - total_start_time

    print(f"\n{'='*90}")
    print(f"FINAL SUMMARY")
    print(f"{'='*90}")
    print(f"{'Dataset':<15} {'Facts':>12} {'Time':>10} {'Size':>12} {'Output':<40}")
    print("-" * 90)

    total_facts = 0
    total_size = 0
    for res in all_results:
        if "error" in res:
            print(f"{res['dataset']:<15} {'ERROR':>12} {'':<10} {'':<12} {res.get('error', '')[:40]}")
        else:
            print(f"{res['dataset']:<15} {res['facts_written']:>12,} "
                  f"{format_time(res['time_sec']):>10} {format_size(res['file_size_mb']):>12} "
                  f"{os.path.basename(res['output_path']):<40}")
            total_facts += res['facts_written']
            total_size += res['file_size_mb']

    if len(all_results) > 1:
        print("-" * 90)
        print(f"{'TOTAL':<15} {total_facts:>12,} {format_time(total_elapsed):>10} "
              f"{format_size(total_size):>12}")

    print(f"{'='*90}")


if __name__ == "__main__":
    main()

# Performance Investigation Findings

## Date: December 15, 2025

## Summary

Investigation into OOM errors and performance issues with `test_eval_perf.py` when using `mode='reduce-overhead'` with `fullgraph=True`.

---

## Issue 1: OOM with 100 queries × 100 corruptions

### Root Cause
When processing 100 queries with 100 corruptions each, the total candidates per mode is:
```
100 × (1 + 100) = 10,100 candidates
```

The original code tried to process all 10,100 candidates in a single batch, causing:
- Tensor allocation of `[10100, 120, 26, 3]` for derived states
- Memory requirement: ~12 GB (exceeds 8 GB GPU)

### Solution Implemented
1. **Chunking in `evaluate_policy()`**: Process queries in chunks of `fixed_batch_size` (default: 512)
2. **Padding to fixed size**: Pad smaller batches to `fixed_batch_size` to avoid CUDA graph recompilation
3. **Memory-aware batch sizing**: `compute_optimal_batch_size()` caps at 512 for best throughput

---

## Issue 2: CUDA Graph Recompilation (~20-30s per unique batch size)

### Root Cause
`torch.compile(mode='reduce-overhead')` uses CUDA graphs which are **size-specific**. Each unique batch size triggers a full recompilation.

### Solution Implemented
1. **Fixed batch size padding**: All batches padded to same `fixed_batch_size`
2. **Warmup with exact size**: Warmup uses the same size that evaluation will use
3. **Store `fixed_batch_size` in evaluator**: Ensures consistency between warmup and eval

### Verification
```python
# After warmup with 512:
Batch size 512 (repeated): 0.27s  # Fast - uses cached graph
Batch size 512 (varied):   1.02s  # Fast - same size, different content
Batch size 110 (no pad):   2.97s  # SLOW - triggers recompilation!
```

---

## Issue 3: Batch Size 1024 Pathologically Slow

### Finding
```
Batch size 100:  0.079s, 0.79 ms/query
Batch size 256:  0.122s, 0.47 ms/query  
Batch size 512:  0.208s, 0.41 ms/query
Batch size 1024: 3.450s, 3.37 ms/query  ← 8x slower per query!
```

### Root Cause
GPU memory bandwidth saturation with large tensors:
- `derived_states: [B, 120, 26, 3]` 
- At B=1024: ~75 MB per tensor, multiple such tensors
- Exceeds L2 cache, causes memory thrashing

### Solution
Cap `compute_optimal_batch_size()` at 512 for optimal throughput.

---

## Issue 4: No Per-Query Early Exit (Major Performance Issue)

### Finding
For 512 varied queries with `max_steps=20`:
```
Step 0:  0/512 done,   time=81ms
Step 1:  294/512 done, time=46ms
...
Step 11: 502/512 done, time=42ms  ← 98% done!
...
Step 19: 512/512 done, time=42ms
Early exit at step 19
```

**Each step takes ~42-45ms regardless of how many queries are done!**

### Root Cause
The compiled `step_with_policy()` processes ALL B queries on every step, even if most are already done. The `state.done` mask only affects:
- Log prob accumulation (zeroed for done)
- Reward accumulation (zeroed for done)

But the expensive operations still run:
1. `_compute_derived_functional()` - unification for all B
2. Policy forward pass - embedding + MLP for all B
3. State transitions - tensor operations for all B

### Impact
- Repeated query (2 steps): 0.16s
- Varied queries (avg 5 steps, max 20): 0.9s
- **5.6x slowdown** due to wasted computation on done queries

---

## Current Performance (After Fixes)

### Configuration: 100 queries × 100 corruptions × 1 mode ('both')

| Metric | Value |
|--------|-------|
| Warmup/Compile | 5.49s |
| Evaluation | 18.19s |
| Total | 23.68s |
| Per original query | 181.9 ms |
| Per candidate | 1.80 ms |

### Breakdown
- 10,100 total candidates (100 queries × 101 candidates × 1 mode)
- 20 chunks of 512 (with padding)
- ~0.9s per chunk × 20 = 18s
- Each chunk runs ~20 steps × ~45ms/step ≈ 0.9s

---

## Future Optimizations

### 1. **Per-Query Early Exit** (High Impact - est. 2-4x speedup)

**Current**: All B queries processed every step
**Proposed**: Skip computation for done queries

Options:
a) **Masked computation**: Zero out done queries before expensive ops
b) **Dynamic batching**: Compact active queries each step (breaks CUDA graphs)
c) **Warp-level masking**: Custom CUDA kernel that skips done warps

Implementation sketch for (a):
```python
# In step_functional, before unification:
active_mask = ~state.done
if not active_mask.all():
    # Only compute derived for active queries
    active_current = current_states[active_mask]
    active_derived, active_counts, active_var = self._compute_derived_functional(...)
    # Scatter back
    derived[active_mask] = active_derived
```

**Challenge**: CUDA graphs require static shapes, so (b) won't work with `reduce-overhead`.

### 2. **Smaller Default Batch Size** (Medium Impact)

Current 512 may be too large for some workloads. Consider:
- 256 for small evaluations (faster warmup)
- 512 for large evaluations (better throughput)

### 3. **Compile Caching** (UX Improvement)

PyTorch's `torch.compile` caches are per-process. Consider:
- Using `torch.compile(..., mode='reduce-overhead')` with explicit cache directory
- Pre-warming common batch sizes at startup

### 4. **Reduce max_depth for Evaluation**

SHOULD NOT DO THIS; CHANGES THE SEMANTICS

### 5. **Batch Corruption Generation**

Currently corruptions are generated per-chunk. Could pre-generate all corruptions as one tensor operation. 
(No, potentially there are millions of negatives)

### 6. **Profile Unification** (Investigation)

The `_compute_derived_functional()` may have optimization opportunities:
- Fused kernels for fact/rule matching
- Better memory layout for coalesced access
- Reduced padding overhead

---

## Code Changes Made

### `model_eval_optimized.py`
1. Simplified to single `_evaluate_policy_batch()` function
2. `evaluate_policy()` handles chunking and padding
3. Merged `eval_corruptions_optimized` into `evaluate_with_corruptions`
4. `compute_optimal_batch_size()` caps at 512

### `tests/test_eval_perf.py`
1. Warmup uses exact `fixed_batch_size`
2. Evaluation uses same `fixed_batch_size` from warmup
3. Stores `fixed_batch_size` in evaluator for consistency

---

## Recommendations

1. **Use `mode='reduce-overhead'`** for production (faster after warmup)
2. **Use `mode='default'`** for debugging (no CUDA graph restrictions)
3. **Always warmup before timing** evaluation
4. **Use `fixed_batch_size=512`** for 8GB GPUs
5. **Consider `max_depth=10`** for faster evaluation if accuracy permits

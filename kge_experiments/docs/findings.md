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

### Solution Implemented
Applied step-level masking in `step_functional()` - all computation is masked with `active = ~state.done`:
- State transitions: `torch.where(active, next_states, current_states)`
- Depth updates: `torch.where(active, depths + 1, depths)`  
- Termination checks: `newly_done = active & (is_true | is_false | ...)`
- Rewards: `torch.where(active & is_true, 1.0, 0.0)`
- History updates: Only for active envs
- Derived state computation: Uses `active_mask` to skip unification for done envs

**Key design**: All masking uses `torch.where` (no branching) for CUDA graph compatibility with `fullgraph=True`.

```python
# In step_functional:
was_done = state.done
active = ~was_done  # [n] - envs that need computation

# All operations masked with active
new_current = torch.where(active.view(n,1,1), next_states, state.current_states)
new_depths = torch.where(active, state.depths + 1, state.depths)
newly_done = active & (is_true | is_false | is_end | is_depth_limit)
rewards = torch.where(active & is_true, ones, zeros)

# Derived states - skip unification for done envs
still_active = ~new_done
new_derived, ... = self._compute_derived_functional(..., active_mask=still_active)
new_derived = torch.where(still_active.view(n,1,1,1), new_derived, state.derived_states)
```

**Note**: The policy forward pass in `_step_with_policy_impl` still runs on all B queries.
Full optimization would require policy-level masking or custom CUDA kernels.

---

## Current Performance (After Fixes)

### Configuration: 100 queries × 50 corruptions × 1 mode

| Metric | Value |
|--------|-------|
| Warmup/Compile | 5.15s |
| Evaluation | 2.10s |
| Total | 7.25s |
| Per original query | 21.0 ms |
| Per candidate | 0.41 ms |

### Breakdown
- 5,100 total candidates (100 queries × 51 candidates × 1 mode)
- 10 chunks of 512 (with padding)
- ~0.21s per chunk
- Significant speedup from Issue 4 optimization

---

## Future Optimizations

### 1. **Policy-Level Masking** (Medium Impact - additional optimization)

Current step-level masking skips:
- ✅ State transitions
- ✅ Termination checks  
- ✅ History updates
- ✅ Derived state computation (unification)

Not yet masked:
- ❌ Policy forward pass (embedding lookup + MLP) in `_step_with_policy_impl`

Options for policy masking:
a) **Zero out logits for done queries**: Simple but still computes embeddings
b) **Custom masked embedding**: Skip lookup for done queries
c) **Warp-level masking**: Custom CUDA kernel that skips done warps

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

### API Refactoring (Latest)

**`ppo_optimized.py`** - Now the central location for evaluation:
1. `compute_metrics_from_ranks()` - MRR/Hits@K computation (moved from model_eval_optimized)
2. `compute_optimal_batch_size()` - Adaptive batch sizing capped at 512
3. `PPOOptimized.fixed_batch_size` property - Getter/setter for consistent batch sizes
4. `PPOOptimized._pad_queries()` - Helper to pad query tensors to fixed_batch_size
5. `PPOOptimized.evaluate_policy()` - Core trajectory evaluation (no padding, expects exact batch size)
6. `PPOOptimized.evaluate_with_corruptions()` - Full MRR evaluation, handles ALL chunking/padding internally

**Usage Pattern:**
```python
# Create PPOOptimized with fixed batch size
ppo = PPOOptimized(env, ...)
ppo.fixed_batch_size = 512  # Or use compute_optimal_batch_size()

# Warmup to compile CUDA graphs
warmup_queries = ppo._pad_queries(sample_queries)
ppo.evaluate_policy(warmup_queries)

# Evaluate - handles any number of queries with chunking/padding
results = ppo.evaluate_with_corruptions(queries, n_negatives=100)
```

**`model_eval_optimized.py`** - Simplified to re-exports only:
```python
from ppo_optimized import compute_metrics_from_ranks, compute_optimal_batch_size
```

**`tests/test_eval_perf.py`**:
1. Creates `PPOOptimized` instances for evaluation
2. Uses `ppo.fixed_batch_size` and `ppo.evaluate_with_corruptions()`
3. Warmup uses proper batch_size parameter

### Key Design Decisions

1. **evaluate_policy() doesn't pad** - Expects queries at exact fixed_batch_size
2. **evaluate_with_corruptions() handles everything** - Chunking, padding, metric computation
3. **_pad_queries() is public helper** - Available for manual warmup scenarios
4. **No backward compatibility** - Clean API, use PPOOptimized directly

---

## Evaluation Parity Notes

### Root Cause of Small MRR Differences

When comparing `model_eval.py` (original) vs `ppo_optimized.py` (optimized) evaluation:

1. **Perfect parity** when all queries fit in one chunk (same RNG progression)
2. **Small differences (~0.003 MRR)** for multi-chunk evaluation due to:
   - Original: Uses `batch_size_env` for chunking queries
   - Optimized: Uses `chunk_queries` for chunking
   - Different chunk sizes → different `torch.randint` calls → different corruption samples

3. **Tie-breaking RNG differences**:
   - Original: Single `np.random.RandomState(0)` accumulated across all batches
   - Optimized: Fresh `np.random.RandomState(0)` per chunk
   - Affects ranking when log-probs are equal (rare but possible)

### How to Achieve Perfect Parity

For tests requiring exact parity:
1. Set `chunk_queries = batch_size_env` (same chunk size)
2. Seed torch RNG before each evaluation: `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`
3. Process all queries in single chunk if possible

Example:
```python
config.chunk_queries = config.batch_size_env  # MUST match!
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
results = ppo.evaluate_with_corruptions(queries, sampler, ...)
```

---

## Recommendations

1. **Use `mode='reduce-overhead'`** for production (faster after warmup)
2. **Use `mode='default'`** for debugging (no CUDA graph restrictions)
3. **Always warmup before timing** evaluation
4. **Use `fixed_batch_size=512`** for 8GB GPUs
5. **Consider `max_depth=10`** for faster evaluation if accuracy permits
6. **For parity tests**: Set `chunk_queries = batch_size_env`

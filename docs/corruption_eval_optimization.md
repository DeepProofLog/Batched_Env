# Corruption Evaluation Pipeline - Technical Documentation

## Overview

This document covers the optimized corruption evaluation pipeline for ranking metrics (MRR, Hits@K) implemented in `model_eval_optimized.py`. The pipeline is designed to efficiently evaluate large-scale scenarios (10,000+ queries, 80,000 corruptions per query) within an 8GB VRAM budget.

## Performance Summary

| Metric | Before Optimization | After Optimization | Speedup |
|--------|---------------------|-------------------|---------|
| First run | N/A | ~7.5s (warmup) | - |
| Steady-state (100q × 50c) | 16.2s | **1.16s** | **14x** |
| Per query | 162ms | 11.6ms | 14x |
| Warmup time | - | ~8s (one-time) | - |

## Key Components

### 1. CompiledEvaluator (`model_eval_optimized.py`)

The core class that wraps the trajectory evaluation with CUDA graph optimization.

```python
from model_eval_optimized import CompiledEvaluator, eval_corruptions_fast

# Create evaluator with fixed batch size
evaluator = CompiledEvaluator(
    env=env,                    # EvalOnlyEnvCompiled instance
    policy_logits_fn=policy_fn, # Function: obs -> logits [B, S]
    batch_size=510,             # Fixed batch size (chunk_queries × (1 + K))
    max_steps=10,               # Max trajectory length
    deterministic=True          # Greedy action selection
)

# One-time warmup (~8s)
evaluator.warmup(sample_queries[:20])

# Fast evaluation (~1.16s for 100 queries with K=50)
results = eval_corruptions_fast(
    evaluator, queries, sampler,
    n_corruptions=50,
    corruption_modes=['tail'],
    chunk_queries=10
)
```

### 2. EvalOnlyEnvCompiled (`env_eval_compiled.py`)

A pure-functional environment designed for `torch.compile`:
- Immutable state using `EvalState` NamedTuple
- `step_functional()` - pure function, no side effects
- `evaluate_trajectory_compiled()` - unrolled loop for fixed `max_steps`

### 3. UnificationEngineVectorized (`unification_vectorized.py`)

Fully compilable unification engine with fixed tensor shapes:
- Fixed output shape: `[B, 120, 12, 3]`
- Masked operations for variable-length results
- Pre-computed fact/rule indices for O(1) lookup

---

## CUDA Graph Optimization - Root Cause & Solution

### The Problem

When using `torch.compile(mode='reduce-overhead')`, CUDA graphs are used to minimize CPU overhead. However, CUDA graphs record **tensor storage memory addresses**, not just shapes.

**Symptom:** Passing new tensor objects each call (even with identical shapes) causes:
- Graph invalidation
- Re-recording on every call
- 26+ seconds instead of 55ms

### The Solution

Use a **static input buffer** with `copy_()`:

```python
class CompiledEvaluator:
    def __init__(self, ...):
        # Static buffer - CUDA graph records this address
        self._input_buffer = torch.zeros(batch_size, 3, dtype=torch.long, device=device)
    
    def __call__(self, queries):
        # Copy data INTO static buffer (address unchanged)
        self._input_buffer.copy_(queries)
        
        # CUDA graph replays correctly
        return compiled_fn(self._input_buffer)
```

### Why This Works

1. **Tensor address is constant** - `self._input_buffer` is allocated once
2. **`copy_()` is in-place** - only values change, not the tensor object
3. **CUDA graph can replay** - all memory addresses match the recorded graph

---

## Known Issues & Limitations

### 1. MRR Discrepancy Between Original and Compiled Environments

**Status:** Known issue, partially investigated

**Symptom:** The `validate_mrr_correctness` test shows different success rates:
- Original environment: ~75% success
- Compiled environment: ~5% success (with the test's policy)

**Root Cause:** Differences in proof detection and state handling between `EvalOnlyEnv` and `EvalOnlyEnvCompiled`. The vectorized unification engine has subtle differences in:
- How substitution chains are resolved
- True atom detection after fact unification
- Handling of `was_done` vs `_done` flags

**Impact:** For relative ranking (MRR), this may still be valid if the ranking order is preserved. The absolute success rate difference suggests the compiled version may be stricter or more lenient in proof detection.

**Investigation needed:**
- Compare action sequences step-by-step
- Verify derived states are identical
- Check proof detection logic in `get_derived_states_compiled`

### 2. Batch Size Sensitivity

Certain batch sizes cause CUDA graph issues:
- **Optimal:** 510, 1000, 200
- **Problematic:** Some batch sizes around 500-510 occasionally fail

**Workaround:** Use chunk sizes that result in batch sizes of 510 or 1020.

### 3. First-Run Overhead

The first call after warmup still takes ~7.5s (vs 1.16s steady-state). This is due to:
- Additional JIT compilation
- Memory allocation patterns settling

---

## Future Work

### High Priority

1. **Fix MRR Discrepancy**
   - Debug step-by-step trajectory differences
   - Ensure proof detection logic is identical
   - Add comprehensive parity tests

2. **Better Batch Size Handling**
   - Investigate why certain batch sizes fail
   - Auto-detect optimal batch size
   - Do not add fallback to eager mode if CUDA graphs fail, investigate the problem

### Medium Priority

3. **Reduce First-Run Overhead**
   - Extended warmup with diverse queries
   - Pre-allocate all intermediate tensors

4. **Memory Optimization**
   - Stream processing for very large evaluations
   - Gradient checkpointing if needed for backprop

### Low Priority

5. **Multi-GPU Support**
   - Distribute queries across GPUs
   - Aggregate results

6. **Async Evaluation**
   - Overlap CPU work with GPU execution
   - Pipeline corruption generation with evaluation

---

## Usage Examples

### Basic Usage

```python
import torch
from model_eval_optimized import CompiledEvaluator, eval_corruptions_fast

# Setup (see test_eval_compiled.py for full setup)
evaluator = CompiledEvaluator(env, policy_fn, batch_size=510, max_steps=10)
evaluator.warmup(queries[:20])

# Evaluate
results = eval_corruptions_fast(evaluator, queries, sampler, n_corruptions=50)
print(f"MRR: {results['MRR']:.4f}")
```

### With Real Policy

```python
from model_eval_optimized import create_policy_logits_fn

# Wrap actor network to extract logits
policy_fn = create_policy_logits_fn(actor_network, deterministic=True)

evaluator = CompiledEvaluator(env, policy_fn, batch_size=510, max_steps=10)
```

### Large-Scale Evaluation

```python
# For 10,000 queries with 80,000 corruptions each
evaluator = CompiledEvaluator(env, policy_fn, batch_size=1020, max_steps=20)
evaluator.warmup(queries[:50])

# Process in chunks (memory-efficient)
results = eval_corruptions_fast(
    evaluator, queries,  # 10,000 queries
    sampler,
    n_corruptions=80000,  # Will be processed in chunks
    chunk_queries=20,     # 20 queries per chunk
)
```

---

## File Reference

| File | Description |
|------|-------------|
| `model_eval_optimized.py` | Main evaluation pipeline with `CompiledEvaluator` |
| `env_eval_compiled.py` | Pure-functional environment for compilation |
| `unification_vectorized.py` | Fixed-shape unification engine |
| `tests/test_eval_compiled.py` | Tests including `validate_mrr_correctness` |

---

## Debugging Tips

### Enable CUDA Graph Logging

```python
import os
os.environ['TORCH_LOGS'] = '+cudagraphs'
```

### Check Tensor Addresses

```python
print(f"data_ptr: {tensor.data_ptr()}")  # Should stay constant for static buffers
```

### Verify No Recompilation

```python
os.environ['TORCH_LOGS'] = '+recompiles'
```

### Profile Per-Component

```python
torch.cuda.synchronize()
t0 = time.time()
# ... operation ...
torch.cuda.synchronize()
print(f"Time: {(time.time()-t0)*1000:.1f}ms")
```

---

## Changelog

- **2024-12-12:** Initial optimized implementation with 14x speedup
- Fixed CUDA graph caching with static input buffer
- Added `CompiledEvaluator` class
- Added `eval_corruptions_fast` function

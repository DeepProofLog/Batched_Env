# Evaluation Optimization Documentation

## 1. Performance Summary

The goal of this optimization was to speed up the evaluation loop which involves complex logic (unification, policy forward passes, environment steps).

| Implementation | Runtime (per query) | Warmup/Compile Time | MRR Correctness | Notes |
|----------------|---------------------|---------------------|-----------------|-------|
| **Original** | ~162 ms | ~0s | ✅ Correct (0.82) | Slowest, pure Python loop. |
| **Eager Optimized** | ~20-30 ms* | ~0s | ❓ Discrepancy | Uses vectorized environment/unification but no `torch.compile`. |
| **Compiled (Default)** | **~11.6 ms** | **17s - 192s** | ❌ **Buggy (0.47)** | 14x Faster, but currently has a critical correctness bug. |

*\*Estimated based on partial eager tests.*

### Detailed Timing
*   **Steady State Speedup**: The compiled version achieves a **14x speedup** over the original (1.16s vs 16.2s for 100 queries x 50 corruptions).
*   **Warmup/Compilation**:
    *   `fullgraph=True`: ~192s (Initial one-time cost).
    *   `fullgraph=False`: ~17s (For small batches) to ~260s (For large batches). Highly variable.
    *   **Separate Compilation** (Failed experiment): ~226s.

---

## 2. Architecture: How the Optimized Version Works

The optimized pipeline replaces standard Python logic with tensor-based operations suitable for CUDA graphs.

### Key Components

1.  **`UnificationEngineVectorized`** (`unification_vectorized.py`):
    *   Replaces iterative unification with masked tensor operations.
    *   Operates on fixed-size tensors (padding to `max_atoms` and `max_states`).
    *   Returns derivations for the entire batch in parallel.

2.  **`EvalOnlyEnvCompiled`** (`env_eval_compiled.py`):
    *   A "functional" environment designed for `torch.compile`.
    *   **State**: Uses `EvalObs` (NamedTuple) instead of mutable objects.
    *   **Step**: `step_functional` is a pure function taking (state, action) -> (next_state, reward, done).
    *   **Trajectory**: `evaluate_trajectory_compiled` unrolls the interaction loop for a fixed `max_depth` (e.g., 20 steps).

3.  **`CompiledEvaluator`** (`model_eval_optimized.py`):
    *   Wraps the environment and policy.
    *   Manages **Static Input Buffers**: To prevent CUDA graph recompilation, inputs are copied into a pre-allocated static buffer `self._input_buffer`.

---

## 3. Compilation Modes & Eager Execution

The `CompiledEvaluator` supports different modes via the `compile_mode` and `fullgraph` arguments.

### Eager Optimized
*   **How**: Set `compile_mode=None` or just don't wrap with `torch.compile`.
*   **Pros**: No compilation latency, easier debugging.
*   **Cons**: Slower than compiled (Python overhead in the 20-step loop).

### Compiled Modes
1.  **`default`** (Recommended):
    *   Good balance of compile time and runtime.
    *   Used in current benchmarks.
2.  **`reduce-overhead`**:
    *   Uses CUDA Graphs aggressively.
    *   **Pros**: Lowest runtime latency (eliminates CPU launch overhead).
    *   **Cons**: extremely slow compilation, sensitive to graph breaks.

### Full Graph (`fullgraph=True`)
*   **Goal**: Capture the entire 20-step loop + Policy + Environment in a single graph.
*   **Reality**:
    *   Achieves best runtime stability.
    *   **Compilation Time**: ~192s (Very slow due to graph size: ~40k nodes).
    *   **Constraint**: Code must be strictly functional (no side effects, no dynamic control flow depending on data values).

---

## 4. Warmup & Graph Issues

### The Warmup Problem
The first call to a compiled function triggers the tracing and optimization process.
*   **Graph Size Explosion**: Unrolling 20 steps of a policy+unification loop creates a massive computational graph.
*   **Optimization Time**: `torch.compile` spends most of its time optimizing this massive graph (fusion, scheduling).

### Optimizations Applied
To reduce the ~209s compilation time, we applied:
```python
os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'
os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING'] = '0'
inductor_config.compile_threads = 4
torch.set_float32_matmul_precision('high')
```
**Result**: Reduced time from 209s -> 192s (~8% improvement).

---

## 5. Known Issues & Failed Experiments

### 🔴 Critical Bug: MRR Discrepancy
The compiled version consistently underperforms the original.
*   **Original MRR**: ~0.82
*   **Compiled MRR**: ~0.47
*   **Status**: **Unresolved**. The drop is systematic across seeds.
*   **Suspects**:
    *   Policy/Environment mismatch in observation handling.
    *   `UnificationEngineVectorized` correctness edge cases.
    *   Numerical precision differences (TF32 vs FP32).

### ❌ Failed: Separate Compilation (`model_eval_optimized_fast.py`)
**Idea**: Compile the Policy (~30s) and Trajectory Loop (~30s) separately to get ~60s total.
**Result**: **Failed (226s)**.
*   **Why**: Breaking the graph into two parts introduced "Graph Breaks".
*   `torch.compile` handles graph breaks by falling back to Python, or creating multiple subgraphs.
*   Optimizing a graph with breaks turned out to be *slower* than optimizing one giant `fullgraph`.

---

## 6. Testing Guide

### Primary Test Script
`tests/test_eval_optimized.py` is the main entry point.

**Usage**:
```bash
# 1. Performance Benchmark (Original vs Compiled)
python tests/test_eval_optimized.py --performance --n-test-queries 20 --n-corruptions 50

# 2. Correctness Check (Multi-seed) - Run this to debug MRR
python tests/test_eval_optimized.py --seed-test --num-seeds 3

# 3. Graph Break Analysis
python tests/test_eval_optimized.py --check-compile

# 4. Fast Smoke Test (Check compilation time)
python tests/test_eval_optimized.py --compiled-smoke
```

### Debugging Tips
*   Use `--skip-compiled` to run just the vectorized unification tests.
*   Use `--fullgraph` to toggle `fullgraph=True/False`.
*   Unused/Deleted tests: `test_fast_compilation.py`, `debug_mrr.py`, `quick_debug.py`.

### Future Work Checklist
1.  **Fix MRR Bug**: Isolate the divergence. Compare step-by-step state traces between Eager-Original and Eager-Optimized.
2.  **Warmup Optimization**: Investigate if `torch.export` (AOTInductor) can pre-compile the graph offline.
3.  **Memory**: `fullgraph` is memory hungry. Verify behavior on larger batch sizes (currently tuned for ~510).
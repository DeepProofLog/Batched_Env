# Evaluation Optimization Documentation

## 1. Performance Summary

The optimized pipeline uses **single-step compilation**: compile one policy+step transition and loop in Python.

### Why Single-Step Compilation?

| Approach | Compile Time | Runtime | Notes |
|----------|-------------|---------|-------|
| Full Trajectory | ~180-200s | ~73 ms/q | Slow compile, same results |
| **Single-Step** | **~5-10s** | ~76 ms/q | Fast compile, same results ✓ |

**Decision**: Use single-step only. Full trajectory compilation was removed - same results with 10-20x faster compile.

### Benchmark Results (family dataset, 8GB VRAM)

| Mode | Queries | Corruptions | Warmup (s) | Runtime (s) | ms/query |
|------|---------|-------------|------------|-------------|----------|
| **Original** | 50 | 50 | 0.0 | 32.40 | 648 |
| **Optimized** | 50 | 50 | ~5-10s | ~1.5 | ~30 |

**Runtime speedup**: ~8-10x faster than Original

### Summary

| Implementation | Runtime (per query) | Compile Time | Notes |
|----------------|---------------------|--------------|-------|
| **Original** | ~650 ms | 0s | Slowest, pure Python loop |
| **Optimized (single-step)** | **~76 ms** | ~5-10s | Best balance: fast compile + fast runtime |



---

## 2. Architecture: How the Optimized Version Works

The optimized pipeline replaces standard Python logic with tensor-based operations suitable for CUDA graphs.

### Key Components

1.  **`UnificationEngineVectorized`** (`unification_vectorized.py`):
    *   Replaces iterative unification with masked tensor operations.
    *   Operates on fixed-size tensors (padding to `max_atoms` and `max_states`).
    *   Returns derivations for the entire batch in parallel.

2.  **`EvalOnlyEnvOptimized`** (`env_optimized.py`):
    *   A "functional" environment designed for `torch.compile()`.
    *   **State**: Uses `EnvState` NamedTuple instead of mutable objects.
    *   **Step**: `step_functional` is a pure function (state, action) → (next_state, obs, reward).
    *   **Policy Step**: `step_with_policy_functional` combines policy + step for compilation.
    *   **Trajectory**: `evaluate_trajectory` loops in Python, calling compiled single-step.

3.  **`FastEvaluator`** (`model_eval_optimized.py`):
    *   Wraps the environment and policy.
    *   Compiles single step via `create_compiled_step_fn`.
    *   Manages static input buffers for CUDA graph stability.

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

## 5. Known Issues & Resolved Experiments

### 🔴 Critical Bug: MRR Discrepancy
The compiled version consistently underperforms the original.
*   **Original MRR**: ~0.82
*   **Compiled MRR**: ~0.47
*   **Status**: **Unresolved**. The drop is systematic across seeds.
*   **Suspects**:
    *   Policy/Environment mismatch in observation handling.
    *   `UnificationEngineVectorized` correctness edge cases.
    *   Numerical precision differences (TF32 vs FP32).

### ✅ Resolved: Single-Transition Compilation (`env_eval_fast.py`)
**Idea**: Instead of compiling the full 20-step trajectory, compile only a **single transition** (policy + step).

**Previous Attempt** (`model_eval_optimized_fast.py`): Failed (226s) because it tried to compile policy and trajectory separately, causing graph breaks.

**New Approach** (`env_eval_fast.py`): Compile only `step_with_policy_functional()` which combines:
1. Policy forward pass (logits)
2. Action selection (argmax/sampling)
3. Environment step (`step_functional`)

Then loop over 20 steps in Python, calling the compiled single-step function each iteration.

**Result**: **Success!**
*   **Warmup (cold cache)**: ~18.5s (vs ~189s for full trajectory) → **10x faster**
*   **Runtime**: ~76 ms/query (vs ~79 ms/query for full trajectory) → Similar performance
*   **Why it works**: The single-step graph is ~20x smaller (~2k nodes vs ~40k nodes), so compilation is much faster. The Python loop overhead is minimal since each step is still fully compiled.

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
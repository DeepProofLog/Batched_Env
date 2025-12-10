# Unification Engine Optimization Plan: Bridging the Gap to 35s

**Current Status**: 
- **Baseline**: 94.19s
- **Current**: 47.34s (**~50% Speedup**)
- **Target**: <35.00s (Gap: ~12.3s)

We have successfully implemented aggressive "Phase 1-3" optimizations, including cache locality, vectorized logic, boolean masking (removing `nonzero`), and sort-free hashing. Performance has plateaued because we are now **latency-bound by PyTorch kernel launch overhead** rather than throughput-bound by computation. The unification engine executes many small, data-dependent operations (scatter, gather, index_select) that are difficult for the standard PyTorch eager runtime to optimize further.

To reach the sub-35s target, we must move beyond standard PyTorch operations into **Kernel Fusion** and **Structural Optimization**.

---

## 1. High-Impact Optimizations (The "Big Guns")

### A. Custom Kernel Fusion with Triton / CUDA
**Impact**: High (~10-15s reduction) | **Effort**: High
**Rationale**: The current bottlenecks (`unify_with_rules`, `apply_substitutions`) involve chains of memory-bound elementwise operations (`where`, `and`, `expand`). PyTorch launches separate kernels for each. Fusing these into single kernels avoids VRAM round-trips and launch latency.

*   **Substitutions Kernel**: Fuse the `match -> gather -> where` logic into a single Triton kernel.
    *   *Current*: ~2.6s (still involves large materialized masks).
    *   *Target*: <1.0s.
*   **Pruning Kernel**: Fuse `check_facts -> update_mask -> scatter_proofs` into one operation.
    *   *Current*: ~2.1s (heavily uses boolean indexing).
    *   *Target*: <0.8s.

### B. Static Shape Bucketing (Enable CUDA Graphs)
**Impact**: Medium-High (~5-8s reduction) | **Effort**: High
**Rationale**: `torch.compile` and CUDA Graphs failed previously due to dynamic shapes (number of derived states varies wildly). By padding derived states to fixed "buckets" (e.g., 64, 128, 256), we can stabilize shapes and enable graph capture.

*   **Strategy**: Instead of `[N_varying, M, 3]`, pad to `[N_bucket, M, 3]`.
*   **Significance**: Removes CPU-side shape analysis overhead in every step.

---

## 2. Algorithmic Optimizations (Logic Changes)

### C. "Lazy" Standardization
**Impact**: Medium (~2-3s reduction) | **Effort**: Medium
**Rationale**: `standardize_derived_states` (~2.1s) is expensive because it canonicalizes *every* derived state immediately.
*   **Proposal**: Only standardize states *before* adding them to the global memory/cache. During the intermediate search (within a batch), use a cheaper strict equality check or handle raw variable IDs if local context allows.

### D. Hierarchical Unification (Pre-filtering)
**Impact**: Medium (~2-4s reduction) | **Effort**: Medium
**Rationale**: We check every query against every rule head.
*   **Proposal**: Implement a coarse Bloom filter or bitmask for rule heads.
    *   If a query's predicate/args don't match the signature of a rule block, skip the entire vectorized unification block.
    *   Reduces the `N` in `N*R` operations.

---

## 3. Micro-Optimizations (Low Hanging Fruit Harvested, but Remnants Remain)

### E. Zero-Copy Integers (uint32)
**Impact**: Low-Medium (~1-2s) | **Effort**: Medium
**Rationale**: We currently use `int64` everywhere. Switching to `int32` halves memory bandwidth for index-heavy operations (gather/scatter).
*   **Constraint**: Requires verifying that vocab size and batch size fit in 4 billion (likely yes).

### F. Specialized "One-to-One" Paths
**Impact**: Low (~1s) | **Effort**: Low
**Rationale**: `unify_one_to_one` is still generic.
*   **Proposal**: Hard-code paths for specific common rules (e.g., Symmetry `p(X,Y) :- p(Y,X)`) to completely bypass the generic "args -> match -> sub" pipeline.

---

## Impact Estimates Summary

| Optimization Strategy | Est. Speedup | Complexity | Priority |
| :--- | :--- | :--- | :--- |
| **Custom Triton Kernels** | **10-15s** | High | **1** |
| **Static Shape Bucketing** | 5-8s | High | 2 |
| **Lazy Standardization** | 2-3s | Medium | 3 |
| **Hierarchical Filtering** | 2-4s | Medium | 4 |
| **Int32 Precision** | 1-2s | Medium | 5 |

**Conclusion**: To guarantee <35s, we should pursue **Triton Kernels** for the hottest paths (`apply_substitutions`, `prune_and_collapse`). This addresses the fundamental "small kernel launch" bottleneck.

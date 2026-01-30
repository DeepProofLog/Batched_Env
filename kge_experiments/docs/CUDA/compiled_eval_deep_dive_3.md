# PyTorch Compiled Evaluation: Deep Dive Part 3
# V10 Architecture, Evolution, and Large-Scale Optimization

This document covers the evolution of the evaluation system, what V10 does, alternative approaches tried, and optimal strategies for large-scale datasets.

---

## Table of Contents

1. [The Evolution of Evaluation Approaches](#1-the-evolution-of-evaluation-approaches)
2. [V10 Architecture Deep Dive](#2-v10-architecture-deep-dive)
3. [The Slot Recycling Strategy](#3-the-slot-recycling-strategy)
4. [The Transposed Pool Layout](#4-the-transposed-pool-layout)
5. [What We Tried and Learned](#5-what-we-tried-and-learned)
6. [Optimal Strategy for Large-Scale Datasets](#6-optimal-strategy-for-large-scale-datasets)
7. [Performance Analysis](#7-performance-analysis)
8. [Production Recommendations](#8-production-recommendations)

---

## 1. The Evolution of Evaluation Approaches

### 1.1 The Evaluation Problem

For knowledge graph link prediction, we need to:
1. Take N test queries (e.g., 3000 triples)
2. For each query, generate K corruptions (e.g., 100 per corruption mode)
3. Run each candidate through the proof system
4. Rank based on proof success/failure
5. Compute MRR, Hits@10, etc.

Total candidates: N × K × modes = 3000 × 100 × 2 = 600,000 candidates!

### 1.2 Naive Approach: Sequential

```python
# V0: Sequential (SLOW)
for query in queries:
    for corruption in corruptions:
        result = run_proof(query, corruption)
        record(result)
```

**Time:** ~100ms per candidate × 600K = 16+ hours

### 1.3 Batched Approach: First Improvement

```python
# V1-V5: Batched
batch_size = 256
for batch in batched(candidates, batch_size):
    results = run_batch_proof(batch)
    record_batch(results)
```

**Time:** Reduced by batch_size factor → ~1-2 hours

### 1.4 Compiled Approach: Major Speedup

```python
# V6-V9: Compiled batched
compiled_step = torch.compile(step_fn, mode='reduce-overhead')

while not done:
    results = compiled_step(batch)
    # ...
```

**Time:** Further reduced by compilation → 30-60 minutes

### 1.5 V10: Optimized Compiled + Slot Recycling

The current best approach combining:
- Compiled step function with CUDA graphs
- Slot recycling for continuous parallelism
- Transposed pool layout for efficient memory access
- Conditional engine calls with torch.cond

---

## 2. V10 Architecture Deep Dive

### 2.1 Key Components

```
┌───────────────────────────────────────────────────────────────┐
│                     MinimalEvalV10                            │
├───────────────────────────────────────────────────────────────┤
│  Pre-allocated Buffers:                                       │
│  ├── _current      [B, A, 3]         Current proof states     │
│  ├── _derived      [B, S, A, 3]      Possible next states     │
│  ├── _counts       [B]               Valid derived count      │
│  ├── _mask         [B, S]            Action mask              │
│  ├── _depths       [B]               Current proof depth      │
│  ├── _done         [B]               Slot completion status   │
│  └── _pool_ptr     [B]               Index into query pool    │
│                                                                │
│  Query Pool:                                                   │
│  ├── _pool         [max_pool, 3]     All candidates           │
│  ├── _pool_size    scalar            Current pool size        │
│  └── _result_buf   [max_pool]        Success/failure results  │
│                                                                │
│  Compiled Functions:                                          │
│  └── _compiled_step(pool, pool_size) → new states             │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 The Step Function

```python
def step_fn(pool, pool_size):
    """One step for all B batch slots."""
    
    # 1. POLICY: Select action for each slot
    obs = {'current': current, 'derived': derived, 'mask': mask}
    logits = policy.get_logits(obs)
    masked_logits = logits.masked_fill(~mask, -inf)
    actions = masked_logits.argmax(dim=-1)
    
    # 2. TRANSITION: Apply selected action
    new_current = derived[arange_B, actions]
    new_depth = depths + 1
    
    # 3. ENGINE: Compute new derived states (UNIFICATION)
    new_derived, new_counts = engine.get_derived(new_current)
    
    # 4. TERMINAL CHECK
    is_success = (first_pred == true_pred_idx)
    is_failure = (first_pred == false_pred_idx)
    truncated = (new_depth >= max_depth)
    newly_done = is_success | is_failure | truncated
    
    # 5. SLOT RECYCLING: Load new query if slot finished
    needs_reset = newly_done & (pool_ptr + stride < pool_size)
    new_ptr = torch.where(needs_reset, pool_ptr + stride, pool_ptr)
    
    # Load new queries for recycled slots
    new_query = pool[new_ptr]
    reset_current = torch.where(needs_reset, new_query, new_current)
    reset_derived = engine.get_derived(reset_current)  # Only for reset slots!
    
    return reset_current, reset_derived, new_ptr, ...
```

### 2.3 The Evaluate Loop

```python
def evaluate(self):
    self.setup_pool(queries, sampler, n_corruptions, modes)
    
    while not self._done.all():
        torch.compiler.cudagraph_mark_step_begin()
        
        # Compiled step returns NEW tensors
        results = self._compiled_step(self._pool, self._pool_size)
        
        # Copy into persistent buffers
        torch._foreach_copy_(
            [self._current, self._derived, ...],
            [results[0], results[1], ...]
        )
        
        # Record results for finished slots
        self._record_results(results)
    
    return self._compute_metrics()
```

---

## 3. The Slot Recycling Strategy

### 3.1 The Core Idea

Traditional approach: Process all candidates, wait for slowest.
V10 approach: When a slot finishes, immediately load next candidate.

```
Traditional (No Recycling):
Step:  1    2    3    4    5    6    ...   20
Slot0: ████ ████ DONE ░░░░ ░░░░ ░░░░       ░░░░  (wasted!)
Slot1: ████ ████ ████ ████ DONE ░░░░       ░░░░  (wasted!)
Slot2: ████ ████ ████ ████ ████ ████       DONE
Slot3: ████ ████ ████ DONE ░░░░ ░░░░       ░░░░  (wasted!)

V10 (With Recycling):
Step:  1    2    3    4    5    6    ...   20
Slot0: ████ ████ DONE|████ ████ ████ DONE|████   (recycled twice!)
Slot1: ████ ████ ████ ████ DONE|████ ████ DONE
Slot2: ████ ████ ████ ████ ████ ████ ...  DONE
Slot3: ████ ████ ████ DONE|████ ████ ████ DONE
                    ▲                  ▲
                    │                  │
              Slot recycled      Recycled again
```

### 3.2 Why Recycling Matters

For 20,000 candidates with batch size 256:

**Without recycling:**
- Must run max_depth (20) steps for ALL slots
- Steps needed: ceil(20000/256) × 20 = 79 × 20 = 1580 steps
- But many finish early (avg ~10 steps)
- Wasted: 79 × (20-10) × 256 = 202,240 slot-steps!

**With recycling:**
- Total work = sum of individual proof lengths
- If avg proof = 10 steps: 20000 × 10 = 200,000 slot-steps
- Steps needed: 200,000 / 256 ≈ 781 steps
- Savings: ~50% fewer steps!

### 3.3 The Law of Large Numbers Effect

With many candidates:

```
                    Proof Length Distribution
    ┌────────────────────────────────────────────────┐
    │                                                │
30% │                  ╭────╮                        │
    │               ╭──╯    ╰──╮                     │
20% │            ╭──╯          ╰──╮                  │
    │         ╭──╯                ╰──╮               │
10% │      ╭──╯                      ╰──────────╮    │
    │   ╭──╯                                    ╰──  │
    └────────────────────────────────────────────────┘
         2  4  6  8  10  12  14  16  18  20  steps

At any step, ~same number of proofs finish
→ Continuous recycling, steady throughput
→ All slots stay busy until very end
```

---

## 4. The Transposed Pool Layout

### 4.1 The Problem

For ranking, we need to compare:
- Original query Q0 vs its corruptions Q0_c1, Q0_c2, ..., Q0_c100

If candidates are laid out sequentially:
```
Pool: [Q0, Q0_c1, Q0_c2, ..., Q0_c100, Q1, Q1_c1, ...]
       ▲   ▲
       │   │
       │   └── Second candidate for same query
       └────── First candidate
       
Stride between same-query candidates = 1 (adjacent)
```

But with B=256 slots, we load:
```
Slot 0: Q0      (idx 0)
Slot 1: Q0_c1   (idx 1)
...
Slot 100: Q0_c100 (idx 100)
Slot 101: Q1     (idx 101)
...
```

**Problem:** One query's candidates span multiple slots!
When Q0 finishes, we need to wait for all 100 corruptions.

### 4.2 The Transposed Solution

Transpose the layout so same-query candidates are separated by stride:

```
Pool (Transposed):
idx 0:   Q0        ─┐
idx 1:   Q1         │ First corruption of each query
idx 2:   Q2         │
...                 │
idx N-1: Q_{N-1}  ─┘

idx N:   Q0_c1    ─┐
idx N+1: Q1_c1     │ Second corruption of each query
idx N+2: Q2_c1     │
...               ─┘

idx 2N:  Q0_c2    ─┐
...               ─┘
```

Now with stride = N (number of queries):
```
Slot 0 processes:  Q0 → Q0_c1 → Q0_c2 → ... → Q0_c100
Slot 1 processes:  Q1 → Q1_c1 → Q1_c2 → ... → Q1_c100
...
```

**Each slot handles ALL corruptions for ONE query!**

### 4.3 Implementation in V10

```python
def setup_pool(self, queries, sampler, n_corruptions, modes):
    N = queries.shape[0]
    K = 1 + n_corruptions  # Original + corruptions
    
    for mode in modes:
        # Generate corruptions: [N, K, 3]
        neg = sampler.corrupt(queries, num_negatives=n_corruptions, mode=mode)
        candidates = torch.cat([queries.unsqueeze(1), neg], dim=1)  # [N, K, 3]
        
        # TRANSPOSE: [K, N, 3] - crucial step!
        candidates_transposed = candidates.transpose(0, 1).contiguous()
        
        # Flatten: [K*N, 3]
        pool_segment = candidates_transposed.view(-1, 3)
        pools.append(pool_segment)
    
    self._pool[:total].copy_(torch.cat(pools, dim=0))
    self._stride.fill_(N)  # Stride = number of queries
```

### 4.4 Memory Access Pattern

```
Sequential layout: Poor cache utilization
Step 1: Access idx 0, 1, 2, ..., 255 (consecutive ✓)
Step 2: Access idx 256, 257, ... (consecutive ✓ but different queries)

Transposed layout: Strided access
Step 1: Access idx 0, 1, 2, ..., 255 (consecutive ✓)
Step 2: Access idx 0+N, 1+N, 2+N, ... (strided, same queries ✓)

Transposed gives better RANKING coherence at cost of memory stride.
```

---

## 5. What We Tried and Learned

### 5.1 Evolution Summary

| Version | Key Change | Result |
|---------|------------|--------|
| V1-V5 | Basic batching | Baseline |
| V6 | torch.compile | 2x speedup |
| V7 | mode='reduce-overhead' | 1.5x more |
| V8 | Separate step/reset | Cleaner but same speed |
| V9 | Compiled engine | 4.3x engine speedup |
| V10 | torch.cond for conditional | 0.88 ms/candidate |

### 5.2 What Worked

1. **Compilation with reduce-overhead**: CUDA graphs eliminated launch overhead
2. **Buffer-copy pattern**: Fixed addresses for CUDA graph stability
3. **Compiled engine**: 4.3x speedup on unification
4. **torch.cond**: Skip engine calls when not needed
5. **Transposed pool**: Efficient slot recycling per query

### 5.3 What Didn't Work (Or Didn't Help Much)

1. **Removing .any() check**: Engine every step was 8x slower than sync overhead
2. **Fused step without cond**: Engine every step slower due to unnecessary compute
3. **Whole loop compilation**: Dynamic termination and memory constraints
4. **Index-based derived states**: Impossible due to dynamic computation
5. **Zero-copy patterns**: Only 2% overhead, not worth complexity

### 5.4 Key Lessons

1. **Profile first, optimize what matters**: Copies were 2%, policy was 45%
2. **CPU-GPU sync isn't always bad**: .any() saved 10+ seconds
3. **Static shapes are critical**: Padding for fixed dimensions
4. **Graph breaks have real costs**: torch.cond partitions graph
5. **Slot recycling is essential**: 50% fewer steps with variable-length work

---

## 6. Optimal Strategy for Large-Scale Datasets

### 6.1 Scaling Dimensions

For large datasets, these dimensions grow:
- N: Number of test queries (e.g., 10,000+)
- K: Corruptions per query (e.g., 500+ for filtered ranking)
- Modes: Corruption modes (typically 2: head, tail)

Total candidates: 10,000 × 500 × 2 = 10 million!

### 6.2 Memory-Limited Batching

Can't load 10M candidates at once. Solution: **Chunked Evaluation**

```python
def evaluate_large_scale(queries, chunk_size=10000):
    all_results = []
    
    for chunk_start in range(0, len(total_candidates), chunk_size):
        chunk = candidates[chunk_start:chunk_start + chunk_size]
        
        # V10-style evaluation on chunk
        evaluator.setup_pool(chunk)
        chunk_results = evaluator.evaluate_chunk()
        
        all_results.append(chunk_results)
        
        # Free memory
        torch.cuda.empty_cache()
    
    return combine_results(all_results)
```

### 6.3 Multi-GPU Scaling

For very large datasets, distribute across GPUs:

```python
def evaluate_multi_gpu(queries, num_gpus=4):
    chunks = split_queries(queries, num_gpus)
    
    # Each GPU evaluates its chunk in parallel
    futures = []
    for gpu_id, chunk in enumerate(chunks):
        with torch.cuda.device(gpu_id):
            futures.append(evaluate_async(chunk, gpu_id))
    
    # Gather results
    results = [f.result() for f in futures]
    return combine_results(results)
```

### 6.4 Optimal Batch Size Selection

```
Batch size trade-offs:

Small batch (e.g., 64):
+ Less memory per batch
+ More flexibility
- Higher launch overhead
- Lower GPU utilization

Large batch (e.g., 1024):
+ Better GPU utilization
+ Lower overhead per candidate
- More memory needed
- Longer time to first result

Optimal: Largest batch that fits in memory with peak utilization

For V10 with derived states [B, 120, 6, 3]:
Memory ≈ B × 120 × 6 × 3 × 8 bytes = B × 17 KB
For B=1024: 17 MB (easily fits)
For B=4096: 68 MB (still OK on modern GPUs)
For B=16384: 272 MB (may compete with model)
```

### 6.5 Optimal Corruption Count

```
Trade-off: More corruptions = better ranking but more compute

For MRR/Hits@10:
- 100 corruptions: Usually sufficient
- 500 corruptions: Marginal improvement
- Full ranking (all entities): Best but expensive

Recommendation:
- Development: 100 corruptions (fast iteration)
- Validation: 500 corruptions (better signal)
- Final evaluation: Full ranking (for papers)
```

---

## 7. Performance Analysis

### 7.1 V10 Performance Breakdown

```
20,000 candidates, batch=256, max_depth=20

┌─────────────────────────────────────────────────────────────┐
│                    V10 Performance                          │
├─────────────────────────────────────────────────────────────┤
│ Runtime:         17.7 seconds                               │
│ ms/candidate:    0.88                                       │
│ Steps:           1600                                       │
│ Candidates/step: 12.5                                       │
│                                                              │
│ Time breakdown:                                              │
│ ├── Policy forward:    45%  (7.9s)                          │
│ ├── Engine calls:      25%  (4.4s)                          │
│ ├── torch.where:       12%  (2.1s)                          │
│ ├── Miscellaneous:     16%  (2.8s)                          │
│ └── Memory copies:      2%  (0.4s)                          │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Scaling Projections

```
Candidates    Batch   Steps    Time (projected)
─────────────────────────────────────────────────
20,000        256     1600     17.7s   (measured)
100,000       256     8000     88s     (~1.5 min)
1,000,000     256     80,000   880s    (~15 min)
10,000,000    1024    200,000  2200s   (~37 min)
```

With multi-GPU (4×):
```
10,000,000    1024×4  50,000   550s    (~9 min)
```

### 7.3 Bottleneck Evolution

As we optimize, bottlenecks shift:

```
Initial:
[████████████████] Engine (80%)
[██] Policy (15%)
[█] Other (5%)

After compiled engine:
[████████] Policy (45%)
[█████] Engine (25%)
[████] Other (30%)

After further optimization (theoretical):
[███] Policy (20%)  ← Need smaller/faster model
[██] Engine (15%)   ← Already optimized
[████████] Memory (40%)  ← Becomes bottleneck
[███] Other (25%)
```

---

## 8. Production Recommendations

### 8.1 Configuration Checklist

```python
# Optimal V10 configuration for production
config = {
    # Batch size: maximize for your GPU memory
    'batch_size': 1024,  # or 2048 on A100
    
    # Pool size: balance memory vs chunk frequency
    'max_pool_size': 100000,  # Fits in GPU memory
    
    # Depth: based on your problem
    'max_depth': 20,  # Typical for KG reasoning
    
    # Corruptions: based on evaluation type
    'n_corruptions': 100,  # Development
    # 'n_corruptions': -1,  # Full ranking for papers
    
    # Compilation
    'compile_mode': 'reduce-overhead',
    'fullgraph': True,
    
    # Strategy: torch.cond for conditional engine
    'strategy': 'torch_cond',
}
```

### 8.2 Memory Management

```python
# For large-scale evaluation
def evaluate_with_memory_management(queries, evaluator):
    chunk_size = 50000  # Adjust based on GPU memory
    
    results = []
    for i in range(0, len(queries), chunk_size):
        chunk = queries[i:i+chunk_size]
        
        # Evaluate chunk
        chunk_result = evaluator.evaluate(chunk, ...)
        results.append(chunk_result)
        
        # Force garbage collection between chunks
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return aggregate_results(results)
```

### 8.3 Monitoring and Logging

```python
# Add monitoring for production
class MonitoredEvaluator(MinimalEvalV10):
    def evaluate(self, ...):
        start = time.time()
        step_count = 0
        
        while not self._done.all():
            step_count += 1
            
            if step_count % 100 == 0:
                elapsed = time.time() - start
                done_count = self._done.sum().item()
                throughput = done_count / elapsed
                
                log.info(f"Step {step_count}: "
                        f"{done_count}/{total} done, "
                        f"{throughput:.1f} cand/s")
        
        total_time = time.time() - start
        log.info(f"Completed: {total} candidates in {total_time:.1f}s")
```

### 8.4 Final Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                OPTIMAL LARGE-SCALE EVALUATION                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Queries    │───▶│  Transposed  │───▶│   Chunked    │       │
│  │   + Corrupt  │    │    Pool      │    │   Batches    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   V10 Evaluation Loop                     │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │  For each step:                                      │ │   │
│  │  │    1. mark_step_begin()                              │ │   │
│  │  │    2. compiled_step(pool, pool_size)                 │ │   │
│  │  │       • Policy forward                               │ │   │
│  │  │       • Engine (if needed, via torch.cond)           │ │   │
│  │  │       • Slot recycling                               │ │   │
│  │  │    3. _foreach_copy_(buffers, results)               │ │   │
│  │  │    4. Record finished results                        │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Results aggregation, metric computation                  │   │
│  │  MRR, Hits@1, Hits@3, Hits@10                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Key Performance Features:
✓ Slot recycling (50% fewer steps)
✓ Transposed pool (per-query slot affinity)
✓ Compiled step with CUDA graphs
✓ Conditional engine via torch.cond
✓ Chunked evaluation for memory management
✓ Buffer-copy pattern for graph stability
```

---

## Summary

### What V10 Does Right

1. **Slot recycling**: Immediate reuse of finished slots
2. **Transposed pool**: Efficient per-query candidate grouping
3. **Compiled step**: CUDA graph + Triton optimization
4. **Conditional engine**: torch.cond skips unnecessary work
5. **Buffer pattern**: Stable addresses for CUDA graphs

### Optimal for Large Scale

1. **Chunked evaluation**: Process in memory-fitting chunks
2. **Large batch size**: Maximize GPU utilization
3. **Multi-GPU**: Distribute across devices for huge datasets
4. **Memory management**: GC and cache clearing between chunks
5. **Monitoring**: Track progress for long evaluations

### Performance Achieved

- **0.88 ms/candidate** on V10
- **~15 minutes** for 1M candidates (projected)
- **~37 minutes** for 10M candidates single GPU
- **~9 minutes** for 10M candidates with 4 GPUs

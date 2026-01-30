# PyTorch Compiled Evaluation: Deep Dive Part 2
# Zero-Copy Strategies and Buffer Management

This document is Part 2 of the expert deep dive, focusing specifically on memory copy optimization strategies, why they matter, when they don't matter, and the patterns used in V10.

---

## Table of Contents

1. [The Copy Problem](#1-the-copy-problem)
2. [Where Copies Happen in Compiled Code](#2-where-copies-happen-in-compiled-code)
3. [The Three Copy Patterns](#3-the-three-copy-patterns)
4. [Double Buffering: Zero-Copy Pattern](#4-double-buffering-zero-copy-pattern)
5. [Index-Based Access: Avoiding Query Copies](#5-index-based-access-avoiding-query-copies)
6. [Why Copying Wasn't Our Major Bottleneck](#6-why-copying-wasnt-our-major-bottleneck)
7. [The Strategy We Used](#7-the-strategy-we-used)
8. [When to Optimize for Copies](#8-when-to-optimize-for-copies)
9. [Implementation Considerations](#9-implementation-considerations)

---

## 1. The Copy Problem

### 1.1 The Fundamental Question

When using `torch.compile(mode='reduce-overhead')` with CUDA graphs, a natural question arises:

> **If CUDA graphs require fixed memory addresses, and my computation produces new tensors, how do I get the results into my fixed buffers without copying?**

This is the central tension in compiled evaluation loops:

```
Compiled Function:
┌─────────────────────────────────────────────────────────────┐
│ Input tensors (fixed addresses) ──▶ Computation ──▶ Output │
│                                                             │
│ But outputs are NEW tensors with NEW addresses!             │
│ Next iteration needs to read from fixed buffer addresses... │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The Memory Model

Understanding the memory model is critical:

```
Iteration N:
┌───────────────────────┐         ┌───────────────────────┐
│ self._derived         │         │ new_derived           │
│ Address: 0x7f001000   │  ───▶   │ Address: 0x7f002000   │
│ [old data]            │ compile │ [new data]            │
└───────────────────────┘  step   └───────────────────────┘
                                            │
                                    .copy_()│
                                            ▼
Iteration N+1:                     ┌───────────────────────┐
┌───────────────────────┐          │ self._derived         │
│ self._derived         │  ◀───── │ Address: 0x7f001000   │
│ Address: 0x7f001000   │          │ [new data]            │
│ [new data]            │          └───────────────────────┘
└───────────────────────┘
        │
        ▼
   Next compiled step reads from here
```

The `copy_()` operation transfers data from the result tensor (at a temporary address) into the persistent buffer (at a fixed address).

### 1.3 Why This Matters

Memory bandwidth is a precious resource on GPUs. Copying large tensors consumes:

1. **Memory bandwidth**: Data must be read and written
2. **Time**: Proportional to tensor size
3. **Power**: Memory operations are energy-intensive

For a tensor of size `[B, S, A, 3]` = `[256, 120, 6, 3]` = 552,960 elements:
- At 8 bytes per element (int64): ~4.4 MB per copy
- At 1600 iterations: ~7 GB of memory traffic just for copies!

---

## 2. Where Copies Happen in Compiled Code

### 2.1 Explicit Copies (Our Code)

These are the copies WE write:

```python
# After compiled step returns
torch._foreach_copy_(
    [self._current, self._derived, self._counts, ...],
    [new_cur, new_derived, new_counts, ...]
)
```

We can see these in profiler output as `_foreach_copy_` or `aten::copy_`.

### 2.2 Implicit Copies (PyTorch Internals)

With `mode='reduce-overhead'`, PyTorch internally copies arguments into "placeholder" tensors before replaying the CUDA graph:

```
Your call:
    result = compiled_step(self._derived)
                              │
                              ▼
PyTorch internally:
    placeholder.copy_(self._derived)  # Hidden copy!
    result = graph.replay()
```

This is visible in profiler as `cudagraph_trees.py:_copy_inputs_and_remove_from_src`.

### 2.3 Memory Allocation Overhead

Even if we avoid copies, allocation itself has overhead:

```python
# Inside compiled function
new_derived = torch.full((B, S, A, 3), pad, ...)  # Allocation!
```

With CUDA graphs, allocations are captured and replayed efficiently. But the first capture incurs allocation time.

---

## 3. The Three Copy Patterns

### 3.1 Pattern 1: Captured References + Explicit Copy

This is what V10 uses:

```python
class MinimalEvalV10:
    def __init__(self):
        self._derived = torch.zeros(B, S, A, 3)  # Persistent buffer
    
    def compile(self):
        derived = self._derived  # Capture REFERENCE
        
        def step_fn():
            # Read from captured reference
            logits = policy(derived)
            new_derived = engine.compute(...)
            return new_derived
        
        self._compiled = torch.compile(step_fn, ...)
    
    def evaluate(self):
        result = self._compiled()
        self._derived.copy_(result)  # EXPLICIT copy
```

**Pros:**
- Clear and explicit
- Full control over what gets copied
- Easy to debug

**Cons:**
- Copy overhead

### 3.2 Pattern 2: Arguments + Internal Copy

Pass tensors as arguments instead of capturing:

```python
def compile(self):
    def step_fn(derived_input):  # Argument, not captured
        logits = policy(derived_input)
        new_derived = engine.compute(...)
        return new_derived
    
    self._compiled = torch.compile(step_fn, ...)

def evaluate(self):
    derived = initial_derived
    while not done:
        # PyTorch internally copies derived into placeholder
        derived = self._compiled(derived)  # Use returned directly
```

**Pros:**
- No explicit copy code
- Slightly cleaner

**Cons:**
- PyTorch still copies internally
- Less control over copy behavior
- Can be harder to debug

### 3.3 Pattern 3: Double Buffering + In-Place

The most advanced pattern, achieving true zero-copy:

```python
class ZeroCopyEval:
    def __init__(self):
        # TWO buffers for each state
        self._derived_A = torch.zeros(B, S, A, 3)
        self._derived_B = torch.zeros(B, S, A, 3)
        self._read_from_A = True  # Which buffer to read
    
    def compile(self):
        derived_A = self._derived_A
        derived_B = self._derived_B
        
        def step_fn_A_to_B():
            # Read from A, write to B
            logits = policy(derived_A)
            engine.compute_inplace(output=derived_B)  # In-place write!
            return ...
        
        def step_fn_B_to_A():
            # Read from B, write to A
            logits = policy(derived_B)
            engine.compute_inplace(output=derived_A)
            return ...
        
        self._step_A_to_B = torch.compile(step_fn_A_to_B, ...)
        self._step_B_to_A = torch.compile(step_fn_B_to_A, ...)
    
    def evaluate(self):
        while not done:
            if self._read_from_A:
                self._step_A_to_B()
                self._read_from_A = False  # Just flip a boolean!
            else:
                self._step_B_to_A()
                self._read_from_A = True
```

**Pros:**
- Zero data copying
- Maximum performance

**Cons:**
- Double memory usage
- More complex code
- Requires in-place engine support
- Two compiled graphs instead of one

---

## 4. Double Buffering: Zero-Copy Pattern

### 4.1 The Core Idea

Instead of copying data between a temporary and a persistent buffer, we maintain TWO persistent buffers and alternate between them:

```
Iteration 1:
┌─────────────┐      ┌─────────────┐
│  Buffer A   │─────▶│   READ      │
│  [old data] │      │   (input)   │
└─────────────┘      └─────────────┘

┌─────────────┐      ┌─────────────┐
│  Buffer B   │◀─────│   WRITE     │
│  [new data] │      │   (output)  │
└─────────────┘      └─────────────┘

Iteration 2:
┌─────────────┐      ┌─────────────┐
│  Buffer B   │─────▶│   READ      │  (Was write buffer)
│  [new data] │      │   (input)   │
└─────────────┘      └─────────────┘

┌─────────────┐      ┌─────────────┐
│  Buffer A   │◀─────│   WRITE     │  (Was read buffer)
│  [     ]    │      │   (output)  │
└─────────────┘      └─────────────┘
```

### 4.2 Why It Works with CUDA Graphs

CUDA graphs require fixed addresses. With double buffering:
- Buffer A always at address 0x7f001000
- Buffer B always at address 0x7f002000

The graph that reads from A and writes to B is captured once. The graph that reads from B and writes to A is captured once. We alternate between replaying these two graphs.

No address changes, no copies needed!

### 4.3 The "Pointer Swap"

After each iteration, we don't copy data. We just change which buffer is considered "current":

```python
# This is NOT a data copy - just reassigning Python variables
self._read_buffer, self._write_buffer = self._write_buffer, self._read_buffer
```

This is O(1) - literally just swapping two pointers in Python.

### 4.4 Requirements for Double Buffering

1. **In-place write capability**: The computation must be able to write directly to a pre-allocated output buffer:
   ```python
   # Need this:
   engine.compute(input, output=buffer)  # Writes to buffer
   
   # Not this:
   result = engine.compute(input)  # Returns new tensor
   ```

2. **Separate read and write buffers**: The buffer being read cannot be the same as the buffer being written (that would be a mutation during the graph execution).

3. **Two compiled graphs**: One for A→B direction, one for B→A direction.

### 4.5 Memory Trade-off

Double buffering uses 2x memory for the buffered tensors:

```python
# Single buffer: 4.4 MB
self._derived = torch.zeros(B, S, A, 3)

# Double buffer: 8.8 MB
self._derived_A = torch.zeros(B, S, A, 3)
self._derived_B = torch.zeros(B, S, A, 3)
```

For V10 with `[256, 120, 6, 3]` derived states:
- Single: ~4.4 MB
- Double: ~8.8 MB
- Increase: +4.4 MB (negligible compared to model/GPU memory)

---

## 5. Index-Based Access: Avoiding Query Copies

### 5.1 The Problem with Query Copying

In the evaluation loop, we process queries one after another:

```python
for query in query_pool:
    self._current.copy_(query)  # Copy query into buffer
    result = compiled_step()
```

### 5.2 The Index-Based Solution

Instead of copying query data, pre-load ALL queries and use indices:

```python
class IndexBasedEval:
    def __init__(self):
        # All queries pre-loaded
        self._query_pool = torch.zeros(TOTAL_QUERIES, A, 3)
        # Current index per batch slot
        self._query_idx = torch.zeros(B, dtype=torch.long)
    
    def compile(self):
        query_pool = self._query_pool
        
        def step_fn(query_idx):
            # Get queries by indexing (advanced indexing, no full copy)
            current = query_pool[query_idx]
            
            # Process...
            result = process(current)
            
            # Update index, not data
            new_idx = query_idx + 1
            return result, new_idx
        
        self._compiled = torch.compile(step_fn, ...)
    
    def evaluate(self):
        idx = initial_indices
        while not done:
            result, idx = self._compiled(idx)
```

### 5.3 Why Indexing is Cheaper Than Copying

```python
# Full copy: Reads ALL data, writes ALL data
buffer.copy_(source)  # O(N) memory operations

# Indexing: Reads only needed elements
selected = pool[indices]  # O(B) memory operations, B << N
```

For a pool of 20,000 queries but batch of 256:
- Full copy overhead: O(20,000 × 3 × 8) = 480 KB per operation
- Index overhead: O(256 × 3 × 8) = 6 KB per operation

80x reduction in memory traffic for query access!

### 5.4 Why Index-Based Access Works for Queries

Queries are **pre-known** - we load all 20,000 candidates into memory BEFORE the evaluation loop starts:

```python
# BEFORE the loop - all data is known
self._query_pool[:total] = all_queries  # Load once

# DURING the loop - just use indices
current = pool[idx]  # No data copy, just indexing
```

The reason this works:

```
BEFORE EVALUATION:
┌─────────────────────────────────────────────────────────────┐
│  Query Pool (20,000 × 3)                                    │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐     │
│  │ Q0│ Q1│ Q2│ Q3│ Q4│ Q5│ Q6│ Q7│ Q8│ Q9│...│Q99│...│     │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘     │
│    ▲   ▲               ▲                       ▲            │
│    │   │               │                       │            │
│  idx[0]=0  idx[1]=1  idx[2]=4              idx[B-1]=99      │
│                                                              │
│  Indices tell us WHICH queries to process                   │
│  All data already exists in memory                          │
└─────────────────────────────────────────────────────────────┘
```

### 5.5 Why Index-Based Access DOESN'T Work for Derived States

Derived states are **computed dynamically** by the engine. They don't exist until we compute them:

```python
# Derived states depend on the CURRENT state
derived = engine.get_derived(current_state)

# The derived states for state X don't exist until we ask for them
# We can't pre-compute all possible derived states because:
# 1. There are exponentially many possible states
# 2. Each state can lead to different derived states
# 3. The proof search is dynamic - we can't predict which states we'll visit
```

#### The Engine Computation Explained

The unification engine takes a **current state** (a set of logical atoms) and produces **derived states** (possible next states after applying rules):

```
Current State: [parent(X, john), male(john)]
                        │
                        ▼
                   ┌─────────┐
                   │ ENGINE  │  ← Applies unification rules
                   │         │  ← Matches with knowledge base
                   │         │  ← Generates all possible derivations
                   └─────────┘
                        │
                        ▼
Derived States: [
    [grandparent(X, Z), parent(X, john), male(john)],  # Applied rule 1
    [ancestor(X, john), parent(X, john), male(john)],  # Applied rule 2
    [true],                                             # Proof found!
    ...up to 120 possible derived states
]
```

The derived states depend on:
1. **The current state** (which we just computed)
2. **The knowledge base** (facts and rules)
3. **The unification algorithm** (which rules apply)

Since the current state changes at every step, the derived states must be recomputed at every step.

#### Why We Can't Pre-Compute All Derived States

Consider the state space:

```
# For a simple problem:
# - 100 constants
# - 10 predicates
# - Max 3 atoms per state
# - Max arity 2

Possible states ≈ (predicates × constants²)³
                ≈ (10 × 100²)³
                ≈ 10^15 states!

Pre-computing derived for all states would require:
- 10^15 × 120 × 6 × 3 × 8 bytes
- ≈ 10^20 bytes
- ≈ 100 Exabytes (!)
```

This is obviously impossible. We MUST compute derived states on-demand.

#### The Fundamental Difference

```
QUERIES (Index-Based OK):           DERIVED STATES (Must Compute):
┌───────────────────────────┐       ┌───────────────────────────┐
│ Pre-known                 │       │ Computed dynamically      │
│ Finite set (20,000)       │       │ Depends on current state  │
│ Loaded once               │       │ Changes every step        │
│ Just need to select       │       │ Must call engine          │
│ Index access: O(B)        │       │ Engine call: O(B×rules)   │
└───────────────────────────┘       └───────────────────────────┘
         │                                    │
         ▼                                    ▼
   current = pool[idx]               derived = engine(current)
   (cheap indexing)                  (expensive computation)
```

#### What About Caching/Memoization?

You might think: "Can't we cache engine results?"

```python
# Hypothetical caching
cache = {}

def get_derived_cached(state):
    key = hash(state)
    if key in cache:
        return cache[key]
    result = engine.get_derived(state)
    cache[key] = result
    return result
```

Problems:
1. **Cache size**: Exponential number of possible states
2. **Cache misses**: Each proof path visits different states
3. **Hashing overhead**: Converting tensor to hashable key is expensive
4. **Memory pressure**: Cache competes with model/buffer memory
5. **GPU caching**: PyTorch tensors don't hash naturally on GPU

In practice, caching rarely helps because:
- Most states are visited only once (tree-like proof search)
- The engine computation is similar cost to cache lookup
- Cache management overhead adds complexity

---

## 6. Why Copying Wasn't Our Major Bottleneck

### 6.1 The V10 Profiler Results

From our GPU profiling:

```
Operation                    GPU Time    % of Total
──────────────────────────────────────────────────
Policy Forward (GEMM)        11.9s       45%
Engine Operations            6.6s        25%
torch.where operations       3.2s        12%
Miscellaneous Triton         4.3s        16%
_foreach_copy_               0.5s        2%     ← Small!
──────────────────────────────────────────────────
Total                        26.5s       100%
```

**Copy operations were only 2% of GPU time!**

### 6.2 Why Copies Were Fast in V10

Several factors made copying not a bottleneck:

1. **GPU Memory Bandwidth**: Modern GPUs (like the one used) have very high memory bandwidth (500+ GB/s). A 4.4 MB copy takes microseconds.

2. **Kernel Fusion**: `_foreach_copy_` batches multiple copies into one kernel, reducing launch overhead.

3. **Hiding Latency**: Copies overlap with other GPU work due to CUDA streams.

4. **Relative Cost**: The engine and policy were SO expensive that copies were noise in comparison.

### 6.3 When Copies ARE the Bottleneck

Copies become problematic when:

1. **Tensors are very large**: If `derived` was 100x bigger, copies would dominate.

2. **Copies happen frequently**: In a tight loop with 10,000+ iterations.

3. **Other operations are cheap**: If policy and engine were 10x faster, copies would be significant.

4. **Memory bandwidth saturated**: If many kernels compete for memory bandwidth.

### 6.4 The Bottleneck Hierarchy in V10

```
┌─────────────────────────────────────────────────────────────┐
│             V10 BOTTLENECK HIERARCHY                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  #1 Policy Forward (45%)                                     │
│  ├── GEMM (matrix multiply)                                  │
│  ├── LayerNorm                                               │
│  └── Activation (ReLU)                                       │
│                                                              │
│  #2 Engine Operations (25%)                                  │
│  ├── Scatter/Gather indexing                                 │
│  └── Unification logic                                       │
│                                                              │
│  #3 torch.where operations (12%)                             │
│  ├── Conditional selection                                   │
│  └── Mask operations                                         │
│                                                              │
│  #4 Miscellaneous (16%)                                      │
│  └── Various fused kernels                                   │
│                                                              │
│  #5 Memory Copies (2%)  ◀── NOT the bottleneck!              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

To achieve significant speedups, we focused on #1 and #2, not #5.

---

## 7. The Strategy We Used

### 7.1 The V10 Approach

Given that copies weren't the bottleneck, we chose the **Captured References + Explicit Copy** pattern because:

1. **Simplicity**: Easy to understand and debug
2. **Correctness**: Clear ownership of data
3. **Flexibility**: Easy to modify for experimentation
4. **Sufficient Performance**: Copy overhead was acceptable

### 7.2 The Code Pattern

```python
class MinimalEvalV10:
    def __init__(self):
        # Pre-allocate buffers
        self._current = torch.zeros(B, A, 3)
        self._derived = torch.zeros(B, S, A, 3)
        self._counts = torch.zeros(B)
        # ... more buffers
    
    def compile(self):
        # Capture references (not copies!)
        current = self._current
        derived = self._derived
        
        def step_fn(pool, pool_size):
            # Read from captured buffers
            obs = {'current': current, 'derived': derived}
            logits = policy(obs)
            
            # Compute new states
            new_current = ...
            new_derived = engine.get_derived(...)
            
            # Return NEW tensors
            return new_current, new_derived, ...
        
        self._compiled = torch.compile(step_fn, mode='reduce-overhead')
    
    def evaluate(self):
        while not done:
            torch.compiler.cudagraph_mark_step_begin()
            
            # Get new tensors from compiled step
            new_cur, new_der, ... = self._compiled(self._pool, self._pool_size)
            
            # Explicit copy into persistent buffers
            torch._foreach_copy_(
                [self._current, self._derived, ...],
                [new_cur, new_der, ...]
            )
```

### 7.3 Why We Didn't Use Double Buffering

1. **Copy overhead was only 2%**: Not worth the complexity
2. **Would require engine modification**: For in-place writes
3. **Double the memory**: For derived states
4. **Two graphs instead of one**: More complexity
5. **Marginal benefit**: Maybe 1-2% speedup, not worth it

### 7.4 Why We Didn't Use Index-Based Queries

The query pool was already accessed efficiently via indexing in the step function:

```python
# In step_fn
new_queries = pool[pool_idx]  # Already using indices!
```

No full copy was happening for queries anyway.

---

## 8. When to Optimize for Copies

### 8.1 Decision Framework

```
Is copy overhead > 10% of total time?
├── NO  ──▶ Don't optimize copies, focus elsewhere
└── YES ──▶ Continue...
            │
            ├─ Can you use index-based access?
            │   └─ For pre-known data like queries: Use indices
            │
            ├─ Can you modify the compute kernel for in-place?
            │   └─ If yes and memory allows: Use double buffering
            │
            └─ Neither possible?
                └─ Accept the copy overhead, optimize compute instead
```

### 8.2 Signs That Copies Are the Bottleneck

1. **Profiler shows > 10% time in copy operations**
2. **GPU utilization is low** (bandwidth-bound, not compute-bound)
3. **Increasing batch size doesn't help** (more data to copy)
4. **Reducing model size doesn't help proportionally** (copies dominate)

### 8.3 Signs That Copies Are NOT the Bottleneck

1. **Copy operations < 5% of time** (like V10)
2. **GPU compute utilization is high**
3. **Time is dominated by GEMM/convolutions**
4. **Reducing model size gives proportional speedup**

---

## 9. Implementation Considerations

### 9.1 If You Need Double Buffering

1. **Modify Engine for In-Place Output**:
   ```python
   class EngineWrapper:
       def get_derived_inplace(self, input, output):
           # Compute and write directly to output
           result = self._compute(input)
           output.copy_(result)  # In-place write
   ```

2. **Create Two Graphs**:
   ```python
   def compile(self):
       def step_A_to_B():
           engine.get_derived_inplace(buffer_A, buffer_B)
       
       def step_B_to_A():
           engine.get_derived_inplace(buffer_B, buffer_A)
       
       self._step_AtoB = torch.compile(step_A_to_B)
       self._step_BtoA = torch.compile(step_B_to_A)
   ```

3. **Alternate Graphs in Loop**:
   ```python
   use_A = True
   while not done:
       if use_A:
           self._step_AtoB()
       else:
           self._step_BtoA()
       use_A = not use_A
   ```

### 9.2 If You Need Index-Based Access

1. **Pre-allocate Query Pool**:
   ```python
   self._query_pool = torch.zeros(MAX_QUERIES, A, 3)
   ```

2. **Load All Queries Once**:
   ```python
   def setup(self, queries):
       self._query_pool[:len(queries)] = queries
   ```

3. **Use Indices in Step**:
   ```python
   def step_fn(query_idx):
       current = self._query_pool[query_idx]
       return process(current)
   ```

### 9.3 Measuring Copy Overhead

Use the GPU profiler to identify copy operations:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA]
) as prof:
    # Your evaluation loop
    pass

# Look for these in output:
# - aten::copy_
# - _foreach_copy_
# - Memcpy DtoD (Device -> Device)
print(prof.key_averages().table(sort_by='cuda_time_total'))
---

## 10. Inner Loop vs Whole Loop Compilation

This section explores a critical architectural decision: Should you compile just the step inside the loop, or compile the entire loop?

### 10.1 The Two Approaches

**Approach A: Compile Inner Step (Current V10)**
```python
def compile(self):
    def step_fn(pool, pool_size):
        # Single step logic
        logits = policy(current)
        new_state = select_action(logits)
        new_derived = engine(new_state)
        return new_state, new_derived, ...
    
    self._compiled_step = torch.compile(step_fn, mode='reduce-overhead')

def evaluate(self):
    for step in range(max_steps):
        # Each step: CUDA graph replay + copy
        torch.compiler.cudagraph_mark_step_begin()
        results = self._compiled_step(pool, pool_size)
        torch._foreach_copy_([buffers], [results])  # Copy every step!
        
        if done:
            break
```

**Approach B: Compile Entire Loop**
```python
def compile(self):
    def full_loop_fn(pool, pool_size, max_steps):
        current = initial
        derived = initial_derived
        
        for step in range(max_steps):  # Loop INSIDE compiled function
            logits = policy(current)
            new_state = select_action(logits)
            derived = engine(new_state)
            current = new_state
            
            if check_done():  # Can't early exit!
                pass  # Must continue anyway
        
        return results
    
    self._compiled_loop = torch.compile(full_loop_fn, mode='reduce-overhead')

def evaluate(self):
    # Single call, no intermediate copies!
    results = self._compiled_loop(pool, pool_size, max_steps)
```

### 10.2 Comparison Table

| Aspect | Inner Step Compiled | Whole Loop Compiled |
|--------|---------------------|---------------------|
| CUDA graph replays | 1600 replays | 1 replay |
| Copies | 1600 × `_foreach_copy_` | 0 intermediate copies |
| Python overhead | Loop in Python | Loop in compiled code |
| Early exit | ✓ (can break when done) | ✗ (must run all iterations) |
| Dynamic iteration count | ✓ (run until done) | ✗ (fixed iteration count) |
| Control flow | Easy | Limited by compiler |
| Memory | Minimal | Must hold all intermediate states |
| Debugging | Easy | Hard (inside compiled code) |

### 10.3 Why Whole Loop Compilation is Attractive

**Eliminates per-step overhead:**

```
Inner Step (1600 steps):
┌─────────────────────────────────────────────────────────────┐
│ For each of 1600 steps:                                     │
│   1. Python loop overhead (~1μs)                            │
│   2. cudagraph_mark_step_begin() (~5μs)                     │
│   3. CUDA graph replay (~50μs)                              │
│   4. _foreach_copy_ (~100μs)                                │
│   5. Check if done (~10μs)                                  │
│                                                              │
│   Total overhead per step: ~166μs                           │
│   Total overhead: 1600 × 166μs = 266ms                      │
└─────────────────────────────────────────────────────────────┘

Whole Loop Compiled:
┌─────────────────────────────────────────────────────────────┐
│ Single CUDA graph containing:                               │
│   1. All 1600 iterations fused                              │
│   2. No Python in between                                   │
│   3. No intermediate copies                                 │
│                                                              │
│   Total overhead: ~50μs (single graph launch)               │
│   Savings: ~266ms                                           │
└─────────────────────────────────────────────────────────────┘
```

### 10.4 Why Whole Loop Compilation is HARD

#### Problem 1: Dynamic Termination

The evaluation loop should exit when ALL batch slots are done. But this requires reading tensor values:

```python
for step in range(max_steps):
    ...
    if self._done.all():  # GRAPH BREAK! Reads tensor value
        break
```

With whole loop compilation:
- Can't have data-dependent control flow
- Can't break early
- Must run fixed number of iterations

**Workaround:** Run a fixed `max_steps` that's guaranteed to be enough:

```python
# Run max_steps even if done early
for step in range(max_steps):
    # Mask out done slots instead of breaking
    active = ~done
    new_state = torch.where(active, computed, current)
```

But this wastes compute on already-done slots.

#### Problem 2: Graph Size

A loop with 1600 iterations, each doing:
- Policy forward (8 layers)
- Engine call (complex indexing)
- Multiple torch.where operations

Results in a MASSIVE graph:
- 1600 × (policy + engine + masks)
- Could be millions of operations
- Graph capture time: potentially minutes
- GPU memory for graph: potentially GBs

#### Problem 3: No Intermediate Inspection

With inner step compilation:
```python
for step in range(max_steps):
    result = compiled_step()
    
    # Can inspect intermediate results!
    print(f"Step {step}: {result['done'].sum()} done")
    
    # Can record metrics mid-loop
    if step % 100 == 0:
        log_metrics(result)
```

With whole loop compilation:
```python
result = compiled_loop()  # Black box!
# Can only see final result
# No intermediate visibility
```

#### Problem 4: Variable-Length Computation

Different queries take different numbers of steps to resolve:
- Query A: Proved in 3 steps
- Query B: Proved in 15 steps
- Query C: Truncated at 20 steps

With whole loop:
```python
# Must run 20 steps for ALL queries
# Even if Query A was done after 3 steps
# Wasted compute: 17 steps × (cost of Query A slot)
```

With inner step:
```python
# After Query A is done, its slot gets recycled for a new query
# Maximum parallelism maintained
```

### 10.5 Unrolled Loop: A Middle Ground

Instead of one step or all steps, compile k steps:

```python
def compile(self):
    def k_steps_fn(current, derived, k=20):  # Unroll 20 steps
        for _ in range(k):
            logits = policy(current)
            action = logits.argmax(-1)
            current = derived[action]
            derived = engine(current)
        return current, derived
    
    self._compiled_k_steps = torch.compile(k_steps_fn, mode='reduce-overhead')

def evaluate(self):
    for batch in range(max_steps // k):
        torch.compiler.cudagraph_mark_step_begin()
        current, derived = self._compiled_k_steps(current, derived)
        
        # Copy and check every k steps (less frequent!)
        self._current.copy_(current)
        self._derived.copy_(derived)
        
        if self._done.all():
            break
```

**Benefits:**
- Reduce copies by k-fold (e.g., 80 copies instead of 1600 for k=20)
- Still allows early exit every k steps
- Graph size is k× not 1600×
- More compute between syncs

**Drawbacks:**
- Wasted compute for slots that finish mid-k
- Graph is larger (k× single step)
- Complexity: choosing optimal k

### 10.6 Memory Implications

**Inner Step (V10 current):**
```
Memory per step:
- Input buffers: B × S × A × 3 = 4.4 MB (read)
- Output tensors: B × S × A × 3 = 4.4 MB (new each step)
- Total live: ~9 MB + model

After copy_():
- Output tensors can be freed
- Memory stable at ~9 MB + model
```

**Whole Loop Compiled:**
```
Memory for full loop:
- Input buffers: 4.4 MB
- Intermediate states: 1600 × 4.4 MB = 7 GB (!!)
  (Each iteration produces tensors that may be needed)
- OR if reusing: still need space for graph

This can EASILY exceed GPU memory!
```

PyTorch's Inductor tries to optimize memory, but with 1600 loop iterations, it's challenging.

### 10.7 What V10 Chose and Why

V10 uses **Inner Step Compilation** because:

1. **Dynamic termination**: Need to recycle slots when proofs finish
2. **Memory efficiency**: Can't hold 1600 iterations of state
3. **Debugging**: Can inspect intermediate results
4. **Flexibility**: Easy to modify step logic
5. **Reasonable overhead**: Copy overhead was only 2%

The 2% copy overhead is acceptable compared to:
- 45% policy forward (actual bottleneck)
- 25% engine (actual bottleneck)

### 10.8 When Whole Loop Compilation Makes Sense

Whole loop compilation CAN work when:

1. **Fixed, small iteration count**: e.g., 10-20 steps max
2. **No early termination needed**: All queries take similar time
3. **Small state size**: Intermediate states fit in memory
4. **Uniform workload**: No slot recycling complexity

Example: Simple RL with fixed episode length:
```python
# Fixed 100 steps, no early termination
@torch.compile(mode='reduce-overhead')
def episode_fn(state):
    for _ in range(100):
        action = policy(state)
        state = env.step(action)  # Simple state update
    return state

# This CAN work because:
# - Fixed iteration count
# - Small state (just observation)
# - No complex slot management
```

### 10.9 Hybrid Approach for V10

A potential future optimization:

```python
def evaluate(self):
    # Phase 1: Bulk processing (whole loop for simple cases)
    easy_batch = queries[:easy_count]
    easy_results = compiled_short_loop(easy_batch, max_steps=5)
    
    # Phase 2: Complex queries (inner step for flexibility)
    hard_batch = queries[easy_count:]
    for step in range(max_steps):
        result = compiled_step(hard_batch)
        ...
```

Identify "easy" queries (short proofs) and batch them with whole loop. Handle "hard" queries with flexible inner step.

---

## Summary

### Key Takeaways

1. **Copies happen in compiled code** due to the CUDA graph fixed-address requirement

2. **Three patterns exist**: Captured + Copy, Arguments, Double Buffering

3. **Double buffering achieves zero-copy** but requires in-place kernel support

4. **In V10, copies were only 2%** of GPU time - not the bottleneck

5. **We used the simple pattern** (captured + explicit copy) because:
   - Complexity wasn't justified
   - Other operations (policy, engine) dominated
   - Clarity and maintainability mattered

6. **Optimize copies only when they're > 10%** of total time

### The V10 Optimization Philosophy

> "Optimize the bottleneck, not the noise."

Copy overhead was noise. Policy forward and engine operations were the bottlenecks. That's where we focused our optimization effort (compiled engine, torch.cond, etc.).

### Future Work

If V10 is further optimized and policy/engine become faster, copies might become significant. At that point:

1. Implement index-based query access (moderate effort, moderate gain)
2. Consider double buffering (high effort, high gain)
3. Profile again to verify new bottlenecks

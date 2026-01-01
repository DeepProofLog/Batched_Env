# PyTorch Compiled Evaluation: Expert Deep Dive

## Part 1: Foundations - Memory, Tensors, and Addresses

This document provides an expert-level understanding of PyTorch tensor memory management, compilation with `torch.compile`, CUDA graphs, and optimization strategies. Based on extensive work optimizing the V10 evaluation system.

---

## Table of Contents

1. [Tensor Memory Fundamentals](#1-tensor-memory-fundamentals)
2. [Copying vs Views vs Cloning](#2-copying-vs-views-vs-cloning)
3. [In-Place Operations and Mutations](#3-in-place-operations-and-mutations)
4. [CUDA Graphs and Memory Addresses](#4-cuda-graphs-and-memory-addresses)
5. [torch.compile Internals](#5-torchcompile-internals)
6. [Aliasing: The Silent Killer](#6-aliasing-the-silent-killer)
7. [Compilation Strategies](#7-compilation-strategies)
8. [CPU-GPU Synchronization Deep Dive](#8-cpu-gpu-synchronization-deep-dive)
9. [Profiling and Bottleneck Analysis](#9-profiling-and-bottleneck-analysis)
10. [TensorDict and Structured State](#10-tensordict-and-structured-state)
11. [V10 Architecture Analysis](#11-v10-architecture-analysis)
12. [Best Practices Checklist](#12-best-practices-checklist)

---

## 1. Tensor Memory Fundamentals

### 1.1 What is a Tensor Really?

A PyTorch tensor is NOT the data itself - it's a **view** into a contiguous block of memory. A tensor consists of:

```
┌─────────────────────────────────────────────────────────────┐
│                     TENSOR OBJECT                           │
├─────────────────────────────────────────────────────────────┤
│  data_ptr      → Points to actual memory buffer             │
│  shape         → (B, S, A, 3) - logical dimensions          │
│  stride        → (S*A*3, A*3, 3, 1) - memory layout         │
│  dtype         → torch.long, torch.float32, etc.            │
│  device        → cuda:0, cpu                                │
│  storage       → The underlying Storage object              │
│  requires_grad → Whether to track gradients                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    GPU MEMORY                               │
│  Address: 0x7f8a12340000                                    │
│  Size: B * S * A * 3 * sizeof(long)                         │
│  [data][data][data][data][data][data]...                    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The data_ptr() Function

Every tensor has a `data_ptr()` that returns the memory address of its first element:

```python
x = torch.zeros(256, 120, 6, 3, device='cuda')
print(f"Memory address: {x.data_ptr()}")
# Output: Memory address: 140234567890432

# After in-place operation - SAME address
x.fill_(1)
print(f"After fill_: {x.data_ptr()}")
# Output: After fill_: 140234567890432  (unchanged!)

# After creating new tensor - DIFFERENT address
y = x + 1
print(f"y address: {y.data_ptr()}")
# Output: y address: 140234567891456  (different!)
```

### 1.3 Why Addresses Matter for CUDA Graphs

CUDA graphs capture a sequence of GPU operations and their memory addresses. When you replay a graph:

1. The EXACT SAME memory addresses are used
2. Operations read from and write to those specific addresses
3. If you've changed what's at those addresses, you get garbage

```python
# CUDA Graph capture
with torch.cuda.graph(g):
    y = model(x)  # x at address A, y at address B

# Replay expects:
# - Input at address A
# - Output written to address B
g.replay()  # Works!

# If x is reallocated:
x = torch.zeros_like(x)  # NEW address C!
g.replay()  # WRONG! Graph still reads from address A
```

---

## 2. Copying vs Views vs Cloning

### 2.1 The Spectrum of Tensor Operations

```
┌────────────────────────────────────────────────────────────────────┐
│                    TENSOR OPERATION SPECTRUM                        │
├──────────────┬─────────────┬─────────────┬────────────────────────┤
│   View       │   Shallow   │    Clone    │     Deep Copy          │
│              │    Copy     │             │                        │
├──────────────┼─────────────┼─────────────┼────────────────────────┤
│ Same memory  │ New tensor  │ New memory  │ New memory +           │
│ Same data    │ Same data   │ Copied data │ Copied data            │
│ No copy      │ No copy     │ Full copy   │ (recursive for nested) │
├──────────────┼─────────────┼─────────────┼────────────────────────┤
│ x.view()     │ x[:]        │ x.clone()   │ copy.deepcopy(x)       │
│ x.reshape()  │ x.detach()  │ x.detach(). │ TensorDict.clone()     │
│ x.transpose()│             │    clone()  │                        │
│ x.unsqueeze()│             │             │                        │
└──────────────┴─────────────┴─────────────┴────────────────────────┘
```

### 2.2 Views: Zero-Copy Tensor Operations

A view shares memory with the original tensor:

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = x.view(6)  # Same memory!

print(f"x.data_ptr() = {x.data_ptr()}")
print(f"y.data_ptr() = {y.data_ptr()}")
# Both print the SAME address!

y[0] = 999
print(x[0, 0])  # Prints 999! x was modified through y
```

**View operations:**
- `view()`, `reshape()` (when contiguous)
- `transpose()`, `permute()`, `t()`
- `squeeze()`, `unsqueeze()`
- `expand()` (broadcasts without copying)
- Slicing: `x[0]`, `x[:, 1:3]`, `x[::2]`

### 2.3 clone() vs copy_()

These are fundamentally different:

```python
# clone() - Creates NEW tensor with NEW memory, copies data
x = torch.tensor([1, 2, 3], device='cuda')
y = x.clone()
print(x.data_ptr() == y.data_ptr())  # False! Different memory

# copy_() - Copies data INTO existing tensor
z = torch.empty(3, device='cuda')
z.copy_(x)
print(z.data_ptr() == x.data_ptr())  # False! Different memory
# But z now contains [1, 2, 3]
```

**Key differences:**

| Aspect | `clone()` | `copy_()` |
|--------|-----------|-----------|
| Returns | New tensor | None (in-place) |
| Memory | Allocates new | Uses existing |
| Address stability | New address each time | Target address unchanged |
| CUDA graph safe | Creates new address | Preserves target address ✓ |

### 2.4 torch._foreach_copy_: Batched Copying

For copying multiple tensors efficiently:

```python
# Individual copies (slower)
self._current.copy_(new_cur)
self._derived.copy_(new_der)
self._counts.copy_(new_cnt)

# Batched copy (faster, single kernel launch)
torch._foreach_copy_(
    [self._current, self._derived, self._counts],
    [new_cur, new_der, new_cnt]
)
```

**Why it's faster:**
- Single kernel launch vs multiple
- Better memory access patterns
- Reduced CPU overhead

---

## 3. In-Place Operations and Mutations

### 3.1 What is an In-Place Operation?

In-place operations modify the tensor's DATA without changing its memory address:

```python
x = torch.zeros(3)
print(f"Before: {x.data_ptr()}")

x.add_(1)  # In-place add (note the underscore!)
print(f"After add_: {x.data_ptr()}")  # SAME address!

x[:] = torch.ones(3)  # In-place assignment
print(f"After [:]=: {x.data_ptr()}")  # SAME address!

x.fill_(5)  # In-place fill
print(f"After fill_: {x.data_ptr()}")  # SAME address!
```

### 3.2 The Underscore Convention

PyTorch uses `_` suffix for in-place operations:

```python
# Out-of-place (returns new tensor)    # In-place (modifies tensor)
y = x.add(1)                           x.add_(1)
y = x.mul(2)                           x.mul_(2)
y = x.fill(5)                          x.fill_(5)
y = x.zero()                           x.zero_()
y = torch.where(c, a, b)               # No in-place version!
```

### 3.3 Why Mutations Break CUDA Graphs

When `torch.compile(mode='reduce-overhead')` captures a CUDA graph:

```python
def step_fn():
    self._current[:, 0, :] = new_query  # MUTATION!
    return result

compiled = torch.compile(step_fn, mode='reduce-overhead')
```

**Problem:** The graph captures operations reading from `self._current` at address A. But the mutation changes the data at address A before the graph is replayed. On replay:
1. Graph reads from address A (expecting old data)
2. But address A now has different data from previous iteration
3. Results are incorrect!

**Solution: Return new tensors, copy outside:**

```python
def step_fn():
    # Create NEW tensor instead of mutating
    new_current = torch.where(mask, new_query, self._current)
    return new_current  # Return NEW tensor

compiled = torch.compile(step_fn, mode='reduce-overhead')
result = compiled()
self._current.copy_(result)  # Mutation OUTSIDE compiled region
```

---

## 4. CUDA Graphs and Memory Addresses

### 4.1 How CUDA Graphs Work

```
Step 1: CAPTURE                    Step 2: REPLAY
┌─────────────────────────┐       ┌─────────────────────────┐
│ Record operations:      │       │ Execute cached ops:     │
│                         │       │                         │
│ kernel1(addr_A)         │  ──▶  │ kernel1(addr_A)         │
│ kernel2(addr_A, addr_B) │       │ kernel2(addr_A, addr_B) │
│ kernel3(addr_B, addr_C) │       │ kernel3(addr_B, addr_C) │
│                         │       │                         │
│ Addresses are FROZEN!   │       │ Uses SAME addresses!    │
└─────────────────────────┘       └─────────────────────────┘
```

### 4.2 torch.compile with reduce-overhead

When you use `mode='reduce-overhead'`:

```python
@torch.compile(mode='reduce-overhead', fullgraph=True)
def step_fn(x):
    return model(x)
```

PyTorch:
1. Traces the function
2. Compiles to optimized kernels
3. **Captures as CUDA graph** (if possible)
4. On subsequent calls, replays the graph

### 4.3 When CUDA Graphs Break

```python
# ❌ Dynamic shapes
def bad1(x):
    return x[:x.shape[0]//2]  # Shape depends on input!

# ❌ Mutations
def bad2():
    self.buffer[:] = new_data  # Mutates existing tensor!

# ❌ Control flow based on tensor values
def bad3(x):
    if x.sum() > 0:  # Requires reading tensor value!
        return x * 2
    return x

# ❌ Python side effects
def bad4(x):
    print(x.shape)  # Python print!
    return x * 2
```

### 4.4 The Buffer-Copy Pattern

The solution used in V10:

```python
class MinimalEvalV10:
    def __init__(self):
        # Pre-allocate buffers with FIXED addresses
        self._current = torch.zeros(B, A, 3)  # Address A
        self._derived = torch.zeros(B, S, A, 3)  # Address B
    
    def compile(self):
        current = self._current  # Capture reference
        derived = self._derived
        
        def step_fn():
            # READ from fixed addresses
            new = process(current, derived)
            # Return NEW tensor (different address)
            return new
        
        self._compiled = torch.compile(step_fn, mode='reduce-overhead')
    
    def evaluate(self):
        result = self._compiled()  # Returns new tensor at address X
        self._current.copy_(result)  # Copy INTO fixed buffer (address A)
        # Next iteration reads from address A again!
```

---

## 5. torch.compile Internals

### 5.1 The Compilation Pipeline

```
Python Function
      │
      ▼
┌─────────────────┐
│    Dynamo       │  ← Traces Python bytecode
│   (Tracer)      │  ← Captures tensor operations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FX Graph      │  ← Intermediate representation
│                 │  ← Nodes represent operations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Inductor     │  ← Generates optimized code
│   (Backend)     │  ← Fuses operations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Triton Kernels │  ← Custom CUDA kernels
│  + CUDA Graphs  │  ← Captured for replay
└─────────────────┘
```

### 5.2 Compilation Modes

```python
# Default mode - balanced
torch.compile(fn)

# Reduce overhead - uses CUDA graphs for minimal launch overhead
torch.compile(fn, mode='reduce-overhead')

# Max autotune - tries many implementations, picks fastest
torch.compile(fn, mode='max-autotune')

# Full graph - errors if graph break occurs
torch.compile(fn, fullgraph=True)

# No dynamic shapes - assumes shapes are static
torch.compile(fn, dynamic=False)
```

### 5.3 Graph Breaks

A "graph break" occurs when Dynamo cannot continue tracing:

```python
def fn(x):
    y = x * 2          # ✓ Traced
    print(x.shape)     # ✗ GRAPH BREAK! Python print
    z = y + 1          # ✓ New graph starts
    return z
```

**Common causes:**
- Python print/logging
- `if tensor.item() > 0:` (data-dependent control flow)
- Unsupported operations
- Calling non-compilable functions
- Dynamic shapes

**Detection:**
```python
import torch._dynamo as dynamo
dynamo.config.verbose = True

# Or set environment variable
# TORCH_LOGS="+dynamo"
```

### 5.4 Compiling Nested Functions

You can compile functions that call other functions:

```python
def inner(x):
    return x * 2

def outer(x):
    y = inner(x)  # This gets inlined and compiled together!
    return y + 1

compiled = torch.compile(outer)  # Both inner and outer compiled
```

**But be careful with class methods:**

```python
class Model:
    def forward(self, x):
        return self.layer(x)
    
    def layer(self, x):
        return x * self.weight

# Option 1: Compile the whole forward
compiled_forward = torch.compile(model.forward)

# Option 2: Compile specific methods
model.layer = torch.compile(model.layer)
```

---

## 6. Aliasing: The Silent Killer

### 6.1 What is Aliasing?

Aliasing occurs when two names/references point to the same memory:

```python
x = torch.tensor([1, 2, 3])
y = x  # ALIASING! y and x point to same memory

y[0] = 999
print(x[0])  # 999! x was modified through y
```

### 6.2 Input-Output Aliasing in torch.cond

`torch.cond` has strict rules about aliasing:

```python
def branch_fn(a, b, c):
    return a, b, c  # ❌ Returns inputs directly = ALIASING

# Error: Input-to-output aliasing detected
```

**The fix:**

```python
def branch_fn(a, b, c):
    return a.clone(), b.clone(), c.clone()  # ✓ New tensors
```

### 6.3 Why Aliasing Breaks Compilation

When a compiled function returns an alias:
1. The graph expects input at address A
2. The graph produces output at address A (same tensor!)
3. On next iteration, input is modified, but graph cached the address
4. Chaos ensues

### 6.4 Detecting Aliasing

```python
def check_aliasing(x, y):
    # Check if same underlying storage
    return x.storage().data_ptr() == y.storage().data_ptr()

x = torch.tensor([1, 2, 3])
y = x
z = x.clone()

print(check_aliasing(x, y))  # True (aliased)
print(check_aliasing(x, z))  # False (independent)
```

---

## 7. Compilation Strategies

### 7.1 Strategy Comparison (V10)

| Strategy | torch.cond | fused |
|----------|------------|-------|
| Engine calls | Conditional (~10% of steps) | Every step |
| CUDA graphs | Breaks (3 partitions) | Works |
| Runtime | ~21s | ~28s |
| ms/candidate | 1.04 | 1.38 |
| Complexity | Higher | Lower |

### 7.2 torch.cond Deep Dive

`torch.cond` enables conditional execution in compiled graphs:

```python
result = torch.cond(
    predicate,   # Boolean tensor (scalar)
    true_fn,     # Called if pred is True
    false_fn,    # Called if pred is False
    operands     # Tuple of inputs to both functions
)
```

**Requirements:**
1. `predicate` must be a scalar boolean tensor
2. Both branches must have same input signature
3. Both branches must return same number and shapes of tensors
4. NO input-output aliasing (use .clone())
5. NO mutations of inputs or globals

**Example from V10:**

```python
def reset_branch(new_current, derived, counts, ...):
    # Engine call happens here
    reset_derived, reset_counts = engine.get_derived(reset_queries)
    return torch.where(...), torch.where(...), ...

def no_reset_branch(new_current, derived, counts, ...):
    # Skip engine - just return (cloned!)
    return new_current.clone(), derived.clone(), counts.clone(), ...

result = torch.cond(
    needs_reset.any(),  # Predicate
    reset_branch,
    no_reset_branch,
    (new_current, derived, counts, ...)
)
```

### 7.3 Why torch.cond Breaks CUDA Graphs

```
Without torch.cond:              With torch.cond:
┌────────────────────┐           ┌────────────────────┐
│   Single CUDA      │           │   CUDA Graph 1     │
│   Graph            │           ├────────────────────┤
│   (all ops)        │           │   CPU: evaluate    │
│                    │           │   predicate        │──▶ SYNC!
└────────────────────┘           ├────────────────────┤
                                 │   CUDA Graph 2a    │
                                 │   OR 2b (branch)   │
                                 └────────────────────┘
```

The conditional requires:
1. Evaluating the predicate (CPU must read GPU value)
2. Choosing which branch to execute
3. This forces CPU-GPU synchronization

---

## 8. CPU-GPU Synchronization Deep Dive

### 8.1 Asynchronous Execution Model

```
CPU Timeline:
────●───────●───────●───────●───────●───────▶
   launch  launch  launch  launch  launch
   kernel1 kernel2 kernel3 kernel4 kernel5

GPU Timeline:
════════════●══════●══════●══════●══════●═══▶
            exec1  exec2  exec3  exec4  exec5
            
           ◄─────────────────────────────────▶
                    GPU runs in parallel
```

### 8.2 Synchronization Points

Certain operations force the CPU to wait for GPU:

```python
# These all cause synchronization:
x.item()           # Copy single value to CPU
x.tolist()         # Copy entire tensor to CPU
x.numpy()          # Convert to numpy
print(x)           # Prints need values
if x.any():        # Boolean check needs value
x.sum().item()     # Reduction + copy

# Explicit sync:
torch.cuda.synchronize()
```

### 8.3 Profiler Evidence

From our V10 profiling:

```
Name                                CPU time    % 
aten::_local_scalar_dense           483ms       78%
```

This is the `.any()` call! Each call costs ~8.8ms of sync time.

### 8.4 The .any() Trade-off

```python
# With .any() check:
if needs_reset.any():  # Sync cost: ~0.1ms
    engine_call()       # Only called 200 times

# Without .any() check:
engine_call()           # Called 1600 times!
```

**Math:**
- With `.any()`: 200 × 0.1ms (sync) + 200 × 7ms (engine) = 1.42s
- Without `.any()`: 1600 × 7ms (engine) = 11.2s

**Conclusion:** Sync is expensive but engine is more expensive. Keep `.any()`.

---

## 9. Profiling and Bottleneck Analysis

### 9.1 CPU Profiler (cProfile)

Shows Python function call times:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... your code ...
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('tottime').print_stats(20)
```

**What it shows:**
- Time spent in Python functions
- Number of calls
- Per-call time

**Limitation:** For GPU code, shows time waiting for GPU, not actual GPU work.

### 9.2 GPU Profiler (torch.profiler)

Shows actual CUDA kernel times:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
) as prof:
    # ... your code ...

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
```

### 9.3 V10 Bottleneck Analysis

**GPU Profile (26.5s total):**

| Operation | Time | % | Notes |
|-----------|------|---|-------|
| cutlass GEMM | 7.0s | 27% | Policy matmul |
| triton layer_norm | 4.9s | 18% | Policy norms |
| scatter_gather | 3.2s | 12% | Engine indexing |
| Engine triton | 3.4s | 13% | Unification |
| Compiled region | 26.3s | 99% | Total GPU |

**Interpretation:**
- Policy forward (GEMM + norms) = 45% of GPU time
- Engine operations = 25% of GPU time
- Memory copies (`_foreach_copy_`) = <1%

### 9.4 Reading Triton Kernel Names

The profiler shows fused kernel names:

```
triton_per_fused_add_addmm_native_layer_norm_relu_view_6
```

Breakdown:
- `triton_per_` - Persistent kernel
- `fused_` - Multiple ops fused
- `add_addmm_native_layer_norm_relu_view` - The fused ops
- `_6` - Variant number

---

## 10. TensorDict and Structured State

### 10.1 Why TensorDict?

Managing many state tensors is error-prone:

```python
# Manual approach (V10)
self._current = torch.zeros(...)
self._derived = torch.zeros(...)
self._counts = torch.zeros(...)
# ... many more ...

# Copy all
self._current.copy_(new_cur)
self._derived.copy_(new_der)
# Easy to forget one!
```

### 10.2 TensorDict Benefits

```python
from tensordict import TensorDict

state = TensorDict({
    'current': torch.zeros(B, A, 3),
    'derived': torch.zeros(B, S, A, 3),
    'counts': torch.zeros(B),
}, batch_size=[B])

# Clone entire state
new_state = state.clone()

# Update all at once
state.update(new_state)

# Works with torch.compile!
```

### 10.3 TensorDict and Compilation

TensorDict is designed for compilation:

```python
@torch.compile
def step(state: TensorDict) -> TensorDict:
    new_current = process(state['current'])
    return TensorDict({
        'current': new_current,
        'derived': state['derived'],
    })
```

---

## 11. V10 Architecture Analysis

### 11.1 File: eval_minimal_v10.py

Key components:

```python
class MinimalEvalV10:
    """
    Strategy parameter chooses execution mode:
    - 'torch_cond': Conditional engine (21s, 1.04 ms/cand)
    - 'fused': Engine every step (28s, 1.38 ms/cand)
    """
    
    def __init__(self, ..., strategy='torch_cond'):
        # Pre-allocate ALL buffers here
        self._current = torch.full((B, A, 3), pad, ...)
        self._derived = torch.full((B, S, A, 3), pad, ...)
        # These addresses are FIXED for CUDA graph compatibility
    
    def compile(self):
        # Capture buffer references (not copies!)
        current = self._current
        derived = self._derived
        
        def step_fn(pool, pool_size):
            # READ from captured references
            # RETURN new tensors (don't mutate!)
            return new_current, new_derived, ...
        
        self._compiled_step = torch.compile(step_fn, ...)
    
    def evaluate(self):
        while not done:
            # Mark graph boundary
            torch.compiler.cudagraph_mark_step_begin()
            
            # Call compiled function (returns NEW tensors)
            results = self._compiled_step(self._pool, self._pool_size)
            
            # Copy INTO buffers (mutations OUTSIDE compiled region)
            torch._foreach_copy_([self._current, ...], [results[0], ...])
```

### 11.2 Memory Layout

```
Buffer Addresses (Fixed at __init__):
┌─────────────────────────────────────────────────┐
│ self._current    → 0x7f0001000000 (B×A×3)       │
│ self._derived    → 0x7f0002000000 (B×S×A×3)     │
│ self._counts     → 0x7f0003000000 (B)           │
│ self._pool       → 0x7f0004000000 (max_pool×3)  │
│ self._result_buf → 0x7f0005000000 (max_pool)    │
└─────────────────────────────────────────────────┘

Compiled Step Returns (New addresses each call):
┌─────────────────────────────────────────────────┐
│ new_current  → 0x7f0010000000 (temporary)       │
│ new_derived  → 0x7f0011000000 (temporary)       │
│ new_counts   → 0x7f0012000000 (temporary)       │
└─────────────────────────────────────────────────┘

After copy_():
┌─────────────────────────────────────────────────┐
│ self._current ← data from new_current           │
│ (address unchanged: 0x7f0001000000)             │
└─────────────────────────────────────────────────┘
```

---

## 12. Best Practices Checklist

### 12.1 For CUDA Graph Compatibility

- [ ] Pre-allocate ALL buffers in `__init__`
- [ ] Capture buffer REFERENCES in compiled function (not copies)
- [ ] Return NEW tensors from compiled function (no mutations inside)
- [ ] Use `copy_()` or `_foreach_copy_()` OUTSIDE compiled region
- [ ] Use `torch.compiler.cudagraph_mark_step_begin()` at loop start
- [ ] Ensure STATIC shapes for all tensors
- [ ] Avoid `.item()`, `.any()`, `.sum()` inside compiled function

### 12.2 For torch.compile

- [ ] Use `mode='reduce-overhead'` for inference
- [ ] Use `fullgraph=True` to catch graph breaks early
- [ ] Use `dynamic=False` for static shapes
- [ ] Avoid Python control flow based on tensor values
- [ ] Avoid print statements inside compiled functions

### 12.3 For torch.cond

- [ ] Predicate must be scalar boolean tensor
- [ ] Both branches return same number/shapes of tensors
- [ ] NO input-output aliasing (use `.clone()`)
- [ ] NO mutations of inputs inside branches
- [ ] Be aware it breaks CUDA graphs

### 12.4 For Performance

- [ ] Profile with GPU profiler, not just CPU profiler
- [ ] Minimize sync points (`.item()`, `.any()`)
- [ ] Use `_foreach_copy_` for batched copies
- [ ] Consider trade-offs (sync overhead vs compute)
- [ ] Compile expensive functions separately

---

## Summary

The key insights from V10 optimization:

1. **Memory addresses are critical** for CUDA graphs
2. **Mutations inside compiled functions** break CUDA graphs
3. **Buffer-copy pattern** solves the mutation problem
4. **torch.cond** enables conditional execution but breaks CUDA graphs
5. **CPU-GPU sync** from `.any()` is expensive but sometimes necessary
6. **GPU profiler** reveals true bottlenecks (CPU profiler can mislead)
7. **Engine compilation** provides 4.3x speedup

The final V10 achieves **0.88 ms/candidate** using torch.cond strategy.

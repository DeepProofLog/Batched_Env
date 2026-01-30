# PyTorch Compiled Evaluation: Expert Deep Dive (V2)

## Part 1: Foundations - Memory, Tensors, and Addresses

This document provides an expert-level understanding of PyTorch tensor memory management, compilation with `torch.compile`, CUDA graphs, and optimization strategies. Based on extensive work optimizing the V10 evaluation system.

---

## Table of Contents

1. [Tensor Memory Fundamentals](#1-tensor-memory-fundamentals)
2. [Copying vs Views vs Cloning](#2-copying-vs-views-vs-cloning)
3. [In-Place Operations and Mutations](#3-in-place-operations-and-mutations)
4. [CUDA Graphs and Memory Addresses](#4-cuda-graphs-and-memory-addresses)
5. [cudagraph_mark_step_begin Explained](#5-cudagraph_mark_step_begin-explained)
6. [torch.compile Internals](#6-torchcompile-internals)
7. [Aliasing: The Silent Killer](#7-aliasing-the-silent-killer)
8. [Conditional Execution: .any() vs .where vs torch.cond](#8-conditional-execution-comparison)
9. [CPU-GPU Synchronization Deep Dive](#9-cpu-gpu-synchronization-deep-dive)
10. [Profiling and Bottleneck Analysis](#10-profiling-and-bottleneck-analysis)
11. [TensorDict Deep Dive](#11-tensordict-deep-dive)
12. [V10 Architecture Analysis](#12-v10-architecture-analysis)
13. [Best Practices Checklist](#13-best-practices-checklist)

---

## 1. Tensor Memory Fundamentals

### 1.1 What is a Tensor Really?

A PyTorch tensor is NOT the data itself - it's a **metadata wrapper** around a contiguous block of memory:

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

---

## 2. Copying vs Views vs Cloning

### 2.1 Complete Tensor Operations Reference

This is the DEFINITIVE table of all tensor copy/view operations:

| Operation | Memory Address | Data | Use Case | CUDA Graph Safe |
|-----------|---------------|------|----------|-----------------|
| **Views (Zero-Copy)** |
| `x.view(shape)` | Same | Shared | Reshape without copy | ✓ (if stable) |
| `x.reshape(shape)` | Same or New* | Shared or Copied | Flexible reshape | Depends |
| `x.transpose(0,1)` | Same | Shared | Reorder dims | ✓ |
| `x.permute(dims)` | Same | Shared | Reorder dims | ✓ |
| `x.squeeze()` | Same | Shared | Remove dim=1 | ✓ |
| `x.unsqueeze(0)` | Same | Shared | Add dimension | ✓ |
| `x.expand(shape)` | Same | Shared | Broadcast | ✓ |
| `x[0]`, `x[:, 1:3]` | Same** | Shared | Slicing | ✓ |
| `x.T` | Same | Shared | Transpose 2D | ✓ |
| **References** |
| `y = x` | Same | Shared | Python assignment | ✓ |
| `x.detach()` | Same | Shared | Remove grad history | ✓ |
| **Full Copies** |
| `x.clone()` | **New** | **Copied** | Independent copy | Returns new addr |
| `x.detach().clone()` | **New** | **Copied** | Copy without grad | Returns new addr |
| `y.copy_(x)` | y unchanged | **Copied** | Copy into existing | ✓ (stable target) |
| `torch._foreach_copy_([a,b],[x,y])` | a,b unchanged | **Copied** | Batched copy | ✓ |
| `x.to(device)` | New if diff device | Copied | Move to device | Depends |
| `x.contiguous()` | Same or New | Same or Copied | Ensure contiguous | Depends |

*`reshape` copies data when the tensor is not contiguous and the new shape requires it.
**Slices share storage but may have different stride/offset.

### 2.2 Views: Zero-Copy Tensor Operations

A view shares memory with the original tensor. This is efficient but dangerous:

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = x.view(6)  # Same memory!

print(f"x.data_ptr() = {x.data_ptr()}")  # 0x7f001000
print(f"y.data_ptr() = {y.data_ptr()}")  # 0x7f001000 (SAME!)

y[0] = 999
print(x[0, 0])  # 999! x was modified through y
```

**Danger:** Modifying a view modifies the original!

### 2.3 clone() vs copy_() - Deep Comparison

These are the two most important operations to understand:

#### `clone()` - Creates NEW Independent Tensor

```python
x = torch.tensor([1, 2, 3], device='cuda')
y = x.clone()

# Properties of y:
print(x.data_ptr() == y.data_ptr())  # False - DIFFERENT memory
print(torch.equal(x, y))              # True - same values
print(x.storage().data_ptr() == y.storage().data_ptr())  # False

# Modifying y does NOT affect x
y[0] = 999
print(x[0])  # Still 1! x is unchanged
```

**What clone() does internally:**
1. Allocates NEW memory on the same device
2. Copies all data from source to new memory
3. Returns new tensor pointing to new memory

#### `copy_()` - Copies Data INTO Existing Tensor

```python
x = torch.tensor([1, 2, 3], device='cuda')
z = torch.empty(3, device='cuda')  # Pre-allocated buffer
print(f"z address before: {z.data_ptr()}")  # 0x7f002000

z.copy_(x)  # Copy x's DATA into z's MEMORY

print(f"z address after: {z.data_ptr()}")   # 0x7f002000 (UNCHANGED!)
print(z)  # tensor([1, 2, 3])
```

**What copy_() does internally:**
1. Takes target tensor's memory address (does NOT reallocate)
2. Copies data from source to target's memory
3. Returns None (in-place operation)

#### The Critical Difference for CUDA Graphs

```
clone():                              copy_():
┌───────────┐                        ┌───────────┐
│ x at 0x01 │                        │ x at 0x01 │
│ [1, 2, 3] │                        │ [1, 2, 3] │
└─────┬─────┘                        └─────┬─────┘
      │ .clone()                           │ z.copy_(x)
      ▼                                    ▼
┌───────────┐                        ┌───────────┐
│ y at 0x02 │  NEW ADDRESS!          │ z at 0x03 │  ADDRESS UNCHANGED!
│ [1, 2, 3] │                        │ [1, 2, 3] │  (was 0x03 before)
└───────────┘                        └───────────┘
```

**For CUDA graphs:**
- `clone()`: BAD inside compiled - creates new address each call
- `copy_()`: GOOD - preserves target address for graph replay

### 2.4 torch._foreach_copy_: Batched Copying

```python
# Individual copies (3 kernel launches)
self._current.copy_(new_cur)
self._derived.copy_(new_der)
self._counts.copy_(new_cnt)

# Batched copy (1 kernel launch)
torch._foreach_copy_(
    [self._current, self._derived, self._counts],  # Targets
    [new_cur, new_der, new_cnt]                     # Sources
)
```

**Why faster:** Single kernel launch, better memory coalescing.

### 2.5 TensorDict Copy Operations

TensorDict provides high-level copy semantics:

```python
from tensordict import TensorDict

state = TensorDict({
    'current': torch.zeros(B, A, 3),
    'derived': torch.zeros(B, S, A, 3),
}, batch_size=[B])

# .clone() - Deep copy, new addresses for all tensors
new_state = state.clone()  

# .update() - In-place update (like nested copy_)
state.update(new_state)  # Copies data INTO state's tensors

# .copy_() - Same as update, explicit copy_ semantics
state.copy_(new_state)
```

---

## 3. In-Place Operations and Mutations

### 3.1 What is an In-Place Operation?

In-place operations modify data without changing memory address:

```python
x = torch.zeros(3)
addr_before = x.data_ptr()

x.add_(1)  # In-place add
x[:] = torch.ones(3)  # In-place assignment
x.fill_(5)  # In-place fill

assert x.data_ptr() == addr_before  # Address NEVER changes
```

### 3.2 The Underscore Convention

```python
# Out-of-place (NEW tensor)    # In-place (MODIFIES tensor)
y = x.add(1)                   x.add_(1)
y = x.mul(2)                   x.mul_(2)
y = torch.where(c, a, b)       # No in-place version!
```

### 3.3 Why Mutations Break CUDA Graphs - Full Explanation

This is subtle but critical. Let's trace through exactly what happens:

#### First, understand CUDA graph capture:

```python
# During capture, PyTorch records:
with torch.cuda.graph(g):
    y = x * 2  # Records: "multiply tensor at address A, write to address B"
```

The graph stores: "Read from 0x7f001000, multiply by 2, write to 0x7f002000"

#### Now, the problem with mutations INSIDE a compiled function:

```python
def step_fn():
    # Assume self._current is at address 0x7f001000
    self._current[:, 0, :] = new_query  # MUTATION
    result = process(self._current)
    return result

compiled = torch.compile(step_fn, mode='reduce-overhead')
```

**What happens during first call (capture):**
1. Graph captures: "Write new_query into 0x7f001000"
2. Graph captures: "Read from 0x7f001000, process it"
3. Returns result

**What happens during second call (replay):**
1. Graph replays: "Write new_query into 0x7f001000"
2. BUT: self._current already has DIFFERENT data from iteration 1!
3. The mutation mixes iteration 1's residual data with new_query
4. Wrong results!

#### The x[:] = ... case explained:

```python
x = torch.zeros(3)
x[:] = torch.ones(3)  # This is ALSO a mutation!
```

Why is `x[:] = y` a mutation but `x = y` is not?

```python
x = torch.zeros(3)  # x points to address A with [0,0,0]
x[:] = torch.ones(3)  # Writes [1,1,1] INTO address A
# x still points to address A, now contains [1,1,1]

x = torch.zeros(3)  # x points to address A with [0,0,0]
x = torch.ones(3)   # x now points to NEW address B with [1,1,1]
# Address A still contains [0,0,0], but x no longer references it
```

#### The correct pattern:

```python
def step_fn():
    # READ from self._current (address A)
    # CREATE NEW tensor (address B)
    new_current = torch.where(mask, new_query, self._current)
    return new_current  # Return address B

compiled = torch.compile(step_fn, mode='reduce-overhead')

# OUTSIDE compiled region:
result = compiled()
self._current.copy_(result)  # Mutation happens HERE, outside the graph
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
│                         │       │                         │
│ Addresses are FROZEN!   │       │ Uses SAME addresses!    │
└─────────────────────────┘       └─────────────────────────┘
```

### 4.2 The Buffer-Copy Pattern

```python
class MinimalEvalV10:
    def __init__(self):
        # FIXED addresses - never reallocated
        self._current = torch.zeros(B, A, 3)  # Address 0x7f001000
    
    def compile(self):
        current = self._current  # Capture REFERENCE (not copy!)
        
        def step_fn():
            # Read from 0x7f001000 (captured address)
            new = process(current)
            return new  # New address each call - OK!
        
        self._compiled = torch.compile(step_fn, mode='reduce-overhead')
    
    def evaluate(self):
        result = self._compiled()  # Returns tensor at new address
        self._current.copy_(result)  # Copy INTO 0x7f001000
```

**Why "capture REFERENCE not copy"?**

```python
# WRONG - captures a copy
def compile_wrong(self):
    current_copy = self._current.clone()  # New address each time!
    def step_fn():
        return process(current_copy)  # Uses cloned address
    # Problem: current_copy doesn't update when self._current.copy_() is called!

# CORRECT - captures reference
def compile_correct(self):
    current = self._current  # Just a Python reference, same object
    def step_fn():
        return process(current)  # Uses self._current's address
    # self._current.copy_() updates the data, step_fn sees it!
```

---

## 5. cudagraph_mark_step_begin Explained

### 5.1 What is it?

`torch.compiler.cudagraph_mark_step_begin()` tells PyTorch: "A new iteration is starting, you can safely replay the CUDA graph now."

### 5.2 Why is it needed?

Without it, PyTorch might:
1. Queue up operations from multiple iterations
2. batch them incorrectly
3. Replay graphs at wrong boundaries

### 5.3 Example Usage

```python
while not done:
    # Mark: "New iteration starting"
    torch.compiler.cudagraph_mark_step_begin()
    
    # This compiled function will be captured as a CUDA graph
    result = compiled_step()
    
    # These mutations happen AFTER the graph
    self._state.copy_(result)
    
    # Check termination (outside compiled region)
    if self._done.all():
        break
```

### 5.4 When to Use It

**Always use at the start of each loop iteration when:**
- Using `torch.compile(mode='reduce-overhead')`
- The compiled function uses pre-allocated buffers
- You're doing repeated iterations

### 5.5 What Happens Without It

```python
# WITHOUT mark_step_begin - potential issues:
while not done:
    result = compiled_step()  # Graph might not replay correctly
    self._state.copy_(result)
```

PyTorch may not know where one iteration ends and another begins, leading to:
- Incorrect graph segmentation
- Operations from iteration N mixed with N+1
- Random failures

---

## 6. torch.compile Internals

### 6.1 The Compilation Pipeline

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
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Inductor     │  ← Generates optimized Triton code
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Triton Kernels │  ← Custom CUDA kernels
│  + CUDA Graphs  │  ← Captured for replay
└─────────────────┘
```

### 6.2 Compiling Nested Functions - Common Errors

#### Error 1: Double Compilation

```python
# BAD - nested compile calls
@torch.compile
def outer(x):
    return inner(x)

@torch.compile  # Already compiled!
def inner(x):
    return x * 2

# Can cause: "Segmentation fault" or "CUDA error"
```

**Solution:** Only compile the outermost function:

```python
def outer(x):
    return inner(x)

def inner(x):
    return x * 2

compiled_outer = torch.compile(outer)  # inner gets inlined
```

#### Error 2: Capturing Compiled Functions

```python
# BAD - capturing an already-compiled function
class Model:
    def __init__(self):
        self.layer = torch.compile(some_layer)  # Pre-compiled
    
    def forward(self, x):
        return self.layer(x)

compiled_forward = torch.compile(self.forward)  # Tries to compile compiled!
```

**Solution:** Compile at the highest level only:

```python
class Model:
    def forward(self, x):
        return self.layer(x)  # Don't compile layer separately

compiled_model = torch.compile(model.forward)  # Everything compiled together
```

#### Error 3: Method Compilation with Self

```python
# CAREFUL - self capture can be tricky
class Model:
    def compile(self):
        def step_fn():
            return self.process(self.buffer)  # 'self' captured
        
        self.compiled = torch.compile(step_fn)
```

This works BUT: if `self` attributes change between compile and call, behavior may be unexpected.

---

## 7. Aliasing: The Silent Killer

### 7.1 What is Aliasing?

Two references to the same memory:

```python
x = torch.tensor([1, 2, 3])
y = x  # y IS x (aliased)
y[0] = 999
print(x[0])  # 999!
```

### 7.2 Why torch.cond Requires Clone (Not Copy)

In torch.cond, you CANNOT return inputs directly:

```python
def no_reset_branch(a, b, c):
    return a, b, c  # ERROR: Input-to-output aliasing!
```

**Why not just return the inputs?**

The compiler needs to track dataflow. If output IS the input:
1. Compiler can't tell if data changed
2. Graph optimization breaks
3. Memory analysis fails

**Why clone() and not copy_()?**

```python
def no_reset_branch(a, b, c):
    # WRONG - copy_ doesn't return a tensor
    result = torch.empty_like(a)
    result.copy_(a)  # But what about b and c?
    return result, ???  # Can't return None!

    # CORRECT - clone returns new tensors
    return a.clone(), b.clone(), c.clone()
```

`copy_()` is in-place - it returns None. You can't return it.
`clone()` returns a new tensor - exactly what torch.cond needs.

---

## 8. Conditional Execution: .any() vs .where vs torch.cond

### 8.1 The Options Compared

| Method | CPU Sync? | CUDA Graph | Branches | Best For |
|--------|-----------|------------|----------|----------|
| `.any()` + Python `if` | YES | Breaks outside | Real skip | Expensive ops |
| `torch.where` | NO | Works | Both always run | Element-wise |
| `torch.cond` | YES* | Partitions | Real skip | Branch is expensive |
| Masking | NO | Works | Both compute | Cheap ops |

*torch.cond evaluates predicate which requires sync

### 8.2 .any() with Python if

```python
if needs_reset.any():  # CPU sync! Reads GPU value
    engine_call()       # Skipped if no resets
```

**Pros:** Truly skips expensive operations
**Cons:** Forces CPU-GPU sync (~0.1ms per call)

### 8.3 torch.where (Element-wise Conditional)

```python
# BOTH new_query and current are evaluated/exist
result = torch.where(
    mask.view(B, 1, 1),  # Condition
    new_query,            # If True
    current               # If False
)
```

**Pros:** No sync, works with CUDA graphs
**Cons:** Both branches must already exist as tensors

### 8.4 torch.cond (Control Flow)

```python
result = torch.cond(
    needs_reset.any(),  # Predicate (causes sync!)
    reset_branch,        # Function if True
    no_reset_branch,     # Function if False
    operands
)
```

**Pros:** Truly skips branch, works in compiled code
**Cons:** Partitions CUDA graph (loses some optimization)

### 8.5 Masking (Compute All, Select Results)

```python
# Compute BOTH paths
result_a = expensive_op_a(x)
result_b = expensive_op_b(x)

# Select based on mask
result = torch.where(condition, result_a, result_b)
```

**Pros:** No sync, perfect CUDA graph
**Cons:** Wastes compute on unused branch

### 8.6 Decision Tree

```
Is the operation expensive (>1ms)?
├── NO → Use torch.where or masking
└── YES → How often is it skipped?
    ├── Rarely (<10%) → Just run it every time
    └── Often (>10%) → 
        ├── Does CUDA graph matter?
        │   ├── YES → Use .any() outside compiled
        │   └── NO → Use torch.cond
```

### 8.7 V10 Choice

V10 uses **torch.cond**:
- Engine call is expensive (~7ms)
- Skipped 90% of the time
- Saves: 0.9 * 1600 * 7ms = 10+ seconds
- Worth the CUDA graph partition

---

## 9. CPU-GPU Synchronization Deep Dive

### 9.1 Asynchronous Execution

```
CPU: launch ─── launch ─── launch ─── launch ────────▶
       │          │          │          │
GPU:   ═══════════●══════════●══════════●════════════▶
                  execute    execute    execute
```

CPU immediately returns after launching. GPU executes in background.

### 9.2 Sync Points

```python
# All these force sync:
value = x.item()      # Must read GPU value
if x.any():           # Must evaluate condition
print(x)              # Must get values to print
x.numpy()             # Must copy to CPU
```

### 9.3 The .any() Trade-off Analysis

```python
# Option A: Check with sync
if needs_reset.any():  # 0.1ms sync
    engine_call()       # 7ms, called 200 times

# Total: 200 * 0.1 + 200 * 7 = 1420ms

# Option B: No check, always call
engine_call()  # 7ms, called 1600 times

# Total: 1600 * 7 = 11200ms
```

**Winner:** Option A by ~10 seconds!

---

## 10. Profiling and Bottleneck Analysis

### 10.1 GPU Profile Breakdown (V10)

**Total: 26.5s CUDA time**

| Category | Time | % | Operations |
|----------|------|---|------------|
| Policy Forward | 11.9s | 45% | GEMM, LayerNorm, ReLU |
| Engine | 6.6s | 25% | scatter_gather, indexing |
| torch.where ops | 3.2s | 12% | Conditional selection |
| Memory copies | 0.5s | 2% | _foreach_copy_ |
| Other Triton | 4.3s | 16% | Fused kernels |

**The "Other 30%":**
- `torch.where` operations: 12%
- Fused triton kernels for masking: 8%
- Memory allocation overhead: 4%
- CUDA graph management: 2%
- Miscellaneous: 4%

### 10.2 Reading Triton Kernel Names

```
triton_per_fused_add_addmm_native_layer_norm_relu_view_6
```

- `triton_` - Generated by Triton compiler
- `per_` - Persistent kernel (stays in cache)
- `fused_` - Multiple ops combined
- `add_addmm_...` - The operations fused together
- `_6` - Variant number (multiple versions tried)

---

## 11. TensorDict Deep Dive

### 11.1 Installation

```bash
pip install tensordict
```

### 11.2 Creating TensorDicts

```python
from tensordict import TensorDict

# Basic creation
state = TensorDict({
    'current': torch.zeros(B, A, 3),
    'derived': torch.zeros(B, S, A, 3),
    'counts': torch.zeros(B),
}, batch_size=[B])

# Nested TensorDict
state = TensorDict({
    'observation': TensorDict({
        'visual': torch.zeros(B, 84, 84),
        'state': torch.zeros(B, 10),
    }, batch_size=[B]),
    'action': torch.zeros(B, 4),
}, batch_size=[B])
```

### 11.3 Accessing Data

```python
# Dict-style access
current = state['current']

# Attribute access
current = state.current

# Nested access
visual = state['observation', 'visual']
# or
visual = state['observation']['visual']
```

### 11.4 Copy Operations

```python
# .clone() - Full deep copy, NEW addresses
new_state = state.clone()
print(state['current'].data_ptr() == new_state['current'].data_ptr())  # False

# .copy_() - In-place copy, preserves addresses
target = state.clone()  # Create target with same structure
target.copy_(source)    # Copy data INTO target
print(target['current'].data_ptr())  # Same as before copy_!

# .update() - Same as copy_ but can add new keys
state.update({'current': new_current, 'new_key': new_tensor})
```

### 11.5 In-Place Operations

```python
# Apply in-place operation to all tensors
state.apply_(lambda x: x.fill_(0))

# Conditional in-place update
mask = torch.tensor([True, False, True, ...])
state['current'][mask] = new_values  # Only updates masked positions

# Masked assignment
state.masked_fill_(mask, 0)  # Set masked positions to 0
```

### 11.6 TensorDict with torch.compile

```python
@torch.compile
def step(state: TensorDict) -> TensorDict:
    new_current = process(state['current'])
    # Return NEW TensorDict (don't mutate input!)
    return TensorDict({
        'current': new_current,
        'derived': compute_derived(new_current),
    }, batch_size=state.batch_size)

# Usage
new_state = step(state)
state.update(new_state)  # Copy into original
```

---

## 12. V10 Architecture Analysis

### 12.1 Strategy Parameter

```python
evaluator = MinimalEvalV10(
    policy=policy,
    engine=engine,
    strategy='torch_cond',  # or 'fused'
)
```

### 12.2 Memory Layout

```
Pre-allocated buffers (fixed addresses):
┌─────────────────────────────────────────────────┐
│ self._current    → 0x7f0001000000               │
│ self._derived    → 0x7f0002000000               │
│ self._pool       → 0x7f0004000000               │
└─────────────────────────────────────────────────┘

Compiled step returns (new addresses each call):
┌─────────────────────────────────────────────────┐
│ new_current  → 0x7f0010000000 (temporary)       │
│ new_derived  → 0x7f0011000000 (temporary)       │
└─────────────────────────────────────────────────┘

After _foreach_copy_:
Data from temporaries copied INTO fixed buffers.
Addresses unchanged, data updated.
```

---

## 13. Best Practices Checklist

### 13.1 Buffer Pattern

- [ ] Pre-allocate ALL state tensors in `__init__`
- [ ] **Capture buffer REFERENCES** (not copies) when compiling:
  ```python
  current = self._current  # Reference, not clone!
  ```
- [ ] Return NEW tensors from compiled function
- [ ] Use `copy_()` or `_foreach_copy_()` OUTSIDE compiled region
- [ ] This is the **buffer-copy pattern**: read from buffers, return new, copy back

### 13.2 Compilation

- [ ] Use `mode='reduce-overhead'` for inference loops
- [ ] Use `fullgraph=True` to catch breaks early
- [ ] **Compile expensive functions separately** if they cause issues:
  ```python
  engine.compile()  # Compile engine methods independently
  eval.compile()    # Then compile the main loop
  ```
- [ ] Avoid nested `@torch.compile` decorators

### 13.3 CUDA Graphs

- [ ] Use `torch.compiler.cudagraph_mark_step_begin()` at loop start
- [ ] Ensure static shapes (use padding if needed)
- [ ] No mutations inside compiled functions
- [ ] No Python control flow based on tensor values

### 13.4 Sync Points

- [ ] Minimize `.item()`, `.any()` in hot paths
- [ ] When needed, use outside compiled region
- [ ] Consider trade-offs (sync cost vs compute saved)

---

## Summary

Key insights from V10 optimization:

1. **Memory addresses are critical** for CUDA graphs
2. **Buffer-copy pattern**: Pre-allocate, return new, copy back
3. **Clone vs Copy**: Clone for new tensor, copy_ for preserving address
4. **torch.cond**: Enables skipping but breaks CUDA graphs
5. **Sync trade-offs**: Sometimes worth it to avoid expensive compute
6. **Profile with GPU profiler**: CPU profiler misleads for GPU code

The final V10 achieves **0.88 ms/candidate** using torch.cond strategy.

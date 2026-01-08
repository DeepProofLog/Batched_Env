# CUDA Compilation Optimization Guide

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Description](#problem-description)
3. [Root Cause Analysis](#root-cause-analysis)
4. [CUDA Graphs Deep Dive](#cuda-graphs-deep-dive)
5. [Solutions Implemented](#solutions-implemented)
6. [Code Changes](#code-changes)
7. [Performance Results](#performance-results)
8. [Best Practices](#best-practices)
9. [Debugging Techniques](#debugging-techniques)
10. [Common Pitfalls](#common-pitfalls)
11. [References](#references)

---

## Executive Summary

This document details the optimization journey of enabling `torch.compile()` with `mode='reduce-overhead'` for PPO (Proximal Policy Optimization) training. The `reduce-overhead` mode uses CUDA graphs to minimize CPU-GPU synchronization overhead, but requires careful handling of tensor memory addresses.

### Key Results

| Metric | Before Optimization | After Optimization | Improvement |
|--------|--------------------|--------------------|-------------|
| Training time per `learn()` | ~134 seconds | **1.6 seconds** | **83x faster** |
| Warmup time | ~22 seconds | ~9.6 seconds | 2.3x faster |
| CUDA graph recordings | 129 per iteration | 1 (cached) | Stable |
| `run_backward` time | 10.8s | 0.80s | 13x faster |

---

## Problem Description

### Initial Symptoms

When running `profile_optimized_learn.py` with different compile modes:

```bash
# This worked fine
python tests/profile_optimized_learn.py --compile-mode default

# This would hang or run extremely slowly
python tests/profile_optimized_learn.py --compile-mode reduce-overhead
```

The `reduce-overhead` mode would either:
1. Appear to hang indefinitely
2. Run 100x slower than expected
3. Show excessive CUDA graph re-recordings in logs

### Profiling Output (Before Fix)

```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   320   10.818    0.034   12.045    0.038 {method 'run_backward' of 'torch._C._EngineBase'}
   321    9.383    0.029    9.383    0.029 {built-in method torch.clamp}
   768    1.018    0.001    1.018    0.001 graphs.py:139(replay)
```

The 9.4 seconds for `torch.clamp` was a red flag - this operation should take microseconds.

---

## Root Cause Analysis

### What is `reduce-overhead` Mode?

`torch.compile(model, mode='reduce-overhead')` enables **CUDA Graph Trees**, which:

1. **Record** GPU operations into a graph during the first execution
2. **Replay** the entire graph in subsequent executions
3. **Eliminate** CPU-GPU kernel launch overhead

### The Core Problem: Memory Address Instability

CUDA graphs capture the **exact memory addresses** of input and output tensors. If any tensor's address changes between recordings, the graph must be re-recorded.

#### Problem 1: TensorDict Creates New Tensors

```python
# BAD: TensorDict creates new tensor objects each time
def evaluate_actions(self, obs: TensorDict, actions: torch.Tensor):
    features = self.extract_features(obs)  # obs["feature"] creates new tensor
    # ...
```

Each call to `obs["feature"]` potentially returns a new tensor object with a different memory address, even if the underlying data is the same.

#### Problem 2: Missing CUDA Graph Step Boundaries

Without explicit step boundaries, PyTorch's CUDA graph system doesn't know when one training iteration ends and the next begins:

```python
# BAD: No step boundary marking
for epoch in range(n_epochs):
    for batch in dataloader:
        loss = compute_loss(batch)
        loss.backward()
        optimizer.step()
```

This causes the graph system to treat the entire training loop as one continuous operation, leading to graph instability.

#### Problem 3: Gradient Memory Recycling

```python
# BAD: set_to_none=True changes gradient memory addresses
optimizer.zero_grad(set_to_none=True)
```

When `set_to_none=True`, gradients are deallocated and reallocated, changing their memory addresses.

#### Problem 4: CUDA Graph Output Overwrites

```python
# BAD: Tensors inside CUDA graphs get overwritten on replay
metrics["loss"] = loss.detach()  # loss tensor will be overwritten!
```

CUDA graphs reuse the same memory for outputs. If you store a reference to an output tensor, it will be overwritten when the graph replays.

### Diagnostic Evidence

Using `TORCH_LOGS=+cudagraph_trees`:

```bash
TORCH_LOGS="+cudagraph_trees" python tests/profile_optimized_learn.py --compile-mode reduce-overhead
```

Output showed:
```
[DEBUG] Recording CUDA graph tree...
[DEBUG] Recording CUDA graph tree...  # 129 times!
[DEBUG] Recording CUDA graph tree...
```

Each re-recording added significant overhead, explaining the slowdown.

---

## CUDA Graphs Deep Dive

### How CUDA Graphs Work

```
Traditional Execution:
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│ CPU │───▶│ GPU │───▶│ CPU │───▶│ GPU │  (repeated kernel launches)
└─────┘    └─────┘    └─────┘    └─────┘

CUDA Graph Execution:
┌─────┐    ┌───────────────────────────┐
│ CPU │───▶│     GPU Graph Replay      │  (single launch, all kernels)
└─────┘    └───────────────────────────┘
```

### CUDA Graph Tree Structure

PyTorch's `reduce-overhead` mode creates a **tree of graphs**:

```
                    ┌─────────────┐
                    │  Root Node  │
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │Forward 1│  │Forward 2│  │Forward 3│
        └────┬────┘  └────┬────┘  └────┬────┘
             ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │Backward1│  │Backward2│  │Backward3│
        └─────────┘  └─────────┘  └─────────┘
```

Different execution paths (e.g., different batch sizes, different control flow) create different branches in the tree.

### Requirements for Stable CUDA Graphs

1. **Fixed tensor shapes** - All input/output tensors must have the same shape
2. **Fixed memory addresses** - Tensor storage addresses must not change
3. **Deterministic control flow** - Same code path each iteration
4. **No CPU-dependent operations** - No `.item()`, `.cpu()`, or Python conditionals based on tensor values during the graph

---

## Solutions Implemented

### Solution 1: Raw Tensor Interface

**Problem:** TensorDict creates new tensor objects, destabilizing CUDA graphs.

**Solution:** Create a method that accepts raw tensors directly.

```python
# model.py - Added evaluate_actions_raw() method
def evaluate_actions_raw(
    self,
    obs_batch: torch.Tensor,
    obs_variable_batch: torch.Tensor, 
    obs_constants_batch: torch.Tensor,
    actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Raw tensor version for CUDA graph compatibility.
    Bypasses TensorDict to maintain stable memory addresses.
    """
    # Directly use tensors instead of extracting from TensorDict
    embedded = self.embedder(
        obs_batch, obs_variable_batch, obs_constants_batch
    )
    features = self.mlp_extractor.forward_actor(embedded)
    features_critic = self.mlp_extractor.forward_critic(embedded)
    
    # ... rest of evaluation
    return log_prob, values, entropy
```

### Solution 2: CUDA Graph Step Boundaries

**Problem:** PyTorch doesn't know where one iteration ends and the next begins.

**Solution:** Mark step boundaries explicitly.

```python
# ppo_optimized.py - Training loop
for epoch in range(self.n_epochs):
    for batch_idx in range(num_batches):
        # Mark the start of a new CUDA graph step
        torch.compiler.cudagraph_mark_step_begin()
        
        # Now the training step
        loss_dict = loss_module(batch_obs, batch_actions, ...)
        total_loss.backward()
        optimizer.step()
```

### Solution 3: Gradient Memory Stability

**Problem:** `set_to_none=True` deallocates gradient memory.

**Solution:** Keep gradient tensors allocated with `set_to_none=False` and pre-warm gradients.

```python
# Keep gradient addresses stable
self.optimizer.zero_grad(set_to_none=False)

# Pre-warm gradients during initialization
def _warmup_gradients(self):
    """Initialize gradient tensors with stable memory addresses."""
    dummy_obs = torch.zeros(1, obs_dim, device=self.device)
    dummy_actions = torch.zeros(1, dtype=torch.long, device=self.device)
    
    # Forward pass
    _, values, _ = self.policy.evaluate_actions_raw(
        dummy_obs, dummy_var, dummy_const, dummy_actions
    )
    
    # Backward pass to allocate gradients
    values.sum().backward()
    
    # Zero without deallocating
    self.optimizer.zero_grad(set_to_none=False)
```

### Solution 4: Clone Tensors from CUDA Graph Memory

**Problem:** CUDA graph outputs get overwritten on replay.

**Solution:** Clone tensors before storing them.

```python
# BAD
pg_losses.append(pg_loss.detach())

# GOOD - Clone to copy data out of CUDA graph memory
pg_losses.append(pg_loss.detach().clone())
```

### Solution 5: Pre-allocated Batch Tensors

**Problem:** Creating new tensors for each batch destabilizes graphs.

**Solution:** Pre-allocate output tensors and use in-place operations.

```python
# rollout_optimized.py
class RolloutBufferOptimized:
    def __init__(self, buffer_size, obs_dim, device, batch_size=None):
        # Pre-allocate batch tensors if batch_size is known
        if batch_size is not None:
            self._batch_obs = torch.empty(batch_size, obs_dim, device=device)
            self._batch_actions = torch.empty(batch_size, dtype=torch.long, device=device)
            # ... other pre-allocated tensors
    
    def get(self, indices):
        # Use index_select with pre-allocated output tensor
        torch.index_select(self.obs, 0, indices, out=self._batch_obs)
        torch.index_select(self.actions, 0, indices, out=self._batch_actions)
        return self._batch_obs, self._batch_actions, ...
```

### Solution 6: Compile Mode Parameter

**Problem:** Need to conditionally apply optimizations based on compile mode.

**Solution:** Pass compile mode through the system.

```python
# ppo_optimized.py
class PPOOptimized:
    def __init__(self, ..., compile_mode: str = "default"):
        self.compile_mode = compile_mode
        
        if compile_mode == "reduce-overhead":
            self._warmup_gradients()

# Loss module uses raw tensors for reduce-overhead
class PPOLossModule(nn.Module):
    def forward(self, obs, obs_var, obs_const, actions, ...):
        if self.use_raw_tensors:
            log_prob, values, entropy = self.policy.evaluate_actions_raw(
                obs, obs_var, obs_const, actions
            )
        else:
            # Traditional TensorDict path
            ...
```

---

## Code Changes

### File: `model.py`

Added `evaluate_actions_raw()` method to `ActorCriticPolicy`:

```python
def evaluate_actions_raw(
    self,
    obs_batch: torch.Tensor,
    obs_variable_batch: torch.Tensor,
    obs_constants_batch: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate actions using raw tensors (CUDA graph compatible).
    
    This method bypasses TensorDict to maintain stable tensor memory
    addresses required by CUDA graphs in reduce-overhead mode.
    
    Args:
        obs_batch: Observation features [batch_size, obs_dim]
        obs_variable_batch: Variable indices [batch_size, n_vars]
        obs_constants_batch: Constant indices [batch_size, n_consts]
        actions: Action indices [batch_size]
    
    Returns:
        Tuple of (log_prob, values, entropy)
    """
    # Direct embedding computation (bypasses extract_features)
    embedded = self.features_extractor.embedder(
        obs_batch, obs_variable_batch, obs_constants_batch
    )
    
    # Actor-critic forward passes
    latent_pi = self.mlp_extractor.forward_actor(embedded)
    latent_vf = self.mlp_extractor.forward_critic(embedded)
    
    # Value prediction
    values = self.value_net(latent_vf)
    
    # Action distribution
    distribution = self._get_action_dist_from_latent(latent_pi)
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()
    
    return log_prob, values.flatten(), entropy
```

### File: `ppo_optimized.py`

#### Constructor Changes

```python
class PPOOptimized:
    def __init__(
        self,
        policy: ActorCriticPolicy,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
        compile_mode: str = "default",  # NEW PARAMETER
    ):
        self.compile_mode = compile_mode
        # ... other initialization
        
        # Pre-warm gradients for reduce-overhead mode
        if compile_mode == "reduce-overhead":
            self._warmup_gradients()
```

#### Gradient Warmup Method

```python
def _warmup_gradients(self):
    """
    Pre-warm gradient tensors with stable memory addresses.
    
    CUDA graphs require fixed memory addresses for all tensors.
    This method performs a dummy forward/backward pass to allocate
    gradient tensors before training begins.
    """
    # Create minimal dummy inputs
    dummy_obs = torch.zeros(
        1, self.obs_dim, device=self.device, dtype=torch.float32
    )
    dummy_obs_var = torch.zeros(
        1, self.n_variables, device=self.device, dtype=torch.long
    )
    dummy_obs_const = torch.zeros(
        1, self.n_constants, device=self.device, dtype=torch.long
    )
    dummy_actions = torch.zeros(1, device=self.device, dtype=torch.long)
    
    # Forward pass
    _, values, _ = self.policy.evaluate_actions_raw(
        dummy_obs, dummy_obs_var, dummy_obs_const, dummy_actions
    )
    
    # Backward pass to allocate gradients
    values.sum().backward()
    
    # Zero gradients without deallocating
    self.optimizer.zero_grad(set_to_none=False)
```

#### Loss Module Forward Pass

```python
class PPOLossModule(nn.Module):
    def __init__(self, policy, clip_range, clip_range_vf, ent_coef, vf_coef, 
                 use_raw_tensors=False):
        super().__init__()
        self.policy = policy
        self.use_raw_tensors = use_raw_tensors
        # ... other attributes
    
    def forward(
        self,
        obs: torch.Tensor,
        obs_variable: torch.Tensor,
        obs_constants: torch.Tensor,
        actions: torch.Tensor,
        old_values: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute PPO loss using raw tensors for CUDA graph compatibility.
        """
        # Use raw tensor interface for stable memory addresses
        log_prob, values, entropy = self.policy.evaluate_actions_raw(
            obs, obs_variable, obs_constants, actions
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss with clipping
        ratio = torch.exp(log_prob - old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio, 1 - self.clip_range, 1 + self.clip_range
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Value loss with optional clipping
        if self.clip_range_vf is not None:
            values_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_range_vf, self.clip_range_vf
            )
            value_loss_1 = F.mse_loss(values, returns, reduction='none')
            value_loss_2 = F.mse_loss(values_clipped, returns, reduction='none')
            value_loss = torch.max(value_loss_1, value_loss_2).mean()
        else:
            value_loss = F.mse_loss(values, returns)
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss 
            + self.vf_coef * value_loss 
            + self.ent_coef * entropy_loss
        )
        
        # Compute auxiliary metrics
        with torch.no_grad():
            clip_fraction = torch.mean(
                (torch.abs(ratio - 1) > self.clip_range).float()
            )
            approx_kl = torch.mean((ratio - 1) - torch.log(ratio))
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
        }
```

#### Training Loop with Step Boundaries

```python
def learn(self, total_timesteps: int, callback=None):
    """Train the policy using PPO with CUDA graph support."""
    
    # Compile the loss module
    compiled_loss = torch.compile(
        self.loss_module,
        mode=self.compile_mode,
        fullgraph=True,
    )
    
    iteration = 0
    while self.num_timesteps < total_timesteps:
        # Collect rollout
        self._collect_rollout()
        
        # Compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantage(
            self.policy, self.gamma, self.gae_lambda
        )
        
        # Training epochs
        for epoch in range(self.n_epochs):
            # Get random permutation for batches
            indices = torch.randperm(self.n_steps, device=self.device)
            
            for batch_idx in range(num_batches):
                # === CRITICAL: Mark CUDA graph step boundary ===
                torch.compiler.cudagraph_mark_step_begin()
                
                # Get batch indices
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data (using pre-allocated tensors)
                batch_data = self.rollout_buffer.get(batch_indices)
                
                # Zero gradients WITHOUT deallocating
                self.optimizer.zero_grad(set_to_none=False)
                
                # Compute loss
                loss_dict = compiled_loss(*batch_data)
                
                # Backward pass
                loss_dict["loss"].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # === CRITICAL: Clone tensors out of CUDA graph memory ===
                pg_losses.append(loss_dict["policy_loss"].detach().clone())
                value_losses.append(loss_dict["value_loss"].detach().clone())
                entropy_losses.append(loss_dict["entropy_loss"].detach().clone())
                clip_fractions.append(loss_dict["clip_fraction"].detach().clone())
                approx_kls.append(loss_dict["approx_kl"].detach().clone())
        
        iteration += 1
```

### File: `rollout_optimized.py`

#### Pre-allocated Batch Tensors

```python
class RolloutBufferOptimized:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple,
        action_dim: int,
        device: str = "cuda",
        batch_size: Optional[int] = None,
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.batch_size = batch_size
        
        # Main storage tensors
        self.obs = torch.zeros(buffer_size, *obs_shape, device=device)
        self.obs_variable = torch.zeros(
            buffer_size, n_variables, device=device, dtype=torch.long
        )
        self.obs_constants = torch.zeros(
            buffer_size, n_constants, device=device, dtype=torch.long
        )
        self.actions = torch.zeros(buffer_size, device=device, dtype=torch.long)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)
        
        # Pre-allocate batch output tensors for CUDA graph stability
        if batch_size is not None:
            self._batch_obs = torch.empty(
                batch_size, *obs_shape, device=device
            )
            self._batch_obs_variable = torch.empty(
                batch_size, n_variables, device=device, dtype=torch.long
            )
            self._batch_obs_constants = torch.empty(
                batch_size, n_constants, device=device, dtype=torch.long
            )
            self._batch_actions = torch.empty(
                batch_size, device=device, dtype=torch.long
            )
            self._batch_old_values = torch.empty(batch_size, device=device)
            self._batch_old_log_prob = torch.empty(batch_size, device=device)
            self._batch_advantages = torch.empty(batch_size, device=device)
            self._batch_returns = torch.empty(batch_size, device=device)
    
    def get(self, indices: torch.Tensor) -> tuple:
        """
        Get a batch of data using pre-allocated tensors.
        
        Uses torch.index_select with out= parameter for in-place
        operations that maintain stable memory addresses.
        """
        if self._batch_obs is not None:
            # Use pre-allocated tensors with in-place index_select
            torch.index_select(self.obs, 0, indices, out=self._batch_obs)
            torch.index_select(
                self.obs_variable, 0, indices, out=self._batch_obs_variable
            )
            torch.index_select(
                self.obs_constants, 0, indices, out=self._batch_obs_constants
            )
            torch.index_select(
                self.actions, 0, indices, out=self._batch_actions
            )
            torch.index_select(
                self.values, 0, indices, out=self._batch_old_values
            )
            torch.index_select(
                self.log_probs, 0, indices, out=self._batch_old_log_prob
            )
            torch.index_select(
                self.advantages, 0, indices, out=self._batch_advantages
            )
            torch.index_select(
                self.returns, 0, indices, out=self._batch_returns
            )
            
            return (
                self._batch_obs,
                self._batch_obs_variable,
                self._batch_obs_constants,
                self._batch_actions,
                self._batch_old_values,
                self._batch_old_log_prob,
                self._batch_advantages,
                self._batch_returns,
            )
        else:
            # Fallback to regular indexing
            return (
                self.obs[indices],
                self.obs_variable[indices],
                self.obs_constants[indices],
                self.actions[indices],
                self.values[indices],
                self.log_probs[indices],
                self.advantages[indices],
                self.returns[indices],
            )
```

---

## Performance Results

### Benchmark Configuration

```bash
python tests/profile_optimized_learn.py \
    --compile-mode reduce-overhead \
    --total-timesteps 10 \
    --n-steps 64 \
    --batch-size-env 64 \
    --n-epochs 2 \
    --batch-size 128
```

### Before Optimization (Cold Start, No Fixes)

```
Total time: ~134 seconds (stuck in graph re-recording)

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      320   10.818    0.034   12.045    0.038 {method 'run_backward'}
      321    9.383    0.029    9.383    0.029 {built-in method torch.clamp}
      768    1.018    0.001    1.018    0.001 graphs.py:139(replay)

CUDA graph recordings per training iteration: 129
```

### After Optimization (Warm, All Fixes Applied)

```
Total time: 1.616 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       64    0.800    0.012    0.866    0.014 {method 'run_backward'}
        6    0.152    0.025    0.152    0.025 {method 'item'}
     1560    0.049    0.000    0.049    0.000 {built-in method torch._C._nn.linear}
      192    0.010    0.000    0.026    0.000 graphs.py:139(replay)

CUDA graph recordings: 1 (cached and reused)
```

### Speedup Analysis

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| `run_backward` | 10.82s | 0.80s | **13.5x** |
| `torch.clamp` | 9.38s | <0.001s | **9380x** (was sync artifact) |
| `graphs.py:replay` | 1.02s | 0.026s | **39x** |
| Total training | 134s | 1.6s | **83x** |

### Memory Usage

The pre-allocated tensors add some upfront memory but eliminate allocation overhead during training:

```
Additional memory for batch_size=128:
- Batch tensors: ~8 tensors × 128 elements × 4 bytes = ~4KB
- Total overhead: Negligible compared to model parameters
```

---

## Best Practices

### 1. Always Use Raw Tensors in Compiled Code Paths

```python
# ❌ BAD: TensorDict creates new tensor objects
def forward(self, obs: TensorDict):
    features = obs["features"]  # New tensor each time!

# ✅ GOOD: Raw tensors have stable addresses
def forward(self, features: torch.Tensor):
    # features tensor address is stable
```

### 2. Mark Step Boundaries in Training Loops

```python
for epoch in range(n_epochs):
    for batch in dataloader:
        # ✅ Mark step boundary at the START of each iteration
        torch.compiler.cudagraph_mark_step_begin()
        
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### 3. Keep Gradient Memory Stable

```python
# ❌ BAD: Deallocates gradient memory
optimizer.zero_grad(set_to_none=True)

# ✅ GOOD: Keeps gradient tensors allocated
optimizer.zero_grad(set_to_none=False)
```

### 4. Clone Outputs Before Storing

```python
# ❌ BAD: Reference to CUDA graph memory (will be overwritten)
losses.append(loss.detach())

# ✅ GOOD: Clone copies data to new memory
losses.append(loss.detach().clone())
```

### 5. Pre-allocate Batch Tensors

```python
# ❌ BAD: Creates new tensors each iteration
batch = data[indices]

# ✅ GOOD: Reuse pre-allocated tensor
torch.index_select(data, 0, indices, out=self._batch)
```

### 6. Avoid Dynamic Shapes

```python
# ❌ BAD: Variable batch size
for batch in variable_size_batches:
    model(batch)

# ✅ GOOD: Fixed batch size
for batch in fixed_size_batches:  # Always same size
    model(batch)
```

### 7. Minimize CPU-GPU Synchronization

```python
# ❌ BAD: .item() forces synchronization
if loss.item() > threshold:
    break

# ✅ GOOD: Keep comparisons on GPU or defer to end
# Or batch all .item() calls at the end of training
```

### 8. Pre-warm the Compiled Function

```python
# Run a few warmup iterations before measuring performance
for _ in range(3):
    torch.compiler.cudagraph_mark_step_begin()
    _ = compiled_model(dummy_input)
    
# Now the graph is cached and ready
```

---

## Debugging Techniques

### 1. Enable CUDA Graph Logging

```bash
TORCH_LOGS="+cudagraph_trees" python train.py
```

This shows when graphs are recorded vs replayed:
```
[DEBUG] Recording CUDA graph tree...  # BAD if repeated
[DEBUG] Replaying CUDA graph...       # GOOD
```

### 2. Check for Graph Re-recordings

```python
import torch._dynamo as dynamo

# Reset compilation cache
dynamo.reset()

# Count compilations
compilation_count = 0
original_compile = torch.compile

def counting_compile(*args, **kwargs):
    global compilation_count
    compilation_count += 1
    return original_compile(*args, **kwargs)

torch.compile = counting_compile
```

### 3. Profile with cProfile

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run training
model.learn(total_timesteps=1000)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(40)
```

### 4. Check Tensor Memory Addresses

```python
def check_address_stability(tensor, name):
    """Track if tensor address changes."""
    addr = tensor.data_ptr()
    if not hasattr(check_address_stability, 'addresses'):
        check_address_stability.addresses = {}
    
    if name in check_address_stability.addresses:
        if check_address_stability.addresses[name] != addr:
            print(f"WARNING: {name} address changed!")
    
    check_address_stability.addresses[name] = addr
```

### 5. Isolate the Problem

```python
# Test each component separately
def test_forward_only():
    for _ in range(100):
        torch.compiler.cudagraph_mark_step_begin()
        out = compiled_forward(input)
    
def test_backward_only():
    for _ in range(100):
        torch.compiler.cudagraph_mark_step_begin()
        out = compiled_forward(input)
        out.sum().backward()
```

---

## Common Pitfalls

### Pitfall 1: Using TensorDict in Compiled Code

**Symptom:** Excessive graph re-recordings

**Solution:** Use raw tensors instead of TensorDict

### Pitfall 2: Forgetting Step Boundaries

**Symptom:** First iteration slow, subsequent iterations also slow

**Solution:** Add `torch.compiler.cudagraph_mark_step_begin()` at iteration start

### Pitfall 3: Dynamic Control Flow

**Symptom:** Graph breaks, fallback to eager mode

```python
# ❌ BAD: Python control flow based on tensor value
if loss.item() > 1.0:  # Breaks the graph!
    learning_rate *= 0.5
```

**Solution:** Use `torch.where()` or defer to outside compiled region

### Pitfall 4: In-place Operations on Graph Inputs

**Symptom:** Errors about tensor modification

```python
# ❌ BAD: Modifying input tensor
def forward(self, x):
    x += 1  # In-place modification!
    return self.layer(x)

# ✅ GOOD: Create new tensor
def forward(self, x):
    x = x + 1  # New tensor
    return self.layer(x)
```

### Pitfall 5: Storing References to Graph Outputs

**Symptom:** Stored values mysteriously change

```python
# ❌ BAD
all_losses = []
for batch in dataloader:
    loss = compiled_model(batch)
    all_losses.append(loss)  # All point to same memory!

# ✅ GOOD
all_losses = []
for batch in dataloader:
    loss = compiled_model(batch)
    all_losses.append(loss.clone())  # Independent copies
```

### Pitfall 6: Profiler Attribution Errors

**Symptom:** Simple operations like `torch.clamp` appear very slow

**Explanation:** CUDA operations are asynchronous. The profiler may attribute time to the first synchronization point (like `clamp`) when the actual work was in previous operations.

**Solution:** Use NVIDIA Nsight Systems for accurate GPU profiling

```bash
nsys profile python train.py
```

---

## References

### PyTorch Documentation

- [torch.compile Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [CUDA Graphs in PyTorch](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [TorchDynamo Deep Dive](https://pytorch.org/docs/stable/torch.compiler_deepdive.html)

### Key Functions

```python
# Compile a model with CUDA graphs
torch.compile(model, mode='reduce-overhead', fullgraph=True)

# Mark iteration boundaries for CUDA graph trees
torch.compiler.cudagraph_mark_step_begin()

# Reset compilation cache (useful for debugging)
torch._dynamo.reset()

# Check if function was compiled
torch._dynamo.is_compiling()
```

### Environment Variables

```bash
# Enable CUDA graph logging
TORCH_LOGS="+cudagraph_trees"

# Enable all dynamo logging
TORCH_LOGS="+dynamo"

# Disable CUDA graphs (for debugging)
TORCH_COMPILE_DISABLE=1

# Force synchronous CUDA execution (for debugging)
CUDA_LAUNCH_BLOCKING=1
```

### Useful Debugging Commands

```bash
# Profile with CUDA graphs disabled (baseline)
python train.py --compile-mode default

# Profile with CUDA graphs enabled
python train.py --compile-mode reduce-overhead

# Profile with maximum verbosity
TORCH_LOGS="+dynamo,+cudagraph_trees" python train.py --compile-mode reduce-overhead 2>&1 | tee debug.log
```

---

## Appendix: Complete Working Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_with_cuda_graphs():
    device = "cuda"
    model = SimpleModel(128, 256, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Compile with CUDA graphs
    compiled_model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
    
    # Pre-allocate batch tensor
    batch_size = 64
    input_tensor = torch.randn(batch_size, 128, device=device)
    target_tensor = torch.randint(0, 10, (batch_size,), device=device)
    
    # Pre-warm gradients
    output = compiled_model(input_tensor)
    loss = F.cross_entropy(output, target_tensor)
    loss.backward()
    optimizer.zero_grad(set_to_none=False)
    
    # Training loop
    losses = []
    for iteration in range(100):
        # Mark step boundary
        torch.compiler.cudagraph_mark_step_begin()
        
        # Zero gradients (keep memory allocated)
        optimizer.zero_grad(set_to_none=False)
        
        # Forward pass
        output = compiled_model(input_tensor)
        loss = F.cross_entropy(output, target_tensor)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Clone loss before storing
        losses.append(loss.detach().clone())
    
    # Convert to CPU only at the end
    final_losses = [l.item() for l in losses]
    print(f"Final loss: {final_losses[-1]:.4f}")

if __name__ == "__main__":
    train_with_cuda_graphs()
```

---

## Changelog

| Date | Change | Impact |
|------|--------|--------|
| 2024-12-16 | Added `evaluate_actions_raw()` to model.py | Enabled CUDA graph stability |
| 2024-12-16 | Added `cudagraph_mark_step_begin()` to training loop | Fixed graph re-recording |
| 2024-12-16 | Changed to `set_to_none=False` in optimizer | Stable gradient addresses |
| 2024-12-16 | Added `.clone()` to metric storage | Fixed output overwriting |
| 2024-12-16 | Added `_warmup_gradients()` method | Pre-allocated gradient memory |
| 2024-12-16 | Pre-allocated batch tensors in rollout buffer | Stable batch memory addresses |

---

*Document last updated: December 16, 2024*

# Agent2 Optimization Report

## Summary

This report documents the performance optimizations made to `ppo.py` in the backup version compared to the last commit (3807151). The optimizations achieve a **2.3x speedup** for training and **2.2x speedup** for evaluation.

## Performance Comparison

### Training (profile_learn.py)

| Metric | Last Commit | Backup Version | Improvement |
|--------|-------------|----------------|-------------|
| **Runtime** | 22.3s | 9.6s | **2.3x faster** |
| **Steps/sec** | 733.6 | 1707.6 | **2.3x faster** |
| **Rollout FPS** | 2319 | 4651 | **2.0x faster** |
| **Warmup time** | 34.4s | 31.1s | 10% faster |
| **Target (<=16s)** | FAIL | PASS | |

### Evaluation (profile_eval.py)

| Metric | Last Commit | Backup Version | Improvement |
|--------|-------------|----------------|-------------|
| **Runtime** | 9.15s | 4.15s | **2.2x faster** |
| **ms/candidate** | 1.79 | 0.81 | **2.2x faster** |

### GPU Profile

| Metric | Last Commit | Backup Version | Change |
|--------|-------------|----------------|--------|
| **Self CPU time** | 22.1s | 8.3s | **2.7x less** |
| **Self CUDA time** | 4.7s | 2.2s | **2.1x less** |
| **scatter_gather kernel** | 1.72s (37%) | 0.57s (26%) | **3x faster** |

## Bottlenecks (Current)

The main GPU bottlenecks in the optimized version are:

1. **scatter_gather_elementwise_kernel**: 26% of CUDA time (down from 37%)
2. **cutlass GEMM kernels**: ~4% of CUDA time
3. **triton fused layer_norm/relu**: ~2.6% of CUDA time

## Key Optimizations

### 1. Fused Gradient Clipping (`fused_clip_grad_norm_`)

**Location**: `ppo.py:50-91`

Replaces PyTorch's `clip_grad_norm_` with a fused version that avoids GPU-CPU synchronization:

```python
def fused_clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.cat([g.flatten() for g in grads]).norm(norm_type)
    clip_coef = (total_norm + 1e-6).reciprocal_().mul_(max_norm)
    clip_coef.clamp_(max=1.0)
    torch._foreach_mul_(grads, clip_coef)
    return total_norm
```

**Impact**: Eliminates CPU sync during gradient clipping.

### 2. Flash Attention Enabled

**Location**: `ppo.py:40-42`

```python
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

**Impact**: Faster attention computation in transformer layers.

### 3. Cached Parameter List

**Location**: `ppo.py:286`

```python
self._cached_params = list(self.policy.parameters())
```

**Impact**: Avoids repeated `list()` call on every training iteration.

### 4. Deferred Episode Statistics

**Location**: `ppo.py:733-766`

Changed from processing done episodes on every step to only processing when callback is provided:

```python
# OLD: Always processed done episodes (sync on every step)
done_indices = torch.nonzero(new_state['step_dones']).flatten()
if num_dones > 0:
    # CPU sync here

# NEW: Only process when callback needs it
if on_step_callback is not None:
    # Only sync when callback is provided
```

**Impact**: Eliminates unnecessary GPU-CPU sync during rollout collection.

### 5. Loss Module Returns Individual Tensors

**Location**: `ppo.py:145-147`

```python
# OLD: torch.stack synchronization
return torch.stack([loss, policy_loss, value_loss, entropy_loss, approx_kl, clip_fraction])

# NEW: No synchronization
return loss, policy_loss, value_loss, entropy_loss, approx_kl.detach(), clip_fraction
```

**Impact**: Avoids `torch.stack` synchronization overhead.

### 6. Removed Unused Code

- Removed `_output` buffer and `_ensure_output_buffer` method from `PPOLossModule`
- Removed `_device` parameter from `PPOLossModule.__init__`

## Failed Optimization Attempt: CUDA Graph for Backward Pass

An attempt was made to capture the entire training step (forward + backward + optimizer) as a CUDA graph to eliminate CPU dispatch overhead. This approach failed due to:

1. **Conflict with torch.compile**: When `compile=True` (default), manual CUDA graphs conflict with torch.compile's internal graph management
2. **OOM errors**: When `compile=False`, capturing the backward pass requires ~1.5GB extra memory to cache intermediate tensors

The CUDA graph implementation was reverted as it caused a **13% slowdown** for training and **25% slowdown** for evaluation due to failed capture attempts adding overhead.

## Verification

All tests pass after optimizations:
- **17/17 compiled parity tests**
- **138/138 SB3 parity tests**

Verbose mode and callbacks verified working:
- `on_training_start`, `on_training_end`, `on_iteration_start` callbacks
- `prepare_batch_infos` callback for done episodes
- Per-epoch loss printing
- Rollout FPS and training time logging

## Files Changed

- `kge_experiments/ppo.py` - Main optimization changes
- `kge_experiments/backup_optimized/ppo.py` - Backup of optimized version

## How to Profile

```bash
# Training performance
python tests/profile_learn.py --total-timesteps 1

# Evaluation performance
python tests/profile_eval.py --n-queries 50 --n-corruptions 50

# GPU profiling
python tests/profile_learn.py --use-gpu-profiler --total-timesteps 1
```

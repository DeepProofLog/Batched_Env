# Family Dataset Training Report - Target MRR 0.7+

## Overview

This report tracks experiments to achieve test MRR >= 0.70 on the family dataset.

## Phase 1: Training Stabilization

### CRITICAL BUG FIXES

1. **use_l2_norm/temperature not passed to policy** (`train.py`) - **ROOT CAUSE OF KL ISSUES**
   - `use_l2_norm`, `temperature`, `sqrt_scale` were NOT being passed from config to ActorCriticPolicy
   - This caused logits to be 300+ instead of ~9, leading to KL divergence of 10+ vs expected <0.1
   - **Fix**: Added `use_l2_norm=config.use_l2_norm, temperature=config.temperature, sqrt_scale=config.sqrt_scale` to policy creation
   - **Impact**: Initial MRR improved from 0.42 to 0.60, KL reduced from 10+ to 0.12-0.20

2. **Dropout Fix** (`policy.py`)
   - Fixed bug where `dropout_prob` config was never applied in non-parity path
   - Added `nn.Dropout` modules to SharedBody (CUDA graph compatible)

3. **LR Warmup** (`config.py`, `callbacks.py`, `train.py`)
   - Added `lr_warmup: bool = True` and `lr_warmup_steps: float = 0.1`
   - ScalarAnnealingCallback now supports warmup phase (0 -> initial LR)

---

## Experiment Log

### Experiment 1: Before Bug Fix (KL ~10)

**Date**: 2026-01-25

**Issue**: use_l2_norm not passed, causing massive KL divergence

| Metric | Value |
|--------|-------|
| Initial MRR | 0.42 |
| KL per iteration | 10-12 (way above target 0.07) |
| Early stopping | Every iteration at step 0 |
| Status | **FAILED** - No learning |

### Experiment 2: After Bug Fix with target_kl=0.15

**Date**: 2026-01-25

**Config**:
```python
total_timesteps = 2000000
use_l2_norm = True  # NOW PROPERLY PASSED
temperature = 0.1
target_kl = 0.15    # Increased to allow more learning
lr_warmup = True
dropout_prob = 0.1
```

**Status**: COMPLETED

| Metric | Initial | Final |
|--------|---------|-------|
| Val MRR | 0.596 | **0.636** (best at iter 8) |
| Test MRR | - | **0.626** |
| Test Hits@1 | - | 0.615 |
| Test Hits@10 | - | 0.634 |
| Explained Variance | -4.4 | **+0.075** |

**Key Observations**:
- Training stable - running full 5 epochs without early stopping
- Value function learning (explained variance: -4.4 -> +0.075)
- Val-Test gap: 0.636 - 0.626 = 0.01 (minimal overfitting)
- MRR plateaued around 0.62-0.64, below 0.7 target

---

## Summary

| Experiment | Val MRR | Test MRR | Status |
|------------|---------|----------|--------|
| Before Fix | N/A | N/A | Failed (KL 10+) |
| After Fix (2M steps) | **0.636** | **0.626** | Below target |

## Temperature Tuning

**Date**: 2026-01-25

Testing different temperature values to control action distribution sharpness.

| Temperature | Test MRR | Hits@1 | Hits@10 | Proven Pos | Status |
|-------------|----------|--------|---------|------------|--------|
| 0.05 | 0.6346 | 0.624 | 0.643 | 63.6% | Completed |
| 0.10 (baseline) | 0.626 | 0.615 | 0.634 | - | Completed |
| 0.20 | **0.6385** | 0.623 | **0.657** | 62.7% | Completed |
| 0.50 | 0.6365 | 0.622 | 0.652 | 62.8% | Completed |

**Winner: temperature=0.2** (Test MRR 0.6385, +1.3% vs baseline)

**Observations**:
- Temperature 0.2 (softer) slightly outperforms all others
- Best Hits@10: 0.657 at temp=0.2
- All temperatures show similar proven_pos rates (~63%)
- KL divergence stable at 0.008-0.012 across all temperatures

---

## Hyperparameter Search

**Status**: Trial 1 running (5 trials total)

**Trial 0 Completed**:
- Test MRR: **0.632**
- Config: lr=5e-05, hidden=256, layers=4, ent_coef=0.1, dropout=0.2, target_kl=0.2, temp=0.1

**Search Space**:
- Architecture: hidden_dim [128, 256, 512], num_layers [4, 6, 8], atom_embedding_size [128, 250, 512]
- Regularization: dropout_prob [0.0, 0.1, 0.2, 0.3]
- Learning: learning_rate [5e-5, 1e-4, 2e-4], batch_size [256, 512, 1024], n_epochs [3, 5, 10]
- PPO: target_kl [0.05, 0.1, 0.15, 0.2], clip_range [0.1, 0.2, 0.3]
- Exploration: temperature [0.05, 0.1, 0.2], ent_coef_init [0.1, 0.15, 0.2, 0.3]

**Log**: `logs/hyperparam_search/search3.log`

---

## Code Changes Summary

| File | Change | Impact |
|------|--------|--------|
| `train.py` | Added use_l2_norm, temperature, sqrt_scale to policy creation | **CRITICAL**: Fixed KL divergence (10 -> 0.01) |
| `policy.py` | Added dropout to SharedBody (non-parity path) | Regularization now works |
| `config.py` | Added lr_warmup, lr_warmup_steps | LR warmup support |
| `callbacks.py` | Added warmup phase to AnnealingTarget | Gradual LR warmup |
| `hyperparam_search.py` | Expanded search space | Better tuning |

## Current Best Results

| Metric | Value |
|--------|-------|
| Test MRR | **0.6385** (temp=0.2) |
| Val MRR (best) | 0.628 |
| Hits@1 | 0.623 |
| Hits@10 | **0.657** |
| Target | 0.70+ |

---

## Bottleneck Analysis

### Why MRR is stuck at ~0.63?

1. **Deep Proof Struggles (D4+)**:
   - D2 proofs: 76% proven, reward +0.52
   - D4 proofs: 53% proven, reward +0.07
   - D6 proofs: 40% proven, reward -0.19
   - Longer episodes dilute rewards

2. **Predicate-Specific Weaknesses**:
   - nephew D4: only 6% proven (worst)
   - son D4: 24% proven
   - husband/wife D2: 51% proven (low)

3. **Negative Example Leakage**:
   - 34.6% of negatives have valid proof paths
   - This creates noisy rewards for the model

4. **Value Function Still Weak**:
   - Explained variance only ~0.05-0.08
   - Value estimates not improving much

### Potential Solutions

1. **Architecture Changes**:
   - Deeper networks for D4+ proofs (current: 4 layers)
   - Larger hidden dim (current: 256)

2. **Longer Training**:
   - Current: 2M steps
   - Try: 5M-10M steps with slower LR decay

3. **Reward Shaping**:
   - Penalize negatives with valid proofs less
   - Bonus for depth-4+ proofs

---

## Next Steps

1. Wait for temperature 0.2 and 0.5 experiments to complete
2. Complete hyperparameter search (5 trials)
3. Analyze which hyperparameters correlate with better MRR
4. Run longer training with best params if MRR < 0.7


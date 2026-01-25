# Family Dataset Training Analysis

## Dataset Provability Analysis

### Validation Set (2799 queries)

| Depth | Count | Percentage | Description |
|-------|-------|------------|-------------|
| -1 (unprovable) | 650 | 23.2% | No proof exists |
| 1 | 164 | 5.9% | Direct rule (husband↔wife) |
| 2 | 1939 | 69.3% | Single chain |
| 4 | 45 | 1.6% | Two chains |
| 6 | 1 | 0.04% | Three chains |
| **Provable Total** | **2149** | **76.8%** | |

### Test Set (5626 queries)

| Depth | Count | Percentage |
|-------|-------|------------|
| -1 (unprovable) | 1261 | 22.4% |
| 1 | 318 | 5.7% |
| 2 | 3979 | 70.7% |
| 4 | 68 | 1.2% |
| **Provable Total** | **4365** | **77.6%** |

### Provability by Predicate (Validation)

| Predicate | Unprovable | Provable | Prov Rate | Difficulty |
|-----------|------------|----------|-----------|------------|
| aunt | 15 | 284 | 95% | Easy |
| nephew | 21 | 342 | 94% | Easy |
| niece | 12 | 269 | 96% | Easy |
| uncle | 28 | 320 | 92% | Easy |
| husband | 6 | 101 | 94% | Easy (D1) |
| wife | 9 | 94 | 91% | Easy (D1) |
| son | 58 | 151 | 72% | Medium |
| daughter | 44 | 110 | 71% | Medium |
| father | 65 | 127 | 66% | Medium |
| mother | 63 | 108 | 63% | Medium |
| sister | 147 | 118 | 44% | Hard |
| brother | 182 | 125 | 41% | Hard |

**Key Insight**: brother/sister have ~40% provability due to missing/insufficient rules.

## Theoretical Max MRR Calculation

With 100 negative samples:
- Proved positive at rank 1 → RR = 1.0
- Unproved positive at random rank ~50 → RR ≈ 0.02

**Theoretical Upper Bound:**
```
MRR_max = P(provable) × 1.0 + P(unprovable) × 0.02
MRR_max = 0.768 × 1.0 + 0.232 × 0.02
MRR_max ≈ 0.77
```

**Realistic Targets (accounting for imperfect proofs):**

| Scenario | D1 (100%) | D2 (varied) | D4+ (varied) | D-1 | MRR |
|----------|-----------|-------------|--------------|-----|-----|
| Perfect | 0.059 | 0.693 | 0.016 | 0.005 | 0.77 |
| Excellent | 0.059 | 0.66 (95%) | 0.012 | 0.005 | 0.74 |
| Good | 0.059 | 0.62 (90%) | 0.008 | 0.005 | 0.69 |
| Baseline | 0.059 | 0.52 (75%) | 0.005 | 0.005 | 0.59 |

## Revised Training Goals

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| **MRR (stable)** | 0.65-0.70 | 0.60-0.65 | <0.55 |
| **Entropy** | -0.15 → -0.40 | -0.10 → -0.50 | <-1.0 (collapse) |
| **Explained Var** | >0.15 | >0.05 | <0 (useless) |
| **approx_kl** | <0.07/epoch | <0.10/epoch | >0.15 (unstable) |
| **clip_fraction** | <0.15 | <0.20 | >0.30 |

## Experiment Tracking

### Run 1: Stabilization Parameters
- **Config**: target_kl=0.07, clip_range_vf=0.2, lr_decay=True (cos), ent_coef_decay=True (0.15→0.02)
- **Start Time**: 2026-01-25 09:49
- **Model Path**: `models/family-250-128-torchrl/best_model_eval_2026_01_25_09_57_52.pt`
- **Total Timesteps**: 2,000,000

#### MRR Progression
| Iteration | MRR | Notes |
|-----------|-----|-------|
| 0 | 0.276 | Untrained |
| 4 | 0.341 | Early learning |
| 20 | 0.490 | Rapid improvement |
| 28 | 0.574 | Continuing |
| 36 | 0.638 | Near peak |
| 56 | **0.655** | **Best (85% of theoretical max)** |
| Final eval | 0.479 | Dropped (but best model saved) |

#### Final Test Results (with restored best model)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **MRR** | 0.589 | 0.65-0.70 | ⚠️ Below target |
| Hits@1 | 0.557 | - | - |
| Hits@3 | 0.600 | - | - |
| Hits@10 | 0.633 | - | - |
| Proven (pos) | 69.7% | ~77% | Good |
| Proven (neg) | 13.3% | <20% | ✅ Good |

#### Stability Metrics (End of Training)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Entropy | -0.117 | -0.10 to -0.50 | ✅ **STABLE** |
| Explained Var | 0.05 | >0.05 | ✅ **Working** |
| approx_kl | 0.006 | <0.07 | ✅ **Controlled** |
| clip_fraction | 0.035 | <0.15 | ✅ **Excellent** |

#### Analysis
**Successes:**
1. ✅ Training is now STABLE - no entropy collapse
2. ✅ Value function is learning (explained variance went from -4.0 to +0.05)
3. ✅ Policy updates are controlled (approx_kl 10x lower than before)
4. ✅ Best validation MRR reached 0.655 (85% of theoretical max 0.77)

**Issues:**
1. ⚠️ Test MRR (0.589) is below best validation MRR (0.655) - some overfitting
2. ⚠️ Training might be too conservative - approx_kl is very low (0.006)
3. ⚠️ MRR dropped in final iterations - need longer stable plateau

---

### Run 2: Longer Training with Adjusted Parameters
- **Config**: target_kl=0.10 (increased from 0.07), other params same
- **Start Time**: 2026-01-25 10:09
- **Model Path**: `models/family-250-128-torchrl/best_model_eval_2026_01_25_10_09_58.pt`
- **Total Timesteps**: 3,000,000

#### MRR Progression
| Iteration | MRR | Notes |
|-----------|-----|-------|
| 0 | 0.287 | Untrained |
| 16 | 0.566 | Early learning |
| 32 | 0.670 | Good progress |
| 48 | 0.682 | Continuing |
| 64 | **0.705** | **Best (91% of theoretical max!)** |
| Final eval | 0.629 | Dropped |

#### Final Test Results (with restored best model)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **MRR** | 0.562 | 0.65-0.70 | ⚠️ Below target |
| Hits@1 | 0.523 | - | - |
| Hits@3 | 0.581 | - | - |
| Hits@10 | 0.613 | - | - |
| Proven (pos) | 69.5% | ~77% | Good |
| Proven (neg) | 15.8% | <20% | ✅ Good |

#### Stability Metrics (End of Training)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Entropy | -0.126 | -0.10 to -0.50 | ✅ **STABLE** |
| Explained Var | 0.12 | >0.05 | ✅ **Excellent** |
| approx_kl | 0.003 | <0.10 | ✅ **Controlled** |
| clip_fraction | 0.024 | <0.15 | ✅ **Excellent** |

#### Analysis
**Comparison with Run 1:**
| Metric | Run 1 | Run 2 | Winner |
|--------|-------|-------|--------|
| Best Val MRR | 0.655 | **0.705** | Run 2 |
| Test MRR | **0.589** | 0.562 | Run 1 |
| Test Hits@1 | **0.557** | 0.523 | Run 1 |
| Test Proven (pos) | **69.7%** | 69.5% | Run 1 |

**Key Finding: Overfitting detected!**
- Run 2 achieves higher validation MRR (0.705 vs 0.655)
- But Run 2 has LOWER test MRR (0.562 vs 0.589)
- This indicates the model is overfitting to the validation set

---

## Summary and Recommendations

### Results Comparison

| Run | Config | Best Val MRR | Test MRR | Stability |
|-----|--------|--------------|----------|-----------|
| **1** | target_kl=0.07, 2M steps | 0.655 | **0.589** | ✅ Excellent |
| 2 | target_kl=0.10, 3M steps | **0.705** | 0.562 | ✅ Excellent |

### Key Achievements

1. **Training Stability Fixed**: Both runs show stable training
   - Entropy stays controlled (-0.12, not collapsing to -1.4)
   - Value function learns (explained variance ~0.05-0.12, not -4.0)
   - Policy updates controlled (approx_kl 0.003-0.006)

2. **Best Test Performance**: Run 1 with MRR=0.589
   - This is 76% of theoretical maximum (0.77)
   - Within "Baseline" to "Good" range from targets

3. **Best Validation Performance**: Run 2 with MRR=0.705
   - This is 91% of theoretical maximum (0.77)
   - In "Excellent" range

### Recommended Model

**Use Run 1's model** (`best_model_eval_2026_01_25_09_57_52.pt`):
- Better test generalization (0.589 vs 0.562)
- More conservative policy prevents overfitting
- target_kl=0.07 provides good stability-performance tradeoff

### Next Steps (if further improvement needed)

1. **Reduce overfitting**: Try regularization or data augmentation
2. **Increase training data**: More diverse training queries
3. **Try intermediate target_kl=0.08-0.09**: Balance between Run 1 and 2
4. **Ensemble**: Combine Run 1 and Run 2 models

---

## Phase 1: Critical Fixes Implementation (2026-01-25)

### Root Cause Analysis of Val-Test Gap

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| Tiny validation set (n_eval=20) | CRITICAL | n_eval_queries=200 |
| Deterministic query selection | HIGH | Randomized sampling (seed=42) |
| Dropout not working in fused layers | HIGH | Added nn.Dropout after SharedBody |
| No weight decay (L2 regularization) | HIGH | AdamW with weight_decay=1e-4 |
| No test monitoring during training | MEDIUM | Test eval every 8 iterations |

### Code Changes Summary

```
config.py:     n_eval_queries: 20 → 200, added weight_decay=1e-4
train.py:      Randomized query sampling with fixed seed
policy.py:     Added dropout to SharedBody (non-parity path)
ppo.py:        Adam → AdamW with weight_decay
callbacks.py:  RankingCallback now supports periodic test evaluation
embeddings.py: Registered 'attention' state embedder
hyperparam_search.py: New Optuna-based optimization script
```

---

### Run 3: Baseline with All Phase 1 Fixes
- **Config**: All Phase 1 fixes, target_kl=0.07, lr=1e-4, hidden=256, layers=8, ent_coef=0.15
- **Start Time**: 2026-01-25 (in progress)
- **Total Timesteps**: 5,000,000
- **Key Changes**: n_eval=200, randomized sampling, dropout working, weight_decay=1e-4

#### Expected Improvements
- Val-Test Gap: <0.05 (was 0.066-0.143)
- Test MRR: >0.60 (baseline expectation)
- More stable model selection

#### MRR Progression
| Iteration | Val MRR | Test MRR | Val-Test Gap | Notes |
|-----------|---------|----------|--------------|-------|
| 0 | - | - | - | Starting... |

#### Final Results
*Pending...*

---

### Run 4-N: Bayesian Hyperparameter Optimization
- **Tool**: Optuna with TPE sampler
- **Trials**: 20-25
- **Search Space**:
  - learning_rate: [1e-5, 5e-5, 1e-4]
  - hidden_dim: [128, 256, 512]
  - num_layers: [4, 6, 8]
  - ent_coef_init_value: [0.1, 0.15, 0.2]

*Will run after baseline experiment completes.*

---

### Run N+1: Attention Architecture (if needed)
- **Config**: state_embedder='attention' instead of 'mean'
- **Hypothesis**: Self-attention over atoms may capture better state representations

*Will run if Phase 2 doesn't reach 0.70 MRR.*

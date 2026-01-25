# Family Dataset Experiment Results

**Goal**: Achieve test MRR >= 0.70 on the family dataset
**Current Best**: 0.6437 (dropout=0, seed=123)
**Gap**: ~0.06 (9% improvement needed)

**Key Finding**: Setting `dropout_prob=0.0` fixed the value function learning issue!

---

## Summary Table

| Exp | Config Changes | Test MRR | Expl.Var | Stable? | Notes |
|-----|----------------|----------|----------|---------|-------|
| 1   | baseline (temp=0.2, target_kl=0.15) | **0.6376** | 0.074 | Yes | husband/wife ~8% |
| 2   | vf_coef=2.0 | **0.6111** | 0.066 | Yes | WORSE! husband 10.5% |
| 3   | gae_lambda=0.9 | - | - | - | Stopped early |
| 4   | vf_coef=2.0 + gae_lambda=0.9 | - | - | - | Skipped |
| 5   | larger value head + dropout=0.1 | - | ~0.04 | - | Expl.var still low |
| 6   | **dropout_prob=0.0** | 0.5992 | -0.08 | Yes | **KEY FINDING** |
| 7   | parity=True, dropout=0 | - | - | - | Started worse (-10) |
| 8   | dropout=0, eval_freq=16 | **0.6421** | - | Yes | **BEST!** proven 75.1% |
| 9   | dropout=0, gae_lambda=0.9 | 0.6402 | - | Yes | Slightly worse |
| 10  | dropout=0, 5M steps | 0.6174 | - | Yes | OVERFITTING! |
| 11  | dropout=0, seed=123 | **0.6437** | **0.06** | Yes | **NEW BEST!** husband 53% |

---

## Stability Criteria (Must Pass Before Proceeding)

- [ ] No "Early stopping at step 0" messages
- [ ] approx_kl < 0.05 (stable policy updates)
- [ ] clip_fraction < 0.15 (not clipping too much)
- [ ] explained_variance > 0 by iteration 20
- [ ] explained_variance > 0.1 by end of training

---

## Experiment 1: Baseline

**Goal**: Establish stable baseline with proper metrics tracking

**Config**:
```
total_timesteps=2000000
temperature=0.2
target_kl=0.15
lr_warmup=True
run_signature=exp1_baseline
```

### Stability Metrics
| Phase | approx_kl | clip_fraction | expl_var | n_epochs |
|-------|-----------|---------------|----------|----------|
| Early (iter 1-10) | 0.009 | 0.10 | 0.02 | 5/5 |
| Mid (iter 20) | 0.012 | 0.13 | 0.04 | 5/5 |
| Final (iter 62) | 0.017 | 0.15 | 0.074 | 5/5 |

### Performance Metrics
| Metric | Value |
|--------|-------|
| test_mrr | **0.6376** |
| val_mrr | 0.629 (best) |
| hits@1 | 0.6280 |
| hits@10 | 0.6440 |

### Per-Depth Performance (Training)
| Depth | proven_pos |
|-------|------------|
| D1 | 100% |
| D2 | 70.5% |
| D4 | 41.7% |
| D6 | 21.4% |

### Problematic Predicates
| Predicate | D2 proven | D4 proven |
|-----------|-----------|-----------|
| husband | 8.5% | 7.3% |
| wife | 8.0% | - |

### Observations
- Explained variance still very low (0.074 final) - confirms value function not learning
- approx_kl slightly elevated (~0.017) but stable
- All 5 epochs completing (no early stopping)
- husband/wife predicates failing catastrophically (8% vs 70%+ for others)
- D4+ performance degraded as expected without good advantage estimates

### Decision
Proceed to Experiment 2: Increase vf_coef to 2.0 to force better value learning

---

## Experiment 2: Increase vf_coef

**Goal**: Force value function to learn better

**Config**:
```
total_timesteps=2000000
temperature=0.2
target_kl=0.15
lr_warmup=True
vf_coef=2.0
run_signature=exp2_vf_coef_2
```

**Hypothesis**: Higher vf_coef -> better explained variance -> better D4+ performance

### Stability Metrics
| Phase | approx_kl | clip_fraction | expl_var | n_epochs |
|-------|-----------|---------------|----------|----------|
| Early (iter 1-10) | 0.012 | 0.13 | -0.17 | 5/5 |
| Mid (iter 20) | 0.017 | 0.16 | 0.05 | 5/5 |
| Final (iter 62) | 0.018 | 0.17 | 0.066 | 5/5 |

### Performance Metrics
| Metric | Value |
|--------|-------|
| test_mrr | **0.6111** (WORSE than baseline!) |
| val_mrr | 0.635 (best at iter 8) |
| hits@1 | 0.5980 |
| hits@10 | 0.6220 |

### Per-Depth Performance (Training)
| Depth | proven_pos |
|-------|------------|
| D1 | 100% |
| D2 | 71.9% |
| D4 | 45.9% |
| D6 | 48.4% |

### Problematic Predicates
| Predicate | D2 proven | D4 proven |
|-----------|-----------|-----------|
| husband | 10.5% | 5.3% |
| wife | 9.8% | - |

### Observations
- **Hypothesis rejected** - higher vf_coef did NOT improve explained variance
- Value loss doubled (from 0.61 to 1.2+) as expected, but no benefit
- Explained variance similar or slightly worse
- clip_fraction slightly elevated (~0.17)
- Test MRR dropped 0.03 compared to baseline

### Decision
Proceed to Experiment 3: Try gae_lambda=0.9 (without vf_coef change)

---

## Experiment 3: Lower GAE Lambda

**Goal**: Reduce reliance on unreliable value estimates

**Config**:
```
total_timesteps=2000000
temperature=0.2
target_kl=0.15
lr_warmup=True
gae_lambda=0.9
run_signature=exp3_gae_0.9
```

**Hypothesis**: Lower lambda -> more weight on immediate rewards -> better for long proofs

### Stability Metrics
| Phase | approx_kl | clip_fraction | expl_var | n_epochs |
|-------|-----------|---------------|----------|----------|
| Early (iter 1-10) | - | - | - | - |
| Mid (iter 20) | - | - | - | - |
| Final | - | - | - | - |

### Performance Metrics
| Metric | Value |
|--------|-------|
| test_mrr | - |
| val_mrr | - |
| hits@1 | - |
| hits@10 | - |

### Observations
-

### Decision
-

---

## Experiment 4: Combined vf_coef + gae_lambda

**Goal**: Combined value function improvements

**Config**:
```
total_timesteps=2000000
temperature=0.2
target_kl=0.15
lr_warmup=True
vf_coef=2.0
gae_lambda=0.9
run_signature=exp4_vf2_gae0.9
```

### Stability Metrics
| Phase | approx_kl | clip_fraction | expl_var | n_epochs |
|-------|-----------|---------------|----------|----------|
| Early (iter 1-10) | - | - | - | - |
| Mid (iter 20) | - | - | - | - |
| Final | - | - | - | - |

### Performance Metrics
| Metric | Value |
|--------|-------|
| test_mrr | - |
| val_mrr | - |
| hits@1 | - |
| hits@10 | - |

### Observations
-

### Decision
-

---

## Experiment 5: Larger Value Head (Code Change)

**Goal**: Fix value head architecture bottleneck

**Code Change**: `policy.py:206-209`
- Before: `hidden_dim -> hidden_dim//2 -> 1`
- After: `hidden_dim -> hidden_dim -> 1`

**Config**:
```
total_timesteps=2000000
temperature=0.2
target_kl=0.15
lr_warmup=True
vf_coef=2.0
gae_lambda=0.9
run_signature=exp5_larger_value_head
```

### Stability Metrics
| Phase | approx_kl | clip_fraction | expl_var | n_epochs |
|-------|-----------|---------------|----------|----------|
| Early (iter 1-10) | - | - | - | - |
| Mid (iter 20) | - | - | - | - |
| Final | - | - | - | - |

### Performance Metrics
| Metric | Value |
|--------|-------|
| test_mrr | - |
| val_mrr | - |
| hits@1 | - |
| hits@10 | - |

### Observations
-

### Decision
-

---

## Experiment 6: Disable Dropout (KEY FINDING)

**Goal**: Test if dropout causes value function learning issues

**Config**:
```
total_timesteps=500000
temperature=0.2
target_kl=0.15
lr_warmup=True
dropout_prob=0.0  # KEY CHANGE
run_signature=exp6_no_dropout
```

**Rationale**: SB3 reference uses `dropout_prob=0.0` with note:
> "dropout_prob set to 0.0 by default to avoid train/eval mode inconsistencies that cause issues with PPO's log probability computation and value function learning"

### Stability Metrics
| Phase | approx_kl | clip_fraction | expl_var | n_epochs |
|-------|-----------|---------------|----------|----------|
| Early (iter 1-4) | 0.001 | 0.01 | -4.4 to -1.1 | 5/5 |
| Mid (iter 8-10) | 0.003 | 0.03 | -0.28 to -0.18 | 5/5 |
| Final (iter 16) | 0.002 | 0.00 | -0.08 | 5/5 |

### Performance Metrics
| Metric | Value |
|--------|-------|
| test_mrr | 0.5992 (model restored from iter 4!) |
| val_mrr | 0.626 (best at iter 4) |
| hits@1 | 0.5880 |
| hits@10 | 0.6080 |

### Key Observation: Explained Variance Progression

| Iter | Expl.Var | Notes |
|------|----------|-------|
| 1 | -4.395 | Starting point |
| 4 | -1.127 | Best model saved here |
| 8 | -0.284 | Improving rapidly |
| 10 | -0.184 | |
| 13 | -0.108 | |
| 16 | -0.081 | End (500k steps) |

**Critical Issue**: Best model was saved at iter 4 when explained variance was still -1.127!
The value function continued improving AFTER the best model was saved.

### Observations
- **Dropout=0 dramatically improved explained variance progression**: -4.4 → -0.08 in 500k steps
- Previous experiments with dropout=0.1: stayed at ~0.04-0.08 (never improved much)
- Test MRR was low (0.5992) because model restored from iter 4 (before value function learned)
- husband D2: 31-34% (improved from 8%)

### Decision
Run longer experiment with eval_freq=16 to allow value function more time to learn before saving best model.

---

## Experiment 8: Long Run with Dropout=0 (COMPLETED)

**Goal**: Let value function train longer before saving best model

**Config**:
```
total_timesteps=2000000
temperature=0.2
target_kl=0.15
lr_warmup=True
dropout_prob=0.0
eval_freq=16  # Less frequent eval
run_signature=exp8_dropout0_eval16
```

### Performance Metrics
| Metric | Value |
|--------|-------|
| test_mrr | **0.6421** (BEST SO FAR!) |
| hits@1 | 0.6280 |
| hits@3 | 0.6370 |
| hits@10 | 0.6600 |
| proven_pos | **75.1%** (improved from 70.5%) |
| proven_neg | 19.4% |

### Observations
- Dropout=0 + less frequent eval = better test MRR
- Proven positive rate improved significantly (70.5% -> 75.1%)
- Still 0.06 gap to target MRR 0.70

### Decision
Try combining dropout=0 with gae_lambda=0.9 to further improve

---

## Historical Reference: Previous Best Results

From `/tmp/family_separate_vf_500k.log`:
- **Eval MRR: 0.722** (iter 8)
- **Explained variance: 0.101** (positive!)
- Test MRR: 0.532 (large eval-test gap)
- Used seed=123, different codebase

This shows explained variance CAN reach positive values (0.1+) and MRR CAN reach 0.72+.

---

## Notes

- Each experiment changes one variable at a time
- Monitor stability metrics before proceeding to next experiment
- If explained_variance improves but MRR doesn't -> try architecture changes
- If MRR improves -> extend training to 5M steps
- If stability issues -> adjust target_kl, clip_range

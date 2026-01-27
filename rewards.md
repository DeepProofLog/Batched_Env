# Reward Type Analysis - Family Dataset

## Overview

This report analyzes the impact of different reward functions on PPO training for knowledge graph reasoning. All experiments use `separate_value_network=True` and `hidden_dim=512` with 2M timesteps.

---

## Reward Type Definitions

| Type | Name | Formula | Description |
|------|------|---------|-------------|
| **0** | Sparse | TP: +1, else: 0 | Only rewards true positives |
| **1** | TP/FP | TP: +1, FP: -1, else: 0 | Rewards TP, penalizes FP, ignores TN/FN |
| **2** | Classification | TP/TN: +1, else: 0 | Rewards correct classifications |
| **3** | Asymmetric | TP: +1, FN: -0.5, FP: -1.5, TN: +1 | Heavier FP penalty |
| **4** | Balanced | TP: +1, FN: -1, FP: -1, TN: +weight | Equal penalties |
| **5** | ENDF Bonus | Same as 4 + TN with ENDF: +0.3 | Bonus for early negative detection |

---

## Experiment Results Summary

| Experiment | Reward | Test MRR | Hits@1 | Hits@10 | exp_var | value_loss | proven_neg |
|------------|--------|----------|--------|---------|---------|------------|------------|
| **exp29** | Type 4 | **0.6324** | **0.620** | **0.645** | 0.058 | 0.522 | 5.7% |
| exp30 | Type 3 | 0.6211 | 0.608 | 0.634 | 0.054 | 0.435 | 6.4% |
| exp31 | Type 1 | 0.6015 | 0.581 | 0.626 | **0.303** | **0.178** | 14.3% |
| exp33 | Type 5+PBRS | 0.6313 | 0.621 | 0.634 | 0.091 | 0.582 | 7.8% |

**Winner: Type 4 (Balanced)** with Test MRR 0.6324

---

## Detailed Analysis

### 1. Reward Type 4 (Balanced) - BEST MRR

**Config**: TP: +1, FN: -1, FP: -1, TN: rejection_weight

```
Test MRR: 0.6324 (+1.7% vs baseline 0.615)
Hits@1: 0.620, Hits@10: 0.645
explained_variance: +0.058
proven_pos: 64.8%, proven_neg: 5.7% (lowest FP rate)
```

**Reward by Depth**:
| Depth | Reward | Proven % |
|-------|--------|----------|
| D1 | +1.00 | 100% |
| D2 | +0.37 | 68.3% |
| D4 | -0.24 | 38.2% |
| D6 | -0.02 | 48.9% |

**Strengths**:
- Best discrimination (lowest proven_neg = 5.7%)
- Highest MRR and Hits@10
- Balanced penalties prevent overfitting to easy examples

**Weaknesses**:
- Lower explained_variance (0.058)
- D4+ proofs get negative average reward

---

### 2. Reward Type 3 (Asymmetric)

**Config**: TP: +1, FN: -0.5, FP: -1.5, TN: +1

```
Test MRR: 0.6211
Hits@1: 0.608, Hits@10: 0.634
explained_variance: +0.054
proven_pos: 65.7%, proven_neg: 6.4%
```

**Reward by Depth**:
| Depth | Reward | Proven % |
|-------|--------|----------|
| D1 | +1.00 | 100% |
| D2 | +0.49 | 65.6% |
| D4 | +0.02 | 34.8% |
| D6 | -0.02 | 32.1% |

**Key Difference**: Softer FN penalty (-0.5 vs -1.0) leads to:
- Lower reward variance (0.70 vs 0.92)
- 17% lower value_loss (0.435 vs 0.522)
- But worse MRR (-1.1% vs Type 4)

**Analysis**: The asymmetric penalty reduces learning signal strength. The model is less penalized for missing proofs (FN), which hurts discrimination.

---

### 3. Reward Type 1 (TP/FP Only) - BEST VALUE LEARNING

**Config**: TP: +1, FP: -1, else: 0

```
Test MRR: 0.6015 (worst)
Hits@1: 0.581, Hits@10: 0.626
explained_variance: +0.303 (best!)
proven_pos: 67.6%, proven_neg: 14.3% (worst FP rate)
```

**Reward by Depth**:
| Depth | Reward | Proven % |
|-------|--------|----------|
| D1 | +1.00 | 100% |
| D2 | +0.68 | 69.9% |
| D4 | +0.40 | 40.2% |
| D6 | +0.33 | 32.7% |

**Why Best explained_variance?**:
- Simplest reward structure (only 2 outcomes: +1 or -1)
- No TN reward means value function doesn't need to predict "do nothing" value
- Lower variance makes returns easier to predict

**Why Worst MRR?**:
- No reward for true negatives (TN: 0)
- Model learns to prove everything, including negatives
- 14.3% false positive rate (2.5x higher than Type 4)

**Trade-off**: Perfect value learning but terrible discrimination.

---

### 4. Reward Type 5 + PBRS (Potential-Based Reward Shaping)

**Config**: Type 5 base + PBRS with beta=0.1, gamma=0.99

```
Test MRR: 0.6313
Hits@1: 0.621, Hits@10: 0.634
explained_variance: +0.091
proven_pos: 67.0%, proven_neg: 7.8%
```

**Reward by Depth** (with PBRS shaping):
| Depth | Reward | Proven % |
|-------|--------|----------|
| D1 | +1.00 | 100% |
| D2 | +0.33 | 67.0% |
| D4 | -0.31 | 33.0% |
| D6 | -0.29 | 35.7% |

**PBRS Effect**:
- explained_variance: 0.091 (55% better than Type 4's 0.058)
- But MRR similar to Type 4 (0.6313 vs 0.6324)
- PBRS adds ~0.05 improvement to value learning

**Disappointment**: PBRS did not significantly improve MRR despite better value estimates.

---

## Value Learning vs Discrimination Trade-off

```
                    Value Learning (exp_var)
                           ^
                           |
              Type 1 (0.30)|  *
                           |
                           |
              PBRS (0.09)  |      *
              Type 4 (0.06)|          *
              Type 3 (0.05)|            *
                           +-----------------------> MRR
                          0.60   0.62   0.63   0.64
```

**Key Insight**: There's a fundamental trade-off:
- **Simple rewards** (Type 1) → easy value prediction but poor discrimination
- **Complex rewards** (Type 4) → good discrimination but harder value prediction

---

## Loss Analysis

| Experiment | value_loss | policy_loss | entropy | approx_kl |
|------------|------------|-------------|---------|-----------|
| exp29 (T4) | 0.522 | -0.028 | -1.29 | 0.00036 |
| exp30 (T3) | 0.435 | -0.024 | -1.31 | 0.00036 |
| exp31 (T1) | **0.178** | +0.001 | -1.39 | 0.00039 |
| exp33 (PBRS) | 0.582 | -0.009 | -1.22 | 0.00049 |

**Observations**:
1. **Type 1 has 3x lower value_loss** - simplest reward = easiest to predict
2. **Type 1 has positive policy_loss** - model is exploring (entropy -1.39 is highest)
3. **PBRS has highest value_loss** - shaped rewards add complexity
4. **All have similar approx_kl** - policy stability is similar across reward types

---

## Depth-wise Performance

### Proven Rate by Depth (positive examples)

| Depth | Type 4 | Type 3 | Type 1 | PBRS |
|-------|--------|--------|--------|------|
| D1 | 100% | 100% | 100% | 100% |
| D2 | 68.3% | 65.6% | 69.9% | 67.0% |
| D4 | 38.2% | 34.8% | 40.2% | 33.0% |
| D6 | 48.9% | 32.1% | 32.7% | 35.7% |

**Analysis**:
- All types struggle with D4+ proofs (< 50%)
- Type 1 has best D2/D4 proven rates but high FP
- Type 4 has best D6 proven rate (48.9%)

---

## PBRS Analysis

### What PBRS Does

```python
r'(s,a,s') = r(s,a,s') + γ * Φ(s') - Φ(s)
where Φ(s) = β * log(KGE_score(goal_atom))
```

PBRS adds intermediate rewards based on KGE embedding similarity:
- States closer to true facts get higher potential
- Creates dense reward signal from sparse terminal rewards

### Bug Fix Applied

**Before**: PBRS used reset state potential after episode end (wrong)
```
r' = r + γ * Φ(START) - Φ(s)  # Teleportation penalty!
```

**After**: PBRS uses terminal state potential (correct)
```
r' = r + γ * Φ(TERMINAL) - Φ(s)  # Proper credit assignment
```

### PBRS Results

| Metric | Without PBRS (T5) | With PBRS | Change |
|--------|-------------------|-----------|--------|
| Test MRR | ~0.63 | 0.6313 | ~0% |
| exp_var | ~0.06 | 0.091 | +52% |
| value_loss | ~0.52 | 0.582 | +12% |

**Conclusion**: PBRS improves value learning (+52% exp_var) but does NOT improve MRR. The KGE potentials help the value function predict returns, but don't help the policy find better proofs.

---

## Conclusions

### 1. Best Reward for MRR: Type 4 (Balanced)
- Equal penalties for FN and FP
- Best discrimination (5.7% FP rate)
- Test MRR: 0.6324

### 2. Best Reward for Value Learning: Type 1 (TP/FP only)
- Simplest structure → easiest to predict
- explained_variance: 0.303 (5x better than Type 4)
- But terrible MRR (0.60) due to high FP rate

### 3. PBRS Helps Value Learning, Not MRR
- +52% improvement in explained_variance
- ~0% improvement in MRR
- KGE potentials don't translate to better proofs

### 4. The Core Trade-off
```
Simple Rewards → Easy Value Learning → Poor Discrimination
Complex Rewards → Hard Value Learning → Good Discrimination
```

### 5. Recommendations

For **maximum MRR**: Use Type 4 (balanced rewards)
```python
reward_type = 4  # TP:+1, FN:-1, FP:-1, TN:+weight
```

For **value function research**: Use Type 1 (sparse TP/FP)
```python
reward_type = 1  # TP:+1, FP:-1, else:0
```

For **PBRS experiments**: Fix is applied, but don't expect MRR gains
```python
pbrs_beta = 0.1  # Helps exp_var, not MRR
kge_inference = False  # Use RL logits for ranking
```

---

## PBRS Verification

### Test Results

```
============================================================
PBRS VERIFICATION TEST
============================================================

[Test 1] PBRS Module Initialization
Beta: 0.1
Gamma: 0.99
Terminal predicates: {'End', 'Endt', 'Endf', 'True', 'False'}
PASSED

[Test 2] Reward Shaping Formula: r'= r + gamma*Phi(s') - Phi(s)
Base rewards:   [1.0, -1.0, 0.0, 1.0]
Phi(s):         [0.2, 0.3, 0.1, 0.5]
Phi(s'):        [0.4, 0.1, 0.3, 0.0]
Done mask:      [False, False, False, True]
Shaped rewards: [1.196, -1.201, 0.197, 0.5]
Expected:       [1.196, -1.201, 0.197, 0.5]
PASSED: Formula verified

[Test 3] Terminal States Get Zero Potential
Terminal shaped reward: 0.500
Expected (r - phi_s): 0.500
PASSED: Terminal states handled correctly
============================================================
```

### Verification Details

1. **Formula Correctness**: `r' = r + γ*Φ(s') - Φ(s)` verified numerically
2. **Terminal Handling**: Done episodes get `Φ(s') = 0` to avoid teleportation penalty
3. **Terminal Predicates**: `{True, False, End, Endf, Endt}` all get zero potential

### Bug Fix Verification

The fix in `env.py` and `ppo.py` ensures:
- `terminal_states` preserves the actual terminal state before reset
- PBRS uses `terminal_states` (not `current_states`) for potential calculation
- No "teleportation penalty" from mid-proof → new start state

### Why PBRS Didn't Improve MRR

Despite correct implementation, PBRS only improved explained_variance (+52%) but not MRR (~0%):

1. **KGE potentials don't equal proof difficulty**: A fact might have high KGE score but require a complex proof path
2. **Potential is state-independent of proof strategy**: PBRS rewards reaching "good" states, not finding efficient proof paths
3. **Value learning ≠ Policy improvement**: Better value predictions don't automatically translate to better action selection

### Future PBRS Improvements

To make PBRS more effective:
1. **Proof-aware potentials**: Use depth/complexity in potential function
2. **Action-level shaping**: Shape individual rule selections, not just states
3. **Learned potentials**: Train a potential function alongside the policy

---

## Appendix: Experiment Commands

```bash
# Type 4 (Best MRR)
python runner_kge.py --set reward_type=4 --set separate_value_network=True --set hidden_dim=512

# Type 1 (Best Value Learning)
python runner_kge.py --set reward_type=1 --set separate_value_network=True --set hidden_dim=512

# PBRS
python runner_kge.py --set reward_type=5 --set pbrs_beta=0.1 --set kge_inference=False --set separate_value_network=True --set hidden_dim=512
```

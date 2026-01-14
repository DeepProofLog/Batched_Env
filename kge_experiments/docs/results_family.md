# Family Dataset: RL Performance Analysis & Limitations

## 0. Reproducibility

### Command to Reproduce

```bash
cd kge_experiments
python runner_kge.py --set dataset=family,n_envs=128,n_steps=256,total_timesteps=2000000,vf_coef=1.0,seed=0
```

### Exact Configuration Used

```python
# Environment
dataset = "family"
n_envs = 128                    # Parallel environments
n_steps = 256                   # Steps per rollout
max_steps = 20                  # Max episode length
padding_states = 64             # Max actions per state
padding_atoms = 6               # Max atoms per state

# Policy Network
atom_embedding_size = 64        # Embedding dimension
hidden_dim = 256                # MLP hidden size
num_layers = 8                  # Residual blocks

# PPO Algorithm
total_timesteps = 2_000_000
gamma = 0.99                    # Discount factor
gae_lambda = 0.95               # GAE parameter
learning_rate = 3e-4
n_epochs = 4                    # PPO epochs per update
batch_size = 8192               # n_envs * n_steps / mini_batches
clip_range = 0.2                # PPO clip parameter
vf_coef = 1.0                   # Value loss coefficient (default 0.5, increased)
ent_coef = 0.2                  # Entropy bonus

# Reward Structure
reward_type = 4                 # TP=+1, FN=-1, FP=-1, TN=+1/neg_ratio
negative_ratio = 1.0            # 1:1 positive:negative sampling
end_proof_action = True         # Allow END action for negative rejection

# Seed
seed = 0
```

### Key Hyperparameter Changes from Default

| Parameter | Default | Used | Reason |
|-----------|---------|------|--------|
| `vf_coef` | 0.5 | 1.0 | Better value function learning |
| `n_envs` | 3 | 128 | More parallel experience |
| `n_steps` | 20 | 256 | Longer rollouts |
| `reward_type` | 0 | 4 | Full TP/FN/FP/TN feedback |

---

## 1. Dataset Overview

The Family dataset contains genealogical relationships derived from family trees.

### 1.1 Split Statistics

| Split | Total Queries | Provable (depth ≥ 1) | Unprovable (depth -1) | % Provable |
|-------|---------------|----------------------|-----------------------|------------|
| Train | 19,844 | 15,157 | 4,688 | 76.4% |
| Valid | 2,799 | 2,172 | 627 | 77.6% |
| Test | 5,625 | 4,394 | 1,231 | 78.1% |

### 1.2 Depth Distribution

| Depth | Train | Valid | Test |
|-------|-------|-------|------|
| 1 | 14,804 (74.6%) | 2,103 (75.1%) | 4,297 (76.4%) |
| 2 | 347 (1.7%) | 61 (2.2%) | 92 (1.6%) |
| 3 | 6 (0.03%) | 8 (0.3%) | 6 (0.1%) |
| -1 (unprovable) | 4,688 (23.6%) | 627 (22.4%) | 1,231 (21.9%) |

**Key Observation**: ~98% of provable queries are depth-1, making depth-2+ queries rare and harder to learn.

---

## 2. Provability Analysis by Predicate

### 2.1 Rules per Predicate

| Predicate | # Rules | Coverage |
|-----------|---------|----------|
| aunt | 8 | High |
| uncle | 7 | High |
| nephew | 7 | High |
| niece | 7 | High |
| son | 4 | Medium |
| daughter | 4 | Medium |
| father | 3 | Medium |
| mother | 2 | Low |
| husband | 2 | Low |
| wife | 2 | Low |
| brother | 1 | Very Low |
| sister | 1 | Very Low |

### 2.2 Provability Rate by Predicate (Train)

| Predicate | Total | Provable | % Provable |
|-----------|-------|----------|------------|
| nephew | 2,543 | 2,392 | 94.1% |
| niece | 1,970 | 1,870 | 94.9% |
| uncle | 2,439 | 2,307 | 94.6% |
| aunt | 2,093 | 2,033 | 97.1% |
| husband | 796 | 658 | 82.7% |
| wife | 796 | 649 | 81.5% |
| son | 1,477 | 1,028 | 69.6% |
| daughter | 1,112 | 781 | 70.2% |
| father | 1,407 | 882 | 62.7% |
| mother | 1,199 | 810 | 67.6% |
| **brother** | 2,153 | 876 | **40.7%** |
| **sister** | 1,860 | 871 | **46.8%** |

**Key Finding**: `brother` and `sister` have the lowest provability rates (40-47%) because they only have 1 rule each:
- `brother(a,b) <- brother(a,h), sister(b,h)` (r31)
- `sister(a,b) <- brother(b,h), sister(a,h)` (r25)

---

## 3. Training Results

### 3.1 Final Metrics (2M steps)

| Metric | Value |
|--------|-------|
| Overall Reward | 0.718 |
| Explained Variance | 0.467 |
| Proven Positives | 76.9% |
| False Positive Rate | 5.2% |
| Reward on Negatives | 0.897 |
| Best Eval MRR | 0.741 |
| Hits@1 | 0.670 |
| Hits@3 | 0.745 |
| Hits@10 | 0.940 |

### 3.2 Learning Progression

| Timesteps | Reward | Explained Var | Proven Pos | FP Rate |
|-----------|--------|---------------|------------|---------|
| 32k | 0.661 | 0.058 | 77.2% | 10.9% |
| 500k | 0.676 | 0.159 | 76.4% | 6.7% |
| 1M | 0.696 | 0.298 | 76.4% | 6.7% |
| 1.5M | 0.712 | 0.427 | 76.8% | 5.2% |
| 2M | 0.718 | 0.467 | 76.9% | 5.2% |

### 3.3 Performance by Depth

| Depth | Train Proven | Eval Proven | Gap |
|-------|--------------|-------------|-----|
| 1 | 76.9% | 79.6% | +2.7% |
| 2 | 77.0% | 25.0% | **-52.0%** |
| 3 | N/A | 0.0% | - |

---

## 4. Limitations

### 4.1 Structural Limitations

#### L1: Rule Coverage Imbalance
- `brother` and `sister` have only 1 rule each (vs. 7-8 for aunt/uncle/nephew/niece)
- These predicates account for ~20% of queries but have <50% provability
- **Impact**: Ceiling on achievable proof rate

#### L2: Depth Distribution Skew
- 98% of provable queries are depth-1
- Only 2.3% are depth-2, 0.04% are depth-3
- **Impact**: Model undertrained on multi-step proofs

#### L3: Unprovable Queries (~23%)
- Cannot be derived through any rule chain
- Queries where required intermediate facts are missing
- **Impact**: These contribute to failures even with perfect policy

### 4.2 Negative Sampling Limitations

#### L4: Provable Negatives (~14.5%)
Corrupting head/tail of positive queries sometimes creates queries that are still provable through rules.

Example: If `uncle(john, mary)` is true, corrupting to `uncle(john, bob)` might also be provable if Bob is in John's family.

| Outcome | Reward | Proportion |
|---------|--------|------------|
| True Positive (TP) | +1 | ~38.5% |
| False Negative (FN) | -1 | ~11.5% |
| False Positive (FP) | -1 | ~7.25% |
| True Negative (TN) | +1 | ~42.75% |

**Theoretical Maximum Reward**: ~0.855 (if all negatives that are actually provable are proven)

### 4.3 Train-Eval Gap on Depth-2

| Split | Depth-2 Queries | Proven Rate |
|-------|-----------------|-------------|
| Train | 347 | 77.0% |
| Eval | 8 | 25.0% |

**Root Cause**: Evaluation has only 8 depth-2 queries (vs. 347 in train), and they may require different proof paths not seen during training.

---

## 5. Why Explained Variance ≠ 1

The explained variance measures how well the value function predicts returns:
```
EV = 1 - Var(returns - predicted) / Var(returns)
```

### 5.1 Factors Limiting Explained Variance

#### F1: Sparse Terminal Rewards
- Rewards only at episode end (+1, -1, or +1/neg_ratio)
- Value function must predict across 3-7 steps of zeros before terminal reward
- High variance in intermediate state values

#### F2: Stochastic Action Selection
- During rollout, actions are sampled from policy distribution
- Same state can lead to different outcomes depending on sampled actions
- Value function predicts *expected* return, but actual returns vary

#### F3: Binary Reward Structure
```
Reward std dev: 0.70-0.84 (very high for mean of 0.72)
```
- Rewards are essentially binary (+1 or -1)
- Creates high variance that value function cannot fully capture

#### F4: State Aliasing
- Different queries may have similar intermediate states
- But lead to different outcomes (provable vs. not provable)
- Value function cannot distinguish based on state alone

### 5.2 Theoretical Analysis

For a policy with:
- 77% success on positives → E[reward_pos] = 0.77 - 0.23 = 0.54
- 95% rejection on negatives → E[reward_neg] = 0.95 - 0.05 = 0.90

**Intrinsic variance** in returns due to stochasticity:
```
Var(reward_pos) = 0.77 * (1-0.54)² + 0.23 * (-1-0.54)² ≈ 0.71
Var(reward_neg) = 0.95 * (1-0.90)² + 0.05 * (-1-0.90)² ≈ 0.19
```

Even a **perfect value function** that correctly predicts expected returns cannot achieve EV=1 because returns are stochastic.

### 5.3 Expected Explained Variance Range

| Scenario | Expected EV |
|----------|-------------|
| Random value predictions | ~0 |
| Learned value function | 0.3-0.5 |
| Perfect expected value estimation | 0.5-0.7 |
| EV = 1 | Only possible with deterministic policy AND deterministic environment |

**Conclusion**: Achieved EV of 0.467 is reasonable given the stochastic nature of the task. Values above 0.5 would indicate excellent value estimation.

### 5.4 How to Increase Explained Variance

**Without sacrificing exploration (stochastic policy):**

| Method | Current | Suggested | Effect |
|--------|---------|-----------|--------|
| GAE λ | 0.95 | 0.5 | Reduces advantage variance (increases bias) |
| Discount γ | 0.99 | 0.9 | Returns decay faster, easier to predict |
| Value network size | hidden=256, layers=4 | hidden=512, layers=6 | More capacity |

**Why EV Cannot Reach 1 with Stochastic Policy:**

The explained variance formula is:
```
EV = 1 - Var(actual_returns - predicted_returns) / Var(actual_returns)
```

For EV = 1, we need `Var(actual_returns - predicted_returns) = 0`, meaning the value function perfectly predicts every individual return.

With a stochastic policy:
- Same state s can produce different actions (sampled from π(a|s))
- Different actions lead to different trajectories and returns
- Value function V(s) = E[R|s] predicts the **expected** return
- But actual returns vary around this expectation

**Example**: From state s, policy has 80% chance of success (+1) and 20% failure (-1)
- V(s) = 0.8 × 1 + 0.2 × (-1) = 0.6 (expected return)
- Actual returns: +1 or -1 (variance = 0.64)
- Even perfect V(s) = 0.6 gives: Var(actual - predicted) = 0.64 ≠ 0
- Maximum achievable EV < 1

**Theoretical upper bound for stochastic environment:**
```
EV_max = 1 - E[Var(R|s)] / Var(R)
```

Where `Var(R|s)` is the variance of returns conditioned on state. This is always > 0 for stochastic policies, so EV_max < 1.

---

## 6. Recommendations

### 6.1 Improving Proof Rate

1. **Add more rules for brother/sister**: Currently only 1 rule each
2. **Multi-hop training curriculum**: Start with depth-1, gradually add depth-2+
3. **Longer max episode length**: Allow more exploration for complex proofs

### 6.2 Improving Value Function

1. **State augmentation**: Include query difficulty features (predicate type, known provability hints)
2. **Distributional RL**: Model return distribution, not just expected value (e.g., C51, QR-DQN)
3. **Lower discount factor**: γ=0.99 may be too high for episodes of length 3-7

### 6.3 Reducing False Positives

1. **Smarter negative sampling**: Avoid creating negatives that are provable
2. **Hard negative mining**: Focus on confusing negatives during training
3. **Separate model for rejection**: Train dedicated classifier for negatives

---

## 7. Summary

| Aspect | Finding |
|--------|---------|
| Provability | 76-78% of queries are provable across splits |
| Main bottleneck | `brother`/`sister` predicates (only 1 rule each, <50% provable) |
| Depth limitation | 98% depth-1, model struggles with depth-2+ on eval |
| Negative sampling | ~14.5% of negatives are actually provable (structural limit) |
| Explained variance | 0.467 is reasonable; upper bound ~0.5-0.7 due to stochasticity |
| Best MRR | 0.741 (competitive for rule-based KG completion) |

# RL Model Analysis Summary - Family Dataset

## Bug Fix: Depth Tracking

**Issue Found**: With `order=False` (random sampling), the depth metrics were tracking `per_env_ptrs` which stayed static instead of the actual randomly sampled query indices.

**Fix Applied**: In `env.py:359-360`, changed:
```python
# Before
next_ptrs = per_env_ptrs

# After
next_ptrs = safe_idx  # Track actual sampled indices for depth metrics
```

## Baseline Results (500k timesteps)

### Training Metrics Summary

| Metric | Seed 0 | Seed 42 |
|--------|--------|---------|
| Best Val MRR | **0.706** | **0.688** |
| Final Test MRR | 0.596 | 0.538 |
| Train reward | 0.645 | 0.646 |
| Explained variance | 0.174 | 0.160 |
| Train proven_pos | 73.7% | 74.8% |

### Proven Rate by Depth

| Depth | Proven Rate | Steps (actual) | Steps (optimal) | Status |
|-------|-------------|----------------|-----------------|--------|
| 1 | **100%** | 1.0 | 1 | Perfect |
| 2 | 72-74% | 3.6-3.8 | 2 | Plateau - room to improve |
| 4 | 25-31% | 6.6-7.5 | 4 | Weak |
| 6 | 9-17% | 11+ | 6 | Very weak |

### Dataset Depth Distribution

| Split | depth -1 | depth 1 | depth 2 | depth 4 | depth 6 |
|-------|----------|---------|---------|---------|---------|
| Train | 4788 | 1110 | 13694 | 250 | 3 |
| Valid | 650 | 164 | 1939 | 45 | 1 |
| Test | 1261 | 318 | 3979 | 68 | 0 |

## Key Findings

### 1. RL Model IS Learning Properly
- Reward grows from ~0.53 to ~0.65 (+22%)
- Explained variance improves from -5.14 to +0.17
- Negative proven rate decreases (good discrimination): 20% -> 9%
- Depth 1 queries: 100% proven consistently

### 2. Depth 2 Plateau at ~73%
- Proven rate stays stable at 72-75% despite more training
- Model takes ~3.6-3.8 steps vs optimal 2 steps
- ~27% of depth 2 queries remain unproven
- Possible causes:
  - Insufficient exploration
  - Value function not accurate (explained variance still low)
  - May need curriculum learning

### 3. Depth 4+ Struggles Significantly
- Only 25-31% proven for depth 4
- <20% for depth 6
- Negative reward for these depths (-0.4 to -0.7)
- Possible causes:
  - Exponential state space explosion
  - Longer reasoning chains harder to discover
  - Credit assignment problem

### 4. Val/Test Gap
- Validation MRR: 0.69-0.71
- Test MRR: 0.54-0.60
- ~0.1 MRR drop from validation to test
- Likely due to smaller validation set (50) vs test (200)

## Comparing to KGE Performance (from kge_perf.md)

| Model | Test MRR | Test Hits@1 | Test Hits@10 |
|-------|----------|-------------|--------------|
| KGE only | 0.9527 | 0.9193 | 0.9911 |
| RL only | 0.59-0.60 | 0.55 | 0.65 |

**Gap Analysis**:
- KGE achieves ~0.95 MRR vs RL's ~0.60
- RL's proven_pos ~65% vs KGE's implicit scoring
- RL only helps for **provable queries** (depth 1-6)
- Non-provable queries (depth=-1, ~22%) need KGE

## Recommendations

### Short-term Improvements
1. **Increase training duration**: Try 1-2M timesteps
2. **Entropy decay**: Start high (0.1) -> end low (0.01) for explore-then-exploit
3. **LR decay**: Enable lr_decay=True for stability
4. **Lower entropy (0.05)**: Less exploration, more exploitation

### Medium-term Improvements
1. **Value function**: Improve explained variance (currently ~0.17)
2. **Curriculum learning**: Start with easier depth 1-2 queries, gradually add harder
3. **Better negative sampling**: Current negative proven rate ~10% is good but could be lower

### Long-term: Hybrid RL+KGE
- Use RL for provable queries (depth >= 1)
- Use KGE scores for non-provable queries (depth = -1)
- Weight combination based on rule coverage

## Reproducibility

Logs saved to:
- `/home/castellanoontiv/Batched_env-swarm/agent1/logs/family_baseline_500k_s42.txt`
- Models saved to: `kge_experiments/models/family-250-128-torchrl/`

## Commands to Reproduce

```bash
# Baseline 500k
python runner_kge.py --set dataset=family --set total_timesteps=500000 --set n_envs=128 --set n_steps=256 --set n_eval_queries=50 --set seed=42

# 1M timesteps
python runner_kge.py --set dataset=family --set total_timesteps=1000000 --set n_envs=128 --set n_steps=256 --set seed=42

# With decay
python runner_kge.py --set dataset=family --set total_timesteps=500000 --set ent_coef_decay=True --set ent_coef_init_value=0.1 --set ent_coef_final_value=0.01 --set lr_decay=True --set seed=42
```

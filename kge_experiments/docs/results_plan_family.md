# Family Dataset: Improvement Plan

## Current Baseline

| Metric | Value |
|--------|-------|
| MRR | 0.741 |
| Hits@1 | 0.670 |
| Proven Positives | 76.9% |
| Explained Variance | 0.467 |

**Target: +5% MRR → 0.79+**

---

## Phase 1: Immediate (Before New Training)

### 1.1 Investigate Evaluation Discrepancy
**Issue**: Early training showed +0.56% improvement, later showed -2.4% gap between train/eval

**Tasks**:
- [ ] Compare eval queries vs train queries by predicate distribution
- [ ] Check if eval depth-2 queries are systematically harder
- [ ] Analyze per-predicate performance on eval set
- [ ] Log specific failing queries to identify patterns

**Files to investigate**:
- `data/family/valid_depths.txt` - eval query depths
- `data/family/test_depths.txt` - test query depths
- `env.py` - evaluation logic

**Command**:
```bash
# Analyze eval vs train distribution
python -c "
from data_handler import DataHandler
dh = DataHandler('family', 'data')
# Compare distributions
"
```

### 1.2 Preserve Model Checkpoints
**Action**: Backup current best model before any new experiments

**Command**:
```bash
cp -r models/family-250-128-torchrl models/family-baseline-backup-$(date +%Y%m%d)
```

---

## Phase 2: Fix Hybrid Evaluation (+1-2% MRR expected)

### 2.1 Hybrid Evaluation Consistency
**Issue**: Potential inconsistency between training and evaluation modes

**Tasks**:
- [ ] Verify same corruption scheme used in train vs eval
- [ ] Check negative sampling seed consistency
- [ ] Ensure action masking is identical
- [ ] Compare episode termination conditions

**Key files**:
- `ppo.py:evaluate_policy()` - evaluation loop
- `env.py:step_and_reset()` - step logic
- `nn/sampler.py` - corruption logic

**Potential fixes**:
```python
# In config.py, ensure consistency:
corruption_scheme = ['head', 'tail']  # Same for train and eval
eval_deterministic = True              # Greedy actions in eval
```

---

## Phase 3: Curriculum Learning (+2-3% MRR expected)

### 3.1 Train Easy Depths First
**Rationale**: 98% of queries are depth-1; model learns shallow proofs well but struggles with depth-2+

**Implementation**:

```python
# In config.py or runner_kge.py
curriculum_schedule = [
    (0, 500_000, {'train_depth': [1]}),           # Phase 1: depth-1 only
    (500_000, 1_000_000, {'train_depth': [1, 2]}), # Phase 2: add depth-2
    (1_000_000, 2_000_000, {'train_depth': [1, 2, 3]}), # Phase 3: all depths
]
```

**Alternative - Weighted Sampling**:
```python
# Weight depth-2+ queries higher in sampling
depth_weights = {1: 1.0, 2: 5.0, 3: 10.0}  # Oversample harder queries
```

**Files to modify**:
- `env.py:_sample_queries()` - add depth-aware sampling
- `config.py` - add curriculum parameters
- `ppo.py:learn()` - add curriculum scheduling

### 3.2 Depth-Aware Reward Shaping (Optional)
```python
# Bonus reward for proving depth-2+ queries
depth_bonus = {1: 0.0, 2: 0.1, 3: 0.2}
reward = base_reward + depth_bonus.get(query_depth, 0)
```

---

## Phase 4: Monte Carlo Evaluation (+1% MRR expected)

### 4.1 Replace Greedy with Sampled Evaluation
**Rationale**: Greedy policy may get stuck; sampling allows exploration of alternative proof paths

**Current** (greedy):
```python
actions = distribution.mode()  # Always pick highest prob action
```

**Proposed** (Monte Carlo):
```python
# Run K rollouts per query, take best result
def mc_evaluate(query, K=5):
    results = []
    for _ in range(K):
        actions = distribution.sample()  # Stochastic
        result = run_episode(query, actions)
        results.append(result)
    return max(results, key=lambda r: r.success)
```

**Files to modify**:
- `ppo.py:evaluate_policy()` - add MC option
- `config.py` - add `eval_mc_samples: int = 5`

---

## Phase 5: Deep Proof Learning (+1-2% MRR expected)

### 5.1 Improve Value Estimation for Long Episodes
**Issue**: Value function struggles with depth-2+ (variable length episodes)

**Options**:

**A. Separate Value Heads by Depth**:
```python
class MultiDepthValueHead(nn.Module):
    def __init__(self, hidden_dim):
        self.depth1_head = nn.Linear(hidden_dim, 1)
        self.depth2_head = nn.Linear(hidden_dim, 1)
        self.depth3_head = nn.Linear(hidden_dim, 1)
```

**B. Recurrent Value Function**:
```python
# Use LSTM to track proof progress
self.value_lstm = nn.LSTM(hidden_dim, hidden_dim)
```

**C. Auxiliary Depth Prediction**:
```python
# Multi-task: predict value AND expected remaining steps
depth_pred = self.depth_head(features)  # Auxiliary loss
value_pred = self.value_head(features)
loss = value_loss + 0.1 * depth_loss
```

### 5.2 Increase Exploration for Deep Proofs
```python
# Higher entropy for depth-2+ queries
ent_coef_by_depth = {1: 0.1, 2: 0.3, 3: 0.5}
```

---

## Implementation Order

| Priority | Task | Expected Gain | Effort | Status |
|----------|------|---------------|--------|--------|
| 1 | Preserve checkpoints | Safety | 5 min | DONE |
| 2 | Investigate eval discrepancy | Debug | 1-2 hours | DONE |
| 3 | Fix hybrid consistency | +1-2% | 2-3 hours | DONE |
| 4 | Curriculum learning | +2-3% | 3-4 hours | DONE |
| 5 | Monte Carlo eval | +1% | 1-2 hours | DONE |
| 6 | Deep proof improvements | +1-2% | 4-6 hours | DONE |

**Total expected improvement: +5-10% MRR**

---

## Implementation Details

### Files Modified

1. **env.py** - Core environment with curriculum and depth-weighted sampling
   - `set_train_depths(depths)` - Set per-query depth labels
   - `set_curriculum(allowed_depths)` - Filter queries by depth for curriculum learning
   - `set_depth_weights(depth_weights)` - Enable weighted sampling to oversample hard queries
   - `_update_current_weights()` - Recompute normalized weights when curriculum changes
   - `step_and_reset()` - Uses weighted sampling via `torch.multinomial` when enabled

2. **ppo.py** - PPO training with curriculum scheduling and MC evaluation
   - Curriculum scheduling in `learn()` - Updates allowed depths based on training progress
   - `evaluate_policy_mc()` - Monte Carlo evaluation with multiple stochastic rollouts

3. **config.py** - New configuration options
   - `curriculum: bool = False` - Enable curriculum learning
   - `curriculum_schedule: List[Tuple[float, set]]` - Progress thresholds and depth sets
   - `eval_mc: bool = False` - Enable Monte Carlo evaluation
   - `eval_mc_samples: int = 5` - Number of MC rollouts per query
   - `depth_weighted_sampling: bool = False` - Enable depth-weighted query sampling
   - `depth_weights: Dict[int, float]` - Per-depth sampling weights

4. **builder.py** - Initialization of new features
   - Sets `train_depths` on environment from DataHandler
   - Enables `depth_weighted_sampling` if configured

### Usage Examples

```bash
# Training with curriculum learning
python runner_kge.py --set dataset=family,curriculum=True,total_timesteps=2000000

# Training with depth-weighted sampling (oversample depth-2+)
python runner_kge.py --set dataset=family,depth_weighted_sampling=True,depth_weights={1:1.0,2:5.0,3:10.0}

# Evaluation with Monte Carlo rollouts
python runner_kge.py --set dataset=family,eval_mc=True,eval_mc_samples=5

# Full configuration
python runner_kge.py --set dataset=family,curriculum=True,depth_weighted_sampling=True,total_timesteps=2000000
```

---

## Validation Plan

After each change:
1. Run short training (300k steps) to verify no regression
2. Compare train vs eval metrics
3. Check depth-stratified performance
4. Save checkpoint if improvement

```bash
# Quick validation run
python runner_kge.py --set dataset=family,total_timesteps=300000,eval_freq=50000
```

---

## Risk Mitigation

1. **Always backup** before modifying training code
2. **A/B test** changes individually, not all at once
3. **Log everything** - save full metrics per run
4. **Seed control** - use same seed for fair comparison

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| MRR | 0.741 | 0.79 | 0.82 |
| Hits@1 | 0.670 | 0.72 | 0.75 |
| Depth-2 Eval | 25% | 50% | 70% |
| Explained Var | 0.467 | 0.55 | 0.60 |

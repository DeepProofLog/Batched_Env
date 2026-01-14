# WN18RR Experiment Results: Hybrid KGE+RL for Link Prediction

## Overview

This report documents experiments combining Knowledge Graph Embeddings (KGE) with Reinforcement Learning (RL) for link prediction on the WN18RR dataset. The goal is to improve KGE baseline performance while maintaining interpretability through logical proofs.

## Experimental Setup

- **Dataset**: WN18RR (WordNet 18 Relation Reduced)
- **KGE Model**: RotatE (1024 dimensions)
- **RL Algorithm**: PPO (Proximal Policy Optimization)
- **Training**: 200,000 timesteps, 128 environments, 256 steps per rollout
- **Evaluation**: 100 test queries with 100 negative samples each

## Results

| Configuration | MRR | Hits@1 | Hits@3 | Hits@10 | Proven (pos) |
|--------------|-----|--------|--------|---------|--------------|
| KGE-only (RotatE) | 0.8256 | - | - | - | - |
| RL-only | 0.3742 | 0.355 | 0.360 | 0.390 | 0.350 |
| **Hybrid KGE+RL** | **0.8448** | **0.790** | **0.885** | **0.950** | 0.355 |
| Soft facts + hybrid | 0.8230 | 0.769 | 0.860 | 0.917 | 0.375 |
| Neural bridge (gated) | 0.8223 | 0.768 | 0.857 | 0.918 | 0.384 |

## Key Findings

### 1. Hybrid Approach Improves KGE Baseline

The hybrid KGE+RL approach achieves **+2.3% MRR improvement** over the KGE-only baseline:
- KGE-only: 0.8256 MRR
- Hybrid: 0.8448 MRR

### 2. Hits@10 Target Achieved

All hybrid configurations exceed the 0.9 Hits@10 target:
- Basic hybrid: **0.950** (best)
- Soft facts: 0.917
- Neural bridge: 0.918

### 3. RL Alone is Insufficient

RL-only achieves only 0.374 MRR, demonstrating that the logical reasoning component alone cannot match statistical embedding methods. However, RL provides valuable interpretability when combined with KGE.

### 4. Simpler is Better

The basic hybrid scoring approach outperforms more complex configurations:
- Soft facts (using KGE top-k predictions as additional facts) slightly degrades performance
- Neural bridge (learned fusion of KGE and RL scores) does not improve over basic hybrid

## Best Configuration

```python
# Hybrid KGE+RL scoring parameters
kge_inference = True          # Enable hybrid scoring
kge_only_eval = False         # Use hybrid mode (not KGE-only)
kge_eval_kge_weight = 2.0     # Weight for KGE log-probability scores
kge_eval_rl_weight = 1.0      # Bonus for successfully proven queries
kge_fail_penalty = 0.5        # Penalty for failed proof attempts
```

### Scoring Formula

For each candidate answer:
- **If proof succeeds**: `score = kge_weight * kge_logp + rl_weight`
- **If proof fails**: `score = kge_weight * kge_logp - fail_penalty`

## Analysis: Why Hybrid Works

### KGE Performance by Provability

Analysis revealed that KGE already achieves near-perfect performance on provable queries:
- KGE MRR on provable queries: **0.9747**
- KGE MRR on unprovable queries: **0.7476**

This explains why the hybrid improvement is modest but consistent - the RL component provides a small boost by correctly ranking provable queries higher.

### Predicate-Level Analysis

WN18RR predicates fall into two categories:
1. **Symmetric predicates** (e.g., `_similar_to`, `_verb_group`): High provability (~90%)
2. **Chain predicates** (e.g., `_hypernym`, `_has_part`): Lower provability (~40-60%)

The RL agent excels at symmetric predicates where logical rules directly apply.

## Data Leakage Fix

During experiments, we discovered and fixed a critical **data leakage issue**:

- **Problem**: 1,819 test queries (62%) appeared in the soft facts file
- **Root cause**: Soft facts were loaded from KGE top-k predictions without filtering
- **Fix**: Modified `data_handler.py` to filter valid/test queries from soft facts

```python
# data_handler.py fix (lines 250-265)
if prob_facts_enabled:
    queries_to_filter = set()
    for q in self.valid_queries:
        queries_to_filter.add(q)
    for q in self.test_queries:
        queries_to_filter.add(q)

    self._load_probabilistic_facts(
        dataset_name=ds_name,
        filter_queries=queries_to_filter,  # Prevent data leakage
    )
```

## Reproducibility

To reproduce the best hybrid result:

```bash
cd kge_experiments
python runner_kge.py \
  --set total_timesteps=200000 \
  --set seed=250 \
  --set kge_inference=True \
  --set kge_only_eval=False
```

## Conclusion

The hybrid KGE+RL approach successfully combines:
1. **Statistical accuracy** from KGE embeddings (RotatE)
2. **Interpretability** from RL-based logical proof search

The basic hybrid configuration achieves the best results with MRR=0.8448 and Hits@10=0.950, demonstrating that simple fusion strategies outperform more complex approaches like soft facts injection or neural bridging.

---

*Experiments conducted: January 13, 2026*
*Environment: CUDA, PyTorch, TorchRL*

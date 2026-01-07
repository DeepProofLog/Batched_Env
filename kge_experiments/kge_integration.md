# KGE Integration Modules for DeepProofLog

This document describes the implementation of 4 independent KGE (Knowledge Graph Embedding) integration modules for DeepProofLog. These modules enhance the RL-based proof discovery system by leveraging pre-trained KGE scores at various stages of the pipeline.

## Overview

All modules are designed to be:
- **Independent**: Can be enabled/disabled via config flags
- **Non-invasive**: Core RL code (`env.py`, `unification.py`) remains unchanged
- **Pre-computed**: Use offline KGE scores from `top_k_scores/files/` for efficiency

## Module Descriptions

### 1. Probabilistic Facts (`data_handler.py`)

**Purpose**: Expand the fact database with high-confidence KGE-predicted facts.

**Implementation**:
- Loads facts from `top_k_scores/files/kge_top_{dataset}.txt`
- Filters by top-k per predicate or score threshold
- Adds filtered facts to the logical fact database during data loading

**Config Options**:
```python
prob_facts: bool = False          # Enable probabilistic facts
prob_facts_topk: int = None       # Top-K facts per predicate
prob_facts_threshold: float = None # Minimum KGE score threshold
```

**File**: `data_handler.py:_load_probabilistic_facts()`

---

### 2. PBRS Module (`kge_pbrs.py`)

**Purpose**: Potential-Based Reward Shaping to guide RL exploration.

**Formula**: `r' = r + γ*Φ(s') - Φ(s)` where `Φ(s) = β * log(KGE_score(first_goal))`

**Implementation**:
- `PBRSModule`: Computes potentials from KGE scores
- `PBRSWrapper`: Applies shaping during PPO rollouts
- Pre-loads potentials from top-k files for efficiency
- Integrated in `ppo.py:collect_rollouts()`

**Config Options**:
```python
pbrs_beta: float = 0.0      # PBRS weight (0 = disabled)
pbrs_gamma: float = 0.99    # Discount factor for shaping
pbrs_precompute: bool = True # Use pre-computed potentials
```

**File**: `kge_pbrs.py`

---

### 3. Neural Bridge (`kge_neural_bridge.py`)

**Purpose**: Learn optimal combination of RL and KGE logprobs for ranking.

**Formula**: `score = α * rl_logprobs + (1-α) * kge_logprobs`

**Implementation**:
- `LinearBridge`: Single learnable parameter α ∈ [0,1]
- Post-training fit on validation set to maximize MRR
- Applied during evaluation for candidate ranking
- Integrated in `ppo.py:evaluate()` and `train_neural_bridge()`

**Config Options**:
```python
neural_bridge: bool = False           # Enable neural bridge
neural_bridge_init_alpha: float = 0.5 # Initial α value
neural_bridge_train_epochs: int = 100 # Training epochs
neural_bridge_lr: float = 0.01        # Learning rate
```

**File**: `kge_neural_bridge.py`

---

### 4. Unification Scorer (`kge_unification_scorer.py`)

**Purpose**: Score derived states from unification using KGE.

**Implementation**:
- `UnificationScorer`: Scores derived states with KGE
- `UnificationScorerWrapper`: Filters/reranks by score
- Wrapper approach keeps `unification.py` clean
- Can filter to top-k scored states

**Config Options**:
```python
unification_scoring: bool = False       # Enable scoring
unification_scoring_mode: str = 'offline' # offline or online
unification_top_k: int = None           # Filter to top-k states
```

**File**: `kge_unification_scorer.py`

---

## KGE Fusion: Detailed Explanation

**KGE Fusion** is the default hybrid scoring mechanism used during evaluation when `kge_inference=True`. It combines KGE embedding scores with proof success signals from the RL agent.

### How It Works

During evaluation, each candidate answer is scored using both:
1. **KGE Score**: Pre-trained embedding similarity (e.g., RotatE distance)
2. **Proof Success**: Binary signal from the RL agent's proof attempt

### Scoring Formula

```python
# Base score from KGE embeddings
scores = kge_weight * log(kge_score)  # kge_weight=2.0 by default

# Adjust based on proof outcome
if proof_successful:
    scores += rl_weight      # rl_weight=1.0 (bonus for proven facts)
else:
    scores -= fail_penalty   # fail_penalty=100.0 (heavy penalty)
```

### Why This Works

1. **KGE provides ranking signal**: Even when proofs fail, KGE scores provide meaningful similarity-based ranking among candidates.

2. **Proof success provides precision**: When the RL agent successfully proves a candidate, it receives a bonus that can elevate it above KGE-only rankings.

3. **Fail penalty removes noise**: Failed proof attempts are heavily penalized, effectively removing candidates that contradict the logical rules.

### Config Options

```python
kge_eval_kge_weight: float = 2.0    # Weight for KGE log-scores
kge_eval_rl_weight: float = 1.0     # Bonus for successful proofs
kge_fail_penalty: float = 100.0     # Penalty for failed proofs
kge_only_eval: bool = False         # If True, use pure KGE (no RL signal)
```

### Comparison with Neural Bridge

| Aspect | KGE Fusion | Neural Bridge |
|--------|-----------|---------------|
| Combination | Fixed weights + binary bonus | Learned α on continuous logprobs |
| RL Signal | Binary (success/fail) | Continuous (log probabilities) |
| Adaptability | Fixed per dataset | Learns from validation data |
| Best for | Quick baseline | Optimal per-dataset tuning |

---

## Neural Bridge: Proposed Improvements

The current `LinearBridge` learns a single global α to combine RL and KGE scores. Several improvements could make it more targeted and effective.

### Current Limitation

The current implementation trains on **all queries equally**, regardless of whether:
- The RL agent successfully proved any candidates
- The query type (head vs tail corruption)
- The predicate being queried

### Proposed Approaches

#### 1. Success-Only Training

**Idea**: Only train on queries where at least one candidate had a successful proof.

**Rationale**: When no proofs succeed, RL provides no useful signal—it's just noise. Training only on success cases focuses the bridge on queries where RL actually contributes.

```python
class SuccessFilteredBridge(LinearBridge):
    def train_on_validation(self, val_data):
        # Filter to queries with at least one successful proof
        has_success = val_data['success_mask'].any(dim=1)  # [B]
        filtered_data = {k: v[has_success] for k, v in val_data.items()}
        super().train_on_validation(filtered_data)
```

**Expected Benefit**: Higher α values (more RL weight) since RL signal is actually informative in filtered set.

#### 2. Success-Weighted Training

**Idea**: Weight each query's loss contribution by its proof success rate.

**Rationale**: Queries with more successful proofs have more reliable RL signals.

```python
def weighted_mrr_loss(scores, target_idx, success_rates):
    base_loss = differentiable_mrr_loss(scores, target_idx)
    weights = success_rates.clamp(min=0.1)  # Minimum weight to avoid zeros
    return (base_loss * weights).mean()
```

**Expected Benefit**: Smoother training, better generalization.

#### 3. Per-Predicate Alpha

**Idea**: Learn a separate α for each predicate type.

**Rationale**: Different relations may have different RL vs KGE reliability. For example:
- Symmetric relations (e.g., `married`) may benefit more from RL proofs
- Hierarchical relations (e.g., `hypernym`) may be better captured by KGE

```python
class PredicateBridge(nn.Module):
    def __init__(self, num_predicates, init_alpha=0.5):
        self.alphas = nn.Parameter(torch.full((num_predicates,), logit(init_alpha)))

    def forward(self, rl_logprobs, kge_logprobs, pred_indices):
        alpha = torch.sigmoid(self.alphas[pred_indices])  # [B, 1]
        return alpha * rl_logprobs + (1 - alpha) * kge_logprobs
```

**Expected Benefit**: Specialized weighting per relation type.

#### 4. Conditional Alpha (MLP-Based)

**Idea**: Learn α as a function of query features.

**Rationale**: The optimal weighting may depend on:
- Number of successful proofs for this query
- Average KGE score for candidates
- Query predicate embedding

```python
class ConditionalBridge(nn.Module):
    def __init__(self, hidden_dim=32):
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),  # Features: [avg_rl, avg_kge, success_rate, pred_embed]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, rl_logprobs, kge_logprobs, features):
        alpha = self.mlp(features)  # [B, 1]
        return alpha * rl_logprobs + (1 - alpha) * kge_logprobs
```

**Expected Benefit**: Adaptive weighting based on query characteristics.

#### 5. Gated Bridge with Success Mask

**Idea**: Use different combination strategies for proven vs unproven candidates.

**Rationale**: For proven candidates, RL signal is reliable. For unproven, rely more on KGE.

```python
class GatedBridge(nn.Module):
    def __init__(self, init_alpha_success=0.7, init_alpha_fail=0.2):
        self.alpha_success = nn.Parameter(torch.tensor(logit(init_alpha_success)))
        self.alpha_fail = nn.Parameter(torch.tensor(logit(init_alpha_fail)))

    def forward(self, rl_logprobs, kge_logprobs, success_mask):
        alpha_s = torch.sigmoid(self.alpha_success)
        alpha_f = torch.sigmoid(self.alpha_fail)

        scores_success = alpha_s * rl_logprobs + (1 - alpha_s) * kge_logprobs
        scores_fail = alpha_f * rl_logprobs + (1 - alpha_f) * kge_logprobs

        return torch.where(success_mask, scores_success, scores_fail)
```

**Expected Benefit**: Different trust levels for RL signal based on proof outcome.

### Implementation Priority

| Approach | Complexity | Expected Impact | Recommendation |
|----------|------------|-----------------|----------------|
| Success-Only Training | Low | Medium | **Try first** |
| Success-Weighted | Low | Medium | Try second |
| Gated Bridge | Medium | High | **Most promising** |
| Per-Predicate Alpha | Medium | Medium | Dataset-dependent |
| Conditional Alpha | High | Variable | Risk of overfitting |

### Recommended Next Steps

1. **Implement Success-Only Training** as a config option (`neural_bridge_success_only=True`)
2. **Implement Gated Bridge** as an alternative architecture (`neural_bridge_type='gated'`)
3. **Run comparative experiments** on Family and WN18RR to measure improvements

---

## Experimental Results

### Family Dataset

| Configuration | MRR | Hits@1 | Hits@3 | Hits@10 |
|--------------|-----|--------|--------|---------|
| Baseline (RL only) | 0.5699 | 0.5025 | 0.5925 | 0.6850 |
| + KGE Fusion | **0.9963** | 0.9925 | 1.0000 | 1.0000 |
| + Prob Facts (top-5) | 0.9127 | 0.8500 | 0.9850 | 0.9900 |
| + PBRS (β=0.1) | 0.6337 | 0.5900 | 0.6400 | 0.7050 |
| + Neural Bridge (α=0.32) | **0.9975** | 0.9950 | 1.0000 | 1.0000 |
| Combined (All modules) | **0.9975** | 0.9950 | 1.0000 | 1.0000 |

**Key Findings - Family**:
- KGE fusion alone provides +74.8% MRR improvement
- Neural Bridge learns α=0.32 (32% RL, 68% KGE), achieving best results
- Prob Facts enables 98% proof rate (vs 70% baseline) by expanding fact database
- PBRS provides modest improvement (+11.2%) by shaping exploration

### WN18RR Dataset

| Configuration | MRR | Hits@1 | Hits@3 | Hits@10 |
|--------------|-----|--------|--------|---------|
| Baseline (RL only) | 0.4203 | 0.3300 | 0.4000 | 0.6100 |
| + KGE Fusion | **0.9154** | 0.8900 | 0.9300 | 0.9500 |
| + Prob Facts (top-5) | 0.4396 | 0.3600 | 0.4100 | 0.6100 |
| + Neural Bridge (α=0.31) | 0.8817 | 0.8400 | 0.9000 | 0.9500 |

**Key Findings - WN18RR**:
- KGE fusion provides +117.8% MRR improvement
- Neural Bridge learns α=0.31 (31% RL, 69% KGE)
- Prob Facts shows minimal improvement on this dataset
- KGE embeddings are highly effective for WN18RR relations

---

## Usage Examples

### Basic Usage

```bash
# Baseline (RL only)
python runner_kge.py --set dataset=family --set kge_inference=False

# With KGE fusion at evaluation
python runner_kge.py --set dataset=family --set kge_inference=True

# With probabilistic facts
python runner_kge.py --set dataset=family --set prob_facts=True --set prob_facts_topk=5

# With PBRS during training
python runner_kge.py --set dataset=family --set pbrs_beta=0.1

# With neural bridge (requires kge_inference=True)
python runner_kge.py --set dataset=family --set kge_inference=True --set neural_bridge=True

# Combined (all modules)
python runner_kge.py --set dataset=family \
    --set kge_inference=True \
    --set prob_facts=True --set prob_facts_topk=5 \
    --set pbrs_beta=0.1 \
    --set neural_bridge=True
```

---

## Architecture

```
runner_kge.py
    ↓
train.py (orchestrates modules)
    ├── DataHandler (prob_facts loading)
    ├── PBRSWrapper (reward shaping in rollouts)
    └── NeuralBridge (post-training fitting)
    ↓
ppo.py
    ├── collect_rollouts() → applies PBRS
    ├── evaluate() → uses neural_bridge if enabled
    └── train_neural_bridge() → fits α on validation

kge_pbrs.py         # PBRS module
kge_neural_bridge.py # Neural bridge module
kge_unification_scorer.py # Unification scoring module
```

---

## Pre-computed KGE Scores

All modules use pre-computed scores from:
```
top_k_scores/files/kge_top_{dataset}.txt
```

Format: `predicate(head,tail) score rank`

To generate scores for a new dataset:
```bash
python top_k_scores/generate_kge_topk.py --dataset {dataset} --k 100
```

---

## Summary

| Module | Effect | Best Use Case |
|--------|--------|---------------|
| **KGE Fusion** | +74-118% MRR | Always recommended |
| **Prob Facts** | Expands fact DB | When proof rate is low |
| **PBRS** | +11% MRR | When exploration is poor |
| **Neural Bridge** | Optimal RL/KGE mix | When both signals are valuable |

The **KGE Fusion** and **Neural Bridge** modules provide the most significant improvements, leveraging the high quality of pre-trained KGE embeddings for link prediction. The Neural Bridge consistently learns to weight KGE higher than RL (α ≈ 0.31-0.32), indicating that KGE provides stronger ranking signals for these datasets.

---

## Further Integrations

Beyond the current 4 modules, several additional KGE integrations could further enhance DeepProofLog.

### Training-Time Integrations

#### 1. KGE-Based Curriculum Learning

**Idea**: Start training on "easy" queries (high KGE score for correct answer) and gradually increase difficulty.

**How**:
1. Score all training queries by KGE confidence
2. Train first on top-50% easiest queries
3. Gradually expand to harder queries

```python
kge_difficulty = 1.0 - kge_scores[correct_answer]
curriculum_threshold = min(1.0, epoch / warmup_epochs)
train_mask = kge_difficulty <= curriculum_threshold
```

**Benefit**: More stable early training, better final performance.

> **Note**: Multi-hop KGE scoring (rewarding intermediate proof states) is already implemented via **PBRS**, which shapes rewards at each step based on KGE scores.

### Evaluation-Time Integrations

#### 2. KGE-Filtered Candidate Set

**Idea**: Pre-filter candidates using KGE before running proofs.

**How**: Only attempt proofs for top-k KGE-scored candidates. This is different from Top-K Unification Scorer which filters *proof paths*—this filters *which answers to even attempt*.

```python
# Before proof attempts (query-level filtering)
kge_scores = kge_model.score(query, all_candidates)
top_k_candidates = candidates[kge_scores.topk(k).indices]
# Only run proofs for top_k_candidates (saves compute)
```

**Benefit**: Faster evaluation on large datasets (e.g., FB15K-237 with 14k entities). If KGE already ranks the correct answer in top-100, no need to attempt proofs for all 14k.

> **Note**: KGE-guided action selection (biasing rule choice during training) is conceptually similar to **Top-K Unification Scorer**—both use KGE to prune the search space. The difference is Top-K works on derived states, while action selection would work on rule selection.

#### 3. Ensemble with Multiple KGE Models

**Idea**: Combine scores from multiple KGE architectures (RotatE, ComplEx, TransE).

**How**: Train separate bridges for each KGE model, ensemble predictions.

```python
score = (alpha_rotate * rotate_score +
         alpha_complex * complex_score +
         alpha_rl * rl_score)
```

**Benefit**: More robust ranking from diverse embedding spaces.

### Architecture Integrations

#### 4. KGE-Initialized Policy Embeddings

**Idea**: Initialize the policy's entity/relation embeddings from pre-trained KGE.

**How**: Copy KGE embeddings to policy network's embedding layers at initialization.

```python
policy.entity_embedding.weight.data = kge_model.entity_embedding.weight.data
policy.relation_embedding.weight.data = kge_model.relation_embedding.weight.data
```

**Benefit**: Better initialization, faster convergence, improved generalization.

#### 5. Joint KGE-RL Training

**Idea**: Fine-tune KGE embeddings alongside RL training.

**How**: Add KGE loss term to PPO objective, share embeddings.

```python
total_loss = ppo_loss + lambda_kge * kge_contrastive_loss
# KGE embeddings are updated by both losses
```

**Benefit**: KGE learns from proof structure, RL benefits from KGE gradients.

#### 6. KGE Attention over Rules

**Idea**: Use KGE to compute attention weights over applicable rules.

**How**: Score each rule by how well its conclusion matches KGE predictions.

```python
rule_scores = []
for rule in applicable_rules:
    conclusion = apply_rule(state, rule)
    rule_scores.append(kge_score(conclusion))
attention = softmax(rule_scores)
```

**Benefit**: KGE-informed rule selection without hard filtering.

### Negative Sampling Integrations

#### 7. KGE-Based Hard Negative Mining

**Idea**: Use KGE to select challenging negative samples for evaluation.

**How**: Instead of random negatives, select entities with high KGE scores but incorrect proofs.

```python
# Current: random negatives
negatives = random.sample(all_entities, k)

# Proposed: KGE-guided hard negatives
kge_scores = kge_model.score(query, all_entities)
hard_negatives = all_entities[kge_scores.topk(k * 2).indices]
hard_negatives = [e for e in hard_negatives if e != correct_answer][:k]
```

**Benefit**: More challenging evaluation, better differentiation between methods.

### Implementation Roadmap

| Integration | Complexity | Impact | Priority |
|-------------|------------|--------|----------|
| Gated Neural Bridge | Medium | High | **High** |
| KGE-Initialized Embeddings | Low | Medium | **High** |
| KGE-Filtered Candidates | Low | Medium | High (for large datasets) |
| KGE-Based Curriculum | Medium | Medium | Medium |
| Hard Negative Mining | Low | Medium | Medium |
| Joint KGE-RL Training | High | High | Low |
| Ensemble KGE Models | Medium | Medium | Low |

### Quick Wins

1. **Gated Neural Bridge**: Different α for proven vs unproven—minimal code change, high impact
2. **KGE-Initialized Embeddings**: Copy KGE embeddings to policy at init—no runtime cost
3. **KGE-Filtered Candidates**: For FB15K-237, filter to top-500 candidates before proofs—10x speedup

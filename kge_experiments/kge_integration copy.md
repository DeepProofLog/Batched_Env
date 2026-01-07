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

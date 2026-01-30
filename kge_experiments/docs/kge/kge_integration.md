# KGE Integration Modules for DeepProofLog

This document describes the implementation of KGE (Knowledge Graph Embedding) integration modules for DeepProofLog. These modules enhance the RL-based proof discovery system by leveraging pre-trained KGE scores at various stages of the pipeline.

## Folder Structure (Updated)

All KGE modules are now organized in `kge_module/`:

```
kge_experiments/
└── kge_module/
    ├── __init__.py           # Re-exports all modules
    ├── inference.py          # KGE model loading and scoring
    ├── pbrs.py               # Potential-Based Reward Shaping
    ├── neural_bridge.py      # Learned RL+KGE fusion (Linear, Gated, PerPredicate, MLP)
    ├── rule_attention.py     # KGE attention for rule selection
    ├── embed_init.py         # KGE-initialized embeddings
    ├── ensemble.py           # Multi-model ensemble
    ├── joint.py              # Joint KGE-RL training
    ├── filter.py             # KGE-filtered candidates
    ├── benchmark.py          # Performance timing utilities
    └── pytorch/              # PyTorch KGE models
        ├── model_torch.py
        ├── kge_inference_torch.py
        └── ...
```

## Module Descriptions

### 1. Probabilistic Facts (`data_handler.py`)

**Purpose**: Expand the fact database with high-confidence KGE-predicted facts.

**Config Options**:
```python
prob_facts: bool = False          # Enable probabilistic facts
prob_facts_topk: int = None       # Top-K facts per predicate
prob_facts_threshold: float = None # Minimum KGE score threshold
```

---

### 2. PBRS Module (`kge_module/pbrs.py`)

**Purpose**: Potential-Based Reward Shaping to guide RL exploration.

**Formula**: `r' = r + γ*Φ(s') - Φ(s)` where `Φ(s) = β * log(KGE_score(first_goal))`

**Config Options**:
```python
pbrs_beta: float = 0.0      # PBRS weight (0 = disabled)
pbrs_gamma: float = 0.99    # Discount factor for shaping
pbrs_precompute: bool = True # Use pre-computed potentials
```

---

### 3. Neural Bridge (`kge_module/neural_bridge.py`)

**Purpose**: Learn optimal combination of RL and KGE logprobs for ranking.

**Bridge Types**:

| Type | Formula | Parameters |
|------|---------|------------|
| `linear` | `α * rl + (1-α) * kge` | Single α ∈ [0,1] |
| `gated` | Different α for success/fail | α_success, α_fail |
| `per_predicate` | Different α per predicate | n_predicates × α |
| `mlp` | MLP(rl, kge) | Hidden layers |

**Config Options**:
```python
neural_bridge: bool = False
neural_bridge_type: str = 'linear'  # 'linear', 'gated', 'per_predicate', 'mlp'
neural_bridge_init_alpha: float = 0.5
neural_bridge_init_alpha_success: float = 0.7  # For gated
neural_bridge_init_alpha_fail: float = 0.2     # For gated
neural_bridge_train_epochs: int = 100
neural_bridge_lr: float = 0.01
```

---

### 4. KGE Rule Attention (`kge_module/rule_attention.py`) - NEW

**Purpose**: Score rule conclusions with KGE to guide action selection.

**Formula**: `augmented_logits = action_logits + weight * kge_attention`

Where `kge_attention[i]` is the KGE score of the first atom in derived state i.

**Config Options**:
```python
kge_rule_attention: bool = False
kge_rule_attention_weight: float = 0.5
kge_rule_attention_temperature: float = 1.0
```

---

### 5. KGE-Filtered Candidates (`kge_module/filter.py`)

**Purpose**: Pre-filter candidates by KGE score before running proofs.

**Config Options**:
```python
kge_filter_candidates: bool = False
kge_filter_top_k: int = 100
```

---

### 6. KGE-Initialized Embeddings (`kge_module/embed_init.py`)

**Purpose**: Initialize policy embeddings from pre-trained KGE model.

**Config Options**:
```python
kge_init_embeddings: bool = False
```

---

### 7. Ensemble KGE Models (`kge_module/ensemble.py`)

**Purpose**: Combine scores from multiple KGE architectures.

**Config Options**:
```python
kge_ensemble: bool = False
kge_ensemble_signatures: str = None  # Comma-separated signatures
kge_ensemble_method: str = 'mean'    # 'mean', 'max', 'learned'
```

---

### 8. Joint KGE-RL Training (`kge_module/joint.py`)

**Purpose**: Fine-tune KGE embeddings alongside RL training.

**Config Options**:
```python
kge_joint_training: bool = False
kge_joint_lambda: float = 0.1
kge_joint_margin: float = 1.0
```

---

### 9. Performance Benchmarking (`kge_module/benchmark.py`) - NEW

**Purpose**: Collect timing statistics for KGE modules.

**Usage**:
```python
from kge_module import get_benchmark

bench = get_benchmark()
with bench.time("kge_inference"):
    scores = kge_engine.predict_batch(atoms)

bench.print_report(total_training_time=training_seconds)
```

**Config Options**:
```python
kge_benchmark: bool = False  # Enable timing collection
```

---

## Usage Example

```bash
# Enable gated bridge with benchmark
python runner_kge.py --set dataset=wn18rr \
    --set neural_bridge=True \
    --set neural_bridge_type=gated \
    --set kge_benchmark=True \
    --total_timesteps 200000

# Enable per-predicate bridge
python runner_kge.py --set dataset=wn18rr \
    --set neural_bridge=True \
    --set neural_bridge_type=per_predicate \
    --total_timesteps 200000

# Enable rule attention
python runner_kge.py --set dataset=wn18rr \
    --set kge_rule_attention=True \
    --set kge_rule_attention_weight=0.5 \
    --total_timesteps 200000
```

---

## Import Paths (Updated)

```python
# All modules available from kge_module
from kge_module import (
    KGEInference,
    PBRSModule,
    LinearBridge,
    GatedBridge,
    PerPredicateBridge,
    MLPBridge,
    KGERuleAttention,
    KGEBenchmark,
)

# Or import specific modules
from kge_module.neural_bridge import create_neural_bridge
from kge_module.benchmark import get_benchmark
```

---

## Summary of Changes

| Module | Status | File |
|--------|--------|------|
| Probabilistic Facts | Implemented | `data_handler.py` |
| PBRS | Implemented | `kge_module/pbrs.py` |
| Neural Bridge (Linear) | Implemented | `kge_module/neural_bridge.py` |
| Neural Bridge (Gated) | Implemented | `kge_module/neural_bridge.py` |
| Neural Bridge (Per-Predicate) | NEW | `kge_module/neural_bridge.py` |
| Neural Bridge (MLP) | Implemented | `kge_module/neural_bridge.py` |
| Rule Attention | NEW | `kge_module/rule_attention.py` |
| KGE-Filtered Candidates | Implemented | `kge_module/filter.py` |
| KGE-Init Embeddings | Implemented | `kge_module/embed_init.py` |
| Ensemble | Implemented | `kge_module/ensemble.py` |
| Joint Training | Implemented | `kge_module/joint.py` |
| Benchmarking | NEW | `kge_module/benchmark.py` |

## Deleted Files

The following files were removed as unused:
- `kge_integration.py` (superseded by kge_module/)
- `kge_unification_scorer.py` (unused)
- `kge_base_runner.py` (unused)

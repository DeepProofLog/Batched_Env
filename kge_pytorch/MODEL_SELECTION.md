# KGE Model Selection Guide

This guide helps you choose which models to test based on your needs.

## Quick Start

### Best Overall Models (Recommended)
These typically give the best results across most datasets:

```bash
python kge_pytorch/runner_pykeen.py --dataset family \
    --models RotatE,ComplEx,TuckER,PairRE \
    --epochs 100 --embedding_dim 200
```

### Fast Testing (Newer Models Only)
```bash
python kge_pytorch/runner_pykeen.py --dataset family \
    --models PairRE,QuatE,RotatE,ComplEx \
    --epochs 100 --embedding_dim 200
```

---

## Model Categories

### 🏆 Top Performers (State-of-the-Art)

**RotatE** - Rotation in complex space
- Excellent on most datasets
- Good balance of speed and accuracy
- Works well with diverse relation types

**PairRE** - Paired Relation Embeddings (Newer)
- Often outperforms RotatE
- Handles complex relations well
- Good for symmetric/antisymmetric relations

**QuatE** - Quaternion Embeddings (Newer)
- Uses hypercomplex numbers
- Very good on complex relation patterns
- Higher dimensional representations

**ComplEx** - Complex Embeddings
- Strong baseline model
- Fast training
- Good for symmetric relations

**TuckER** - Tucker Decomposition
- Excellent performance
- Higher computational cost
- Good for datasets with many relations

---

### 🚀 Newer Advanced Models

**AutoSF** - Automatic Scoring Function
- Learns optimal scoring function
- Often achieves SOTA results
- Requires more epochs

**TripleRE** - Triple Relation Embeddings
- Very recent (2021+)
- Strong on complex patterns
- Good generalization

**BoxE** - Box Embeddings
- Geometric approach
- Good interpretability
- Strong on hierarchical data

**CrossE** - Interaction Embeddings
- Uses entity interactions
- Good on sparse datasets
- Moderate training time

---

### 📊 Classic Strong Baselines

**DistMult** - Bilinear Diagonal
- Fast and simple
- Good baseline
- Limited to symmetric relations

**TransE** - Translational Distance
- Original translation model
- Fast training
- Limited to 1-to-1 relations

**ConvE** - Convolutional
- Uses 2D convolutions
- Good on large datasets
- Requires more tuning

**ConvKB** - Convolutional Knowledge Base
- Alternative convolutional approach
- Good pattern detection
- Moderate speed

---

## Testing Strategies

### Strategy 1: Comprehensive Benchmark
Test all top models to find the best for your dataset:

```bash
python kge_pytorch/runner_pykeen.py --dataset family \
    --models RotatE,PairRE,QuatE,ComplEx,TuckER,AutoSF,TripleRE,DistMult,TransE,ConvE \
    --epochs 100 --embedding_dim 200 --batch_size 4096
```

### Strategy 2: Quick Validation (2-3 best models)
```bash
python kge_pytorch/runner_pykeen.py --dataset family \
    --models RotatE,PairRE,ComplEx \
    --epochs 100 --embedding_dim 200
```

### Strategy 3: Newer Models Only
```bash
python kge_pytorch/runner_pykeen.py --dataset family \
    --models PairRE,QuatE,AutoSF,TripleRE,BoxE \
    --epochs 150 --embedding_dim 200
```

### Strategy 4: Fast Models (for large datasets)
```bash
python kge_pytorch/runner_pykeen.py --dataset family \
    --models DistMult,ComplEx,RotatE \
    --epochs 100 --embedding_dim 200 --batch_size 8192
```

---

## Model Comparison Table

| Model      | Year | Speed    | Performance | Best For                          |
|------------|------|----------|-------------|-----------------------------------|
| PairRE     | 2021 | Fast     | Excellent   | Complex relations, general use    |
| QuatE      | 2019 | Moderate | Excellent   | Complex patterns, large embeddings|
| RotatE     | 2019 | Fast     | Excellent   | General purpose, diverse relations|
| AutoSF     | 2020 | Slow     | Excellent   | Max performance, research         |
| TuckER     | 2019 | Moderate | Excellent   | Many relations, high accuracy     |
| ComplEx    | 2016 | Fast     | Very Good   | Symmetric relations, baseline     |
| TripleRE   | 2021 | Moderate | Very Good   | Compositional patterns            |
| ConvE      | 2018 | Moderate | Good        | Large datasets, pattern detection |
| DistMult   | 2015 | Very Fast| Good        | Simple, symmetric relations       |
| TransE     | 2013 | Very Fast| Good        | 1-to-1 relations, simple patterns |

---

## Recommended Settings by Dataset Size

### Small Dataset (< 50K triples)
```bash
--models RotatE,PairRE,ComplEx,TuckER,QuatE
--epochs 200
--embedding_dim 100-200
--batch_size 512-2048
```

### Medium Dataset (50K - 500K triples)
```bash
--models RotatE,PairRE,ComplEx,TuckER
--epochs 100-150
--embedding_dim 200-400
--batch_size 2048-4096
```

### Large Dataset (> 500K triples)
```bash
--models RotatE,ComplEx,DistMult,PairRE
--epochs 50-100
--embedding_dim 200-500
--batch_size 4096-8192
```

---

## Example: Family Dataset (Small)

```bash
# Best accuracy (slower)
python kge_pytorch/runner_pykeen.py --dataset family \
    --models PairRE,QuatE,RotatE,TuckER,ComplEx,AutoSF \
    --epochs 200 --embedding_dim 200 --batch_size 2048

# Balanced (recommended)
python kge_pytorch/runner_pykeen.py --dataset family \
    --models RotatE,PairRE,ComplEx,TuckER \
    --epochs 150 --embedding_dim 200 --batch_size 4096

# Fast baseline
python kge_pytorch/runner_pykeen.py --dataset family \
    --models RotatE,ComplEx,DistMult \
    --epochs 100 --embedding_dim 200 --batch_size 4096
```

---

## Tips

1. **Start with RotatE, PairRE, and ComplEx** - These three models cover most use cases
2. **Use larger embeddings for newer models** - QuatE, PairRE work better with dim=200-400
3. **Increase epochs for AutoSF** - It needs more training time to converge
4. **Use early stopping for long runs** - Add `--use_early_stopping --eval_frequency 10`
5. **Compare MRR and Hits@10** - Don't rely on a single metric

---

## All Available Models in PyKEEN

For a complete list, see: https://pykeen.readthedocs.io/en/stable/reference/models.html

Common models you can try:
- TransE, TransH, TransR, TransD
- DistMult, ComplEx, RotatE, QuatE
- ConvE, ConvKB
- TuckER, RESCAL, SimplE
- PairRE, AutoSF, TripleRE, BoxE, CrossE
- CP, ERMLP, HolE, MuRE, NTN, ProjE, RGCN, SE, StructuredEmbedding, TorusE, TransF, UM

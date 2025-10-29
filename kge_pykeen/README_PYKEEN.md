# PyKEEN Runner for Knowledge Graph Embedding

This script (`pykeen_runner.py`) provides a complete pipeline for training and evaluating Knowledge Graph Embedding (KGE) models using PyKEEN.

## Features

- ✅ **Multiple Model Support**: Train RotatE, ComplEx, TuckER, DistMult, TransE, PairRE, and more
- ✅ **Dataset Format Support**: 
  - Prolog-style facts: `relation(entity1,entity2).`
  - TSV/CSV format: `entity1\trelation\tentity2`
  - PyKEEN built-in datasets (FB15k237, WN18RR, etc.)
- ✅ **Auto-detection**: Automatically detects dataset paths in `./data/<dataset>/`
- ✅ **GPU Support**: CUDA acceleration with `--device cuda`
- ✅ **Comprehensive Metrics**: MRR, MR, Hits@1, Hits@3, Hits@10
- ✅ **Filtered Evaluation**: Uses filtered (realistic) metrics by default
- ✅ **Early Stopping**: Prevents overfitting with patience-based stopping
- ✅ **Model Checkpointing**: Saves best model weights automatically

## Installation

The script requires PyKEEN and handles torchvision compatibility issues automatically:

```bash
pip install pykeen
```

## Usage Examples

### Train on Your Custom Dataset

```bash
# Using dataset name (auto-detects paths in ./data/family/)
python kge_pytorch/pykeen_runner.py \
    --dataset family \
    --models RotatE,ComplEx,TuckER \
    --epochs 50 \
    --embedding_dim 200 \
    --batch_size 256 \
    --device cuda

# Using explicit file paths
python kge_pytorch/pykeen_runner.py \
    --train_path ./data/family/train.txt \
    --valid_path ./data/family/valid.txt \
    --test_path ./data/family/test.txt \
    --models RotatE \
    --epochs 50 \
    --embedding_dim 200 \
    --device cuda
```

### Train on PyKEEN Built-in Datasets

```bash
python kge_pytorch/pykeen_runner.py \
    --dataset FB15k237 \
    --models RotatE,ComplEx \
    --epochs 100 \
    --embedding_dim 1000 \
    --batch_size 1024 \
    --device cuda
```

### Quick Test Run

```bash
# Quick 3-epoch test on countries dataset
python kge_pytorch/pykeen_runner.py \
    --dataset countries_s1 \
    --models RotatE \
    --epochs 3 \
    --batch_size 64 \
    --embedding_dim 50 \
    --device cuda
```

## Available Datasets

Your local datasets in `./data/`:
- `family` - 19,845 training triples, 12 relations, 2,968 entities
- `countries_s1` - 1,115 training triples, 4 relations, 271 entities
- `countries_s2` - 1,067 training triples, 4 relations, 271 entities
- `countries_s3` - 983 training triples, 4 relations, 271 entities

PyKEEN built-in datasets:
- `FB15k237`, `WN18RR`, `UMLS`, `Kinships`, `Nations`, and many more

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `family` | Dataset name (auto-detects path) or PyKEEN dataset |
| `--train_path` | `None` | Override: path to train.txt |
| `--valid_path` | `None` | Override: path to valid.txt |
| `--test_path` | `None` | Override: path to test.txt |
| `--models` | `RotatE` | Comma-separated model names |
| `--embedding_dim` | `1000` | Embedding dimension |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `1024` | Batch size |
| `--learning_rate` | `0.001` | Learning rate |
| `--negative_sample_rate` | `1` | Negative samples per positive |
| `--training_loop` | `sLCWA` | Training loop type (sLCWA or LCWA) |
| `--device` | `cuda` | Device (cuda or cpu) |
| `--output_dir` | `./pykeen_runs` | Output directory |
| `--create_inverse_triples` | `False` | Create inverse relations |

## Output Files

For each training run, the script creates:

```
pykeen_runs/
├── RotatE_2025-10-28_16-41-31/
│   ├── trained_model.pkl         # Saved model weights
│   ├── results.json               # Full evaluation results
│   ├── summary.json               # Quick summary with metrics
│   ├── metadata.json              # Configuration metadata
│   └── training_triples/          # Training data metadata
└── scoreboard_2025-10-28_16-41-37.csv  # All model results
```

### Scoreboard CSV Format

```csv
model,status,mrr,mr,hits@1,hits@3,hits@10,run_dir
RotatE,ok,0.0658,80.48,0.0208,0.0417,0.1458,./pykeen_runs/RotatE_...
ComplEx,ok,0.0127,123.98,0.0,0.0,0.0,./pykeen_runs/ComplEx_...
```

## Supported Models

- **RotatE** - Rotation-based embeddings in complex space
- **ComplEx** - Complex embeddings with Hermitian dot product
- **TuckER** - Tucker decomposition
- **DistMult** - Bilinear diagonal model
- **TransE** - Translational embeddings
- **PairRE** - Paired relation embeddings
- And many more PyKEEN models...

## Dataset Format

### Prolog-Style (Recommended for your datasets)

```prolog
aunt(1369,1287).
aunt(1682,512).
father(john,mary).
```

### TSV/CSV Format

```tsv
entity1	relation	entity2
john	father	mary
```

## Troubleshooting

### Torchvision Compatibility Error

The script automatically handles torchvision compatibility issues by preventing its import. No action needed.

### CUDA Out of Memory

Reduce batch size or embedding dimension:
```bash
--batch_size 128 --embedding_dim 100
```

### Model Not Found

Make sure the model name matches PyKEEN's class names (case-sensitive):
- Correct: `RotatE`, `ComplEx`, `TuckER`
- Incorrect: `rotate`, `complex`, `tucker`

## Performance Tips

1. **Use GPU**: Always use `--device cuda` for faster training
2. **Batch Size**: Increase batch size (256-1024) for larger datasets
3. **Embedding Dimension**: Start with 100-200, increase for larger datasets
4. **Early Stopping**: Script automatically stops when validation improves (patience=10)

## Integration with Your Workflow

This script integrates seamlessly with your existing datasets in `./data/`. All datasets using the Prolog-style format (`.txt` files with facts) are automatically supported.

## Citation

If you use this script with PyKEEN, please cite:

```bibtex
@inproceedings{ali2021pykeen,
    title={PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings},
    author={Ali, Mehdi and others},
    booktitle={Journal of Machine Learning Research},
    year={2021}
}
```

#!/bin/bash
# Test all common KGE models on a dataset, including newer state-of-the-art models

# Configuration
DATASET="${1:-family}"
EPOCHS="${2:-100}"
BATCH_SIZE="${3:-4096}"
EMBEDDING_DIM="${4:-200}"
OUTPUT_DIR="./kge_pytorch/pykeen_runs"

# List of models to test
# Classic models: TransE, DistMult, ComplEx
# Strong performers: RotatE, TuckER, ComplEx
# Newer/Advanced: PairRE, AutoSF, CrossE, TripleRE, QuatE
# Convolutional: ConvE, ConvKB
MODELS="RotatE,ComplEx,TuckER,DistMult,TransE,PairRE,ConvE,QuatE,AutoSF,TripleRE"

echo "========================================="
echo "Testing KGE Models"
echo "========================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Embedding Dim: $EMBEDDING_DIM"
echo "Models: $MODELS"
echo "========================================="
echo ""

# Run the PyKEEN runner
python kge_pytorch/runner_pykeen.py \
    --dataset "$DATASET" \
    --models "$MODELS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embedding_dim "$EMBEDDING_DIM" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "Testing Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="

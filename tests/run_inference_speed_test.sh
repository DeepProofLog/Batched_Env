#!/bin/bash
# Quick test script to compare KGE inference speed on family dataset
# This script runs the inference speed test with sensible defaults

# Default parameters
NUM_ATOMS=1000
BATCH_SIZE=2048
DATASET="family"
DATA_PATH="./data/family/train.txt"

# TensorFlow model (adjust if needed)
TF_CHECKPOINT_DIR="/home/castellanoontiv/checkpoints/"
TF_RUN_SIGNATURE="kinship_family-backward_0_1-no_reasoner-complex-True-256-256-4-rules.txt"

# PyTorch model
PYTORCH_MODEL_DIR="./kge_pytorch/models"

# PyKEEN model (optional - uncomment if you have a trained model)
# PYKEEN_MODEL_DIR="./kge_pykeen/pykeen_runs/rotate_2025-10-28_18-12-10"

echo "================================================================"
echo "KGE Inference Speed Comparison Test"
echo "================================================================"
echo "Dataset: $DATASET"
echo "Number of atoms: $NUM_ATOMS"
echo "Batch size: $BATCH_SIZE"
echo "================================================================"
echo ""

# Build command
CMD="python tests/test_inference_speed.py \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --num_atoms $NUM_ATOMS \
    --batch_size $BATCH_SIZE \
    --tf_checkpoint_dir $TF_CHECKPOINT_DIR \
    --tf_run_signature $TF_RUN_SIGNATURE \
    --pytorch_model_dir $PYTORCH_MODEL_DIR"

# Add PyKEEN model if specified
if [ ! -z "$PYKEEN_MODEL_DIR" ]; then
    CMD="$CMD --pykeen_model_dir $PYKEEN_MODEL_DIR"
fi

# Run the test
$CMD

echo ""
echo "================================================================"
echo "Test completed!"
echo "================================================================"

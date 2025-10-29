#!/bin/bash
# Helper script to check which KGE models are available for testing

echo "================================================================"
echo "Checking Available KGE Models for Inference Speed Testing"
echo "================================================================"
echo ""

# Check TensorFlow models
echo "1. TensorFlow Models (in /home/castellanoontiv/checkpoints/):"
echo "----------------------------------------------------------------"
TF_DIR="/home/castellanoontiv/checkpoints"
if [ -d "$TF_DIR" ]; then
    echo "Available family models:"
    ls -d $TF_DIR/*family* 2>/dev/null | head -5
    if [ $? -ne 0 ]; then
        echo "  ⚠ No family models found"
    fi
else
    echo "  ⚠ Checkpoint directory not found: $TF_DIR"
fi
echo ""

# Check PyTorch models
echo "2. PyTorch Models (in ./kge_pytorch/models/):"
echo "----------------------------------------------------------------"
PT_DIR="./kge_pytorch/models"
if [ -d "$PT_DIR" ]; then
    if [ -f "$PT_DIR/weights.pth" ]; then
        echo "  ✓ Found PyTorch model at: $PT_DIR"
        if [ -f "$PT_DIR/config.json" ]; then
            echo "    Model config:"
            cat "$PT_DIR/config.json" | python3 -m json.tool 2>/dev/null || cat "$PT_DIR/config.json"
        fi
    else
        echo "  ⚠ No weights.pth found in $PT_DIR"
    fi
else
    echo "  ⚠ PyTorch models directory not found: $PT_DIR"
fi
echo ""

# Check PyKEEN models
echo "3. PyKEEN Models (in ./kge_pykeen/pykeen_runs/):"
echo "----------------------------------------------------------------"
PK_DIR="./kge_pykeen/pykeen_runs"
if [ -d "$PK_DIR" ]; then
    echo "Available trained models:"
    for dir in $PK_DIR/*/; do
        if [ -f "$dir/trained_model.pkl" ]; then
            echo "  ✓ $dir"
            if [ -f "$dir/summary.json" ]; then
                echo "    Summary:"
                cat "$dir/summary.json" | python3 -m json.tool 2>/dev/null | grep -E "(model|status|mrr)" | head -5
            fi
        fi
    done
else
    echo "  ⚠ PyKEEN runs directory not found: $PK_DIR"
fi
echo ""

# Recommended command
echo "================================================================"
echo "Recommended Test Command:"
echo "================================================================"
echo ""

# Find first available PyKEEN model
PYKEEN_MODEL=$(find $PK_DIR -name "trained_model.pkl" 2>/dev/null | head -1 | xargs dirname)

if [ ! -z "$PYKEEN_MODEL" ]; then
    echo "python tests/test_inference_speed.py \\"
    echo "    --num_atoms 1000 \\"
    echo "    --batch_size 2048 \\"
    echo "    --pykeen_model_dir $PYKEEN_MODEL"
else
    echo "python tests/test_inference_speed.py \\"
    echo "    --num_atoms 1000 \\"
    echo "    --batch_size 2048"
    echo ""
    echo "Note: No PyKEEN model found, will test TF and PyTorch only"
fi

echo ""
echo "Or simply run: ./tests/run_inference_speed_test.sh"
echo "================================================================"

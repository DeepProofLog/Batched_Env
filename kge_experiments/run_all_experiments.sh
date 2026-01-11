#!/bin/bash
# Comprehensive KGE Integration Experiments
# Run overnight to compare all integration methods
#
# Usage: nohup bash run_all_experiments.sh > experiments.log 2>&1 &

set -e

# Common parameters
# Configuration
FAST_DEV_RUN=false  # Set to false for full training
DATASETS=("nations" "umls" "fb15k237" "pharmkg_full")

if [ "$FAST_DEV_RUN" = true ]; then
    echo "!!! FAST_DEV_RUN ENABLED - Running truncated experiments !!!"
    TIMESTEPS=1000
else
    TIMESTEPS=5000000
fi

# Common parameters (static)
EVAL_QUERIES=100
EVAL_NEG=10
TEST_QUERIES=None
TEST_NEG=100

PYTHON="/home/castellanoontiv/miniconda3/envs/rl/bin/python"
RUNNER="runner_kge.py"

echo "=============================================="
echo "Starting KGE Integration Experiments"
echo "Datasets: ${DATASETS[*]}"
echo "Timesteps per run: $TIMESTEPS"
echo "Started at: $(date)"
echo "=============================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "Processing Dataset: $DATASET"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    COMMON_ARGS="--set total_timesteps=$TIMESTEPS \
        --set n_eval_queries=$EVAL_QUERIES \
        --set eval_neg_samples=$EVAL_NEG \
        --set n_test_queries=$TEST_QUERIES \
        --set test_neg_samples=$TEST_NEG \
        --set dataset=$DATASET \
        --set eval_freq=4 \
        --set save_model=True"

echo "=============================================="
echo "Starting KGE Integration Experiments"
echo "Dataset: $DATASET"
echo "Timesteps: $TIMESTEPS"
echo "Eval: $EVAL_QUERIES queries, $EVAL_NEG corruptions"
echo "Test: $TEST_QUERIES queries, $TEST_NEG corruptions"
echo "Started at: $(date)"
echo "=============================================="


# 1. KGE Only (Pure KGE ranking, no proofs, no RL)
echo ""
echo "[1/10] Running: KGE Only"
echo "=============================================="
$PYTHON $RUNNER \
    --set total_timesteps=0 \
    --set n_eval_queries=$EVAL_QUERIES \
    --set eval_neg_samples=$EVAL_NEG \
    --set n_test_queries=$TEST_QUERIES \
    --set test_neg_samples=$TEST_NEG \
    --set dataset=$DATASET \
    --set kge_inference=True \
    --set kge_only_eval=True \
    --set neural_bridge=False \
    --eval

# 2. RL Only (No KGE at evaluation)
echo ""
echo "[2/10] Running: RL Only (no KGE)"
echo "=============================================="
$PYTHON $RUNNER $COMMON_ARGS \
    --set kge_inference=False \
    --set neural_bridge=False

# 3. Hybrid Baseline (Fixed weights: KGE + RL)
echo ""
echo "[3/10] Running: Hybrid Baseline (KGE + RL fixed weights)"
echo "=============================================="
$PYTHON $RUNNER $COMMON_ARGS \
    --set kge_inference=True \
    --set kge_only_eval=False \
    --set neural_bridge=False

# 4. Neural Bridge - Linear
echo ""
echo "[4/10] Running: Neural Bridge (Linear)"
echo "=============================================="
$PYTHON $RUNNER $COMMON_ARGS \
    --set kge_inference=True \
    --set neural_bridge=True \
    --set neural_bridge_type=linear

# 5. Neural Bridge - Gated
echo ""
echo "[5/10] Running: Neural Bridge (Gated)"
echo "=============================================="
$PYTHON $RUNNER $COMMON_ARGS \
    --set kge_inference=True \
    --set neural_bridge=True \
    --set neural_bridge_type=gated

# 6. Neural Bridge - Per-Predicate
echo ""
echo "[6/10] Running: Neural Bridge (Per-Predicate)"
echo "=============================================="
$PYTHON $RUNNER $COMMON_ARGS \
    --set kge_inference=True \
    --set neural_bridge=True \
    --set neural_bridge_type=per_predicate

# 7. PBRS (Potential-Based Reward Shaping)
echo ""
echo "[7/10] Running: PBRS (beta=0.1)"
echo "=============================================="
$PYTHON $RUNNER $COMMON_ARGS \
    --set kge_inference=True \
    --set pbrs_beta=0.1 \
    --set neural_bridge=False

# 8. PBRS + Gated Bridge (Combined)
echo ""
echo "[8/10] Running: PBRS + Gated Bridge"
echo "=============================================="
$PYTHON $RUNNER $COMMON_ARGS \
    --set kge_inference=True \
    --set pbrs_beta=0.1 \
    --set neural_bridge=True \
    --set neural_bridge_type=gated

    # 9. KGE Filter Top-K (Pre-filter candidates)
    echo ""
    echo "[9/10] Running: KGE Filter Top-K (k=100)"
    echo "=============================================="
    $PYTHON $RUNNER $COMMON_ARGS \
        --set kge_inference=True \
        --set kge_filter_candidates=True \
        --set kge_filter_top_k=100 \
        --set neural_bridge=False

    # 10. KGE Filter + Gated Bridge (Combined)
    echo ""
    echo "[10/10] Running: KGE Filter + Gated Bridge"
    echo "=============================================="
    $PYTHON $RUNNER $COMMON_ARGS \
        --set kge_inference=True \
        --set kge_filter_candidates=True \
        --set kge_filter_top_k=100 \
        --set neural_bridge=True \
        --set neural_bridge_type=gated
    
    echo "Finished experiments for $DATASET"
done

echo ""
echo "=============================================="
echo "All experiments completed for all datasets!"
echo "Finished at: $(date)"
echo "=============================================="
echo ""
echo "Results are in the runs/ folder"

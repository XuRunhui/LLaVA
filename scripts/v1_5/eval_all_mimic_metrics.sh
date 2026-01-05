#!/bin/bash

# Script to evaluate both dev and test sets
# Usage: ./scripts/v1_5/eval_all_mimic_metrics.sh <model_name> [bootstrap_ci]

MODEL_NAME=${1:-"lora_128_dp_e8"}
BOOTSTRAP_CI=${2:-"true"}

if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME not provided"
    echo "Usage: $0 <model_name> [bootstrap_ci]"
    echo "Example: $0 lora_128_dp_e8 true"
    exit 1
fi

# Base paths
EVAL_BASE_DIR="/scratch1/runhuixu/evaluation/llava_llavarad/"
RESULTS_BASE_DIR="${EVAL_BASE_DIR}/eval_results_${MODEL_NAME}"

# Create logs directory
mkdir -p ${EVAL_BASE_DIR}/logs

echo "============================================"
echo "Evaluating MIMIC-CXR Metrics"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Bootstrap CI: $BOOTSTRAP_CI"
echo "============================================"

# Evaluate dev set
echo ""
echo "Evaluating DEV set..."
DEV_RESULTS="${RESULTS_BASE_DIR}/dev_results.jsonl"
DEV_OUTPUT="${RESULTS_BASE_DIR}/dev_metrics"

if [ -f "$DEV_RESULTS" ]; then
    echo "Results file: $DEV_RESULTS"
    echo "Output directory: $DEV_OUTPUT"
    bash scripts/v1_5/eval_metrics_mimic.sh $DEV_RESULTS $DEV_OUTPUT $BOOTSTRAP_CI
else
    echo "Warning: Dev results file not found at $DEV_RESULTS"
    echo "Skipping dev evaluation."
fi

# Evaluate test set
echo ""
echo "============================================"
echo "Evaluating TEST set..."
TEST_RESULTS="${RESULTS_BASE_DIR}/test_results.jsonl"
TEST_OUTPUT="${RESULTS_BASE_DIR}/test_metrics"

if [ -f "$TEST_RESULTS" ]; then
    echo "Results file: $TEST_RESULTS"
    echo "Output directory: $TEST_OUTPUT"
    bash scripts/v1_5/eval_metrics_mimic.sh $TEST_RESULTS $TEST_OUTPUT $BOOTSTRAP_CI
else
    echo "Warning: Test results file not found at $TEST_RESULTS"
    echo "Skipping test evaluation."
fi

echo ""
echo "============================================"
echo "All evaluations complete!"
echo "============================================"
echo "Dev metrics: $DEV_OUTPUT"
echo "Test metrics: $TEST_OUTPUT"
echo "============================================"

#!/bin/bash

#SBATCH --job-name=eval_metrics_rex
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G

set -euo pipefail

module purge
module load gcc/13.3.0
module load cuda/12.6.3
export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw


mkdir -p logs

# # Conda (batch-safe) activation
source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate llavarad
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd /project2/ruishanl_1185/SDP_for_VLM/Xinyang/LLaVA_DP/LLaVA

MODEL_NAME=${1:-"/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/lora_weight_32"}
BOOTSTRAP_CI=${2:-"true"}

if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME not provided"
    echo "Usage: $0 <model_name> [bootstrap_ci]"
    echo "Example: $0 lora_128_dp_e8 true"
    exit 1
fi

# Base paths
EVAL_BASE_DIR="/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/lora_weight_32/eval/results"
RESULTS_BASE_DIR="/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/lora_weight_32/eval/"

# Create logs directory
mkdir -p ${EVAL_BASE_DIR}/logs

echo "============================================"
echo "Evaluating REX Metrics"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Bootstrap CI: $BOOTSTRAP_CI"
echo "============================================"

# Evaluate dev set
echo ""
echo "Evaluating DEV set..."
DEV_RESULTS="${RESULTS_BASE_DIR}/validation_set_eval_with_metrics.jsonl"
DEV_OUTPUT="${RESULTS_BASE_DIR}/dev_metrics"

if [ -f "$DEV_RESULTS" ]; then
    echo "Results file: $DEV_RESULTS"
    echo "Output directory: $DEV_OUTPUT"
    bash scripts/v1_5/eval_metrics_rex.sh $DEV_RESULTS $DEV_OUTPUT $BOOTSTRAP_CI
else
    echo "Warning: Dev results file not found at $DEV_RESULTS"
    echo "Skipping dev evaluation."
fi

# Evaluate test set
echo ""
echo "============================================"
echo "Evaluating TEST set..."
TEST_RESULTS="${RESULTS_BASE_DIR}/none.jsonl"
TEST_OUTPUT="${RESULTS_BASE_DIR}/none"

if [ -f "$TEST_RESULTS" ]; then
    echo "Results file: $TEST_RESULTS"
    echo "Output directory: $TEST_OUTPUT"
    bash scripts/v1_5/eval_metrics_rex.sh $TEST_RESULTS $TEST_OUTPUT $BOOTSTRAP_CI
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

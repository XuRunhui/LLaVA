#!/bin/bash
#SBATCH --job-name=llava_rex_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load gcc/13.3.0
module load cuda/12.6.3
export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

# # Conda (batch-safe) activation
source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh

conda activate /project2/ruishanl_1185/SDP_for_VLM/Xinyang/envs/llava_dp
export PYTHONNOUSERSITE=1

# ====================================
# Model Configuration
# ====================================
# TODO: Update these paths to your trained model checkpoint
MODEL_PATH="/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/dp_lora_weight_128"
MODEL_BASE="liuhaotian/llava-v1.5-7b"  # Base model for LoRA

# ====================================
# Data Configuration
# ====================================
DATA_PATH_DEV="/project2/ruishanl_1185/SDP_for_VLM/datasets/rexgradient/ReXGradient/metadata/rexgradient_valid.json"
DATA_PATH_TEST="/project2/ruishanl_1185/SDP_for_VLM/datasets/rexgradient/ReXGradient/metadata/rexgradient_test.json"
IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/rexgradient/ReXGradient"

# Extract checkpoint name for output directory
CHECKPOINT_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR="/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/dp_lora_weight_128/eval_new/eval_results_${CHECKPOINT_NAME}"

# ====================================
# MIMIC-CXR Filtering Options
# ====================================
# Note: generation_methods is hardcoded to "rule-based" in eval script for test data
FILTER_VIEWS=True         # Filter to only PA/AP views (recommended)
INCLUDE_REASON=True       # Include clinical indication in prompts

# ====================================
# Generation Configuration
# ====================================
TEMPERATURE=0.0           # Greedy decoding for reproducibility (set to 0)
NUM_BEAMS=1               # Beam search (1 = greedy)
MAX_NEW_TOKENS=256      # Maximum length of generated findings
# TOP_P=None                # Top-p sampling (None = disabled)

# ====================================
# Conversation Mode
# ====================================
CONV_MODE="v1"            # LLaVA conversation template version

# Create output directory and logs
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "=========================================="
echo "MIMIC-CXR Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ====================================
# Evaluate on Dev Set
# ====================================
echo "Evaluating on DEV set..."
echo "------------------------------------------"

# Build command with optional top-p argument
CMD="python /project2/ruishanl_1185/SDP_for_VLM/Xinyang/LLaVA_DP/LLaVA/llava/eval/eval_rex_gradient.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --data-file $DATA_PATH_DEV \
    --image-folder $IMAGE_FOLDER \
    --output-file $OUTPUT_DIR/dev_results.jsonl \
    --split dev \
    --filter-views $FILTER_VIEWS \
    --include-reason $INCLUDE_REASON \
    --temperature $TEMPERATURE \
    --num-beams $NUM_BEAMS \
    --max-new-tokens $MAX_NEW_TOKENS"

# Add top-p if defined
if [ -n "${TOP_P+x}" ]; then
    CMD="$CMD --top-p $TOP_P"
fi

CMD="$CMD --conv-mode $CONV_MODE"

# Execute command
eval $CMD

echo ""
echo "DEV set evaluation complete!"
echo "Results: $OUTPUT_DIR/dev_results.jsonl"
echo "Summary: $OUTPUT_DIR/dev_results_summary.json"
echo ""

# ====================================
# Evaluate on Test Set
# ====================================
echo "Evaluating on TEST set..."
echo "------------------------------------------"

# Build command with optional top-p argument
CMD="python /project2/ruishanl_1185/SDP_for_VLM/Xinyang/LLaVA_DP/LLaVA/llava/eval/eval_rex_gradient.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --data-file $DATA_PATH_TEST \
    --image-folder $IMAGE_FOLDER \
    --output-file $OUTPUT_DIR/test_results.jsonl \
    --split test \
    --filter-views $FILTER_VIEWS \
    --include-reason $INCLUDE_REASON \
    --temperature $TEMPERATURE \
    --num-beams $NUM_BEAMS \
    --max-new-tokens $MAX_NEW_TOKENS"

# Add top-p if defined
if [ -n "${TOP_P+x}" ]; then
    CMD="$CMD --top-p $TOP_P"
fi

CMD="$CMD --conv-mode $CONV_MODE"

# Execute command
eval $CMD

echo ""
echo "TEST set evaluation complete!"
echo "Results: $OUTPUT_DIR/test_results.jsonl"
echo "Summary: $OUTPUT_DIR/test_results_summary.json"
echo ""

# ====================================
# Print Final Summary
# ====================================
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "=========================================="
echo "All results saved to: $OUTPUT_DIR"
echo ""
echo "Files created:"
echo "  - dev_results.jsonl (per-sample predictions and metrics)"
echo "  - dev_results_summary.json (aggregate metrics)"
echo "  - test_results.jsonl (per-sample predictions and metrics)"
echo "  - test_results_summary.json (aggregate metrics)"
echo ""
echo "To view summaries:"
echo "  cat $OUTPUT_DIR/dev_results_summary.json"
echo "  cat $OUTPUT_DIR/test_results_summary.json"
echo "=========================================="

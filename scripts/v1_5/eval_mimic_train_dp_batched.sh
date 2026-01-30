#!/bin/bash

#SBATCH --job-name=llava_mimic_eval_train_batched
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Conda activation
source /home1/runhuixu/miniconda3/etc/profile.d/conda.sh
conda activate llava

# ====================================
# Model Configuration
# ====================================
MODEL_PATH=${1:-"/project2/ruishanl_1185/SDP_for_VLM/outputs/llava_llavarad/lora_128_dp_e8"}
MODEL_BASE="liuhaotian/llava-v1.5-7b"  # Base model for LoRA

# ====================================
# Data Configuration
# ====================================
DATA_PATH_TRAIN="/project2/ruishanl_1185/SDP_for_VLM/datasets/physionet.org/files/llava-rad-mimic-cxr-annotation/1.0.0/chat_train_p10_filtered.json"
IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/mimic-cxr-jpg/mimic-cxr-jpg/2.1.0/files/"

# Extract checkpoint name for output directory
CHECKPOINT_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR="/scratch1/runhuixu/evaluation/llava_llavarad/eval_results_${CHECKPOINT_NAME}_train_batched"

# ====================================
# MIMIC-CXR Filtering Options
# ====================================
FILTER_VIEWS=True                    # Filter to only PA/AP views (recommended)
INCLUDE_REASON=${2:-True}            # Include clinical indication in prompts
GENERATION_METHODS=${3:-"gpt4"}      # Which generation method to evaluate: "gpt4", "rule-based", or "all"

# ====================================
# Batching Configuration (NEW!)
# ====================================
BATCH_SIZE=${4:-4}          # Batch size for inference (default: 4)
NUM_WORKERS=4               # Dataloader workers
COMPUTE_LOSS=True           # Set to False to skip loss computation (faster)

# ====================================
# Generation Configuration
# ====================================
TEMPERATURE=0.0           # Greedy decoding for reproducibility (set to 0)
NUM_BEAMS=1               # Beam search (1 = greedy)
MAX_NEW_TOKENS=512        # Maximum length of generated findings

# ====================================
# Conversation Mode
# ====================================
CONV_MODE="v1"            # LLaVA conversation template version

# Create output directory and logs
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "=========================================="
echo "MIMIC-CXR Training Set BATCHED Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Compute loss: $COMPUTE_LOSS"
echo "Output directory: $OUTPUT_DIR"
echo "Generation methods: $GENERATION_METHODS"
echo "Filter views: $FILTER_VIEWS"
echo "Include reason: $INCLUDE_REASON"
echo "=========================================="
echo ""

# ====================================
# Evaluate on Training Set
# ====================================
echo "Evaluating on TRAINING set (batched)..."
echo "------------------------------------------"

python /scratch1/runhuixu/LLaVA/llava/eval/eval_mimic_cxr.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --data-file $DATA_PATH_TRAIN \
    --image-folder $IMAGE_FOLDER \
    --output-file $OUTPUT_DIR/train_results.jsonl \
    --split train \
    --filter-views $FILTER_VIEWS \
    --include-reason $INCLUDE_REASON \
    --generation-methods $GENERATION_METHODS \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --compute-loss $COMPUTE_LOSS \
    --temperature $TEMPERATURE \
    --num-beams $NUM_BEAMS \
    --max-new-tokens $MAX_NEW_TOKENS \
    --conv-mode $CONV_MODE

echo ""
echo "TRAIN set evaluation complete!"
echo "Results: $OUTPUT_DIR/train_results.jsonl"
echo "Summary: $OUTPUT_DIR/train_results_summary.json"
echo ""

# ====================================
# Print Final Summary
# ====================================
echo "=========================================="
echo "BATCHED EVALUATION COMPLETE"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files created:"
echo "  - train_results.jsonl (per-sample predictions and metrics)"
echo "  - train_results_summary.json (aggregate metrics)"
echo ""
echo "To view summary:"
echo "  cat $OUTPUT_DIR/train_results_summary.json"
echo ""
echo "Speedup achieved with batch_size=$BATCH_SIZE"
echo "=========================================="

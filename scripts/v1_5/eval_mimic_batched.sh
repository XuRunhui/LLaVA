#!/bin/bash

#SBATCH --job-name=llava_mimic_eval_batched
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Conda activation
source /home1/runhuixu/miniconda3/etc/profile.d/conda.sh
conda activate llava

# ====================================
# Model Configuration
# ====================================
MODEL_PATH=${1:-"/scratch1/runhuixu/outputs/llava_llavarad/lora_128_gpt4"}
MODEL_BASE="liuhaotian/llava-v1.5-7b"

# ====================================
# Data Configuration
# ====================================
DATA_PATH_DEV="/project2/ruishanl_1185/SDP_for_VLM/datasets/physionet.org/files/llava-rad-mimic-cxr-annotation/1.0.0/chat_dev_p10_filtered.json"
DATA_PATH_TEST="/project2/ruishanl_1185/SDP_for_VLM/datasets/physionet.org/files/llava-rad-mimic-cxr-annotation/1.0.0/chat_test_p10_filtered.json"
IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/mimic-cxr-jpg/mimic-cxr-jpg/2.1.0/files/"

# Extract checkpoint name
CHECKPOINT_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR="/scratch1/runhuixu/evaluation/llava_llavarad/eval_results_${CHECKPOINT_NAME}_batched"

# ====================================
# MIMIC-CXR Filtering Options
# ====================================
FILTER_VIEWS=True
INCLUDE_REASON=True
GENERATION_METHODS="rule-based"  # For dev/test: rule-based

# ====================================
# Batching Configuration (NEW!)
# ====================================
BATCH_SIZE=${2:-4}          # Batch size for inference (1, 2, 4, 8)
NUM_WORKERS=4               # Dataloader workers
COMPUTE_LOSS=True           # Set to False to skip loss computation (faster)

# ====================================
# Generation Configuration
# ====================================
TEMPERATURE=0.0             # Greedy decoding
NUM_BEAMS=1
MAX_NEW_TOKENS=512

# ====================================
# Conversation Mode
# ====================================
CONV_MODE="v1"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "=========================================="
echo "MIMIC-CXR BATCHED Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Compute loss: $COMPUTE_LOSS"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ====================================
# Evaluate on DEV Set
# ====================================
echo "Evaluating on DEV set (batched)..."
echo "------------------------------------------"

python /scratch1/runhuixu/LLaVA/llava/eval/eval_mimic_cxr_batched.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --data-file $DATA_PATH_DEV \
    --image-folder $IMAGE_FOLDER \
    --output-file $OUTPUT_DIR/dev_results.jsonl \
    --split dev \
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
echo "DEV set evaluation complete!"
echo ""

# ====================================
# Evaluate on TEST Set
# ====================================
echo "Evaluating on TEST set (batched)..."
echo "------------------------------------------"

python /scratch1/runhuixu/LLaVA/llava/eval/eval_mimic_cxr_batched.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --data-file $DATA_PATH_TEST \
    --image-folder $IMAGE_FOLDER \
    --output-file $OUTPUT_DIR/test_results.jsonl \
    --split test \
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
echo "TEST set evaluation complete!"
echo ""

# ====================================
# Print Summary
# ====================================
echo "=========================================="
echo "BATCHED EVALUATION COMPLETE"
echo "=========================================="
echo "All results saved to: $OUTPUT_DIR"
echo ""
echo "Files created:"
echo "  - dev_results.jsonl"
echo "  - dev_results_summary.json"
echo "  - test_results.jsonl"
echo "  - test_results_summary.json"
echo ""
echo "Speedup achieved with batch_size=$BATCH_SIZE"
echo "=========================================="

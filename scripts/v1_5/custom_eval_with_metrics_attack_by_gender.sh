#!/bin/bash
#SBATCH --job-name=llava_rex_eval_attack_by_gender
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# ============================================================================
# LLaVA LoRA Finetuned Model Evaluation Script with Metrics
# ============================================================================
# This script evaluates LoRA-finetuned LLaVA models and computes:
#   - Loss and perplexity (if ground truth is available)
#   - Predictions for all samples
#   - Detailed metrics per sample
#   - Summary statistics
#
# Usage: bash scripts/v1_5/eval/custom_eval_with_metrics.sh
# ============================================================================

# ============================================================================
# CONFIGURATION - Modify these paths according to your setup
# ============================================================================

# Model Configurationsource /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
set -euo pipefail

module purge
module load gcc/13.3.0
module load cuda/12.6.3
export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

mkdir -p logs

# Conda (batch-safe) activation
source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate /project2/ruishanl_1185/SDP_for_VLM/envs/llava
export PYTHONNOUSERSITE=1


MODEL_PATH="/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/dp_lora_weight_128"  # Path to your LoRA checkpoint
MODEL_BASE="liuhaotian/llava-v1.5-7b"  # Base modssh://sdp_for_vlm/project2/ruishanl_1185/SDP_for_VLM/outputs/llava_llavarad/lora_128el for LoRA (set to "None" for full model)
# Data Configuration
QUESTION_FILE="/project2/ruishanl_1185/SDP_for_VLM/datasets/rexgradient/ReXGradient/metadata/rexgradient_train_gender_likelihood.json"
IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/rexgradient/ReXGradient"
ANSWERS_FILE="/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/dp_lora_weight_128/eval/ender_likelihood_with_metrics.jsonl"  # Output file

# Conversation Mode
CONV_MODE="v1"  # Options: llava_v1, llava_llama_2, vicuna_v1, etc.

# ============================================================================
# EVALUATION OPTIONS
# ============================================================================

# Compute loss and perplexity (requires ground truth in question file)
COMPUTE_METRICS=true  # Set to false to skip metrics computation

# Filter to only PA/AP views (useful for medical imaging)
FILTER_VIEWS=true  # Set to true to filter views

# Add reason field to prompts (if available in question file)
ADD_REASON=false  # rexgradient already contains reason field. 

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================

TEMPERATURE=0.0  # Temperature for sampling (0 = greedy, >0 = sampling)
TOP_P=1.0  # Top-p (nucleus) sampling parameter
NUM_BEAMS=1  # Number of beams for beam search (1 = greedy)
MAX_NEW_TOKENS=2048  # Maximum number of tokens to generate

# ============================================================================
# PARALLEL PROCESSING (Optional)
# ============================================================================

NUM_CHUNKS=1  # Number of chunks for parallel processing
CHUNK_IDX=0  # Current chunk index (0 to NUM_CHUNKS-1)

# ============================================================================
# BUILD COMMAND
# ============================================================================
cd /project2/ruishanl_1185/SDP_for_VLM/Xinyang/LLaVA_copy
CMD="python -m llava.eval.model_vqa_with_metrics \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --question-file $QUESTION_FILE \
    --answers-file $ANSWERS_FILE \
    --conv-mode $CONV_MODE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --num_beams $NUM_BEAMS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num-chunks $NUM_CHUNKS \
    --chunk-idx $CHUNK_IDX"

# Add model base if using LoRA
if [ "$MODEL_BASE" != "None" ]; then
    CMD="$CMD --model-base $MODEL_BASE"
fi

# Add optional flags
if [ "$COMPUTE_METRICS" = true ]; then
    CMD="$CMD --compute-metrics"
fi

if [ "$FILTER_VIEWS" = true ]; then
    CMD="$CMD --filter-views"
fi

if [ "$ADD_REASON" = true ]; then
    CMD="$CMD --add-reason"
fi

# ============================================================================
# RUN EVALUATION
# ============================================================================

echo "============================================================================"
echo "LLaVA LoRA Model Evaluation"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Model Path:        $MODEL_PATH"
echo "  Base Model:        $MODEL_BASE"
echo "  Image Folder:      $IMAGE_FOLDER"
echo "  Question File:     $QUESTION_FILE"
echo "  Answers File:      $ANSWERS_FILE"
echo ""
echo "Options:"
echo "  Compute Metrics:   $COMPUTE_METRICS"
echo "  Filter Views:      $FILTER_VIEWS"
echo "  Add Reason Field:  $ADD_REASON"
echo ""
echo "Generation Parameters:"
echo "  Temperature:       $TEMPERATURE"
echo "  Top-p:             $TOP_P"
echo "  Num Beams:         $NUM_BEAMS"
echo "  Max New Tokens:    $MAX_NEW_TOKENS"
echo ""
echo "============================================================================"
echo ""
echo "Running command:"
echo "$CMD"
echo ""
echo "============================================================================"
echo ""

# Run evaluation
eval $CMD

EXIT_CODE=$?

# ============================================================================
# POST-EVALUATION
# ============================================================================

echo ""
echo "============================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo ""
    echo "Output files:"
    echo "  Predictions:  $ANSWERS_FILE"
    echo "  Summary:      ${ANSWERS_FILE%.jsonl}_summary.json"
else
    echo "Evaluation failed with exit code $EXIT_CODE"
fi
echo "============================================================================"
echo ""

exit $EXIT_CODE

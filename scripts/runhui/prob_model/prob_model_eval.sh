#!/bin/bash

# Evaluate LoRA-finetuned LLaVA-Med v1.5 on VQA-RAD test set
# This script evaluates the model and generates predictions

# =============================================================================
# CONFIGURATION
# =============================================================================
if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <EPOCH_NUMBER>"
    exit 1
fi

NUM_EPOCHS=$1
# Path to your LoRA checkpoint directory
# This should contain: adapter_config.json, adapter_model.bin, non_lora_trainables.bin
LORA_MODEL_PATH="/project2/ruishanl_1185/SDP_for_VLM/outputs/prob_model"

# Base model (LLaVA-Med v1.5)
BASE_MODEL="microsoft/llava-med-v1.5-mistral-7b"

# VQA-RAD test data paths
TEST_DATA="/project2/ruishanl_1185/SDP_for_VLM/datasets/CheXpert_Synthesized_VQA/prob_model_data/llava_test.json"
IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/CheXpert_Synthesized_VQA/synthesized"

# Output path for predictions
OUTPUT_DIR="/project2/ruishanl_1185/SDP_for_VLM/outputs/prob_model/eval"
mkdir -p $OUTPUT_DIR
ANSWERS_FILE="$OUTPUT_DIR/prob_model_predictions_$NUM_EPOCHS.jsonl"

Checkpoint_DIR="$OUTPUT_DIR/epoch_$NUM_EPOCHS"
mkdir -p $Checkpoint_DIR
find /project2/ruishanl_1185/SDP_for_VLM/outputs/prob_model -maxdepth 1 -type f -exec cp {} $Checkpoint_DIR \;

# Evaluation parameters
CONV_MODE="mistral_instruct"  # Conversation template for Mistral
TEMPERATURE=0.0  # Use greedy decoding for deterministic results
MAX_NEW_TOKENS=512  # Maximum length of generated answer

# =============================================================================
# RUN EVALUATION
# =============================================================================

cd "$(dirname "$0")/.." || exit

echo "=========================================="
echo "Evaluating LoRA Model on VQA-RAD Test Set"
echo "=========================================="
echo "Base model: $BASE_MODEL"
echo "LoRA checkpoint: $LORA_MODEL_PATH"
echo "Test data: $TEST_DATA"
echo "Image folder: $IMAGE_FOLDER"
echo "Output file: $ANSWERS_FILE"
echo "=========================================="

# Check if test data exists
if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test data not found at $TEST_DATA"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Error: Image folder not found at $IMAGE_FOLDER"
    exit 1
fi

if [ ! -d "$LORA_MODEL_PATH" ]; then
    echo "Error: LoRA model not found at $LORA_MODEL_PATH"
    exit 1
fi

# Run evaluation
# python ../LLaVA/llava/eval/rad_vqa_eval_fixed.py \
#     --model-path $LORA_MODEL_PATH \
#     --model-base $BASE_MODEL \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --question-file $TEST_DATA \
#     --image-folder $IMAGE_FOLDER \
#     --answers-file $ANSWERS_FILE \
#     --conv-mode $CONV_MODE \
#     --temperature $TEMPERATURE \
#     --max_new_tokens $MAX_NEW_TOKENS
python /project2/ruishanl_1185/SDP_for_VLM/runhui/LLaVA/llava/eval/rad_vqa_eval_fixed.py \
    --model-path $LORA_MODEL_PATH \
    --model-base $BASE_MODEL \
    --question-file $TEST_DATA \
    --image-folder $IMAGE_FOLDER \
    --answers-file $ANSWERS_FILE \
    --conv-mode $CONV_MODE \
    --temperature $TEMPERATURE \
    --max-new-tokens $MAX_NEW_TOKENS 

echo "=========================================="
echo "Evaluation completed!"
echo "Predictions saved to: $ANSWERS_FILE"
echo "=========================================="


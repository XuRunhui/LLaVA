#!/bin/bash

# Evaluate LoRA-finetuned LLaVA-Med v1.5 on VQA-RAD test set
# This script evaluates the model and generates predictions

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to your LoRA checkpoint directory
# This should contain: adapter_config.json, adapter_model.bin, non_lora_trainables.bin
LORA_MODEL_PATH="/project2/ruishanl_1185/SDP_for_VLM/outputs/llava-med-v1.5-PathVQA-finetune/lora"

# Base model (LLaVA-Med v1.5)
BASE_MODEL="microsoft/llava-med-v1.5-mistral-7b"

# VQA-RAD test data paths
TEST_DATA="/project2/ruishanl_1185/SDP_for_VLM/datasets/PathVQA/annotations/llava_test.jsonl"
IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/PathVQA/images"

# Output path for predictions
OUTPUT_DIR="/project2/ruishanl_1185/SDP_for_VLM/outputs/llava-med-v1.5-PathVQA-finetune/lora/eval"
mkdir -p $OUTPUT_DIR
ANSWERS_FILE="$OUTPUT_DIR/path_vqa_llava_predictions.jsonl"

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


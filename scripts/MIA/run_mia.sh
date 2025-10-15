#!/bin/bash

# Quick-Start Example: Running a Single MIA Attack
# This script demonstrates how to run the Reference Attack on a small subset

set -e  # Exit on error

echo "=========================================="
echo "MIA Quick-Start Example"
echo "=========================================="
echo ""

# =============================================================================
# STEP 0: Configuration
# =============================================================================

# Model paths (CHANGE THESE TO YOUR PATHS!)
MODEL_PATH="/project/ruishanl_1185/SDP_for_VLM/outputs/llava-med-v1.5-vqarad-finetune/lora"
BASE_MODEL="microsoft/llava-med-v1.5-mistral-7b"

# Data paths (CHANGE THESE TO YOUR PATHS!)
FULL_TRAIN_DATA="data/vqa_rad/train.json"
IMAGE_FOLDER="data/VQA_RAD_Image_Folder"

# Output directory
OUTPUT_DIR="results/mia_quickstart"
mkdir -p $OUTPUT_DIR

# Attack parameters
GRANULARITY=50
SIMILARITY_METRIC="rouge2_f"

# =============================================================================
# STEP 1: Split Data
# =============================================================================

# echo "Step 1: Splitting data into member/non-member sets..."

# python - <<EOF
# import json
# import random

# # Load full training data
# with open('$FULL_TRAIN_DATA', 'r') as f:
#     data = json.load(f)

# # Shuffle and split (80% member, 20% non-member)
# random.seed(42)
# random.shuffle(data)
# split_idx = int(len(data) * 0.8)

# member_data = data[:split_idx]
# non_member_data = data[split_idx:]

# # Save splits
# with open('$OUTPUT_DIR/member_data.json', 'w') as f:
#     json.dump(member_data, f, indent=2)

# with open('$OUTPUT_DIR/non_member_data.json', 'w') as f:
#     json.dump(non_member_data, f, indent=2)

# print(f"Member data: {len(member_data)} samples")
# print(f"Non-member data: {len(non_member_data)} samples")
# print(f"Saved to: $OUTPUT_DIR/")
# EOF

# echo ""

# =============================================================================
# STEP 2: Generate Conversations
# =============================================================================

echo "Step 2: Generating conversations for member data..."

python scripts/mia_conversation_generation.py \
    --model-path $MODEL_PATH \
    --model-base $BASE_MODEL \
    --conv-mode mistral_instruct \
    --input-json-path $OUTPUT_DIR/member_data.json \
    --image-folder $IMAGE_FOLDER \
    --output-json-path $OUTPUT_DIR/member_conversations.json \
    --temperatures 0.1 \
    --repeat 1 \
    --max-new-tokens 512

echo ""
echo "Generating conversations for non-member data..."

python scripts/mia_conversation_generation.py \
    --model-path $MODEL_PATH \
    --model-base $BASE_MODEL \
    --conv-mode mistral_instruct \
    --input-json-path $OUTPUT_DIR/non_member_data.json \
    --image-folder $IMAGE_FOLDER \
    --output-json-path $OUTPUT_DIR/non_member_conversations.json \
    --temperatures 0.1 \
    --repeat 1 \
    --max-new-tokens 512

echo ""

# =============================================================================
# STEP 3: Calculate Similarity
# =============================================================================

echo "Step 3: Calculating similarity scores..."

python scripts/mia_similarity_calculation.py \
    --conversation-json-path $OUTPUT_DIR/member_conversations.json \
    --similarity-json-path $OUTPUT_DIR/member_similarity.json \
    --temperatures 0.1

python scripts/mia_similarity_calculation.py \
    --conversation-json-path $OUTPUT_DIR/non_member_conversations.json \
    --similarity-json-path $OUTPUT_DIR/non_member_similarity.json \
    --temperatures 0.1

echo ""

# =============================================================================
# STEP 4: Run Reference Attack
# =============================================================================

echo "Step 4: Running Reference Attack..."

python scripts/mia_reference_attack.py \
    --member-similarity-file $OUTPUT_DIR/member_similarity.json \
    --non-member-similarity-file $OUTPUT_DIR/non_member_similarity.json \
    --granularity $GRANULARITY \
    --temperature 0.1 \
    --similarity-metric $SIMILARITY_METRIC \
    --output-file $OUTPUT_DIR/attack_results.json

echo ""

# =============================================================================
# STEP 5: Display Results
# =============================================================================

echo "=========================================="
echo "ATTACK COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/attack_results.json"
echo ""

if [ -f "$OUTPUT_DIR/attack_results.json" ]; then
    python -c "
import json

with open('$OUTPUT_DIR/attack_results.json', 'r') as f:
    results = json.load(f)

print('Reference Attack Results:')
print(f'  AUC:       {results[\"auc\"]:.4f} ± {results[\"auc_std\"]:.4f}')
print(f'  Accuracy:  {results[\"accuracy\"]:.4f} ± {results[\"accuracy_std\"]:.4f}')
print(f'  F1 Score:  {results[\"f1\"]:.4f}')
print(f'  Precision: {results[\"precision\"]:.4f}')
print(f'  Recall:    {results[\"recall\"]:.4f}')
"
fi

echo ""
echo "=========================================="
echo ""
echo "To run other attacks, see: scripts/run_mia_pipeline.sh"

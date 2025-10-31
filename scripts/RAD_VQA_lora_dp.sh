#!/bin/bash

# Fine-tuning LLaVA-Med v1.5 (Mistral-7B) on VQA-RAD dataset with Differential Privacy
# This script adds differential privacy guarantees to protect patient data privacy
#
# Key DP Parameters:
# - epsilon (ε): Privacy budget (lower = more private, typical: 1-10)
# - delta (δ): Failure probability (typically 1/n where n is dataset size)
# - max_grad_norm: Gradient clipping threshold (typical: 0.1-1.0)

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SETUP
# =============================================================================
if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <EPOCH_NUMBER> [EPSILON]"
    echo "  EPOCH_NUMBER: Number of training epochs"
    echo "  EPSILON: Privacy budget (default: 3.0, lower = more private)"
    echo ""
    echo "Examples:"
    echo "  bash $0 3           # Train for 3 epochs with ε=3.0 (strong privacy)"
    echo "  bash $0 5 8.0       # Train for 5 epochs with ε=8.0 (moderate privacy)"
    exit 1
fi

# Model and data paths
MODEL_NAME="microsoft/llava-med-v1.5-mistral-7b"
DATA_PATH="/project2/ruishanl_1185/SDP_for_VLM/datasets/vqa_rad/train.json"
IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/vqa_rad/VQA_RAD_Image_Folder"
OUTPUT_DIR="/project2/ruishanl_1185/SDP_for_VLM/outputs/llava-med-v1.5-vqarad-finetune/lora_64_dp"

# Vision tower (must match the pretrained model)
VISION_TOWER="openai/clip-vit-large-patch14-336"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Training epochs and batch size
NUM_EPOCHS=$1
EPSILON=${2:-3.0}  # Default epsilon = 3.0 (strong privacy)

# Batch size configuration for DP
# Note: Smaller batch sizes typically give better privacy/utility tradeoff
# For DP, we recommend batch size = 0.1-1% of dataset size
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8  # Reduced from 16 for better DP guarantees

# Learning rates
LR=2e-4
MM_PROJECTOR_LR=2e-5  # Learning rate for vision-language projector

# Optimization
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
LR_SCHEDULER="cosine"

# Model configuration
MAX_LENGTH=2048  # Mistral supports longer sequences than Llama
IMAGE_ASPECT_RATIO="pad"  # 'pad' or 'square'

# Training strategy - LoRA is recommended for DP training
USE_LORA=True
TUNE_MM_MLP_ADAPTER=False
LORA_ENABLE=True
LORA_R=64  # LoRA rank
LORA_ALPHA=64  # LoRA scaling factor
LORA_DROPOUT=0.05
BITS=16  # 16-bit training recommended for DP (quantization may affect privacy)
GRADIENT_CHECKPOINTING=False  # Disabled for LoRA

FREEZE_MM_MLP_ADAPTER=False
FREEZE_BACKBONE=False

# Precision
BF16=True
FP16=False

# Checkpointing
SAVE_STRATEGY="steps"
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=3

# Logging
LOGGING_STEPS=10
REPORT_TO="wandb"  # Change to "none" to disable wandb

# =============================================================================
# DIFFERENTIAL PRIVACY CONFIGURATION
# =============================================================================

DP_ENABLED=True

# Privacy budget parameters
DP_EPSILON=$EPSILON          # Target privacy budget (ε)
DP_DELTA=1e-5               # Failure probability (δ) - typically 1/n

# Gradient clipping for DP
DP_MAX_GRAD_NORM=1.0        # Clip gradients to this L2 norm
                            # Lower values = stronger privacy but may hurt utility
                            # Typical range: 0.1 - 1.0

# DP sampling strategy
DP_POISSON_SAMPLING=True    # Use Poisson sampling for stronger privacy
DP_SECURE_MODE=False        # Use cryptographically secure RNG (slower)

# Memory management for DP
DP_PHYSICAL_BATCH_SIZE=2    # Physical batch size for memory efficiency
                            # Set to None to use BATCH_SIZE

# Privacy guidance:
# ε < 1:    Very strong privacy (may significantly hurt utility)
# ε = 1-3:  Strong privacy (recommended for sensitive medical data)
# ε = 3-10: Moderate privacy (common in practice)
# ε > 10:   Weak privacy (not recommended for sensitive data)

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================

NUM_GPUS=1
MASTER_PORT=29500

# Memory optimization
LAZY_PREPROCESS=True

# =============================================================================
# VALIDATION AND SETUP
# =============================================================================

cd "$(dirname "$0")/.." || exit

# Check if Opacus is installed
python -c "import opacus" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Opacus not found. Please install it:"
    echo "  pip install opacus"
    exit 1
fi

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data not found at $DATA_PATH"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Error: Image folder not found at $IMAGE_FOLDER"
    exit 1
fi

# Calculate dataset size for privacy analysis
DATASET_SIZE=$(python -c "import json; print(len(json.load(open('$DATA_PATH'))))")
RECOMMENDED_DELTA=$(python -c "print(f'{1.0/$DATASET_SIZE:.2e}')")

echo "=========================================="
echo "LLaVA-Med v1.5 DP Fine-tuning on VQA-RAD"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Dataset size: $DATASET_SIZE samples"
echo "Images: $IMAGE_FOLDER"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS)))"
echo "Learning rate: $LR"
echo ""
echo "=========================================="
echo "DIFFERENTIAL PRIVACY CONFIGURATION"
echo "=========================================="
echo "Privacy enabled: $DP_ENABLED"
echo "Target ε (epsilon): $DP_EPSILON"
echo "Target δ (delta): $DP_DELTA"
echo "  (recommended δ for this dataset: $RECOMMENDED_DELTA)"
echo "Max gradient norm: $DP_MAX_GRAD_NORM"
echo "Poisson sampling: $DP_POISSON_SAMPLING"
echo ""
echo "Privacy level interpretation:"
if (( $(echo "$DP_EPSILON < 1" | bc -l) )); then
    echo "  ✓ VERY STRONG privacy (ε < 1)"
elif (( $(echo "$DP_EPSILON <= 3" | bc -l) )); then
    echo "  ✓ STRONG privacy (ε ≤ 3) - Recommended for medical data"
elif (( $(echo "$DP_EPSILON <= 10" | bc -l) )); then
    echo "  ⚠ MODERATE privacy (ε ≤ 10)"
else
    echo "  ⚠ WEAK privacy (ε > 10) - Not recommended"
fi
echo "=========================================="
echo ""
echo "Training mode: LoRA (rank=$LORA_R, alpha=$LORA_ALPHA)"
echo "=========================================="
echo ""
read -p "Press Enter to start training or Ctrl+C to cancel..."

# =============================================================================
# RUN TRAINING WITH DIFFERENTIAL PRIVACY
# =============================================================================

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    /project2/ruishanl_1185/SDP_for_VLM/runhui/LLaVA/llava/train/train_dp.py \
    --model_name_or_path $MODEL_NAME \
    --version mistral_instruct \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio $IMAGE_ASPECT_RATIO \
    --group_by_modality_length False \
    --bf16 $BF16 \
    --fp16 $FP16 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --evaluation_strategy "no" \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type $LR_SCHEDULER \
    --logging_steps $LOGGING_STEPS \
    --tf32 True \
    --model_max_length $MAX_LENGTH \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --dataloader_num_workers 4 \
    --lazy_preprocess $LAZY_PREPROCESS \
    --report_to $REPORT_TO \
    --tune_mm_mlp_adapter $TUNE_MM_MLP_ADAPTER \
    --freeze_mm_mlp_adapter $FREEZE_MM_MLP_ADAPTER \
    --freeze_backbone $FREEZE_BACKBONE \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --bits $BITS \
    --lora_enable $LORA_ENABLE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --dp_enabled $DP_ENABLED \
    --dp_epsilon $DP_EPSILON \
    --dp_delta $DP_DELTA \
    --dp_max_grad_norm $DP_MAX_GRAD_NORM \
    --dp_poisson_sampling $DP_POISSON_SAMPLING \
    --dp_secure_mode $DP_SECURE_MODE \
    --dp_physical_batch_size $DP_PHYSICAL_BATCH_SIZE

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "Privacy accounting saved in checkpoints"
echo "=========================================="
echo ""
echo "To check final privacy spent:"
echo "  cat $OUTPUT_DIR/checkpoint-*/privacy_accounting.json"

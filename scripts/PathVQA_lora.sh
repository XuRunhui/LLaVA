#!/bin/bash
# Fine-tuning LLaVA-Med v1.5 (Mistral-7B) on VQA-RAD dataset
# This script assumes:
# 1. You have run the VQA-RAD conversion script to create train.json and test.json
# 2. VQA-RAD images are in data/VQA_RAD_Image_Folder/
# 3. Converted data is in data/vqa_rad_llava/train.json

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SETUP
# =============================================================================

# Model and data paths
MODEL_NAME="microsoft/llava-med-v1.5-mistral-7b"
DATA_PATH="/project2/ruishanl_1185/SDP_for_VLM/datasets/PathVQA/annotations/llava_train.json"
IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/PathVQA/images"
OUTPUT_DIR="/project2/ruishanl_1185/SDP_for_VLM/outputs/llava-med-v1.5-PathVQA-finetune/lora"

# Vision tower (must match the pretrained model)
VISION_TOWER="openai/clip-vit-large-patch14-336"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Training epochs and batch size
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM_STEPS=4  # Effective batch size = 4 * 2 = 8 (per GPU: 4 * 2 GPUs * 2 accum = 16 total)

# Learning rates
LR=2e-5
MM_PROJECTOR_LR=2e-5  # Learning rate for vision-language projector

# Optimization
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
LR_SCHEDULER="cosine"

# Model configuration
MAX_LENGTH=2048  # Mistral supports longer sequences than Llama
IMAGE_ASPECT_RATIO="pad"  # 'pad' or 'square'

# Training strategy
# Three training modes:
#   1. Full fine-tuning (tune_mm_mlp_adapter=False, lora_enable=False):
#      - Trains all 7B parameters of Mistral + projector
#      - Best performance, requires most memory (~40-80GB)
#   2. LoRA fine-tuning (tune_mm_mlp_adapter=False, lora_enable=True):
#      - Trains LoRA adapters on Mistral + full projector
#      - Good performance, much less memory (~12-24GB)
#   3. Projector-only (tune_mm_mlp_adapter=True):
#      - ONLY trains projector, freezes Mistral
#      - For stage 1 alignment only

# Choose training mode:
USE_LORA=True  # Set to True for LoRA training, False for full fine-tuning

if [ "$USE_LORA" = "True" ]; then
    TUNE_MM_MLP_ADAPTER=False  # Train both projector and LLM (with LoRA)
    LORA_ENABLE=True
    LORA_R=64  # LoRA rank (higher = more parameters, better performance)
    LORA_ALPHA=16  # LoRA scaling factor
    LORA_DROPOUT=0.05
    BITS=16  # Can use 4 or 8 for quantization to save more memory
    GRADIENT_CHECKPOINTING=False # IMPORTANT: Disable gradient checkpointing for LoRA to avoid DDP conflicts
else
    TUNE_MM_MLP_ADAPTER=False  # Train both projector and LLM (full weights)
    LORA_ENABLE=False
    BITS=16
    GRADIENT_CHECKPOINTING=True
fi

FREEZE_MM_MLP_ADAPTER=False  # Keep projector trainable
FREEZE_BACKBONE=False  # Keep LLM trainable (or LoRA adapters if using LoRA)

# Precision
BF16=True  # Use bfloat16 for better numerical stability
FP16=False

# Checkpointing
SAVE_STRATEGY="steps"
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=3

# Logging
LOGGING_STEPS=10
REPORT_TO="wandb"  # Change to "none" to disable wandb

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================

# Number of GPUs (adjust based on your setup)
NUM_GPUS=1
MASTER_PORT=29500

# Memory optimization
# GRADIENT_CHECKPOINTING=True
LAZY_PREPROCESS=True

# =============================================================================
# RUN TRAINING
# =============================================================================

# Use the LLaVA train.py from the parent LLaVA repo (with our Mistral modifications)
cd "$(dirname "$0")/.." || exit

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data not found at $DATA_PATH"
    echo "Please run the VQA-RAD conversion script first:"
    echo "  python scripts/convert_vqarad_to_llava_improved.py \\"
    echo "    --input data/VQA_RAD_Dataset_Public.json \\"
    echo "    --output-dir data/vqa_rad_llava"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Error: Image folder not found at $IMAGE_FOLDER"
    echo "Please ensure VQA-RAD images are downloaded to $IMAGE_FOLDER"
    exit 1
fi

echo "=========================================="
echo "LLaVA-Med v1.5 Fine-tuning on VQA-RAD"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Images: $IMAGE_FOLDER"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS)))"
echo "Learning rate: $LR"
if [ "$LORA_ENABLE" = "True" ]; then
    echo "Training mode: LoRA (rank=$LORA_R, alpha=$LORA_ALPHA)"
else
    echo "Training mode: Full fine-tuning"
fi
echo "=========================================="

# Run training with torchrun for distributed training
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    /project2/ruishanl_1185/SDP_for_VLM/runhui/LLaVA/llava/train/train.py \
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
    --group_by_modality_length True \
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
    $(if [ "$LORA_ENABLE" = "True" ]; then echo "--lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT"; fi)

echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="

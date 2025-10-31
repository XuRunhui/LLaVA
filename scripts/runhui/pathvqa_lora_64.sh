#!/bin/bash
#SBATCH --job-name=pathvqa_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

################################################################################
# LLaVA-Med v1.5 LoRA Fine-tuning on PathVQA
#
# PathVQA dataset:
#   - Training: ~6,700 samples
#   - Test: ~6,800 samples
#   - Total: ~13,500 samples (MUCH larger than VQA-RAD)
#
# This script uses LoRA for memory-efficient fine-tuning
################################################################################

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

module purge
module load gcc/13.3.0
module load cuda/12.6.3
export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

mkdir -p logs

# Conda activation
source /home1/runhuixu/miniconda3/etc/profile.d/conda.sh
conda activate llava

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model paths
MODEL_NAME="microsoft/llava-med-v1.5-mistral-7b"
DATA_PATH="/project2/ruishanl_1185/SDP_for_VLM/datasets/PathVQA/annotations/llava_train.json"
IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/PathVQA/images"
OUTPUT_DIR="/project2/ruishanl_1185/SDP_for_VLM/outputs/llava-med-v1.5-PathVQA-finetune/lora_64"

# Vision tower
VISION_TOWER="openai/clip-vit-large-patch14-336"

# =============================================================================
# LORA HYPERPARAMETERS (OPTIMIZED FOR PATHVQA)
# =============================================================================

LORA_R=64              # LoRA rank
LORA_ALPHA=64          # INCREASED: Match rank for stronger adaptation
LORA_DROPOUT=0.05      # Standard dropout
BITS=16                # Precision (16-bit = fp16/bf16)

# =============================================================================
# TRAINING HYPERPARAMETERS (OPTIMIZED FOR LARGE DATASET)
# =============================================================================

# Epochs: PathVQA is large, so fewer epochs needed
NUM_EPOCHS=1           # REDUCED: 1 epoch is often enough for 6,700 samples

# Batch size strategy
# PathVQA is large, so we can use larger effective batch
NUM_GPUS=1
BATCH_SIZE=2           # INCREASED: 4 per GPU (adjust based on memory)
GRAD_ACCUM_STEPS=16     # Effective batch = 4 × 1 × 8 = 32

# Learning rates
# LoRA needs HIGHER learning rate than full fine-tuning
LR=2e-4                # INCREASED: 10x higher for LoRA (was 2e-5)
MM_PROJECTOR_LR=2e-5   # Keep projector LR standard

# Optimization
WEIGHT_DECAY=0.01      # Regularization for large dataset (set to 0.0 if underfitting)
WARMUP_RATIO=0.03      # 3% warmup
LR_SCHEDULER="cosine"  # Cosine decay

# Model configuration
MAX_LENGTH=2048
IMAGE_ASPECT_RATIO="pad"

# Training flags
TUNE_MM_MLP_ADAPTER=False   # Train projector with LoRA
FREEZE_MM_MLP_ADAPTER=False
FREEZE_BACKBONE=False

# Precision
BF16=True  # bfloat16 for stability
FP16=False

# Important: Disable gradient checkpointing for LoRA (prevents DDP conflicts)
GRADIENT_CHECKPOINTING=False

# Checkpointing
SAVE_STRATEGY="steps"
SAVE_STEPS=100         # Save every 500 steps (~3 times per epoch)
SAVE_TOTAL_LIMIT=3     # Keep last 3 checkpoints

# Logging
LOGGING_STEPS=10       # Log every 50 steps
REPORT_TO="wandb"      # Change to "none" to disable

# Other
LAZY_PREPROCESS=True
MASTER_PORT=29500

# =============================================================================
# DATA VALIDATION
# =============================================================================

echo "=========================================="
echo "LLaVA-Med v1.5 LoRA Fine-tuning on PathVQA"
echo "=========================================="
echo ""

# Validate paths
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Training data not found at $DATA_PATH"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ Error: Image folder not found at $IMAGE_FOLDER"
    exit 1
fi

# Print dataset statistics
echo "Checking dataset..."
python -c "
import json
import os

# Load training data
with open('${DATA_PATH}', 'r') as f:
    train_data = json.load(f)

print(f'✅ Training data: {len(train_data)} samples')

# Check expected size for PathVQA
expected_min = 6000
expected_max = 7000
if len(train_data) < expected_min:
    print(f'⚠️  Warning: Expected ~6,700 samples, got {len(train_data)}')
elif len(train_data) > expected_max:
    print(f'⚠️  Warning: Expected ~6,700 samples, got {len(train_data)}')
else:
    print(f'✅ Sample count looks correct for PathVQA')

# Check images exist
image_folder = '${IMAGE_FOLDER}'
if os.path.exists(image_folder):
    num_images = len([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f'✅ Image folder: {num_images} images found')
else:
    print(f'❌ Image folder not found')
"

echo ""

# =============================================================================
# TRAINING CONFIGURATION SUMMARY
# =============================================================================

echo "Configuration Summary:"
echo "  Model: ${MODEL_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPU: ${NUM_GPUS}x A100"
echo ""
echo "LoRA Settings:"
echo "  Rank (r): ${LORA_R}"
echo "  Alpha (α): ${LORA_ALPHA}"
echo "  Dropout: ${LORA_DROPOUT}"
echo "  Estimated trainable params: ~70M (vs 7B for full fine-tuning)"
echo ""
echo "Training Settings:"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning rate (LoRA): ${LR}"
echo "  Learning rate (Projector): ${MM_PROJECTOR_LR}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "  Weight decay: ${WEIGHT_DECAY}"
echo ""
echo "Memory Optimization:"
echo "  Gradient checkpointing: ${GRADIENT_CHECKPOINTING}"
echo "  Lazy preprocess: ${LAZY_PREPROCESS}"
echo "  Expected memory: ~20-25GB per GPU"
echo ""

# Calculate training steps
python -c "
import json
with open('${DATA_PATH}', 'r') as f:
    data = json.load(f)
samples = len(data)
batch = ${BATCH_SIZE} * ${NUM_GPUS} * ${GRAD_ACCUM_STEPS}
steps_per_epoch = samples // batch
total_steps = steps_per_epoch * ${NUM_EPOCHS}
print(f'Estimated training:')
print(f'  Steps per epoch: {steps_per_epoch}')
print(f'  Total steps: {total_steps}')
print(f'  Checkpoints: ~{total_steps // ${SAVE_STEPS}} (every ${SAVE_STEPS} steps)')
"

echo ""
echo "=========================================="
echo ""

# # Check for existing checkpoints (resume capability)
# RESUME_FLAG=""
# if [ -d "$OUTPUT_DIR" ]; then
#     CHECKPOINT_DIRS=($(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V))
#     if [ ${#CHECKPOINT_DIRS[@]} -gt 0 ]; then
#         LATEST_CHECKPOINT="${CHECKPOINT_DIRS[-1]}"
#         echo "✅ Found existing checkpoint: $LATEST_CHECKPOINT"
#         echo "   Training will resume from this checkpoint"
#         RESUME_FLAG="--resume_from_checkpoint $LATEST_CHECKPOINT"
#         echo ""
#     fi
# fi

# =============================================================================
# LAUNCH TRAINING
# =============================================================================

echo "Starting LoRA fine-tuning..."
echo ""

cd /project2/ruishanl_1185/SDP_for_VLM/runhui/LLaVA || exit 1

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    llava/train/train.py \
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
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT

# =============================================================================
# POST-TRAINING
# =============================================================================

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""

# List final checkpoints
if [ -d "$OUTPUT_DIR" ]; then
    echo "Final checkpoints:"
    ls -lh "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | tail -5 || echo "  (no checkpoints found)"
    echo ""

    # Find the latest checkpoint
    CHECKPOINT_DIRS=($(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V))
    if [ ${#CHECKPOINT_DIRS[@]} -gt 0 ]; then
        FINAL_CHECKPOINT="${CHECKPOINT_DIRS[-1]}"
        echo "Latest checkpoint: $FINAL_CHECKPOINT"
        echo ""

        # Check LoRA adapter files
        echo "LoRA adapter files:"
        ls -lh "$FINAL_CHECKPOINT"/adapter_* 2>/dev/null || echo "  (adapter files not found)"
        ls -lh "$FINAL_CHECKPOINT"/non_lora_trainables.bin 2>/dev/null || echo "  (non_lora_trainables.bin not found)"
        echo ""
    fi
fi

echo "To evaluate this model:"
echo "  python llava/eval/model_vqa_loader.py \\"
echo "    --model-path $OUTPUT_DIR \\"
echo "    --model-base $MODEL_NAME \\"
echo "    --question-file /path/to/test.json \\"
echo "    --image-folder $IMAGE_FOLDER \\"
echo "    --answers-file results.jsonl"
echo ""
echo "=========================================="

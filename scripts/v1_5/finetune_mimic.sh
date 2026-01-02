#!/bin/bash
#SBATCH --job-name=llava_llavarad
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# module purge
# module load gcc/13.3.0
# module load cuda/12.6.3
# export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

# mkdir -p logs

# # Conda (batch-safe) activation
# source /home1/runhuixu/miniconda3/etc/profile.d/conda.sh
# conda activate llava



MODEL_NAME="liuhaotian/llava-v1.5-7b"
DATA_PATH="/root/autodl-tmp/mimic_cxr_jpg/chat_train_p10_filtered.json"
IMAGE_FOLDER="/root/autodl-tmp/mimic_cxr_jpg/mimic-cxr-jpg/2.1.0/files/p10"
OUTPUT_DIR="/root/autodl-tmp/mimic_cxr_jpg/output/llava_llavarad/lora_128"
# Number of GPUs (adjust based on your setup)
NUM_GPUS=1
MASTER_PORT=29500

# ====================================
# Differential Privacy Configuration
# ====================================
# IMPORTANT: DP training protects patient privacy in medical data
# Set DP_ENABLED=False to train without differential privacy

DP_ENABLED=True              # Set to False to disable DP
DP_EPSILON=8.0               # Privacy budget (lower = more private, e.g., 1.0-10.0)
DP_DELTA=1e-5                # Privacy parameter (typically 1/dataset_size)
DP_MAX_GRAD_NORM=1.0         # Gradient clipping threshold (adjust based on convergence)
DP_GHOST_CLIPPING=True       # Use ghost clipping for memory efficiency (recommended for large models)

# Privacy Budget Guide:
# - ε=1.0: Very strong privacy (may reduce model utility)
# - ε=3.0-8.0: Moderate privacy (good balance)
# - ε=10.0+: Weaker privacy (better utility)
# deepspeed 
    # --deepspeed /project2/ruishanl_1185/SDP_for_VLM/runhui/LLaVA/scripts/zero3.json \
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    /root/LLaVA/llava/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $MODEL_NAME\
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER\
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --dp_enabled $DP_ENABLED \
    --dp_epsilon $DP_EPSILON \
    --dp_delta $DP_DELTA \
    --dp_max_grad_norm $DP_MAX_GRAD_NORM \
    --dp_use_ghost_clipping $DP_GHOST_CLIPPING

#!/bin/bash
#SBATCH --job-name=llava_llavarad_dp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# module purge
# module load gcc/13.3.0
# module load cuda/12.6.3
# export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

# mkdir -p logs

# # Conda (batch-safe) activation
source /home1/runhuixu/miniconda3/etc/profile.d/conda.sh
conda activate llava



MODEL_NAME="liuhaotian/llava-v1.5-7b"
DATA_PATH="/scratch1/runhuixu/evaluation/llava_llavarad/eval_results_lora_128_gpt4_train/chat_train_p10_filtered_prompt_inversion_train.json"
IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/mimic-cxr-jpg/mimic-cxr-jpg/2.1.0/files/"
OUTPUT_DIR="/scratch1/runhuixu/outputs/llava_llavarad/prompt_inversion"
# Number of GPUs (adjust based on your setup)
NUM_GPUS=1
MASTER_PORT=29500

# ====================================
# Differential Privacy Configuration
# ====================================
# IMPORTANT: DP training protects patient privacy in medical data
# Set DP_ENABLED=False to train without differential privacy

DP_ENABLED=False # Set to False to disable DP
DP_EPSILON=8.0               # Privacy budget (lower = more private, e.g., 1.0-10.0)
DP_DELTA=5e-5                # Privacy parameter (typically 1/dataset_size)
DP_MAX_GRAD_NORM=2.0         # Gradient clipping threshold (adjust based on convergence)
DP_GHOST_CLIPPING=True       # Use ghost clipping for memory efficiency (recommended for large models)

# Privacy Budget Guide:
# - ε=1.0: Very strong privacy (may reduce model utility)
# - ε=3.0-8.0: Moderate privacy (good balance)
# - ε=10.0+: Weaker privacy (better utility)

# NOTE: The trainer automatically converts trainable parameters to FP32 for DP
# Non-trainable parts (vision tower, frozen base) remain in BF16 for efficiency

# ====================================
# MIMIC-CXR Data Filtering Options
# ====================================
USE_MIMIC_LOADER=True            # Enable MIMIC-CXR specific data loader
MIMIC_FILTER_VIEWS=True          # Filter to only PA/AP views (recommended)
MIMIC_INCLUDE_REASON=True        # Include clinical indication/reason in prompts
MIMIC_GENERATION_METHODS="gpt4"   # Options: "all", "gpt4", or "rule-based"

    # --tune_mm_mlp_adapter True 
# deepspeed
    # --deepspeed /project2/ruishanl_1185/SDP_for_VLM/runhui/LLaVA/scripts/zero3.json \
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    /scratch1/runhuixu/LLaVA/llava/train/train.py \
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
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
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
    --dp_use_ghost_clipping $DP_GHOST_CLIPPING \
    --use_mimic_loader $USE_MIMIC_LOADER \
    --mimic_filter_views $MIMIC_FILTER_VIEWS \
    --mimic_include_reason $MIMIC_INCLUDE_REASON \
    --mimic_generation_methods $MIMIC_GENERATION_METHODS

bash /scratch1/runhuixu/LLaVA/scripts/v1_5/eval_prompt_attack.sh $OUTPUT_DIR True
# bash eval_all_mimic_metrics.sh $(basename $OUTPUT_DIR)
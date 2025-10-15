#!/bin/bash
#SBATCH --job-name=pathvqa_3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load gcc/13.3.0
module load cuda/12.6.3
export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

mkdir -p logs

# Conda (batch-safe) activation
source /home1/runhuixu/miniconda3/etc/profile.d/conda.sh
conda activate llava

cd /project2/ruishanl_1185/SDP_for_VLM/runhui/LLaVA

bash scripts/PathVQA_lora.sh
bash scripts/eval_pathvqa_lora.sh
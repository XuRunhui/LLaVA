#!/bin/bash
#SBATCH --job-name=llava-vqarad
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          
#SBATCH --mem=32G                  
#SBATCH --time=01:00:00            
#SBATCH --output=logs/%x-%j.out    
#SBATCH --error=logs/%x-%j.err

# ======== 环境设置 ========
module purge
module load gcc/13.3.0
module load cuda/12.6.3            
export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

# 激活你的conda环境
source /home1/runhuixu/miniconda3/bin/activate llava

# ======== 进入项目路径 ========
cd /project2/ruishanl_1185/SDP_for_VLM/runhui/LLaVA/

# ======== 运行训练 ========
# bash scripts/RAD_VQA.sh
# bash scripts/SLAKE_lora.sh
bash scripts/PathVQA_lora.sh
bash scripts/eval_pathvqa_lora.sh
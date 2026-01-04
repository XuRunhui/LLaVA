#!/bin/bash

#SBATCH --job-name=eval_metrics_mimic
#SBATCH --output=/root/autodl-tmp/mimic_cxr_jpg/output/evaluation/logs/eval_metrics_%j.out
#SBATCH --error=/root/autodl-tmp/mimic_cxr_jpg/output/evaluation/logs/eval_metrics_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G

# Evaluation configuration
RESULTS_FILE=$1
OUTPUT_DIR=$2
BOOTSTRAP_CI=${3:-"true"}

# Default values if not provided
if [ -z "$RESULTS_FILE" ]; then
    echo "Error: RESULTS_FILE not provided"
    echo "Usage: $0 <results_file> <output_dir> [bootstrap_ci]"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: OUTPUT_DIR not provided"
    echo "Usage: $0 <results_file> <output_dir> [bootstrap_ci]"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p $(dirname $OUTPUT_DIR)/logs

echo "============================================"
echo "MIMIC-CXR Evaluation Metrics"
echo "============================================"
echo "Results file: $RESULTS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Bootstrap CI: $BOOTSTRAP_CI"
echo "============================================"

# Set up environment
export PYTHONPATH="${PYTHONPATH}:/root/LLaVA"
export CUDA_VISIBLE_DEVICES=0

# Scorers to compute
SCORERS="CheXbert F1-RadGraph BLEU-1 BLEU-4 ROUGE-L"

# Build command
CMD="python /root/LLaVA/llava/eval/evaluate_mimic_metrics.py \
    --results_file $RESULTS_FILE \
    --output_dir $OUTPUT_DIR \
    --scorers $SCORERS"

# Add bootstrap CI flag
if [ "$BOOTSTRAP_CI" = "true" ]; then
    CMD="$CMD --bootstrap_ci"
else
    CMD="$CMD --no_bootstrap_ci"
fi

echo "Running command:"
echo $CMD
echo "============================================"

# Run evaluation
eval $CMD

echo "============================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"

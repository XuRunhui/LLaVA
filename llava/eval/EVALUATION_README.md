# LLaVA LoRA Evaluation with Loss and Perplexity

This guide explains how to evaluate LoRA-finetuned LLaVA models with loss and perplexity computation.

## Overview

The evaluation script ([model_vqa_with_metrics.py](model_vqa_with_metrics.py)) provides:
- **Inference**: Generate predictions on your evaluation dataset
- **Loss Computation**: Calculate cross-entropy loss against ground truth
- **Perplexity**: Compute perplexity metric for model performance
- **Token-level Metrics**: Detailed statistics per token
- **Summary Statistics**: Aggregate metrics across the dataset

## Quick Start

### 1. Prepare Your Evaluation Data

Create a JSONL file with your evaluation questions. Each line should be a JSON object with:

```json
{
  "question_id": "unique_id_1",
  "image": "image1.jpg",
  "text": "What are the findings in this image?",
  "answer": "Ground truth answer here",
  "view": "PA",
  "reason": "Optional clinical indication"
}
```

**Required fields:**
- `question_id`: Unique identifier for the question
- `image`: Filename of the image (relative to IMAGE_FOLDER)
- `text`: The question text

**Optional fields:**
- `answer` or `ground_truth`: Required for loss/perplexity computation
- `view`: Image view (useful for medical imaging, e.g., "PA", "AP", "LATERAL")
- `reason`: Clinical indication or additional context

### 2. Configure the Shell Script

Edit [scripts/v1_5/eval/custom_eval_with_metrics.sh](../../scripts/v1_5/eval/custom_eval_with_metrics.sh):

```bash
# Model paths
MODEL_PATH="./checkpoints/llava-v1.5-13b-task-lora"  # Your LoRA checkpoint
MODEL_BASE="liuhaotian/llava-v1.5-13b"  # Base model

# Data paths
IMAGE_FOLDER="./playground/data/eval/custom/images"
QUESTION_FILE="./playground/data/eval/custom/questions.jsonl"
ANSWERS_FILE="./playground/data/eval/custom/answers_with_metrics.jsonl"

# Enable metrics computation
COMPUTE_METRICS=true  # Requires ground truth in question file
```

### 3. Run Evaluation

```bash
bash scripts/v1_5/eval/custom_eval_with_metrics.sh
```

## Output Files

### 1. Predictions File (`answers_with_metrics.jsonl`)

Each line contains:
```json
{
  "question_id": "unique_id_1",
  "prompt": "What are the findings in this image?",
  "text": "Model's generated answer",
  "answer_id": "random_uuid",
  "model_id": "llava-v1.5-13b",
  "ground_truth": "Ground truth answer",
  "metrics": {
    "loss": 1.234,
    "perplexity": 3.456,
    "num_tokens": 45,
    "avg_token_loss": 1.123,
    "max_token_loss": 2.345,
    "min_token_loss": 0.567
  },
  "view": "PA",
  "reason": "Clinical indication"
}
```

### 2. Summary File (`answers_with_metrics_summary.json`)

Aggregate statistics:
```json
{
  "total_samples": 100,
  "samples_with_metrics": 100,
  "loss": {
    "mean": 1.234,
    "median": 1.123,
    "std": 0.456,
    "min": 0.234,
    "max": 3.456
  },
  "perplexity": {
    "mean": 3.456,
    "median": 3.234,
    "std": 1.234,
    "min": 1.234,
    "max": 10.234
  },
  "config": {
    "model_path": "./checkpoints/llava-v1.5-13b-task-lora",
    "model_base": "liuhaotian/llava-v1.5-13b",
    "question_file": "./playground/data/eval/custom/questions.jsonl",
    "filter_views": false,
    "add_reason": false
  }
}
```

## Advanced Usage

### Command-line Arguments

```bash
python -m llava.eval.model_vqa_with_metrics \
    --model-path <path_to_lora_checkpoint> \
    --model-base <base_model_path> \
    --image-folder <image_directory> \
    --question-file <questions.jsonl> \
    --answers-file <output.jsonl> \
    --conv-mode llava_v1 \
    --temperature 0.0 \
    --compute-metrics \
    --filter-views \
    --add-reason
```

**Arguments:**
- `--model-path`: Path to your LoRA checkpoint or full model
- `--model-base`: Base model for LoRA (omit for full model)
- `--image-folder`: Directory containing evaluation images
- `--question-file`: JSONL file with questions
- `--answers-file`: Output file for predictions and metrics
- `--conv-mode`: Conversation template (llava_v1, vicuna_v1, etc.)
- `--temperature`: Sampling temperature (0 = greedy, >0 = sampling)
- `--top_p`: Nucleus sampling parameter
- `--num_beams`: Beam search width
- `--max_new_tokens`: Maximum tokens to generate
- `--compute-metrics`: Enable loss/perplexity computation
- `--filter-views`: Only evaluate PA/AP views
- `--add-reason`: Include reason field in prompts
- `--num-chunks`: Number of chunks for parallel processing
- `--chunk-idx`: Current chunk index

### Parallel Processing

For large datasets, split evaluation across multiple processes:

```bash
# Process 1 (chunk 0 of 4)
python -m llava.eval.model_vqa_with_metrics \
    --num-chunks 4 --chunk-idx 0 \
    --answers-file answers_chunk0.jsonl \
    ... other args ...

# Process 2 (chunk 1 of 4)
python -m llava.eval.model_vqa_with_metrics \
    --num-chunks 4 --chunk-idx 1 \
    --answers-file answers_chunk1.jsonl \
    ... other args ...

# Combine results
cat answers_chunk*.jsonl > answers_all.jsonl
```

### Without Ground Truth (Inference Only)

If you don't have ground truth answers, omit `--compute-metrics`:

```bash
bash scripts/v1_5/eval/custom_eval_with_metrics.sh
# Set COMPUTE_METRICS=false in the script
```

The script will still generate predictions, just without loss/perplexity metrics.

## Understanding the Metrics

### Loss
Cross-entropy loss measuring how well the model predicts the ground truth tokens. Lower is better.
- **Good**: < 1.0
- **Acceptable**: 1.0 - 2.0
- **Poor**: > 2.0

### Perplexity
Exponentiated loss, measuring model uncertainty. Lower is better.
- **Good**: < 5.0
- **Acceptable**: 5.0 - 10.0
- **Poor**: > 10.0

### Token-level Metrics
- `num_tokens`: Number of tokens in ground truth answer
- `avg_token_loss`: Average loss per token
- `max_token_loss`: Maximum loss for any token
- `min_token_loss`: Minimum loss for any token

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size or use gradient checkpointing
- Use smaller image resolution
- Process fewer samples per chunk with `--num-chunks`

### Missing Ground Truth
- Ensure question file has `answer` or `ground_truth` field
- Or disable metrics with `COMPUTE_METRICS=false`

### LoRA Weights Not Loading
- Verify `--model-path` points to your LoRA checkpoint directory
- Verify `--model-base` points to the base model used during training
- Check that the checkpoint contains `adapter_config.json` and `adapter_model.bin`

### Incorrect Metrics
- Verify ground truth answers are correct and complete
- Check conversation template matches training (`--conv-mode`)
- Ensure images are in the correct folder

## Example Workflow

### 1. After LoRA Finetuning

```bash
# Your training outputs a checkpoint
./checkpoints/llava-v1.5-13b-medical-lora/
├── adapter_config.json
├── adapter_model.bin
└── non_lora_trainables.bin
```

### 2. Prepare Evaluation Data

```bash
./playground/data/eval/medical/
├── images/
│   ├── patient001.jpg
│   ├── patient002.jpg
│   └── ...
└── questions.jsonl  # Your evaluation questions with ground truth
```

### 3. Configure and Run

```bash
# Edit the script
vim scripts/v1_5/eval/custom_eval_with_metrics.sh

# Update paths:
# MODEL_PATH="./checkpoints/llava-v1.5-13b-medical-lora"
# MODEL_BASE="liuhaotian/llava-v1.5-13b"
# IMAGE_FOLDER="./playground/data/eval/medical/images"
# QUESTION_FILE="./playground/data/eval/medical/questions.jsonl"
# COMPUTE_METRICS=true

# Run evaluation
bash scripts/v1_5/eval/custom_eval_with_metrics.sh
```

### 4. Analyze Results

```bash
# View summary
cat ./playground/data/eval/medical/answers_with_metrics_summary.json

# Check individual predictions
head -n 5 ./playground/data/eval/medical/answers_with_metrics.jsonl
```

## Integration with Training

You can run evaluation automatically after training by adding to your training script:

```bash
#!/bin/bash

# 1. Run training
bash scripts/v1_5/finetune_lora.sh

# 2. Run evaluation
bash scripts/v1_5/eval/custom_eval_with_metrics.sh

# 3. Compare with baseline (optional)
python scripts/compare_metrics.py \
    --baseline ./eval/baseline_summary.json \
    --current ./eval/current_summary.json
```

## Citation

If you use this evaluation script in your research, please cite the original LLaVA paper:

```bibtex
@misc{liu2023llava,
    title={Visual Instruction Tuning},
    author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
    publisher={arXiv:2304.08485},
    year={2023},
}
```

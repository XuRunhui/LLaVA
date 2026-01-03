# MIMIC-CXR Evaluation Guide

This guide explains how to evaluate your trained LLaVA model on MIMIC-CXR dev and test sets with loss and perplexity computation.

## Overview

The evaluation script provides:
- **Loss computation**: Cross-entropy loss on ground truth findings
- **Perplexity calculation**: exp(loss) as a measure of model confidence
- **Inference/Generation**: Model predictions for qualitative analysis
- **Automatic filtering**: Uses only rule-based samples for test evaluation
- **Per-sample metrics**: Detailed results for each image
- **Aggregate statistics**: Overall performance summary

## Quick Start

### 1. Update Model Path

Edit `scripts/v1_5/eval_mimic.sh`:

```bash
MODEL_PATH="/path/to/your/checkpoint-XXX"
MODEL_BASE="liuhaotian/llava-v1.5-7b"  # Base model for LoRA
```

### 2. Run Evaluation

```bash
cd /path/to/LLaVA
chmod +x scripts/v1_5/eval_mimic.sh

# Run directly (local)
bash scripts/v1_5/eval_mimic.sh

# Or submit to SLURM
sbatch scripts/v1_5/eval_mimic.sh
```

### 3. Check Results

Results are saved to: `outputs/llava_llavarad/lora_128_dp_e8/eval_results_checkpoint-XXX/`

```bash
# View summary
cat eval_results_checkpoint-XXX/dev_results_summary.json
cat eval_results_checkpoint-XXX/test_results_summary.json

# View per-sample results
head -n 5 eval_results_checkpoint-XXX/dev_results.jsonl
```

## Output Format

### Per-Sample Results (JSONL)

Each line in `dev_results.jsonl` / `test_results.jsonl`:

```json
{
  "id": "p10046166_s50051329",
  "image": "p10/p10046166/s50051329/427446c1.jpg",
  "question": "Provide a description of the findings in the radiology image given the following indication: chest pain",
  "ground_truth": "Lateral view somewhat limited due to overlying motion artifact. The lungs are low in volume...",
  "prediction": "The lungs are clear without focal consolidation, pleural effusion, or pneumothorax...",
  "loss": 2.345,
  "perplexity": 10.432,
  "num_tokens": 42,
  "view": "LATERAL",
  "generate_method": "rule-based",
  "answer_id": "abc123xyz",
  "model_id": "llava-v1.5-7b-lora"
}
```

### Summary Statistics (JSON)

File: `dev_results_summary.json` / `test_results_summary.json`

```json
{
  "split": "test",
  "model_path": "/path/to/checkpoint-XXX",
  "model_name": "llava-v1.5-7b-lora",
  "num_samples": 1234,
  "total_tokens": 52340,
  "average_loss": 2.456,
  "average_perplexity": 11.678,
  "generation_config": {
    "temperature": 0.0,
    "top_p": null,
    "num_beams": 1,
    "max_new_tokens": 512
  },
  "data_config": {
    "filter_views": true,
    "include_reason": true,
    "generation_methods": "rule-based"
  }
}
```

## Configuration Options

### Data Filtering

```bash
# In eval_mimic.sh
FILTER_VIEWS=True         # Filter to only PA/AP views
INCLUDE_REASON=True       # Include clinical indication in prompts
```

**Note**: `generation_methods` is hardcoded to `"rule-based"` in the evaluation script for test data consistency.

### Generation Settings

```bash
TEMPERATURE=0.0           # 0 = greedy (deterministic), >0 = sampling
NUM_BEAMS=1               # Beam search width (1 = greedy)
MAX_NEW_TOKENS=512        # Maximum generated sequence length
```

**Recommended settings**:
- **For reproducibility**: `TEMPERATURE=0.0`, `NUM_BEAMS=1` (greedy decoding)
- **For diversity**: `TEMPERATURE=0.7`, `NUM_BEAMS=3`

## Understanding Metrics

### Loss

- **Cross-entropy loss** on ground truth findings
- Lower is better (model is more confident about ground truth)
- Typical range: 1.0-5.0 for well-trained models

### Perplexity

- **Perplexity = exp(loss)**
- Measures how "surprised" the model is by the ground truth
- Lower is better (model assigns higher probability to ground truth)
- Typical range: 2.7-150 (corresponding to loss 1.0-5.0)

### Token Count

- Number of tokens in the ground truth answer (excluding question)
- Used for weighted averaging of loss/perplexity

## Advanced Usage

### Evaluate on Custom Checkpoint

```bash
python llava/eval/eval_mimic_cxr.py \
    --model-path /path/to/checkpoint \
    --model-base liuhaotian/llava-v1.5-7b \
    --data-file /path/to/data.json \
    --image-folder /path/to/images \
    --output-file results.jsonl \
    --split test \
    --filter-views True \
    --include-reason True \
    --temperature 0.0 \
    --num-beams 1 \
    --max-new-tokens 512 \
    --conv-mode v1
```

### Distributed Evaluation (Multi-GPU)

Split evaluation across multiple GPUs:

```bash
# GPU 0: Process chunk 0 of 4
python llava/eval/eval_mimic_cxr.py \
    --model-path /path/to/checkpoint \
    --data-file data.json \
    --output-file results_chunk0.jsonl \
    --num-chunks 4 \
    --chunk-idx 0 \
    # ... other args ...

# GPU 1: Process chunk 1 of 4
python llava/eval/eval_mimic_cxr.py \
    --model-path /path/to/checkpoint \
    --data-file data.json \
    --output-file results_chunk1.jsonl \
    --num-chunks 4 \
    --chunk-idx 1 \
    # ... other args ...

# Merge results
cat results_chunk*.jsonl > results_full.jsonl
```

### Evaluate Only Dev Set

Comment out the test set section in `eval_mimic.sh`:

```bash
# # ====================================
# # Evaluate on Test Set
# # ====================================
# echo "Evaluating on TEST set..."
# # ... (comment out the test evaluation)
```

## Analyzing Results

### Compute Additional Metrics

After evaluation, you can compute BLEU, ROUGE, etc.:

```python
import json
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Load results
with open('test_results.jsonl') as f:
    results = [json.loads(line) for line in f]

# Compute BLEU
bleu_scores = []
for result in results:
    reference = [result['ground_truth'].split()]
    hypothesis = result['prediction'].split()
    bleu = sentence_bleu(reference, hypothesis)
    bleu_scores.append(bleu)

avg_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU: {avg_bleu:.4f}")

# Compute ROUGE
rouge = Rouge()
rouge_scores = rouge.get_scores(
    [r['prediction'] for r in results],
    [r['ground_truth'] for r in results],
    avg=True
)
print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
```

### Filter by View Type

```python
# Load results
with open('test_results.jsonl') as f:
    results = [json.loads(line) for line in f]

# Filter by view
pa_results = [r for r in results if r['view'] == 'PA']
ap_results = [r for r in results if r['view'] == 'AP']

# Compute per-view metrics
pa_loss = sum(r['loss'] * r['num_tokens'] for r in pa_results) / sum(r['num_tokens'] for r in pa_results)
ap_loss = sum(r['loss'] * r['num_tokens'] for r in ap_results) / sum(r['num_tokens'] for r in ap_results)

print(f"PA view average loss: {pa_loss:.4f}")
print(f"AP view average loss: {ap_loss:.4f}")
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Use smaller batch size (already set to 1)
**Solution 2**: Reduce `max_new_tokens`:

```bash
MAX_NEW_TOKENS=256  # Reduce from 512
```

**Solution 3**: Use quantization:

```python
# In eval_mimic_cxr.py, modify load_pretrained_model call
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    args.model_base,
    model_name,
    load_8bit=True,  # Add this
    device_map="auto"
)
```

### Issue: Model not loading LoRA weights

**Check**:
1. Verify `MODEL_PATH` points to the correct LoRA checkpoint directory
2. Ensure `MODEL_BASE` is set correctly to the base model
3. Check that adapter files exist: `ls $MODEL_PATH/adapter_*`

### Issue: Evaluation is slow

**Solutions**:
1. Use distributed evaluation with `--num-chunks` and `--chunk-idx`
2. Reduce `max_new_tokens` for faster generation
3. Use greedy decoding (`temperature=0, num_beams=1`) instead of beam search

### Issue: Loss is NaN or inf

**Possible causes**:
1. Model not properly loaded (check logs)
2. Data preprocessing issue (check sample in results.jsonl)
3. Mixed precision errors

**Debug**:
```python
# Add to compute_loss_and_perplexity function
print(f"Input IDs shape: {input_ids.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Valid tokens: {(labels != IGNORE_INDEX).sum()}")
```

## Files Created

After running evaluation:

```
eval_results_checkpoint-XXX/
├── dev_results.jsonl          # Per-sample results for dev set
├── dev_results_summary.json   # Aggregate metrics for dev set
├── test_results.jsonl         # Per-sample results for test set
└── test_results_summary.json  # Aggregate metrics for test set
```

## Next Steps

After evaluation, you can:

1. **Compare models**: Evaluate multiple checkpoints and compare metrics
2. **Analyze errors**: Find samples with high loss/perplexity
3. **Compute NLP metrics**: BLEU, ROUGE, CIDEr, etc.
4. **Compute radiology metrics**: F1-RadGraph, CheXbert, etc.
5. **Generate reports**: Create tables/plots for your paper

## Reference

- Evaluation script: [`llava/eval/eval_mimic_cxr.py`](../llava/eval/eval_mimic_cxr.py)
- Data utilities: [`llava/eval/mimic_data_utils.py`](../llava/eval/mimic_data_utils.py)
- Shell script: [`scripts/v1_5/eval_mimic.sh`](../scripts/v1_5/eval_mimic.sh)

# Quick Start: Evaluate LoRA Finetuned LLaVA with Loss & Perplexity

This is a quick reference guide for evaluating your LoRA-finetuned LLaVA models with loss and perplexity metrics.

## 📋 Prerequisites

- LoRA finetuned checkpoint from training
- Base model (the model you finetuned from)
- Evaluation images
- Evaluation questions with ground truth answers

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Your Evaluation Data

Create a JSONL file with your questions:

```jsonl
{"question_id": "q1", "image": "img1.jpg", "text": "What findings are present?", "answer": "Ground truth answer here"}
{"question_id": "q2", "image": "img2.jpg", "text": "Describe the abnormalities.", "answer": "Another ground truth answer"}
```

**Or use the helper script:**

```bash
# Create a template
python llava/eval/prepare_eval_data.py --create-template --output my_questions.jsonl

# Validate your data
python llava/eval/prepare_eval_data.py \
    --validate \
    --question-file my_questions.jsonl \
    --image-folder ./path/to/images
```

### Step 2: Configure the Evaluation Script

Edit `scripts/v1_5/eval/custom_eval_with_metrics.sh`:

```bash
# Update these paths
MODEL_PATH="./checkpoints/your-lora-checkpoint"
MODEL_BASE="liuhaotian/llava-v1.5-13b"  # Or your base model
IMAGE_FOLDER="./path/to/eval/images"
QUESTION_FILE="./path/to/questions.jsonl"
ANSWERS_FILE="./path/to/output.jsonl"

# Enable metrics
COMPUTE_METRICS=true  # Set to false if no ground truth
```

### Step 3: Run Evaluation

```bash
bash scripts/v1_5/eval/custom_eval_with_metrics.sh
```

## 📊 Output

You'll get two files:

### 1. Predictions with Metrics (`output.jsonl`)
```json
{
  "question_id": "q1",
  "text": "Generated answer",
  "ground_truth": "Ground truth answer",
  "metrics": {
    "loss": 1.234,
    "perplexity": 3.456,
    "num_tokens": 45
  }
}
```

### 2. Summary Statistics (`output_summary.json`)
```json
{
  "total_samples": 100,
  "loss": {
    "mean": 1.234,
    "median": 1.123,
    "std": 0.456
  },
  "perplexity": {
    "mean": 3.456,
    "median": 3.234
  }
}
```

## 📈 Understanding Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| **Loss** | < 1.0 | 1.0 - 2.0 | > 2.0 |
| **Perplexity** | < 5.0 | 5.0 - 10.0 | > 10.0 |

Lower is better for both metrics.

## 🔧 Common Issues

### "No ground truth found"
- Add `"answer"` or `"ground_truth"` field to your questions
- Or set `COMPUTE_METRICS=false` to skip metrics

### "LoRA weights not loading"
- Verify `MODEL_PATH` points to your LoRA checkpoint
- Verify `MODEL_BASE` matches the base model from training
- Check for `adapter_config.json` in checkpoint folder

### "Out of memory"
- Use `--num-chunks` for parallel processing
- Reduce image resolution
- Use a smaller base model

## 📁 File Structure

```
LLaVA/
├── checkpoints/
│   └── your-lora-model/          # Your LoRA checkpoint
│       ├── adapter_config.json
│       ├── adapter_model.bin
│       └── non_lora_trainables.bin
├── playground/data/eval/
│   └── your-eval/
│       ├── images/                # Evaluation images
│       │   ├── img1.jpg
│       │   └── img2.jpg
│       └── questions.jsonl        # Questions with ground truth
├── llava/eval/
│   ├── model_vqa_with_metrics.py  # Main evaluation script
│   ├── prepare_eval_data.py       # Data preparation helper
│   └── EVALUATION_README.md       # Detailed documentation
└── scripts/v1_5/eval/
    └── custom_eval_with_metrics.sh  # Shell script to run eval
```

## 📖 Advanced Usage

### Command-line Only (without shell script)

```bash
python -m llava.eval.model_vqa_with_metrics \
    --model-path ./checkpoints/your-lora \
    --model-base liuhaotian/llava-v1.5-13b \
    --image-folder ./images \
    --question-file ./questions.jsonl \
    --answers-file ./output.jsonl \
    --compute-metrics \
    --temperature 0.0
```

### Parallel Processing (for large datasets)

```bash
# Split into 4 chunks, process in parallel
for i in {0..3}; do
    python -m llava.eval.model_vqa_with_metrics \
        --num-chunks 4 --chunk-idx $i \
        --answers-file output_chunk${i}.jsonl \
        ... other args ... &
done
wait

# Combine results
cat output_chunk*.jsonl > output_all.jsonl
```

## 💡 Tips

1. **Always validate your data first** using `prepare_eval_data.py`
2. **Use greedy decoding** (`temperature=0`) for reproducible results
3. **Keep track of your metrics** over different checkpoints to monitor progress
4. **Filter views** if evaluating medical images with specific orientations
5. **Check summary file** for aggregate statistics across all samples

## 📚 Full Documentation

For more details, see:
- [llava/eval/EVALUATION_README.md](llava/eval/EVALUATION_README.md) - Comprehensive guide
- [llava/eval/model_vqa_with_metrics.py](llava/eval/model_vqa_with_metrics.py) - Source code
- [scripts/v1_5/eval/custom_eval_with_metrics.sh](scripts/v1_5/eval/custom_eval_with_metrics.sh) - Shell script

## 🆘 Need Help?

1. Check your data format with validation script
2. Verify all file paths exist
3. Check that base model matches training configuration
4. Review the detailed error messages in terminal output

## Example Workflow

```bash
# 1. Validate your evaluation data
python llava/eval/prepare_eval_data.py \
    --validate \
    --question-file ./eval/questions.jsonl \
    --image-folder ./eval/images

# 2. Edit configuration
vim scripts/v1_5/eval/custom_eval_with_metrics.sh

# 3. Run evaluation
bash scripts/v1_5/eval/custom_eval_with_metrics.sh

# 4. Check results
cat ./eval/output_summary.json
```

That's it! You're ready to evaluate your LoRA-finetuned LLaVA models with loss and perplexity metrics.

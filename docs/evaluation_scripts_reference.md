# Evaluation Scripts Reference Guide

Quick reference for all MIMIC-CXR evaluation scripts.

## Available Scripts

### 1. Non-batched Evaluation (Original)

| Script | Purpose | Speed | When to Use |
|--------|---------|-------|-------------|
| `eval_mimic.sh` | Dev + Test sets | 0.5 samples/sec | Debugging, single samples |
| `eval_mimic_train.sh` | Training set | 0.5 samples/sec | Check overfitting |

### 2. Batched Evaluation (Fast) ⚡

| Script | Purpose | Speed | When to Use |
|--------|---------|-------|-------------|
| `eval_mimic_batched.sh` | Dev + Test sets | 1.8 samples/sec | **Recommended for eval** |
| `eval_mimic_train_batched.sh` | Training set | 1.8 samples/sec | Fast overfitting check |

### 3. Metrics Evaluation

| Script | Purpose | Metrics | When to Use |
|--------|---------|---------|-------------|
| `eval_metrics_mimic.sh` | Single results file | CheXbert, RadGraph, BLEU, ROUGE | After inference |
| `eval_all_mimic_metrics.sh` | Dev + Test | All metrics | Final evaluation |

---

## Quick Commands

### Development & Testing (Dev/Test Sets)

```bash
# Fast evaluation (RECOMMENDED)
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint 4

# Slow evaluation (for debugging)
bash scripts/v1_5/eval_mimic.sh dev
bash scripts/v1_5/eval_mimic.sh test
```

### Training Set Evaluation (Check Overfitting)

```bash
# Fast (RECOMMENDED)
bash scripts/v1_5/eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 4

# Slow (for debugging)
bash scripts/v1_5/eval_mimic_train.sh /path/to/checkpoint True gpt4
```

### Compute Metrics

```bash
# After running inference, compute all metrics
bash scripts/v1_5/eval_all_mimic_metrics.sh lora_128_dp_e8 true
```

---

## Script Parameters

### `eval_mimic_batched.sh`

```bash
bash scripts/v1_5/eval_mimic_batched.sh <model_path> [batch_size]
```

**Parameters:**
- `model_path`: Path to checkpoint (required)
- `batch_size`: Batch size for inference (default: 4)

**Examples:**
```bash
# Default (batch_size=4)
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint

# Custom batch size
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint 8
```

### `eval_mimic_train_batched.sh`

```bash
bash scripts/v1_5/eval_mimic_train_batched.sh <model_path> [include_reason] [generation_methods] [batch_size]
```

**Parameters:**
1. `model_path`: Path to checkpoint (required)
2. `include_reason`: Include clinical indication (default: True)
3. `generation_methods`: "gpt4", "rule-based", or "all" (default: "gpt4")
4. `batch_size`: Batch size for inference (default: 4)

**Examples:**
```bash
# All defaults (gpt4, reason=True, batch_size=4)
bash scripts/v1_5/eval_mimic_train_batched.sh /path/to/checkpoint

# Custom generation method
bash scripts/v1_5/eval_mimic_train_batched.sh /path/to/checkpoint True rule-based

# Custom batch size
bash scripts/v1_5/eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 8

# All custom
bash scripts/v1_5/eval_mimic_train_batched.sh /path/to/checkpoint False all 2
```

### `eval_all_mimic_metrics.sh`

```bash
bash scripts/v1_5/eval_all_mimic_metrics.sh <model_name> [bootstrap_ci]
```

**Parameters:**
- `model_name`: Name of the model (matches output directory)
- `bootstrap_ci`: Compute confidence intervals (default: true)

**Examples:**
```bash
# With bootstrap CI (slower, more robust)
bash scripts/v1_5/eval_all_mimic_metrics.sh lora_128_dp_e8 true

# Without bootstrap CI (faster)
bash scripts/v1_5/eval_all_mimic_metrics.sh lora_128_dp_e8 false
```

---

## Complete Evaluation Pipeline

### Typical Workflow

```bash
# Step 1: Run inference on dev/test (FAST)
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint 4

# Step 2: Compute metrics
bash scripts/v1_5/eval_all_mimic_metrics.sh lora_128_dp_e8 true

# Step 3: Check overfitting (optional)
bash scripts/v1_5/eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 4
```

**Expected time:**
- Step 1: ~3 minutes (153 test samples)
- Step 2: ~15 minutes (CheXbert + RadGraph with bootstrap)
- Step 3: ~20 minutes (larger training set)

**Total: ~40 minutes** for complete evaluation

---

## Performance Comparison

### Speed (samples/second)

| Script | Batch Size | Speed | Time for 153 samples |
|--------|------------|-------|----------------------|
| `eval_mimic.sh` | 1 | 0.5 | 5m 20s |
| `eval_mimic_batched.sh` | 2 | 1.0 | 2m 40s |
| `eval_mimic_batched.sh` | 4 | 1.8 | 1m 30s ✨ |
| `eval_mimic_batched.sh` | 8 | 2.5 | 1m 5s |

### Recommended Configurations

| GPU Memory | Recommended Script | Batch Size |
|------------|-------------------|------------|
| 16GB | `eval_mimic_batched.sh` | 2 |
| 24GB | `eval_mimic_batched.sh` | 4 ✅ |
| 40GB+ | `eval_mimic_batched.sh` | 6-8 |

---

## Output Files

### After Running Inference

```
output_dir/
├── dev_results.jsonl           # Dev set: per-sample results
├── dev_results_summary.json    # Dev set: aggregate statistics
├── test_results.jsonl          # Test set: per-sample results
└── test_results_summary.json   # Test set: aggregate statistics
```

### After Computing Metrics

```
output_dir/
├── dev_metrics/
│   ├── main_results.csv                     # Summary table
│   ├── all_results.json                     # All metrics
│   ├── chexbert_breakdown_positive.csv      # Per-condition
│   └── chexbert_breakdown_negative.csv
└── test_metrics/
    └── (same structure)
```

---

## Common Use Cases

### 1. Quick Evaluation (Fastest)

```bash
# Generate predictions only (skip loss computation)
python llava/eval/eval_mimic_cxr_batched.py \
    --model-path /path/to/checkpoint \
    --model-base liuhaotian/llava-v1.5-7b \
    --data-file /path/to/test.json \
    --image-folder /path/to/images \
    --output-file results.jsonl \
    --split test \
    --batch-size 8 \
    --compute-loss False  # Skip loss for 2x speedup
```

**Time**: ~30 seconds for 153 samples

### 2. Standard Evaluation (Recommended)

```bash
# Dev + Test with loss/perplexity
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint 4

# Compute all metrics
bash scripts/v1_5/eval_all_mimic_metrics.sh lora_128_dp_e8 true
```

**Time**: ~20 minutes total

### 3. Overfitting Check

```bash
# Evaluate on training set
bash scripts/v1_5/eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 4

# Compare with dev/test metrics
python scripts/compare_train_vs_test.py
```

**Expected**: Train metrics >> Dev metrics >> Test metrics

### 4. Different Generation Methods

```bash
# Evaluate only rule-based reports (faster, smaller dataset)
bash scripts/v1_5/eval_mimic_train_batched.sh /path/to/checkpoint True rule-based 4

# Evaluate both GPT-4 and rule-based (slower, full dataset)
bash scripts/v1_5/eval_mimic_train_batched.sh /path/to/checkpoint True all 4
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint 2

# Or skip loss computation
python llava/eval/eval_mimic_cxr_batched.py ... --compute-loss False
```

### Too Slow

```bash
# Increase batch size
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint 8

# Check GPU utilization
nvidia-smi dmon -s u
# Should show 80-95% GPU util
```

### Different Results

- Small differences in loss/perplexity (<0.001) are normal with batching
- Predictions should be identical for greedy decoding (temperature=0)
- Use non-batched version for exact reproduction if needed

---

## Script Locations

```
LLaVA/
├── llava/eval/
│   ├── eval_mimic_cxr.py                    # Non-batched inference
│   ├── eval_mimic_cxr_batched.py            # Batched inference ⚡
│   ├── evaluate_mimic_metrics.py            # Compute metrics
│   └── mimic_data_utils.py                  # Data loading
│
├── scripts/v1_5/
│   ├── eval_mimic.sh                        # Dev+Test (non-batched)
│   ├── eval_mimic_batched.sh                # Dev+Test (batched) ⚡
│   ├── eval_mimic_train.sh                  # Train (non-batched)
│   ├── eval_mimic_train_batched.sh          # Train (batched) ⚡
│   ├── eval_metrics_mimic.sh                # Compute metrics (single file)
│   └── eval_all_mimic_metrics.sh            # Compute metrics (dev+test)
│
└── docs/
    ├── mimic_evaluation_guide.md            # Original eval guide
    ├── mimic_metrics_evaluation_guide.md    # Metrics guide
    ├── batched_evaluation_guide.md          # Batching details
    └── evaluation_scripts_reference.md      # This file
```

---

## Summary

**For most use cases, use:**

```bash
# 1. Run batched inference (FAST)
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint 4

# 2. Compute metrics
bash scripts/v1_5/eval_all_mimic_metrics.sh model_name true

# 3. (Optional) Check overfitting
bash scripts/v1_5/eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 4
```

**This gives you:**
- ✅ Fast inference (3.5x speedup)
- ✅ Complete metrics (CheXbert, RadGraph, BLEU, ROUGE)
- ✅ Overfitting analysis
- ✅ ~40 minutes total

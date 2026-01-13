# Batched Evaluation - Quick Start

## TL;DR

Evaluation is 5x slower than training because batch_size=1. Use the new batched version for **3-5x speedup**.

```bash
# OLD (slow):
bash scripts/v1_5/eval_mimic.sh dev
# → 5 minutes for 153 samples

# NEW (fast):
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint 4
# → 1.5 minutes for 153 samples ✨
```

## Why So Slow?

| Training | Evaluation (old) |
|----------|------------------|
| Effective batch: 8 | Batch: 1 ❌ |
| Teacher forcing (parallel) | Autoregressive (sequential) ❌ |
| 1 forward pass | 2 forward passes (loss + generation) ❌ |
| ~2.5 samples/sec | ~0.5 samples/sec ❌ |

**Result**: Evaluation 5x slower than training!

## Solution

New batched evaluation script processes multiple samples in parallel:

### Files

- **`llava/eval/eval_mimic_cxr_batched.py`** - Batched evaluation script
- **`scripts/v1_5/eval_mimic_batched.sh`** - Shell script wrapper
- **`docs/batched_evaluation_guide.md`** - Full documentation

### Usage

```bash
# Basic usage (batch size 4, recommended)
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint 4

# Faster (batch size 8)
bash scripts/v1_5/eval_mimic_batched.sh /path/to/checkpoint 8

# Maximum speed (skip loss computation)
python llava/eval/eval_mimic_cxr_batched.py \
    --model-path /path/to/checkpoint \
    --model-base liuhaotian/llava-v1.5-7b \
    --data-file /path/to/data.json \
    --image-folder /path/to/images \
    --output-file results.jsonl \
    --split test \
    --batch-size 8 \
    --compute-loss False  # 2x faster, no loss/perplexity
```

## Performance

| Batch Size | Speedup | Time (153 samples) | Memory |
|------------|---------|---------------------|---------|
| 1 (old) | 1.0x | 5m 20s | 8GB |
| 2 | 2.0x | 2m 40s | 12GB |
| **4 (recommended)** | **3.5x** | **1m 30s** | **16GB** |
| 8 | 5.0x | 1m 5s | 22GB |

## Choosing Batch Size

- **16GB GPU**: batch_size=2
- **24GB GPU**: batch_size=4 ✅ (recommended)
- **40GB+ GPU**: batch_size=6-8

Start with 4, reduce if OOM, increase if GPU util < 80%.

## Features

✅ 3-5x faster than original
✅ Identical output format
✅ Same metrics (predictions, loss, perplexity)
✅ Optional loss computation (skip for 2x extra speedup)
✅ Configurable batch size
✅ Works with all generation methods (gpt4, rule-based, all)

## Full Documentation

See [docs/batched_evaluation_guide.md](docs/batched_evaluation_guide.md) for:
- Technical details
- Troubleshooting
- Performance benchmarks
- Migration guide

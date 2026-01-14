# Resume Evaluation Feature Guide

## Overview

The batched evaluation script now supports **resuming from interrupted evaluations**. If your evaluation job crashes, times out, or is interrupted, you can continue from where you left off without re-evaluating already processed samples.

## How It Works

### 1. Automatic Progress Tracking

The script automatically saves results incrementally to the output file:
- Each sample's results are written immediately after processing
- Results are flushed to disk to prevent data loss
- The output file tracks which samples have been processed

### 2. Resume Mode

When you enable resume mode (`--resume True`), the script will:
1. Load the existing output file
2. Identify which samples have already been processed (by `study_id` and `image_id`)
3. Skip those samples and only process remaining ones
4. Append new results to the existing file
5. Compute final statistics across all samples (old + new)

### 3. Safety Features

- **Warning on overwrite**: If you have existing results but don't enable resume mode, you'll get a warning before the file is overwritten
- **Unique sample identification**: Uses combination of `study_id` and `image_id` to ensure correct matching
- **Statistics preservation**: Existing statistics (loss, perplexity, token counts) are preserved and included in final summary

## Usage

### Basic Usage

```bash
# Start a new evaluation (will overwrite existing results)
bash scripts/v1_5/eval_mimic_train_batched.sh \
    /path/to/checkpoint \
    True \
    gpt4 \
    4 \
    False

# Resume from existing results (append mode)
bash scripts/v1_5/eval_mimic_train_batched.sh \
    /path/to/checkpoint \
    True \
    gpt4 \
    4 \
    True
```

### Direct Python Usage

```bash
# Start new evaluation
python llava/eval/eval_mimic_cxr_batched.py \
    --model-path /path/to/checkpoint \
    --model-base liuhaotian/llava-v1.5-7b \
    --data-file /path/to/data.json \
    --image-folder /path/to/images \
    --output-file results.jsonl \
    --batch-size 4 \
    --resume False

# Resume evaluation
python llava/eval/eval_mimic_cxr_batched.py \
    --model-path /path/to/checkpoint \
    --model-base liuhaotian/llava-v1.5-7b \
    --data-file /path/to/data.json \
    --image-folder /path/to/images \
    --output-file results.jsonl \
    --batch-size 4 \
    --resume True
```

## Example Scenarios

### Scenario 1: Job Timeout

Your SLURM job times out after processing 400 out of 1000 samples:

```bash
# Initial run (timed out)
bash eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 4 False
# Processed 400 samples, saved to train_results.jsonl

# Resume to process remaining 600 samples
bash eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 4 True
# Loads existing 400 results
# Processes remaining 600 samples
# Final file has all 1000 results
```

### Scenario 2: Out of Memory Error

Your job crashes due to OOM after 200 samples:

```bash
# Initial run (crashed)
bash eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 8 False
# Crashed after 200 samples with batch_size=8

# Resume with smaller batch size
bash eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 2 True
# Uses smaller batch_size=2 to avoid OOM
# Continues from sample 201
```

### Scenario 3: Multiple Interrupted Runs

You can resume multiple times:

```bash
# Run 1: Process 300 samples, then timeout
bash eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 4 False

# Run 2: Process 300 more (total 600), then crash
bash eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 4 True

# Run 3: Complete remaining 400 samples
bash eval_mimic_train_batched.sh /path/to/checkpoint True gpt4 4 True
```

## Output Format

### Resume Mode Output

When resuming, you'll see:

```
================================================================================
RESUMING EVALUATION
Found 400 already processed samples in /path/to/train_results.jsonl
Existing stats:
  - Average loss: 0.8234
  - Average perplexity: 2.3456
  - Total tokens: 123456
================================================================================

Loading MIMIC-CXR train data...
Skipping 400 already processed samples
Remaining samples to process: 600

Starting batched evaluation on 600 samples...
Batch size: 4
Compute loss: True
Results will be saved to: /path/to/train_results.jsonl
Resume mode: APPEND (starting from sample 401)
================================================================================
```

### Results File

The output JSONL file contains one result per line:

```jsonl
{"study_id": "s12345678", "image_id": "i98765432.dcm", "prediction": "...", "loss": 0.823, "perplexity": 2.28, "valid_tokens": 145}
{"study_id": "s12345679", "image_id": "i98765433.dcm", "prediction": "...", "loss": 0.756, "perplexity": 2.13, "valid_tokens": 152}
...
```

When resuming, new results are appended:
- Lines 1-400: Original results from first run
- Lines 401-1000: New results from resumed run

## Important Notes

### ⚠️ Compatibility Requirements

When resuming, make sure:
1. **Same data file**: Use the same `--data-file` path
2. **Same filtering options**: Keep `--filter-views`, `--include-reason`, `--generation-methods` the same
3. **Same model**: Use the same checkpoint (you can change batch size though)

### ✅ What You CAN Change When Resuming

- `--batch-size`: You can use a different batch size (e.g., reduce if you hit OOM)
- `--num-workers`: Adjust dataloader workers

### ❌ What You CANNOT Change When Resuming

- Data filtering options (`--filter-views`, `--include-reason`, `--generation-methods`)
- Split (`--split`)
- Data file path
- Model checkpoint (unless you want to compare different models on same samples)

### 🔍 Checking Progress

To check how many samples have been processed:

```bash
# Count lines in output file
wc -l /path/to/train_results.jsonl

# View last few results
tail -n 5 /path/to/train_results.jsonl

# Check summary statistics
cat /path/to/train_results_summary.json
```

## Technical Details

### Sample Identification

Samples are identified by the combination of `study_id` and `image_id`:

```python
unique_id = f"{study_id}_{image_id}"
```

This ensures that the correct samples are skipped even if the data ordering changes.

### File Writing

The script uses append mode when resuming:

```python
file_mode = 'a' if args.resume else 'w'
with open(output_file, file_mode) as f:
    # Write results...
    f.flush()  # Ensure immediate write to disk
```

### Statistics Aggregation

When resuming:
1. Load existing results and compute their statistics
2. Initialize counters with existing stats
3. Add new samples' statistics incrementally
4. Final summary includes both old and new samples

## Troubleshooting

### "All samples already processed!"

This means all samples in the dataset have already been evaluated. Either:
- Your evaluation is complete (check the summary file)
- You're using the wrong output file path
- The data filtering resulted in the same samples being identified as processed

### Duplicate Results

If you accidentally run without `--resume True`, the file will be overwritten. Always:
1. Check if results file exists: `ls /path/to/results.jsonl`
2. If it exists and you want to continue, use `--resume True`
3. If you want to start fresh, either delete the file or use a new output path

### Statistics Don't Match

If final statistics seem wrong, make sure you used consistent parameters across all runs. Mixing different filtering options will cause incorrect statistics.

## Best Practices

1. **Use descriptive output filenames** that include checkpoint name and configuration
2. **Enable resume by default** when re-running evaluations on clusters with time limits
3. **Monitor progress** by checking file line counts periodically
4. **Save checkpoint info** in the output directory for reproducibility
5. **Use version control** to track which script version produced which results

## See Also

- [Batched Evaluation Guide](batched_evaluation_guide.md) - Overview of batched evaluation
- [Evaluation Scripts Reference](evaluation_scripts_reference.md) - All evaluation scripts

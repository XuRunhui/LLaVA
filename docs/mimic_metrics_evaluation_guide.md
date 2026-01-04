# MIMIC-CXR Metrics Evaluation Guide

This guide explains how to evaluate your MIMIC-CXR model predictions using CheXbert, RadGraph, BLEU, and ROUGE metrics.

## Overview

The evaluation pipeline computes the following metrics on your model's generated radiology reports:

1. **CheXbert Metrics**: Clinical accuracy metrics based on 14 pathological observations
   - Micro-F1 and Macro-F1 for 14 conditions
   - Micro-F1 and Macro-F1 for 5 most common conditions
   - Two variants: uncertain as negative (-) and uncertain as positive (+)

2. **RadGraph F1**: Measures semantic similarity using entity and relation graphs

3. **BLEU Scores**: N-gram overlap metrics (BLEU-1, BLEU-4)

4. **ROUGE Scores**: Recall-oriented metrics (ROUGE-L, ROUGE-2)

## Quick Start

### Step 1: Install Dependencies

```bash
pip install evaluate sacrebleu rouge-score radgraph scikit-learn pandas numpy scipy statsmodels huggingface_hub
```

### Step 2: Generate Predictions

First, run inference on your dev and test sets:

```bash
# Run inference on dev set
bash scripts/v1_5/eval_mimic.sh dev

# Run inference on test set
bash scripts/v1_5/eval_mimic.sh test
```

This will create JSONL files with predictions:
- `/path/to/output/eval_results_<model_name>/dev_results.jsonl`
- `/path/to/output/eval_results_<model_name>/test_results.jsonl`

### Step 3: Compute Metrics

Evaluate both dev and test sets with a single command:

```bash
bash scripts/v1_5/eval_all_mimic_metrics.sh <model_name> [bootstrap_ci]
```

Example:
```bash
bash scripts/v1_5/eval_all_mimic_metrics.sh lora_128_dp_e8 true
```

Or evaluate a single results file:

```bash
bash scripts/v1_5/eval_metrics_mimic.sh \
    /path/to/results.jsonl \
    /path/to/output_dir \
    true  # bootstrap_ci
```

### Step 4: View Results

Results are saved in multiple formats:

```
output_dir/
├── main_results.csv                      # Main metrics table
├── all_results.json                      # All metrics in JSON
├── chexbert_breakdown_positive.csv       # Per-condition CheXbert (uncertain as +)
└── chexbert_breakdown_negative.csv       # Per-condition CheXbert (uncertain as -)
```

## Input Format

The evaluation script expects a JSONL file where each line contains:

```json
{
  "question_id": "unique_id",
  "image": "path/to/image.jpg",
  "question": "What are the findings?",
  "prediction": "The model's generated report...",
  "ground_truth": "The reference report...",
  "loss": 0.45,
  "perplexity": 1.57,
  "valid_tokens": 128
}
```

**Required fields for evaluation:**
- `prediction`: The model-generated report
- `ground_truth` or `reference`: The reference/gold report

## Detailed Metrics Explanation

### CheXbert Metrics

CheXbert extracts 14 clinical observations from reports and classifies them as:
- **Blank**: Not mentioned
- **Positive**: Present
- **Negative**: Absent
- **Uncertain**: Uncertain presence

**14 Conditions:**
1. Enlarged Cardiomediastinum
2. Cardiomegaly
3. Lung Opacity
4. Lung Lesion
5. Edema
6. Consolidation
7. Pneumonia
8. Atelectasis
9. Pneumothorax
10. Pleural Effusion
11. Pleural Other
12. Fracture
13. Support Devices
14. No Finding

**5 Most Common Conditions:**
- Cardiomegaly
- Edema
- Consolidation
- Atelectasis
- Pleural Effusion

**Metric Variants:**
- `Micro-F1-14`: Micro-average F1 across all 14 conditions (uncertain as negative)
- `Macro-F1-14`: Macro-average F1 across all 14 conditions (uncertain as negative)
- `Micro-F1-5`: Micro-average F1 across 5 common conditions (uncertain as negative)
- `Macro-F1-5`: Macro-average F1 across 5 common conditions (uncertain as negative)
- `*-14+`, `*-5+`: Same metrics but treating uncertain labels as positive

### RadGraph F1

RadGraph constructs entity-relation graphs from reports and computes F1 score based on:
- **Entities**: Medical observations (e.g., "lung opacity")
- **Relations**: Connections between entities (e.g., "located in", "suggestive of")

The `partial` reward level is used, which gives credit for:
- Exact entity matches
- Entities with relations (even if relations don't fully match)

### BLEU Scores

- **BLEU-1**: Unigram overlap (measures word-level accuracy)
- **BLEU-4**: 4-gram overlap (measures fluency and phrase-level accuracy)

### ROUGE Scores

- **ROUGE-L**: Longest common subsequence (measures sentence-level similarity)
- **ROUGE-2**: Bigram overlap (measures phrase-level recall)

## Configuration Options

### Bootstrap Confidence Intervals

By default, bootstrap confidence intervals are computed (500 resamples):

```bash
# With bootstrap CI (slower, more robust)
bash scripts/v1_5/eval_metrics_mimic.sh results.jsonl output/ true

# Without bootstrap CI (faster)
bash scripts/v1_5/eval_metrics_mimic.sh results.jsonl output/ false
```

**With bootstrap CI**, metrics are reported as:
```json
{
  "Micro-F1-14": {
    "median": 0.456,
    "ci_l": 0.423,   // 95% CI lower bound
    "ci_h": 0.489    // 95% CI upper bound
  }
}
```

**Without bootstrap CI**, metrics are single values:
```json
{
  "Micro-F1-14": 0.456
}
```

### Selecting Metrics to Compute

Edit the script to choose which metrics to compute:

```bash
# In eval_metrics_mimic.sh, modify:
SCORERS="CheXbert F1-RadGraph BLEU-1 BLEU-4 ROUGE-L"

# Available scorers:
# - CheXbert
# - F1-RadGraph
# - BLEU-1
# - BLEU-4
# - ROUGE-L
# - ROUGE-2
```

## Python API Usage

You can also use the evaluation script directly from Python:

```python
from llava.eval.evaluate_mimic_metrics import ReportGenerationEvaluator

predictions = [
    "No acute cardiopulmonary process.",
    "Mild cardiomegaly. No focal consolidation."
]

references = [
    "Normal chest radiograph.",
    "Enlarged heart. Clear lungs."
]

evaluator = ReportGenerationEvaluator(
    scorers=['CheXbert', 'F1-RadGraph', 'BLEU-4'],
    bootstrap_ci=False
)

results = evaluator.evaluate(predictions, references)

print(results)
# {
#   'Micro-F1-14': 0.456,
#   'Macro-F1-14': 0.432,
#   'F1-RadGraph': 0.378,
#   'BLEU-4': 0.234,
#   ...
# }
```

## Command-Line Usage

Evaluate a single results file:

```bash
python llava/eval/evaluate_mimic_metrics.py \
    --results_file /path/to/results.jsonl \
    --output_dir /path/to/output \
    --scorers CheXbert F1-RadGraph BLEU-1 BLEU-4 ROUGE-L \
    --bootstrap_ci
```

**Arguments:**
- `--results_file`: Path to JSONL file with predictions and references
- `--output_dir`: Directory to save evaluation results
- `--scorers`: Space-separated list of metrics to compute
- `--bootstrap_ci` / `--no_bootstrap_ci`: Enable/disable bootstrap confidence intervals

## Troubleshooting

### Missing Dependencies

If you get import errors:

```bash
pip install evaluate sacrebleu rouge-score radgraph
pip install scikit-learn pandas numpy scipy statsmodels
pip install huggingface_hub transformers torch
```

### GPU Memory Issues

CheXbert and RadGraph use GPU. If you run out of memory:

1. Reduce batch size in the evaluation script
2. Use CPU by setting `CUDA_VISIBLE_DEVICES=""`
3. Evaluate subsets of data separately

### Empty Predictions

The script automatically filters out empty predictions/references. Check your inference results if you see:

```
Filtered out N empty predictions/references
```

### RadGraph Model Download

On first run, RadGraph model will be downloaded from HuggingFace Hub (~1GB). Ensure you have:
- Internet connection
- Sufficient disk space in `~/.cache/radgraph/`

### CheXbert Model Download

CheXbert model is downloaded from `StanfordAIMI/RRG_scorers` repository. Ensure:
- HuggingFace Hub access
- Model checkpoint is available at `~/.cache/huggingface/hub/`

## Expected Performance

For reference, here are typical metric ranges on MIMIC-CXR:

| Metric | Typical Range | Strong Model |
|--------|---------------|--------------|
| Micro-F1-14 | 0.35-0.50 | > 0.45 |
| Macro-F1-14 | 0.30-0.45 | > 0.40 |
| F1-RadGraph | 0.25-0.40 | > 0.35 |
| BLEU-4 | 0.10-0.25 | > 0.20 |
| ROUGE-L | 0.25-0.40 | > 0.35 |

**Note**: These are approximate ranges. Your mileage may vary based on model architecture, training data, and hyperparameters.

## Integration with Training

You can run evaluation during or after training:

### During Training

Add to your training script:

```python
if epoch % eval_every == 0:
    # Run inference
    os.system(f"bash scripts/v1_5/eval_mimic.sh dev")

    # Compute metrics
    os.system(f"bash scripts/v1_5/eval_metrics_mimic.sh "
              f"{output_dir}/dev_results.jsonl {output_dir}/metrics/ false")
```

### After Training

```bash
# 1. Run full evaluation on dev set
bash scripts/v1_5/eval_mimic.sh dev

# 2. Compute all metrics
bash scripts/v1_5/eval_all_mimic_metrics.sh my_model true

# 3. Compare with baseline
python scripts/compare_metrics.py baseline_metrics/ my_model_metrics/
```

## File Structure

```
LLaVA/
├── llava/eval/
│   ├── rrg_eval/                        # Evaluation package
│   │   ├── chexbert.py                  # CheXbert implementation
│   │   ├── f1radgraph.py                # RadGraph implementation
│   │   ├── rouge.py                     # ROUGE with bootstrap
│   │   ├── factuality_eval.py           # Factuality metrics
│   │   └── factuality_utils.py          # Utility functions
│   ├── evaluate_mimic_metrics.py        # Main evaluation script
│   ├── eval_mimic_cxr.py                # Inference script
│   └── mimic_data_utils.py              # Data loading utilities
├── scripts/v1_5/
│   ├── eval_mimic.sh                    # Run inference
│   ├── eval_metrics_mimic.sh            # Compute metrics (single file)
│   └── eval_all_mimic_metrics.sh        # Compute metrics (dev + test)
└── docs/
    └── mimic_metrics_evaluation_guide.md  # This file
```

## Citation

If you use these evaluation metrics, please cite:

**CheXbert:**
```bibtex
@inproceedings{smit2020chexbert,
  title={CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT},
  author={Smit, Akshay and Jain, Saahil and Rajpurkar, Pranav and Pareek, Anuj and Ng, Andrew Y and Lungren, Matthew P},
  booktitle={EMNLP},
  year={2020}
}
```

**RadGraph:**
```bibtex
@inproceedings{jain2021radgraph,
  title={RadGraph: Extracting Clinical Entities and Relations from Radiology Reports},
  author={Jain, Saahil and Agrawal, Ashwin and Saporta, Adriel and Truong, Steven QH and Duong, Du Nguyen and Bui, Tan and Chambon, Pierre and Zhang, Yuhao and Lungren, Matthew P and Ng, Andrew Y and others},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2021}
}
```

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the example outputs in `examples/`
3. Open an issue with:
   - Error message
   - Input file format
   - Python/CUDA versions
   - Steps to reproduce

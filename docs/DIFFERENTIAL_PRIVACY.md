# Differential Privacy Training for LLaVA

This guide explains how to train LLaVA models with **Differential Privacy (DP)** guarantees to protect sensitive training data (e.g., medical images and patient information).

## Table of Contents

- [What is Differential Privacy?](#what-is-differential-privacy)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [DP Parameters Guide](#dp-parameters-guide)
- [Training Scripts](#training-scripts)
- [Privacy Budget Management](#privacy-budget-management)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## What is Differential Privacy?

**Differential Privacy (DP)** provides mathematical guarantees that a model trained on sensitive data does not reveal information about individual training samples. This is crucial for medical AI applications where patient privacy is paramount.

### Key Concepts

- **ε (epsilon)**: Privacy budget - measures how much privacy loss is allowed
  - Lower ε = stronger privacy
  - Typical values: 1-10 for sensitive data

- **δ (delta)**: Failure probability - the probability that privacy guarantee fails
  - Typically set to 1/n where n is dataset size
  - Example: For 1000 samples, δ = 1e-3

- **Privacy Guarantee**: The training process satisfies (ε, δ)-differential privacy
  - Informally: "It's hard to tell if any specific sample was in the training data"

### DP-SGD Algorithm

We use **DP-SGD** (Differentially Private Stochastic Gradient Descent):
1. **Clip gradients** to bound their L2 norm (max_grad_norm)
2. **Add calibrated noise** to gradients (based on ε, δ)
3. **Track privacy budget** consumed during training

---

## Installation

### 1. Install Opacus

```bash
pip install opacus
```

### 2. Verify Installation

```bash
python -c "import opacus; print(f'Opacus {opacus.__version__} installed successfully')"
```

### 3. Install LLaVA Dependencies

```bash
pip install -e .
```

---

## Quick Start

### Example 1: Train with Strong Privacy (ε=3.0)

```bash
# Train LLaVA-Med on VQA-RAD with strong privacy
bash scripts/RAD_VQA_lora_dp.sh 3 3.0
```

This provides **strong privacy protection** suitable for medical data.

### Example 2: Train with Moderate Privacy (ε=8.0)

```bash
# Train with moderate privacy for less sensitive data
bash scripts/RAD_VQA_lora_dp.sh 5 8.0
```

### Example 3: Custom DP Configuration

```python
# In your Python script
from llava.train.train_dp import train

# Set DP parameters via command line or config
training_args = TrainingArguments(
    # Standard training args
    output_dir="./checkpoints/llava-dp",
    num_train_epochs=3,
    per_device_train_batch_size=4,

    # DP-specific args
    dp_enabled=True,
    dp_epsilon=3.0,
    dp_delta=1e-5,
    dp_max_grad_norm=1.0,
    dp_poisson_sampling=True,
    dp_secure_mode=False,
)

train()
```

---

## DP Parameters Guide

### Core DP Parameters

| Parameter | Type | Default | Description | Recommendations |
|-----------|------|---------|-------------|-----------------|
| `dp_enabled` | bool | False | Enable DP training | Set to `True` |
| `dp_epsilon` | float | 3.0 | Privacy budget (ε) | **Medical data: 1-3**<br>General data: 3-10 |
| `dp_delta` | float | 1e-5 | Failure probability (δ) | **1/dataset_size** |
| `dp_max_grad_norm` | float | 1.0 | Gradient clipping threshold | **0.1-1.0**<br>Lower = more privacy |
| `dp_noise_multiplier` | float | None | Noise scale (auto-computed) | Leave as `None` |
| `dp_poisson_sampling` | bool | True | Use Poisson sampling | **Recommended: True** |
| `dp_secure_mode` | bool | False | Cryptographic RNG | False (slower) |
| `dp_physical_batch_size` | int | None | Memory management | Set if OOM errors |

### Privacy Budget Interpretation

| Epsilon (ε) | Privacy Level | Use Case | Notes |
|------------|---------------|----------|-------|
| < 1 | **Very Strong** | Highly sensitive medical records | May hurt model utility |
| 1-3 | **Strong** | Medical images, patient data | **Recommended for healthcare** |
| 3-10 | **Moderate** | General sensitive data | Common in practice |
| > 10 | **Weak** | Non-sensitive data | Not recommended for privacy |

### Delta (δ) Guidelines

```python
# Rule of thumb: δ should be much smaller than 1/n
dataset_size = len(train_dataset)
recommended_delta = 1.0 / dataset_size

# Examples:
# - 100 samples   → δ = 1e-2
# - 1,000 samples → δ = 1e-3
# - 10,000 samples → δ = 1e-4
# - 100,000 samples → δ = 1e-5
```

### Gradient Clipping (max_grad_norm)

Controls the trade-off between privacy and utility:

- **Lower values (0.1-0.5)**: Stronger privacy, may slow convergence
- **Medium values (0.5-1.0)**: Balanced, **recommended starting point**
- **Higher values (1.0-2.0)**: Weaker privacy, faster convergence

```bash
# Try different clipping values
bash scripts/RAD_VQA_lora_dp.sh 3 3.0  # Uses default max_grad_norm=1.0

# Or modify in script:
DP_MAX_GRAD_NORM=0.5  # Stronger privacy
```

---

## Training Scripts

### 1. VQA-RAD with DP (LoRA)

**Script**: `scripts/RAD_VQA_lora_dp.sh`

```bash
# Basic usage
bash scripts/RAD_VQA_lora_dp.sh <EPOCHS> [EPSILON]

# Examples:
bash scripts/RAD_VQA_lora_dp.sh 3          # ε=3.0 (default)
bash scripts/RAD_VQA_lora_dp.sh 5 8.0      # ε=8.0
bash scripts/RAD_VQA_lora_dp.sh 3 1.5      # ε=1.5 (very strong privacy)
```

**Key features**:
- LoRA fine-tuning for memory efficiency
- Automatic privacy budget calculation
- Privacy accounting saved with checkpoints
- Compatible with medical imaging datasets

### 2. Custom Training with train_dp.py

```bash
torchrun --nproc_per_node=1 llava/train/train_dp.py \
    --model_name_or_path microsoft/llava-med-v1.5-mistral-7b \
    --data_path /path/to/data.json \
    --image_folder /path/to/images \
    --output_dir ./checkpoints/llava-dp \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --lora_enable True \
    --lora_r 64 \
    --dp_enabled True \
    --dp_epsilon 3.0 \
    --dp_delta 1e-5 \
    --dp_max_grad_norm 1.0
```

---

## Privacy Budget Management

### Understanding Privacy Consumption

Privacy budget (ε) is consumed during training:

```
Final ε ≈ (noise_multiplier)^-1 × sqrt(epochs × steps_per_epoch × sampling_rate)
```

**Key insights**:
1. **More epochs** → More privacy consumed
2. **Smaller batches** → Better privacy (but slower training)
3. **Lower noise** → Faster convergence but more privacy consumed

### Monitoring Privacy During Training

The trainer automatically logs privacy metrics:

```python
# Privacy metrics logged to wandb/tensorboard
{
    "privacy/epsilon": 2.85,          # Current privacy spent
    "privacy/delta": 1e-5,            # Target delta
    "privacy/max_grad_norm": 1.0,     # Clipping threshold
    "privacy/remaining_epsilon": 0.15 # Budget remaining
}
```

### Privacy Accounting Files

After each checkpoint, privacy info is saved:

```bash
cat checkpoints/checkpoint-100/privacy_accounting.json
```

```json
{
  "epsilon": 1.45,
  "delta": 1e-5,
  "target_epsilon": 3.0,
  "max_grad_norm": 1.0,
  "noise_multiplier": 0.8,
  "global_step": 100,
  "epoch": 1.0
}
```

### Budget Exhaustion Protection

Training automatically stops if ε exceeds target:

```
WARNING: Privacy budget nearly exhausted! ε=2.85/3.0
ERROR: Privacy budget EXHAUSTED! Current ε=3.05 > target ε=3.0
Stopping training to preserve privacy guarantees.
```

---

## Best Practices

### 1. Choose Appropriate Privacy Level

```python
# Medical imaging / patient data
dp_epsilon = 3.0  # Strong privacy
dp_delta = 1e-5

# Less sensitive medical data
dp_epsilon = 8.0  # Moderate privacy
dp_delta = 1e-4
```

### 2. Optimize Batch Size

```python
# Recommended: batch_size = 0.1-1% of dataset size
dataset_size = 1000
recommended_batch_size = max(32, int(dataset_size * 0.005))  # 50

# For DP training:
per_device_train_batch_size = 4
gradient_accumulation_steps = recommended_batch_size // (4 * num_gpus)
```

### 3. Use LoRA with DP

**LoRA is highly recommended for DP training**:

```bash
# LoRA reduces parameters to train
LORA_ENABLE=True
LORA_R=64
LORA_ALPHA=64

# Benefits:
# ✓ Lower memory usage
# ✓ Faster training
# ✓ Better privacy/utility tradeoff
# ✓ Smaller gradient norms
```

### 4. Disable Incompatible Features

```python
# These features may weaken privacy guarantees:
group_by_modality_length = False  # Disable length grouping
gradient_accumulation_steps = 1   # Minimize (or set physical_batch_size)
```

### 5. Tune Hyperparameters

Start with these values and adjust:

```python
# Conservative starting point (strong privacy)
dp_epsilon = 3.0
dp_max_grad_norm = 1.0
learning_rate = 2e-4

# If model doesn't converge, try:
dp_epsilon = 5.0  # Relax privacy slightly
dp_max_grad_norm = 1.5  # Allow larger gradients
learning_rate = 5e-4  # Increase LR
```

### 6. Validate Privacy Requirements

Before training, compute requirements:

```python
from llava.train.llava_dp_trainer import compute_privacy_requirements

requirements = compute_privacy_requirements(
    dataset_size=1000,
    batch_size=32,
    epochs=3,
    target_epsilon=3.0
)

print(f"Total steps: {requirements['total_steps']}")
print(f"Recommended batch size: {requirements['recommended_batch_size']}")
print(f"Privacy level: {requirements['privacy_guidance']['current']}")
```

---

## Troubleshooting

### Issue 1: "Model has compatibility issues with Opacus"

**Cause**: Some PyTorch modules are not DP-compatible

**Solution**: The trainer automatically fixes most issues:

```python
# Automatic fix applied
Model has 5 compatibility issues with Opacus:
  - BatchNorm layers detected
  - ...
Attempting to fix model for DP compatibility...
Model successfully fixed for DP compatibility!
```

If issues persist, manually replace incompatible layers.

### Issue 2: Training stops with "Privacy budget EXHAUSTED"

**Cause**: Privacy budget consumed faster than expected

**Solutions**:
1. **Increase target epsilon**: `--dp_epsilon 5.0`
2. **Reduce epochs**: Train for fewer epochs
3. **Increase batch size**: Larger batches → less privacy consumed
4. **Increase max_grad_norm**: `--dp_max_grad_norm 1.5`

### Issue 3: Poor model performance

**Cause**: Privacy noise overwhelming signal

**Solutions**:
1. **Relax privacy**: Increase epsilon (e.g., 3.0 → 5.0)
2. **Tune clipping**: Try different `max_grad_norm` values (0.5, 1.0, 1.5)
3. **Increase batch size**: Reduces noise variance
4. **More training**: DP models often need more epochs
5. **Use LoRA**: Better parameter efficiency

### Issue 4: Out of Memory (OOM) errors

**Solutions**:

```bash
# Set physical batch size
DP_PHYSICAL_BATCH_SIZE=2

# Or reduce batch size
BATCH_SIZE=2
GRAD_ACCUM_STEPS=4
```

### Issue 5: "Opacus not found"

```bash
# Install Opacus
pip install opacus

# Verify installation
python -c "import opacus; print(opacus.__version__)"
```

### Issue 6: Slow training speed

**Cause**: DP adds computational overhead (~20-50%)

**Solutions**:
1. **Disable secure_mode**: `--dp_secure_mode False` (default)
2. **Use Poisson sampling**: `--dp_poisson_sampling True` (recommended)
3. **Reduce physical_batch_size**: Only if needed for memory
4. **Use mixed precision**: `--bf16 True`

---

## Advanced Topics

### Multi-GPU DP Training

```bash
# DP works with multi-GPU via DDP
torchrun --nproc_per_node=4 llava/train/train_dp.py \
    --dp_enabled True \
    --dp_epsilon 3.0 \
    # ... other args
```

**Note**: Effective batch size = `per_device_batch_size × num_gpus × grad_accum_steps`

### Comparing Privacy Budgets

| Dataset Size | ε=1 | ε=3 | ε=8 | ε=10 |
|-------------|-----|-----|-----|------|
| 100 | Very Strong | Strong | Moderate | Weak |
| 1,000 | Very Strong | **Strong** | Moderate | Weak |
| 10,000 | Very Strong | **Strong** | **Moderate** | Weak |
| 100,000 | Very Strong | Strong | **Moderate** | Moderate |

**Bold** = Recommended for that dataset size

### DP with Different Training Strategies

| Strategy | DP Compatible | Notes |
|----------|---------------|-------|
| Full fine-tuning | ✓ | High memory, good privacy |
| LoRA | **✓ Recommended** | Best privacy/utility tradeoff |
| QLoRA (4-bit) | ⚠ Partial | Quantization may affect privacy |
| Projector-only | ✓ | Fast, but limited adaptation |

---

## References

- **Opacus**: https://opacus.ai/
- **DP-SGD Paper**: Abadi et al., "Deep Learning with Differential Privacy" (2016)
- **Privacy in ML**: https://developers.google.com/machine-learning/practica/fairness/privacy

---

## Citation

If you use differential privacy training in your research, please cite:

```bibtex
@article{abadi2016deep,
  title={Deep learning with differential privacy},
  author={Abadi, Martin and Chu, Andy and Goodfellow, Ian and McMahan, H Brendan and Mironov, Ilya and Talwar, Kunal and Zhang, Li},
  journal={ACM SIGSAC Conference on Computer and Communications Security},
  year={2016}
}

@misc{opacus,
  title={Opacus: User-Friendly Differential Privacy Library in PyTorch},
  author={Ashkan Yousefpour and others},
  year={2021},
  url={https://opacus.ai}
}
```

---

## Support

For questions or issues:
1. Check this documentation
2. Review the [Troubleshooting](#troubleshooting) section
3. Open an issue on GitHub
4. Consult the [Opacus documentation](https://opacus.ai/docs)

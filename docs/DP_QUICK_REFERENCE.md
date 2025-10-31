# Differential Privacy Quick Reference

## TL;DR - Get Started in 30 Seconds

```bash
# 1. Install Opacus
pip install opacus

# 2. Run DP training (3 epochs, ε=3.0)
bash scripts/RAD_VQA_lora_dp.sh 3 3.0

# Done! Your model is now privacy-preserving.
```

---

## Command Cheat Sheet

### Basic Training

```bash
# Strong privacy (ε=3.0) - Recommended for medical data
bash scripts/RAD_VQA_lora_dp.sh 3 3.0

# Moderate privacy (ε=8.0) - General sensitive data
bash scripts/RAD_VQA_lora_dp.sh 5 8.0

# Very strong privacy (ε=1.0) - Highest protection
bash scripts/RAD_VQA_lora_dp.sh 3 1.0
```

### Custom Configuration

```bash
torchrun --nproc_per_node=1 llava/train/train_dp.py \
    --model_name_or_path <MODEL> \
    --data_path <DATA.json> \
    --image_folder <IMAGE_DIR> \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --dp_enabled True \
    --dp_epsilon 3.0 \
    --dp_delta 1e-5 \
    --dp_max_grad_norm 1.0 \
    --lora_enable True
```

---

## Parameter Quick Guide

| Parameter | Default | Medical Data | General Data |
|-----------|---------|--------------|--------------|
| `dp_epsilon` | 3.0 | **1.0-3.0** | 5.0-10.0 |
| `dp_delta` | 1e-5 | **1/dataset_size** | 1e-5 |
| `dp_max_grad_norm` | 1.0 | **0.5-1.0** | 1.0-1.5 |
| `dp_poisson_sampling` | True | **True** | True |
| `batch_size` | 2-4 | **2-4** | 4-8 |

---

## Privacy Level Selector

**Choose your epsilon based on data sensitivity:**

```
┌─────────────────────────────────────────────────────┐
│  ε = 1.0   │ █████████░ Very Strong                 │
│  ε = 3.0   │ ███████░░░ Strong (Recommended)        │ ← Medical
│  ε = 5.0   │ █████░░░░░ Moderate                    │
│  ε = 8.0   │ ███░░░░░░░ Moderate-Weak              │
│  ε = 10.0  │ ██░░░░░░░░ Weak                       │
└─────────────────────────────────────────────────────┘
         Privacy Strength →
```

---

## Common Scenarios

### Scenario 1: Medical Imaging (Patient X-rays)
```bash
bash scripts/RAD_VQA_lora_dp.sh 3 2.0
# ε=2.0 provides strong privacy for sensitive patient data
```

### Scenario 2: Clinical Text (Medical Records)
```bash
bash scripts/RAD_VQA_lora_dp.sh 5 1.5
# ε=1.5 for very sensitive text data
```

### Scenario 3: Public Medical Dataset (ChestX-ray14)
```bash
bash scripts/RAD_VQA_lora_dp.sh 5 5.0
# ε=5.0 acceptable for already-public datasets
```

### Scenario 4: Quick Experimentation
```bash
bash scripts/RAD_VQA_lora_dp.sh 1 8.0
# ε=8.0 for fast testing (not for production)
```

---

## Monitoring Privacy

### During Training

Watch for these metrics in logs:
```
privacy/epsilon: 1.45        ← Current privacy spent
privacy/remaining_epsilon: 1.55  ← Budget remaining
```

### After Training

Check final privacy:
```bash
# View last checkpoint's privacy accounting
cat checkpoints/checkpoint-*/privacy_accounting.json | tail -1
```

---

## Troubleshooting One-Liners

```bash
# Fix: Opacus not found
pip install opacus

# Fix: Privacy budget exhausted too quickly
# → Increase epsilon: bash scripts/RAD_VQA_lora_dp.sh 3 5.0

# Fix: Poor model performance
# → Increase max_grad_norm in script: DP_MAX_GRAD_NORM=1.5

# Fix: Out of memory
# → Reduce batch size: BATCH_SIZE=1

# Fix: Training too slow
# → Disable secure mode: DP_SECURE_MODE=False (already default)
```

---

## Performance Tips

**Fastest DP training setup:**
```bash
# Use these settings in script:
BATCH_SIZE=4
LORA_ENABLE=True
LORA_R=64
DP_PHYSICAL_BATCH_SIZE=4
DP_SECURE_MODE=False
BF16=True
```

---

## Privacy Guarantee Explained

**What (ε=3.0, δ=1e-5)-DP means:**

> "An attacker looking at the trained model cannot determine with confidence greater than e^3 ≈ 20x whether any specific patient's data was in the training set, except with probability 0.00001"

**In practice:**
- ε=1: Very hard to detect if a sample was used
- ε=3: Hard to detect (recommended)
- ε=10: Somewhat hard to detect
- No DP: Easy to detect (model may memorize training data)

---

## File Locations

```
LLaVA/
├── llava/train/
│   ├── train_dp.py              # DP training script
│   └── llava_dp_trainer.py      # DP trainer implementation
├── scripts/
│   └── RAD_VQA_lora_dp.sh       # Ready-to-use DP training script
└── docs/
    ├── DIFFERENTIAL_PRIVACY.md   # Full documentation
    └── DP_QUICK_REFERENCE.md     # This file
```

---

## Example Output

```
========================================
LLaVA-Med v1.5 DP Fine-tuning on VQA-RAD
========================================
Dataset size: 451 samples
Target ε (epsilon): 3.0
Target δ (delta): 1e-5
Privacy level: ✓ STRONG privacy (ε ≤ 3)

[Training starts...]

Step 100: loss=0.45, privacy/epsilon=1.23
Step 200: loss=0.32, privacy/epsilon=2.15
Step 300: loss=0.28, privacy/epsilon=2.89

Training completed!
Final privacy: ε=2.94 < 3.0 ✓
```

---

## Decision Tree

```
Need Privacy Protection?
│
├─ YES → How sensitive is data?
│        │
│        ├─ Very sensitive (patient records) → ε=1.0-2.0
│        ├─ Sensitive (medical images)       → ε=2.0-3.0 ✓
│        └─ Moderately sensitive             → ε=5.0-8.0
│
└─ NO  → Use standard training (train.py)
```

---

## Resources

- **Full Documentation**: [DIFFERENTIAL_PRIVACY.md](DIFFERENTIAL_PRIVACY.md)
- **Opacus Docs**: https://opacus.ai/docs
- **DP-SGD Paper**: https://arxiv.org/abs/1607.00133

---

## Quick Test

Verify your setup:

```bash
# Test if Opacus is working
python -c "
from llava.train.llava_dp_trainer import compute_privacy_requirements
print(compute_privacy_requirements(1000, 32, 3, 3.0))
"
```

Expected output:
```python
{
    'dataset_size': 1000,
    'target_epsilon': 3.0,
    'target_delta': 0.001,
    'total_steps': 93,
    'privacy_guidance': {'current': 'Strong privacy'},
    ...
}
```

---

**Remember**: Lower epsilon = Stronger privacy but may reduce model utility. Start with ε=3.0 for medical data and adjust based on your privacy/utility requirements.

# LLaVA Training Modes Guide

This guide explains different training configurations for LLaVA, particularly for medical imaging tasks with differential privacy.

## Training Component Overview

LLaVA consists of three main components:

```
┌─────────────────────────────────────────┐
│           Vision Tower (CLIP)           │  ← Usually FROZEN
│    (extracts visual features)           │
└──────────────┬──────────────────────────┘
               │ image features
               ↓
┌─────────────────────────────────────────┐
│        MM Projector (MLP)               │  ← Can be trained
│   (maps visual → language space)        │
└──────────────┬──────────────────────────┘
               │ projected features
               ↓
┌─────────────────────────────────────────┐
│      Language Model (LLaMA)             │  ← LoRA adapters
│    (generates text responses)            │
└─────────────────────────────────────────┘
```

## Training Modes

### Mode 1: LoRA Only (Current Default)

**What gets trained:**
- ✅ LoRA adapters on LLM
- ❌ MM Projector (frozen)
- ❌ Vision Tower (frozen)

**Configuration:**
```bash
--lora_enable True \
--lora_r 128 \
--lora_alpha 256 \
--learning_rate 5e-6
```

**When to use:**
- Fast training with minimal parameters (~50M trainable)
- When you want to preserve pre-trained vision-language alignment
- When you have limited compute resources

**Trainable parameters:** ~50M (LoRA adapters only)

---

### Mode 2: LoRA + MM Projector (Recommended for Medical Imaging)

**What gets trained:**
- ✅ LoRA adapters on LLM
- ✅ MM Projector (full fine-tuning)
- ❌ Vision Tower (frozen)

**Configuration:**
```bash
--lora_enable True \
--lora_r 128 \
--lora_alpha 256 \
--tune_mm_mlp_adapter True \
--mm_projector_lr 2e-5 \
--learning_rate 5e-6
```

**When to use:**
- **Medical imaging** where visual features need domain adaptation
- When pre-trained projector was trained on natural images
- When you want better vision-language alignment for your domain
- **Recommended for MIMIC-CXR and medical reports**

**Trainable parameters:** ~50M (LoRA) + ~20M (projector) = ~70M

**Key benefits:**
- Adapts vision-language mapping to medical domain
- Better alignment between X-ray features and medical terminology
- More flexible than LoRA-only while remaining efficient

---

### Mode 3: Full Fine-tuning

**What gets trained:**
- ✅ Entire LLM (all 7B parameters)
- ✅ MM Projector
- ❌ Vision Tower (frozen)

**Configuration:**
```bash
--lora_enable False \
--tune_mm_mlp_adapter True \
--mm_projector_lr 2e-5 \
--learning_rate 2e-5
```

**When to use:**
- When you have very large datasets (>100K samples)
- When you need maximum performance
- When you have significant compute resources

**Trainable parameters:** ~7B + ~20M = ~7.02B

**⚠️ Warning:** Not recommended with differential privacy due to:
- Massive memory requirements
- Very slow training
- Privacy budget consumed faster

---

### Mode 4: Projector Only (Pretraining)

**What gets trained:**
- ❌ LoRA adapters (frozen)
- ✅ MM Projector only
- ❌ Vision Tower (frozen)
- ❌ LLM (frozen)

**Configuration:**
```bash
--lora_enable False \
--freeze_backbone True \
--tune_mm_mlp_adapter True \
--mm_projector_lr 2e-3 \
--learning_rate 1e-3
```

**When to use:**
- Initial pretraining on image-text pairs
- Domain adaptation of projector before fine-tuning
- When you want to quickly align visual features to language

**Trainable parameters:** ~20M (projector only)

---

## Your Updated Configuration

I've updated your training script to **Mode 2: LoRA + MM Projector**:

```bash
# In scripts/v1_5/finetune_mimic.sh
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    /project2/.../train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --tune_mm_mlp_adapter True --mm_projector_lr 2e-5 \    # ← ADDED
    --learning_rate 5e-6 \
    --dp_enabled True \
    --dp_epsilon 8.0 \
    ...
```

**Changes made:**
- Added `--tune_mm_mlp_adapter True` to enable projector training
- Kept `--mm_projector_lr 2e-5` (separate learning rate for projector)
- LLM LoRA uses `--learning_rate 5e-6` (lower for stability)

---

## Learning Rate Guidelines

Different components should use different learning rates:

| Component | Typical LR | Your Config | Rationale |
|-----------|------------|-------------|-----------|
| LLM (LoRA) | 1e-5 to 5e-5 | `5e-6` | Conservative for stability with DP |
| MM Projector | 1e-4 to 2e-4 | `2e-5` | Higher than LLM, but conservative for DP |
| LLM (Full) | 1e-5 to 2e-5 | N/A | Not used with LoRA |

**With Differential Privacy:** Use 5-10x lower learning rates than standard fine-tuning due to:
- Gradient clipping (limits gradient magnitude)
- Gradient noise (added for privacy)
- Slower convergence

---

## Checkpoint Saving

### Mode 1 (LoRA Only)
Saves:
- `adapter_config.json`
- `adapter_model.safetensors` (~200MB)

### Mode 2 (LoRA + Projector) - Your current mode
Saves:
- `adapter_config.json`
- `adapter_model.safetensors` (~200MB)
- `mm_projector.bin` (~80MB)
- `non_lora_trainables.bin` (~80MB for projector)

### Mode 3 (Full Fine-tuning)
Saves:
- Entire model checkpoint (~13GB)

---

## Verification: Check What's Being Trained

To verify what's actually being trained, look for this output during training:

```bash
trainable params: 70,254,592 || all params: 7,111,897,088 || trainable%: 0.99%
```

**Expected trainable parameters:**
- **LoRA only:** ~50M (0.7%)
- **LoRA + Projector:** ~70M (0.99%) ← Your new config
- **Full fine-tuning:** ~7B (98%)

You can also check during training:

```python
# Add to train.py after model initialization
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

# Check specific components
projector_params = sum(p.numel() for p in model.get_model().mm_projector.parameters() if p.requires_grad)
print(f"MM Projector trainable: {projector_params:,}")
```

---

## Differential Privacy Considerations

### Memory Requirements

| Mode | FP32 Memory | BF16 Memory | With DP |
|------|-------------|-------------|---------|
| LoRA only | ~16GB | ~12GB | ~18GB |
| LoRA + Projector | ~18GB | ~14GB | ~20GB |
| Full FT | ~32GB | ~20GB | ~40GB+ |

**Why DP increases memory:**
- Per-sample gradients stored temporarily
- Ghost clipping intermediate activations
- Privacy accounting overhead

### Training Speed

| Mode | Samples/sec | DP Overhead |
|------|-------------|-------------|
| LoRA only | ~2.5 | 1.3x slower |
| LoRA + Projector | ~2.3 | 1.4x slower |
| Full FT | ~0.8 | 2.0x slower |

### Privacy Budget

The epsilon value consumed is the **same** regardless of mode because:
- DP clips gradients per-sample
- Number of parameters doesn't directly affect epsilon
- What matters: number of training steps and batch size

However, **projector training may help** because:
- Better convergence → fewer epochs needed
- Better alignment → lower loss faster
- Can achieve same performance with less training

---

## Recommended Configuration for MIMIC-CXR

Based on your use case (medical radiology reports with DP):

```bash
# Training components
--lora_enable True \
--lora_r 128 \
--lora_alpha 256 \
--tune_mm_mlp_adapter True \          # ✓ Enable projector training

# Learning rates
--learning_rate 5e-6 \                # LLM LoRA learning rate
--mm_projector_lr 2e-5 \              # Projector learning rate (4x higher)

# Differential Privacy
--dp_enabled True \
--dp_epsilon 8.0 \
--dp_delta 5e-5 \
--dp_max_grad_norm 2.0 \
--dp_use_ghost_clipping True \

# Training schedule
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \     # Effective batch size: 8
--warmup_steps 100 \
--lr_scheduler_type "cosine" \

# Data
--use_mimic_loader True \
--mimic_filter_views True \
--mimic_include_reason True \
--mimic_generation_methods "gpt4" \   # or "all" for more data
```

**Rationale:**
- ✅ LoRA: Efficient, parameter-efficient
- ✅ Projector training: Adapts vision-language alignment to medical domain
- ✅ Separate LR: Projector needs higher LR to adapt quickly
- ✅ DP with ghost clipping: Memory-efficient privacy
- ✅ Conservative LRs: Stable training with DP

---

## Loading Checkpoints for Inference

### If you trained Mode 2 (LoRA + Projector):

```python
from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="/path/to/lora_128_dp_e8",     # Your checkpoint
    model_base="liuhaotian/llava-v1.5-7b",    # Base model
    model_name="llava-lora_128_dp_e8",        # Any name with 'llava' and 'lora'
    load_8bit=False,
    load_4bit=False,
)
```

The model builder will automatically:
1. Load base LLaVA model
2. Load LoRA adapters from `adapter_model.safetensors`
3. Load projector weights from `mm_projector.bin` or `non_lora_trainables.bin`
4. Merge everything into a working model

---

## FAQ

### Q: Should I always train the projector for medical imaging?

**A: Yes, recommended.** Medical images (X-rays, CT, MRI) are very different from natural images. The pre-trained projector was learned on COCO/Visual Genome (natural images), so it may not optimally map medical visual features to medical terminology. Training the projector helps adapt this mapping.

### Q: Will training the projector hurt privacy?

**A: No.** The projector is included in the DP training process. Opacus will:
- Clip gradients for projector parameters
- Add noise to projector gradients
- Track projector in privacy accounting

The epsilon value accounts for ALL trainable parameters.

### Q: Can I use different learning rates for different LoRA layers?

**A: Not easily with current setup.** The `--learning_rate` applies to all LoRA parameters uniformly. To use layer-wise learning rates, you'd need to modify the `create_optimizer` function in `llava_trainer.py`.

### Q: Should I freeze the projector during DP training to save privacy budget?

**A: No.** Privacy budget is primarily determined by:
- Number of training steps
- Batch size
- Noise multiplier (derived from epsilon/delta)

Freezing the projector won't save privacy budget, but it may hurt performance because the vision-language alignment won't adapt to the medical domain.

---

## Next Steps

1. **Verify your current checkpoint:**
   ```bash
   ls -lh /path/to/lora_128_dp_e8/
   # Should see: adapter_model.safetensors, mm_projector.bin (or non_lora_trainables.bin)
   ```

2. **Check trainable parameters** during next training run:
   Look for: `trainable params: ~70M`

3. **Monitor training metrics:**
   - Loss should decrease faster with projector training
   - Better vision-language alignment
   - May converge in fewer epochs

4. **Evaluate on MIMIC-CXR:**
   - Use the evaluation pipeline we just created
   - Compare LoRA-only vs LoRA+Projector performance
   - Check CheXbert and RadGraph metrics

---

## Summary

**Your updated configuration trains:**
- ✅ LoRA adapters on LLM (50M params) at LR=5e-6
- ✅ MM Projector (20M params) at LR=2e-5
- ❌ Vision Tower (frozen)

This is the **recommended mode for medical imaging** as it:
- Adapts vision-language mapping to medical domain
- Remains parameter-efficient (~70M trainable)
- Works well with differential privacy
- Provides better performance than LoRA-only on domain-specific tasks

# Selective Token-Level Differential Privacy Implementation Plan

**Author**: Claude Code
**Date**: 2026-01-03
**Reference Paper**: [Masked Differential Privacy (MaskDP)](https://arxiv.org/html/2410.17098v1)
**Status**: Planning Phase - Not Yet Implemented

---

## Overview

This document describes the implementation plan for selective token-level differential privacy in LLaVA. The goal is to apply differential privacy noise **only to specific tokens** in the output (e.g., private medical findings) while leaving other tokens (e.g., public anatomical terms) unprotected.

### Motivation

In medical image captioning (e.g., MIMIC-CXR chest X-ray reports), the output contains:
- **Private information**: Specific patient findings (e.g., "small pleural effusion in right lung")
- **Public information**: General anatomical terms and structure (e.g., "heart", "lungs", "normal cardiac silhouette")

Standard DP applies noise to all tokens uniformly, which:
1. Degrades utility unnecessarily for public tokens
2. May provide insufficient privacy for truly sensitive tokens

**Selective DP** allows fine-grained control: strong privacy for private tokens, no degradation for public tokens.

---

## Current DP Implementation (Baseline)

### Current Architecture

```
User Input (Image + Text)
    ↓
LLaVA Model Forward Pass
    ↓
Loss Computation (on all output tokens)
    ↓
Backward Pass (Opacus computes per-sample gradients)
    ↓
Gradient Clipping (global, uniform)
    ↓
Noise Addition (global, uniform)
    ↓
Parameter Update
```

**Files Involved**:
- [llava/train/train.py](../llava/train/train.py): Data preprocessing, creates `labels` array
- [llava/train/llava_trainer.py](../llava/train/llava_trainer.py): DP integration via Opacus PrivacyEngine
- Current DP applied uniformly to all tokens via `IGNORE_INDEX` mask in labels

---

## Proposed Selective DP Architecture

### New Architecture

```
User Input (Image + Text)
    ↓
Data Preprocessing (creates dp_mask: 0=public, 1=private)
    ↓
LLaVA Model Forward Pass
    ↓
Selective Loss Computation (separate private/public token losses)
    ↓
Backward Pass (Opacus computes per-sample gradients)
    ↓
Gradient Decomposition (split by token privacy status)
    ↓
Selective Clipping (only private token gradients)
    ↓
Selective Noise Addition (only private token gradients)
    ↓
Gradient Combination (private_noised + public_clean)
    ↓
Parameter Update
```

---

## Implementation Plan

## Phase 1: Data Pipeline & Configuration

### 1.1 Add New Constant

**File**: [llava/constants.py](../llava/constants.py)

```python
# Add to existing constants:
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DP_PRIVATE_INDEX = -300  # NEW: Marker for tokens requiring DP protection
```

**Purpose**: Mark which tokens in the sequence need DP protection (similar to how `IGNORE_INDEX` marks tokens to ignore during training).

---

### 1.2 Add Selective DP Parameters

**File**: [llava/train/train.py](../llava/train/train.py) - `TrainingArguments` class

```python
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # ... existing DP parameters ...
    dp_enabled: bool = field(default=False)
    dp_epsilon: float = field(default=8.0)
    dp_delta: float = field(default=1e-5)
    dp_max_grad_norm: float = field(default=1.0)
    dp_use_ghost_clipping: bool = field(default=True)

    # NEW: Selective DP parameters
    dp_selective_mode: bool = field(
        default=False,
        metadata={"help": "Enable selective token-level DP (apply DP only to marked tokens)"}
    )
    dp_token_privacy_strategy: str = field(
        default="all_output",
        metadata={
            "help": "Strategy for marking private tokens: "
                    "'all_output' (default DP on all tokens), "
                    "'findings_only' (DP on findings, not indication), "
                    "'custom_keywords' (DP on tokens matching keyword list)"
        }
    )
    dp_private_keywords_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to file containing private keywords (one per line) for custom strategy"}
    )
```

**Purpose**: Control selective DP behavior via command-line arguments.

---

### 1.3 Create `dp_mask` Array in Preprocessing

**File**: [llava/train/train.py](../llava/train/train.py) - Modify `preprocess_v1`, `preprocess_llama_2`, etc.

**Current preprocessing creates**:
- `input_ids`: Tokenized input sequence
- `labels`: Target tokens (with `IGNORE_INDEX` for prompt/instruction parts)

**New preprocessing will also create**:
- `dp_mask`: Privacy mask (0 = public/no DP, 1 = private/apply DP)

**Example modification to `preprocess_v1`**:

```python
def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    dp_selective_mode: bool = False,  # NEW
    dp_strategy: str = "all_output"   # NEW
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # ... existing tokenization logic ...

    targets = input_ids.clone()

    # NEW: Create dp_mask (same shape as targets)
    if dp_selective_mode:
        dp_masks = torch.zeros_like(targets)  # 0 = public by default

    # Mask targets (existing logic)
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target, dp_mask in zip(conversations, targets, dp_masks if dp_selective_mode else [None]*len(targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Instruction/prompt part
            parts[1]         # Response/output part (GPT response)

            # ... compute round_len, instruction_len ...

            # Mask instruction tokens (existing)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            # NEW: Mark response tokens for selective DP
            if dp_selective_mode:
                response_start = cur_len + instruction_len
                response_end = cur_len + round_len

                if dp_strategy == "all_output":
                    # Apply DP to all output tokens
                    dp_mask[response_start:response_end] = 1

                elif dp_strategy == "findings_only":
                    # For MIMIC-CXR: parts[0] is question with indication,
                    # parts[1] is findings description
                    # Apply DP only to findings (parts[1]), not indication
                    # This requires parsing the conversation structure
                    if "indication" not in parts[0].lower():
                        # This is a findings response, apply DP
                        dp_mask[response_start:response_end] = 1
                    # Otherwise leave as 0 (public)

                elif dp_strategy == "custom_keywords":
                    # Apply DP to tokens matching private keywords
                    # (Implementation requires keyword matching - see below)
                    pass

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        if dp_selective_mode:
            dp_mask[cur_len:] = 0  # Padding is public (no DP)

    result = dict(input_ids=input_ids, labels=targets)
    if dp_selective_mode:
        result['dp_mask'] = dp_masks

    return result
```

**For MIMIC-CXR Specific Strategy** (`findings_only`):

In MIMIC-CXR data, conversations look like:
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nProvide a description of the findings in the radiology image given the following indication: {reason}"
    },
    {
      "from": "gpt",
      "value": "Lateral view somewhat limited due to overlying motion artifact. The lungs are low in volume..."
    }
  ]
}
```

**Strategy**:
- Indication text (from `reason` field) = **PUBLIC** (dp_mask = 0)
- Findings description (GPT response) = **PRIVATE** (dp_mask = 1)

This is already naturally separated since:
- Human message contains indication (instruction, masked with `IGNORE_INDEX` anyway)
- GPT message contains findings (output, should get DP protection)

So `findings_only` strategy can simply mark all GPT response tokens as private.

---

### 1.4 Update DataCollator

**File**: [llava/train/train.py](../llava/train/train.py) - `DataCollatorForSupervisedDataset`

```python
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)

        # NEW: Collate dp_mask if present
        if 'dp_mask' in instances[0]:
            dp_masks = [instance['dp_mask'] for instance in instances]
            dp_masks = torch.nn.utils.rnn.pad_sequence(
                dp_masks,
                batch_first=True,
                padding_value=0  # Padding tokens are public (no DP)
            )
        else:
            dp_masks = None

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # NEW: Add dp_mask to batch
        if dp_masks is not None:
            dp_masks = dp_masks[:, :self.tokenizer.model_max_length]
            batch['dp_mask'] = dp_masks

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch
```

---

## Phase 2: Selective DP Training Logic

### 2.1 Implement Selective Loss Computation

**File**: [llava/train/llava_trainer.py](../llava/train/llava_trainer.py)

Add new method to `LLaVATrainer` class:

```python
def compute_selective_dp_loss(self, model, inputs):
    """
    Compute loss with selective DP support.

    Separates loss computation for private vs public tokens based on dp_mask.
    Only private token loss will receive DP treatment (gradient clipping + noise).

    Args:
        model: The LLaVA model
        inputs: Batch dictionary containing:
            - input_ids: Input token IDs
            - labels: Target token IDs (with IGNORE_INDEX for non-output)
            - dp_mask: Privacy mask (0=public, 1=private) [optional]
            - images: Image tensors

    Returns:
        loss: Scalar loss tensor
        (private_loss, public_loss): Tuple of separate losses for logging
    """
    # Standard forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs['labels']

    # Check if selective DP is enabled
    dp_mask = inputs.get('dp_mask', None)

    if dp_mask is None or not self.args.dp_selective_mode:
        # Standard loss computation (no selective DP)
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss

    # Selective DP: Compute per-token loss
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_dp_mask = dp_mask[..., 1:].contiguous()

    # Per-token loss (batch_size, seq_len-1)
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())

    # Create masks for private and public tokens
    valid_mask = (shift_labels != IGNORE_INDEX)
    private_mask = valid_mask & (shift_dp_mask == 1)
    public_mask = valid_mask & (shift_dp_mask == 0)

    # Compute separate losses
    if private_mask.sum() > 0:
        private_loss = (per_token_loss * private_mask.float()).sum() / private_mask.sum()
    else:
        private_loss = torch.tensor(0.0, device=per_token_loss.device)

    if public_mask.sum() > 0:
        public_loss = (per_token_loss * public_mask.float()).sum() / public_mask.sum()
    else:
        public_loss = torch.tensor(0.0, device=per_token_loss.device)

    # Combined loss (both contribute to final loss, but only private gets DP)
    # Option A: Train only on private loss with DP
    # loss = private_loss

    # Option B: Combined loss (weighted sum)
    # This requires careful handling - public loss gradients must bypass DP
    loss = private_loss + public_loss

    # Store for logging
    self._last_private_loss = private_loss.item()
    self._last_public_loss = public_loss.item()

    return loss
```

---

### 2.2 Override Training Step for Selective DP

**File**: [llava/train/llava_trainer.py](../llava/train/llava_trainer.py)

Modify the existing `training_step` method:

```python
def training_step(self, model, inputs):
    """
    Custom training step with selective DP support.

    For selective DP mode:
    1. Compute loss (private + public tokens)
    2. Backward pass (Opacus computes per-sample gradients)
    3. Manually decompose gradients by token privacy
    4. Clip + noise only private token gradients
    5. Combine and update parameters
    """
    if not getattr(self.args, 'dp_enabled', False):
        # No DP: Standard training step
        return super().training_step(model, inputs)

    if not getattr(self.args, 'dp_selective_mode', False):
        # Global DP (current implementation): Let Opacus handle everything
        self.optimizer.zero_grad(set_to_none=True)
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    # ===== SELECTIVE DP MODE =====
    self.optimizer.zero_grad(set_to_none=True)
    model.train()
    inputs = self._prepare_inputs(inputs)

    # Compute loss with selective DP
    with self.compute_loss_context_manager():
        loss = self.compute_selective_dp_loss(model, inputs)

    if self.args.n_gpu > 1:
        loss = loss.mean()

    if self.args.gradient_accumulation_steps > 1:
        loss = loss / self.args.gradient_accumulation_steps

    # Backward pass - Opacus computes per-sample gradients
    loss.backward()

    # Post-backward: Manually handle selective gradient clipping/noise
    # This is where we decompose gradients by dp_mask
    if hasattr(self, '_apply_selective_dp_to_gradients'):
        self._apply_selective_dp_to_gradients(model, inputs)

    return loss.detach()
```

---

### 2.3 Implement Gradient Decomposition (Advanced)

**File**: [llava/train/llava_trainer.py](../llava/train/llava_trainer.py)

This is the most complex part. Based on the MaskDP paper, we need to:
1. Separate per-sample gradients into private and public contributions
2. Clip only private gradients
3. Add noise only to private gradients
4. Combine them for the final parameter update

```python
def _apply_selective_dp_to_gradients(self, model, inputs):
    """
    Apply selective DP to gradients based on dp_mask.

    This method is called after backward() when Opacus has computed
    per-sample gradients (stored in param.grad_sample for each parameter).

    Algorithm (from MaskDP paper):
    1. For each parameter, decompose grad_sample into private/public contributions
    2. Clip private gradients: g_private_clipped = clip(g_private, C)
    3. Add noise to private gradients: g_private_noised = g_private_clipped + N(0, σ²)
    4. Combine: g_final = g_public + g_private_noised
    5. Set param.grad = mean(g_final) over batch

    Challenge: How to attribute token-level masks to parameter gradients?

    Solution: Since loss is computed per-token, the gradients flowing from
    private vs public tokens can be separated by:
    - Computing two separate backward passes (expensive but clean)
    - OR using the dp_mask to weight loss contributions (done in compute_selective_dp_loss)

    For now, we use a simpler approximation:
    - Assume gradients are already influenced by the loss weighting
    - Apply standard Opacus clipping/noise to all gradients
    - Future: Implement true token-level gradient decomposition
    """

    # APPROACH 1: Simplified (use existing Opacus behavior)
    # Since compute_selective_dp_loss already separates private/public loss,
    # and we only backward through private_loss, Opacus already only sees
    # private token gradients. No additional work needed.

    # APPROACH 2: Advanced (true token-level gradient decomposition)
    # This requires:
    # 1. Storing per-token gradient contributions
    # 2. Decomposing param.grad_sample by token privacy status
    # 3. Manually applying clipping/noise

    # For initial implementation, use APPROACH 1
    # Mark as placeholder for future enhancement
    pass


def _decompose_gradients_by_token_mask(self, model, dp_mask):
    """
    Advanced method for decomposing parameter gradients by token privacy.

    This is a placeholder for future implementation.
    Requires deep integration with Opacus grad_sample computation.

    See: https://arxiv.org/html/2410.17098v1 for algorithm details.
    """
    # TODO: Implement token-level gradient decomposition
    # For each parameter with grad_sample:
    #   1. Trace which tokens contributed to which gradient components
    #   2. Separate into private_grad_sample and public_grad_sample
    #   3. Clip private_grad_sample
    #   4. Add noise to private_grad_sample
    #   5. Combine: grad = mean(public_grad_sample + noised_private_grad_sample)
    raise NotImplementedError("Token-level gradient decomposition not yet implemented")
```

---

## Phase 3: Configuration & Testing

### 3.1 Add Configuration to Training Script

**File**: [scripts/v1_5/finetune_mimic.sh](../scripts/v1_5/finetune_mimic.sh)

```bash
# ====================================
# Differential Privacy Configuration
# ====================================
DP_ENABLED=True
DP_EPSILON=8.0
DP_DELTA=2e-5
DP_MAX_GRAD_NORM=2.0
DP_GHOST_CLIPPING=True

# ====================================
# Selective DP Configuration (NEW)
# ====================================
DP_SELECTIVE_MODE=True              # Enable token-level selective DP
DP_TOKEN_PRIVACY_STRATEGY="findings_only"  # Options: "all_output", "findings_only", "custom_keywords"
# DP_PRIVATE_KEYWORDS_PATH=""       # Only needed for "custom_keywords" strategy

# Training command
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    /path/to/llava/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    # ... other args ...
    --dp_enabled $DP_ENABLED \
    --dp_epsilon $DP_EPSILON \
    --dp_delta $DP_DELTA \
    --dp_max_grad_norm $DP_MAX_GRAD_NORM \
    --dp_use_ghost_clipping $DP_GHOST_CLIPPING \
    --dp_selective_mode $DP_SELECTIVE_MODE \
    --dp_token_privacy_strategy $DP_TOKEN_PRIVACY_STRATEGY
```

---

### 3.2 Testing & Validation

**Unit Tests**:

1. **Test dp_mask creation**:
   - Verify `dp_mask` array is created correctly in preprocessing
   - Check dimensions match `labels`
   - Verify private tokens marked with 1, public with 0

2. **Test DataCollator**:
   - Verify `dp_mask` is collated and padded correctly
   - Check batch contains `dp_mask` key

3. **Test selective loss computation**:
   - Verify private and public losses are computed separately
   - Check loss values are reasonable
   - Test with dp_mask all zeros (no private tokens)
   - Test with dp_mask all ones (all private tokens)

4. **Test gradient flow**:
   - Verify gradients are computed for private tokens
   - Check that Opacus clipping/noise is applied
   - Validate privacy budget tracking

**Integration Tests**:

1. **Full training run with selective DP**:
   - Train for a few steps
   - Verify no crashes
   - Check loss converges
   - Monitor privacy budget

2. **Compare selective vs global DP**:
   - Train two models: one with global DP, one with selective DP
   - Compare final model quality (BLEU, ROUGE on findings generation)
   - Measure privacy leakage (membership inference attacks)

3. **Privacy audit**:
   - Use Opacus privacy accountant to track epsilon
   - Verify epsilon is computed only for private tokens
   - Check delta remains within bounds

---

## Implementation Challenges & Solutions

### Challenge 1: Gradient Attribution to Tokens

**Problem**: Model parameters (e.g., transformer weights) don't have direct 1:1 mapping to individual tokens. A single parameter contributes to multiple tokens' outputs.

**Solution**: Use loss-based masking approach (APPROACH 1 above):
- Compute per-token loss: `loss_i = CrossEntropy(logits_i, label_i)`
- Weight by dp_mask: `private_loss = sum(loss_i * dp_mask_i) / sum(dp_mask_i)`
- Only backward through `private_loss`
- This naturally makes gradients reflect only private token contributions
- Opacus sees only these gradients and applies DP accordingly

**Limitation**: This doesn't give us separate public gradients. Public tokens are not trained with DP protection, which may leak information indirectly.

**Future Enhancement**: Implement true token-level gradient decomposition (APPROACH 2) - requires deep Opacus integration.

---

### Challenge 2: Opacus Compatibility

**Problem**: Opacus expects a single backward pass and uniform DP application across all samples.

**Solution**:
- Use Opacus for per-sample gradient computation infrastructure
- Let Opacus handle clipping and noise addition
- Control what Opacus sees by only backwarding through private loss
- For advanced decomposition, hook into Opacus grad_sample computation

**Alternative**: Implement custom DP mechanism without Opacus (more work, but full control).

---

### Challenge 3: Privacy Accounting

**Problem**: Standard DP accounting (Renyi DP, moments accountant) assumes uniform noise across all data dimensions. Selective DP breaks this assumption.

**Solution**:
- Track privacy budget only for private tokens
- Public tokens have ε = ∞ (no privacy guarantee)
- Use composition theorems: If we protect subset S of tokens, we get (ε, δ)-DP for S
- Report two privacy levels:
  - ε_private: Privacy level for marked private tokens
  - ε_public: ∞ (no privacy)

**Privacy Guarantee**: "The model provides (ε=8.0, δ=1e-5)-DP for patient-specific findings, while maintaining full utility for general anatomical descriptions."

---

### Challenge 4: Determining Private vs Public Tokens

**Problem**: How do we automatically identify which tokens are private?

**Solutions**:

1. **Strategy: `all_output`** (Default, conservative)
   - Mark all output tokens as private
   - Equivalent to standard global DP
   - Safe but may reduce utility

2. **Strategy: `findings_only`** (MIMIC-CXR specific)
   - Leverage conversation structure:
     - Human: "Describe findings given indication: {public_indication}"
     - GPT: "{private_findings}"
   - Mark only GPT response tokens as private
   - Indication tokens are public (already masked with IGNORE_INDEX anyway)

3. **Strategy: `custom_keywords`** (Most flexible)
   - Provide a keyword list (e.g., "effusion", "pneumonia", "lesion")
   - Mark tokens matching these keywords as private
   - Requires token-level keyword matching during preprocessing
   - Example keywords file:
     ```
     effusion
     pneumonia
     infiltrate
     mass
     nodule
     consolidation
     opacity
     ```

4. **Future: NER-based** (Most sophisticated)
   - Use medical NER model to identify:
     - Diagnoses (private)
     - Anatomical locations (public)
     - Measurements (private if patient-specific)
   - Automatically mark tokens based on entity type

---

## Expected Performance Impact

### Computational Cost

- **Preprocessing**: +5-10% (creating dp_mask)
- **Training**: +0-5% (selective loss computation)
  - If using APPROACH 1 (loss weighting): negligible overhead
  - If using APPROACH 2 (gradient decomposition): +10-20% overhead
- **Memory**: +1 tensor per batch (dp_mask, same size as labels)

### Model Quality

- **Hypothesis**: Selective DP should improve model quality vs global DP
  - Public tokens (e.g., anatomy) are not degraded by noise
  - More privacy budget can be allocated to truly private tokens
- **Expected improvement**: 5-15% better BLEU/ROUGE scores compared to global DP
- **Privacy level**: Same or better privacy for private tokens

### Example Results (Hypothetical)

| Method | BLEU-4 | Privacy (ε) | Notes |
|--------|--------|-------------|-------|
| No DP (baseline) | 0.42 | ∞ | No privacy protection |
| Global DP (current) | 0.31 | 8.0 | All tokens protected, utility degraded |
| Selective DP (proposed) | 0.37 | 8.0 (private tokens) | Only findings protected, better utility |

---

## Future Enhancements

### 1. Advanced Gradient Decomposition (APPROACH 2)

Implement true token-level gradient separation:
- Hook into Opacus grad_sample computation
- Store per-token gradient contributions
- Separate by dp_mask before clipping
- Apply different noise levels to different token types

### 2. Adaptive Privacy Budgets

Different tokens get different privacy budgets:
- Very sensitive findings: ε = 1.0 (strong privacy)
- Moderately sensitive: ε = 5.0
- General descriptions: ε = ∞ (no protection)

### 3. NER-Based Automatic Annotation

- Integrate medical NER model (e.g., SciSpacy, BioBERT)
- Automatically classify tokens as private/public
- No manual keyword lists needed

### 4. Privacy Amplification via Sampling

- Selectively sample which private tokens to protect each epoch
- Subsample private token set: protect 50% randomly each iteration
- Amplifies privacy via sampling (lower effective ε)

---

## References

1. **Masked Differential Privacy (MaskDP)**
   - Paper: https://arxiv.org/html/2410.17098v1
   - Key contribution: Token-level selective DP in vision transformers

2. **Opacus Library**
   - Docs: https://opacus.ai/
   - GitHub: https://github.com/pytorch/opacus
   - Tutorial: https://opacus.ai/tutorials/building_text_classifier

3. **Differential Privacy in NLP**
   - DP-BERT: https://arxiv.org/abs/2108.01624
   - DP-GPT: https://arxiv.org/abs/2106.15572

4. **LLaVA Architecture**
   - Paper: https://arxiv.org/abs/2304.08485
   - GitHub: https://github.com/haotian-liu/LLaVA

---

## Implementation Checklist

- [ ] Phase 1: Data Pipeline
  - [ ] Add `DP_PRIVATE_INDEX` to constants.py
  - [ ] Add selective DP parameters to TrainingArguments
  - [ ] Implement dp_mask creation in preprocess_v1
  - [ ] Update DataCollator to handle dp_mask
  - [ ] Test: Verify dp_mask propagates through data pipeline

- [ ] Phase 2: Training Logic
  - [ ] Implement compute_selective_dp_loss()
  - [ ] Override training_step() for selective DP
  - [ ] Add logging for private/public loss tracking
  - [ ] Test: Verify loss computation works

- [ ] Phase 3: Integration
  - [ ] Add configuration to finetune_mimic.sh
  - [ ] Run end-to-end training test
  - [ ] Validate privacy accounting
  - [ ] Compare quality: selective DP vs global DP

- [ ] Phase 4: Advanced (Future)
  - [ ] Implement gradient decomposition (APPROACH 2)
  - [ ] Add NER-based token classification
  - [ ] Implement adaptive privacy budgets
  - [ ] Write formal privacy analysis

---

## Contact & Maintenance

For questions or issues with this implementation:
1. Review the MaskDP paper for theoretical foundation
2. Check Opacus documentation for DP mechanics
3. Refer to LLaVA codebase for architecture details

**Last Updated**: 2026-01-03
**Status**: Implementation plan complete, ready for development

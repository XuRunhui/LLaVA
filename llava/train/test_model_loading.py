#!/usr/bin/env python3
"""
Test script to load LLaVA-Med v1.5 model using LLaVA v1.5 codebase.

This script tests ONLY the model loading part before running full training.
"""

import sys
sys.path.insert(0, '../LLaVA')  # Add LLaVA repo to path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

print("=" * 80)
print("TESTING: LLaVA-Med v1.5 Model Loading")
print("=" * 80)

# Step 0: Import LLaVA model classes to register them with transformers
print("\n[Step 0] Registering LLaVA model classes with transformers...")
try:
    from llava.model import LlavaMistralForCausalLM, LlavaMistralConfig
    print(f"✓ LlavaMistralForCausalLM and LlavaMistralConfig imported successfully")
    print(f"  Note: This registers 'llava_mistral' model type with transformers")
except Exception as e:
    print(f"✗ Failed to import LLaVA classes: {e}")
    sys.exit(1)

# Step 1: Load config
print("\n[Step 1] Loading model config...")
model_path = "microsoft/llava-med-v1.5-mistral-7b"

try:
    config = AutoConfig.from_pretrained(model_path)
    print(f"✓ Config loaded successfully")
    print(f"  Model type: {config.model_type}")
    print(f"  Architectures: {config.architectures}")

    # Check key config parameters
    if hasattr(config, 'mm_projector_type'):
        print(f"  mm_projector_type: {config.mm_projector_type}")
    if hasattr(config, 'mm_vision_tower'):
        print(f"  mm_vision_tower: {config.mm_vision_tower}")
    if hasattr(config, 'mm_use_im_start_end'):
        print(f"  mm_use_im_start_end: {config.mm_use_im_start_end}")

except Exception as e:
    print(f"✗ Failed to load config: {e}")
    sys.exit(1)

# Step 2: Load tokenizer
print("\n[Step 2] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print(f"✓ Tokenizer loaded successfully")
    print(f"  Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Model max length: {tokenizer.model_max_length}")
except Exception as e:
    print(f"✗ Failed to load tokenizer: {e}")
    sys.exit(1)

# Step 3: Verify model class is available (already imported in Step 0)
print("\n[Step 3] Verifying LLaVA model class...")
print(f"✓ LlavaMistralForCausalLM is available for model loading")

# Step 4: Load model (this is the critical part)
print("\n[Step 4] Loading model...")
print("  This may take a few minutes...")

try:
    # Load with minimal resources for testing
    model = LlavaMistralForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"  # Automatically distribute across available devices
    )
    print(f"✓ Model loaded successfully!")
    print(f"  Model class: {model.__class__.__name__}")
    print(f"  Device: {model.device}")

    # Check model components
    print("\n[Step 5] Checking model components...")

    # Check vision tower
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        print(f"  ✓ Vision tower: {vision_tower.__class__.__name__}")
        if hasattr(vision_tower, 'is_loaded'):
            print(f"    - Is loaded: {vision_tower.is_loaded}")

    # Check mm_projector
    if hasattr(model.get_model(), 'mm_projector'):
        mm_projector = model.get_model().mm_projector
        print(f"  ✓ MM Projector: {mm_projector.__class__.__name__}")

    # Check language model
    print(f"  ✓ Language model backbone: Mistral-7B")

    print("\n" + "=" * 80)
    print("SUCCESS: Model loading test completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. The model loads correctly with LLaVA codebase")
    print("2. Now you can proceed with training setup")
    print("3. Key arguments needed for training:")
    print(f"   --model_name_or_path {model_path}")
    print(f"   --version mistral_instruct")
    print(f"   --vision_tower {config.mm_vision_tower}")
    print(f"   --mm_projector_type {config.mm_projector_type}")
    print(f"   --mm_use_im_start_end {config.mm_use_im_start_end}")

except Exception as e:
    print(f"✗ Failed to load model: {e}")
    print("\nError details:")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
# from llava.eval.run_llava import eval_model

# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )
#!/usr/bin/env python3
"""
Generate conversations for MIA attacks using LLaVA-Med v1.5 (Mistral-based)
Adapted from vlm_mia/LLaVA/conversation_llava.py
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import sys
sys.path.insert(0, '../LLaVA')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def normalize_single_image_token(text, use_im_start_end: bool):
    """Strip and normalize image tokens"""
    text = text.replace(DEFAULT_IM_START_TOKEN, "").replace(DEFAULT_IM_END_TOKEN, "")
    text_wo_img = text.replace(DEFAULT_IMAGE_TOKEN, "").lstrip()
    prefix = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN) if use_im_start_end \
             else DEFAULT_IMAGE_TOKEN
    return f"{prefix}\n{text_wo_img}"


def load_image(image_file):
    """Load image from path"""
    return Image.open(image_file).convert('RGB')


def main(args):
    # Initialize
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    print(f"Loading model from: {model_path}")
    if args.model_base:
        print(f"Using base model: {args.model_base}")

    # Load model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # Ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    torch.set_grad_enabled(False)

    # Load vision tower
    vt = model.get_vision_tower()
    if vt is not None:
        needs_load = getattr(vt, "is_loaded", None)
        if needs_load is None or needs_load is False:
            vt.load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vt.to(device=device, dtype=torch.float16)
        image_processor = vt.image_processor

    if image_processor is None:
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    model = model.to(device)

    # Load input data
    print(f"Loading data from: {args.input_json_path}")
    with open(args.input_json_path, 'r') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Check for existing results
    if os.path.exists(args.output_json_path):
        with open(args.output_json_path, 'r') as f:
            results = json.load(f)
        completed_ids = {item['image_id'] for item in results}
        print(f"Found {len(completed_ids)} completed samples, resuming...")
    else:
        os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)
        results = []
        completed_ids = set()

    # Process each sample
    errors = []
    for idx, item in enumerate(tqdm(data, desc="Generating conversations")):
        item_id = item.get('id', item.get('question_id', idx))

        if item_id in completed_ids:
            continue

        try:
            # Load image
            image_path = os.path.join(args.image_folder, item['image'])
            image = load_image(image_path)
            image_tensor = process_images([image], image_processor, model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.float16, device=device)

            # Prepare result structure
            conversation_result = {"image_id": item_id}

            # Generate for each temperature
            for temperature in args.temperatures:
                conversation_result[f"conversations_{temperature}"] = []

                # Process conversations
                for conv_item in item["conversations"]:
                    if conv_item["from"] == "human":
                        # Extract question
                        question = conv_item["value"].replace("\n<image>", "").replace("<image>\n", "").strip()
                        conversation_result[f"conversations_{temperature}"].append({
                            "from": "human",
                            "value": question
                        })

                        # Generate responses (with repetitions for image-only attack)
                        for repeat_idx in range(args.repeat):
                            # Prepare prompt
                            qs = normalize_single_image_token(question, model.config.mm_use_im_start_end)

                            conv = conv_templates[args.conv_mode].copy()
                            conv.append_message(conv.roles[0], qs)
                            conv.append_message(conv.roles[1], None)
                            prompt = conv.get_prompt()

                            # Tokenize
                            input_ids = tokenizer_image_token(
                                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                            ).unsqueeze(0).to(device)

                            # Generate
                            with torch.inference_mode():
                                output_ids = model.generate(
                                    input_ids,
                                    images=image_tensor,
                                    image_sizes=[image.size],
                                    do_sample=True if temperature > 0 else False,
                                    temperature=temperature if temperature > 0 else 1.0,
                                    max_new_tokens=args.max_new_tokens,
                                    use_cache=True
                                )

                            # Decode
                            outputs = tokenizer.batch_decode(
                                output_ids[:, input_ids.shape[1]:],
                                skip_special_tokens=True
                            )[0].strip()

                            # Save response
                            repeat_label = f"vlm_{repeat_idx + 1}" if args.repeat > 1 else "vlm"
                            conversation_result[f"conversations_{temperature}"].append({
                                "from": repeat_label,
                                "value": outputs
                            })

                    elif conv_item["from"] == "gpt":
                        # Save ground truth
                        conversation_result[f"conversations_{temperature}"].append({
                            "from": "ground truth",
                            "value": conv_item["value"]
                        })

            results.append(conversation_result)

            # Save checkpoint every 100 samples
            if (idx + 1) % 100 == 0:
                with open(args.output_json_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nSaved checkpoint at {idx + 1} samples")

        except Exception as e:
            error_msg = f"Error processing {item_id}: {str(e)}"
            print(f"\n{error_msg}")
            errors.append({'image_id': item_id, 'error': error_msg})

    # Final save
    with open(args.output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Conversation generation complete!")
    print(f"Results saved to: {args.output_json_path}")
    print(f"Total samples: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"{'='*80}")

    if errors:
        error_file = args.output_json_path.replace('.json', '_errors.json')
        with open(error_file, 'w') as f:
            json.dump({"errors": errors}, f, indent=2)
        print(f"Errors saved to: {error_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-base", type=str, default=None, help="Base model path (for LoRA)")
    parser.add_argument("--vision-tower", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct",
                        help="Conversation template")
    parser.add_argument("--input-json-path", type=str, required=True,
                        help="Path to input data (member or non-member)")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-json-path", type=str, required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.1],
                        help="List of temperatures")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of repetitions (>1 for image-only attack)")
    parser.add_argument("--max-new-tokens", type=int, default=512)

    args = parser.parse_args()
    main(args)

"""
MIMIC-CXR Evaluation Script

Evaluate LLaVA model on MIMIC-CXR dev/test data with loss and perplexity computation.
"""

import argparse
import torch
import torch.nn as nn
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from mimic_data_utils import load_mimic_cxr_data


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk of a list split into n chunks"""
    chunks = split_list(lst, n)
    return chunks[k]


class MIMICEvalDataset(Dataset):
    """
    Dataset for MIMIC-CXR evaluation with loss computation support.

    Returns both input_ids for generation and labels for loss computation.
    """

    def __init__(self, data_list, image_folder, tokenizer, image_processor, model_config, args):
        self.data_list = data_list
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.args = args

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_file = sample["image"]
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        # Prepare question with image token
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        # Create conversation for input (question only)
        conv_input = conv_templates[self.args.conv_mode].copy()
        conv_input.append_message(conv_input.roles[0], qs)
        conv_input.append_message(conv_input.roles[1], None)
        prompt_input = conv_input.get_prompt()

        # Create conversation for labels (question + ground truth answer)
        conv_labels = conv_templates[self.args.conv_mode].copy()
        conv_labels.append_message(conv_labels.roles[0], qs)
        conv_labels.append_message(conv_labels.roles[1], ground_truth)
        prompt_labels = conv_labels.get_prompt()

        # Tokenize input (for generation)
        input_ids = tokenizer_image_token(prompt_input, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        # Tokenize full prompt (for loss computation)
        full_input_ids = tokenizer_image_token(prompt_labels, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        # Create labels: mask the question part, only compute loss on answer
        labels = full_input_ids.clone()

        # Find where the assistant's response starts
        # The response starts after the second role marker
        sep = conv_labels.sep + conv_labels.roles[1] + ": "

        # Tokenize the question part to find where to start masking
        question_part = prompt_labels.split(sep)[0] + sep
        question_tokens = tokenizer_image_token(question_part, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        question_len = len(question_tokens)

        # Mask question tokens with IGNORE_INDEX
        labels[:question_len] = IGNORE_INDEX

        # Load and process image
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # Return metadata for logging
        metadata = {
            'id': sample['id'],
            'image': image_file,
            'question': question,
            'ground_truth': ground_truth,
            'view': sample.get('view', 'unknown'),
            'generate_method': sample.get('generate_method', 'unknown'),
        }

        return input_ids, full_input_ids, labels, image_tensor, image.size, metadata


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Note: We use batch_size=1 for simplicity, so no actual batching needed.
    """
    input_ids, full_input_ids, labels, image_tensors, image_sizes, metadata_list = zip(*batch)

    # Since batch_size=1, just return first elements (but keep batch dimension)
    return (
        input_ids[0],
        full_input_ids[0].unsqueeze(0),
        labels[0].unsqueeze(0),
        image_tensors[0].unsqueeze(0),
        list(image_sizes),
        metadata_list[0]
    )


def compute_loss_and_perplexity(model, input_ids, labels, images, image_sizes):
    """
    Compute cross-entropy loss and perplexity for a sample.

    Args:
        model: LLaVA model
        input_ids: Input token IDs (including question + answer)
        labels: Labels with question masked by IGNORE_INDEX
        images: Image tensors
        image_sizes: Original image sizes

    Returns:
        loss: Cross-entropy loss (float)
        perplexity: Perplexity = exp(loss) (float)
        num_tokens: Number of valid tokens used for loss computation
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        img_dtype = next(model.parameters()).dtype
        # Forward pass with labels to compute loss
        outputs = model(
            input_ids=input_ids.to(device),
            labels=labels.to(device),
            images=images.to(device=device, dtype=img_dtype),
            image_sizes=image_sizes,
            return_dict=True
        )

        loss = outputs.loss

        # Compute perplexity
        perplexity = torch.exp(loss)

        # Count valid tokens (not IGNORE_INDEX)
        valid_tokens = (labels != IGNORE_INDEX).sum()

    return loss.item(), perplexity.item(), valid_tokens.item()


def generate_prediction(model, tokenizer, input_ids, images, image_sizes, args):
    """
    Generate prediction using model.generate().

    Args:
        model: LLaVA model
        tokenizer: Tokenizer
        input_ids: Input token IDs (question only, no answer)
        images: Image tensors
        image_sizes: Original image sizes
        args: Arguments with generation parameters

    Returns:
        prediction: Generated text (string)
    """
    with torch.inference_mode():
        device = next(model.parameters()).device
        img_dtype = next(model.parameters()).dtype
        output_ids = model.generate(
            input_ids.unsqueeze(0).to(device),
            images=images.to(device=device, dtype=img_dtype),
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )

    # Decode and strip special tokens
    prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return prediction


def eval_model(args):
    """
    Main evaluation function.

    Loads MIMIC-CXR data, evaluates model, computes loss/perplexity, and saves results.
    """
    # Initialize model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # Force LLaVA LoRA loading if model_base is provided
    # The model builder requires BOTH "llava" and "lora" in the model name
    if args.model_base:
        # Add "llava" if not present (required to enter LLaVA loading path)
        if 'llava' not in model_name.lower():
            model_name = 'llava-' + model_name
        # Add "lora" if not present (required to enter LoRA loading path)
        if 'lora' not in model_name.lower():
            model_name = model_name + '-lora'
        print(f"Model name updated to: {model_name} (to trigger LLaVA LoRA loading)")

    print(f"Loading model from: {model_path}")
    if args.model_base:
        print(f"Using base model: {args.model_base}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device_map="auto"
    )

    # Optional dtype summary (no conversion)
    print("\nDtype summary (no conversion):")
    dtype_counts = {}
    for name, param in model.named_parameters():
        dtype_str = str(param.dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
        if 'lm_head' in name or 'embed' in name or 'mm_projector' in name:
            print(f"  {name[:60]:60s} dtype={param.dtype}")
    print(f"\nDtype summary: {dtype_counts}\n")

    model.eval()  # Set to evaluation mode

    # Load MIMIC-CXR data with filtering
    print(f"\nLoading MIMIC-CXR {args.split} data...")
    print(f"Generation methods: {args.generation_methods}")
    data_list = load_mimic_cxr_data(
        data_path=args.data_file,
        split=args.split,
        filter_views=args.filter_views,
        include_reason=args.include_reason,
        generation_methods=args.generation_methods,
        verbose=True
    )

    # Apply chunking if specified (for distributed evaluation)
    if args.num_chunks > 1:
        data_list = get_chunk(data_list, args.num_chunks, args.chunk_idx)
        print(f"Processing chunk {args.chunk_idx + 1}/{args.num_chunks} ({len(data_list)} samples)")

    # Create dataset and dataloader
    dataset = MIMICEvalDataset(
        data_list,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        args
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time
        num_workers=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Prepare output file
    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize output file (overwrite if exists)
    with open(output_file, 'w') as f:
        pass

    # Initialize metrics
    total_loss = 0.0
    total_perplexity = 0.0
    total_tokens = 0
    num_samples = 0

    print(f"\nStarting evaluation on {len(dataset)} samples...")
    print(f"Results will be saved to: {output_file}")
    print("=" * 80)

    # Evaluation loop
    for batch_idx, (input_ids, full_input_ids, labels, images, image_sizes, metadata) in enumerate(tqdm(dataloader)):
        # try:
        # Compute loss and perplexity
        loss, perplexity, num_tokens = compute_loss_and_perplexity(
            model, full_input_ids, labels, images, image_sizes
        )

        # Generate prediction
        prediction = generate_prediction(
            model, tokenizer, input_ids, images, image_sizes, args
        )

        # Accumulate metrics
        total_loss += loss * num_tokens
        total_perplexity += perplexity * num_tokens
        total_tokens += num_tokens
        num_samples += 1

        # Prepare result
        result = {
            "id": metadata["id"],
            "image": metadata["image"],
            "question": metadata["question"],
            "ground_truth": metadata["ground_truth"],
            "prediction": prediction,
            "loss": loss,
            "perplexity": perplexity,
            "num_tokens": num_tokens,
            "view": metadata["view"],
            "generate_method": metadata["generate_method"],
            "answer_id": shortuuid.uuid(),
            "model_id": model_name,
        }

        # Write result incrementally (in case of crashes)
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')

        # except Exception as e:
        #     print(f"\nError processing sample {metadata['id']}: {e}")
        #     continue

    # Compute average metrics
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        avg_perplexity = total_perplexity / total_tokens
    else:
        avg_loss = 0.0
        avg_perplexity = 0.0

    # Save summary
    summary = {
        "split": args.split,
        "model_path": model_path,
        "model_name": model_name,
        "num_samples": num_samples,
        "total_tokens": total_tokens,
        "average_loss": avg_loss,
        "average_perplexity": avg_perplexity,
        "generation_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
        },
        "data_config": {
            "filter_views": args.filter_views,
            "include_reason": args.include_reason,
            "generation_methods": args.generation_methods,
        }
    }

    summary_file = output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Split:                  {args.split}")
    print(f"Generation methods:     {args.generation_methods}")
    print(f"Filter views (PA/AP):   {args.filter_views}")
    print(f"Include reason:         {args.include_reason}")
    print(f"Samples processed:      {num_samples:,}")
    print(f"Total tokens:           {total_tokens:,}")
    print(f"Average loss:           {avg_loss:.4f}")
    print(f"Average perplexity:     {avg_perplexity:.4f}")
    print("")
    print(f"Results saved to:       {output_file}")
    print(f"Summary saved to:       {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA model on MIMIC-CXR data")

    # Model arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--model-base", type=str, default=None, help="Base model path (for LoRA models)")
    parser.add_argument("--conv-mode", type=str, default="v1", help="Conversation mode")

    # Data arguments
    parser.add_argument("--data-file", type=str, required=True, help="Path to MIMIC-CXR data file (JSON)")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"], help="Dataset split")
    parser.add_argument("--filter-views", type=lambda x: x.lower() == 'true', default=True, help="Filter to PA/AP views only")
    parser.add_argument("--include-reason", type=lambda x: x.lower() == 'true', default=True, help="Include clinical indication in prompts")
    parser.add_argument("--generation-methods", type=str, default="rule-based", choices=["all", "gpt4", "rule-based"],
                        help="Which generation method to use: 'all', 'gpt4', or 'rule-based' (default: rule-based for test/dev)")

    # Output arguments
    parser.add_argument("--output-file", type=str, required=True, help="Path to output JSONL file")

    # Chunking for distributed evaluation
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of chunks to split data into")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Index of current chunk (0-indexed)")

    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")

    args = parser.parse_args()

    eval_model(args)

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from typing import Dict, List, Optional, Tuple

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_question_format(line: Dict) -> Tuple:
    """
    Parse question from different formats.
    Supports:
    1. Simple format: {"text": "...", "answer": "..."}
    2. Conversations format: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

    Returns: (question_id, image_file, question_text, ground_truth)
    """
    # Get question ID
    question_id = line.get("question_id", line.get("id", None))

    # Get image file
    image_file = line.get("image", None)

    # Parse question and answer based on format
    if "conversations" in line:
        # Conversations format
        conversations = line["conversations"]

        # Extract question from human turn
        question_text = None
        ground_truth = None

        for conv in conversations:
            if conv.get("from") == "human":
                # Remove <image> token if present
                question_text = conv.get("value", "").replace("<image>", "").strip()
            elif conv.get("from") == "gpt":
                ground_truth = conv.get("value", "").strip()

        if question_text is None:
            raise ValueError("No 'human' conversation found")

    else:
        # Simple format
        question_text = line.get("text", None)
        ground_truth = line.get("answer", line.get("ground_truth", None))

        if question_text is None:
            raise ValueError("No 'text' field found")

    return question_id, image_file, question_text, ground_truth


def compute_loss_and_perplexity(model, tokenizer, input_ids, images, image_sizes, ground_truth_text):
    """
    Compute loss and perplexity for a given input.

    Args:
        model: The LLaVA model
        tokenizer: The tokenizer
        input_ids: Input token IDs (question)
        images: Processed image tensors
        image_sizes: Image sizes
        ground_truth_text: The expected answer text

    Returns:
        Dictionary containing loss, perplexity, and token-level metrics
    """
    # Tokenize the ground truth answer
    answer_ids = tokenizer(
        ground_truth_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=tokenizer.model_max_length
    ).input_ids[0]

    # Get device from input_ids
    device = input_ids.device

    # Combine input and answer to create full sequence
    # Remove the BOS token from answer_ids if present
    if answer_ids[0] == tokenizer.bos_token_id:
        answer_ids = answer_ids[1:]

    # Move answer_ids to same device as input_ids before concatenation
    answer_ids = answer_ids.to(device)

    # Create full input: [question_tokens, answer_tokens]
    full_input_ids = torch.cat([input_ids.squeeze(0), answer_ids], dim=0).unsqueeze(0)

    # Create labels: ignore question tokens, keep answer tokens
    labels = full_input_ids.clone()
    labels[:, :input_ids.shape[1]] = IGNORE_INDEX  # Mask question tokens

    # Ensure images are on the correct device
    if images is not None:
        images = images.to(device)

    # Forward pass with labels to compute loss
    with torch.no_grad():
        outputs = model(
            input_ids=full_input_ids,
            labels=labels,
            images=images.unsqueeze(0) if images is not None else None,
            image_sizes=image_sizes,
            use_cache=False,
            return_dict=True
        )

    loss = outputs.loss.item() if outputs.loss is not None else float('inf')

    # Compute perplexity
    perplexity = math.exp(loss) if loss < 100 else float('inf')

    # Get token-level loss (optional, for detailed analysis)
    logits = outputs.logits

    # Align labels with logits sequence length (model may expand due to image tokens)
    logits_seq_len = logits.size(1)
    labels_seq_len = labels.size(1)

    if logits_seq_len != labels_seq_len:
        # Adjust labels to match logits length
        if logits_seq_len > labels_seq_len:
            # Pad labels with IGNORE_INDEX
            padding = torch.full(
                (labels.size(0), logits_seq_len - labels_seq_len),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat([labels, padding], dim=1)
        else:
            # Truncate labels
            labels = labels[:, :logits_seq_len]

    # Compute token-level metrics
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Calculate cross entropy per token
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    # Filter out ignored tokens
    valid_tokens = shift_labels.view(-1) != IGNORE_INDEX
    valid_token_losses = token_losses[valid_tokens]

    metrics = {
        'loss': loss,
        'perplexity': perplexity,
        'num_tokens': valid_tokens.sum().item(),
        'avg_token_loss': valid_token_losses.mean().item() if len(valid_token_losses) > 0 else float('inf'),
        'max_token_loss': valid_token_losses.max().item() if len(valid_token_losses) > 0 else float('inf'),
        'min_token_loss': valid_token_losses.min().item() if len(valid_token_losses) > 0 else float('inf'),
    }

    return metrics


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # Check if this is a LLaVA model by looking for config in model_path
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            if 'llava' in str(config.get('architectures', [])).lower():
                if 'llava' not in model_name.lower():
                    model_name = 'llava-' + model_name

    # If not found in model_path and model_base is provided, check base model
    elif args.model_base:
        base_model_path = os.path.expanduser(args.model_base)
        base_model_name = get_model_name_from_path(base_model_path)

        if 'llava' in base_model_name.lower():
            model_name = base_model_name
        else:
            base_config_path = os.path.join(base_model_path, 'config.json')
            if os.path.exists(base_config_path):
                with open(base_config_path, 'r') as f:
                    config = json.load(f)
                    if 'llava' in str(config.get('architectures', [])).lower():
                        model_name = 'llava-' + base_model_name

    print(f"Using model_name: {model_name}")
    print(f"Loading from: {model_path}")
    if args.model_base:
        print(f"With base model: {args.model_base}")

    # Load model with LoRA support
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device_map='auto'
    )

    # Load questions - support both JSON array and JSONL
    question_file_path = os.path.expanduser(args.question_file)

    try:
        # Try loading as JSON array first
        with open(question_file_path, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                questions = json.loads(content)
            else:
                # Load as JSONL
                questions = [json.loads(line) for line in content.split('\n') if line.strip()]
    except json.JSONDecodeError:
        # Fallback to line-by-line JSONL parsing
        questions = [json.loads(q) for q in open(question_file_path, "r") if q.strip()]

    print(f"Loaded {len(questions)} questions from {args.question_file}")

    # Filter by view if requested
    if args.filter_views:
        original_count = len(questions)
        questions = [q for q in questions if q.get('view', '').upper() in ['PA', 'AP']]
        print(f"Filtered dataset: {len(questions)}/{original_count} samples with view='PA' or 'AP'")

    # Chunk for parallel processing
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Metrics tracking
    all_losses = []
    all_perplexities = []
    samples_with_metrics = 0
    parse_errors = 0

    for line in tqdm(questions, desc="Evaluating"):
        try:
            # Parse question format
            idx, image_file, qs, ground_truth = parse_question_format(line)

            if idx is None:
                print(f"Warning: Skipping sample without question_id/id")
                parse_errors += 1
                continue

            if image_file is None:
                print(f"Warning: Skipping sample {idx} without image field")
                parse_errors += 1
                continue

            cur_prompt = qs
        except Exception as e:
            print(f"Error parsing question: {e}")
            parse_errors += 1
            continue

        # Add reason field if requested
        if args.add_reason and "reason" in line and line["reason"]:
            qs = f"{qs}\nWith the indication: {line['reason']}"

        # Prepare input
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Load and process image
        try:
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0]
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            continue

        # Generate response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Compute metrics if ground truth is available and compute_metrics is enabled
        metrics = None
        if args.compute_metrics and ground_truth is not None:
            try:
                metrics = compute_loss_and_perplexity(
                    model,
                    tokenizer,
                    input_ids,
                    image_tensor.half(),
                    [image.size],
                    ground_truth
                )
                all_losses.append(metrics['loss'])
                all_perplexities.append(metrics['perplexity'])
                samples_with_metrics += 1
            except Exception as e:
                print(f"Warning: Could not compute metrics for question {idx}: {e}")
                metrics = None

        ans_id = shortuuid.uuid()
        result = {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }

        # Add ground truth if available
        if ground_truth is not None:
            result["ground_truth"] = ground_truth

        # Add metrics if computed
        if metrics is not None:
            result["metrics"] = metrics

        # Add metadata fields if present
        if "view" in line:
            result["view"] = line["view"]
        if "reason" in line:
            result["reason"] = line["reason"]
        if "orientation" in line:
            result["orientation"] = line["orientation"]
        if "chexpert_labels" in line:
            result["chexpert_labels"] = line["chexpert_labels"]

        ans_file.write(json.dumps(result) + "\n")
        ans_file.flush()

    ans_file.close()

    # Print summary statistics
    if parse_errors > 0:
        print(f"\nWarning: {parse_errors} samples had parsing errors")

    if samples_with_metrics > 0:
        print("\n" + "="*50)
        print("EVALUATION METRICS SUMMARY")
        print("="*50)
        print(f"Total samples evaluated: {len(questions)}")
        print(f"Samples with metrics: {samples_with_metrics}")
        print(f"\nLoss Statistics:")
        print(f"  Mean Loss: {np.mean(all_losses):.4f}")
        print(f"  Median Loss: {np.median(all_losses):.4f}")
        print(f"  Std Loss: {np.std(all_losses):.4f}")
        print(f"  Min Loss: {np.min(all_losses):.4f}")
        print(f"  Max Loss: {np.max(all_losses):.4f}")
        print(f"\nPerplexity Statistics:")
        valid_perplexities = [p for p in all_perplexities if p != float('inf')]
        if valid_perplexities:
            print(f"  Mean Perplexity: {np.mean(valid_perplexities):.4f}")
            print(f"  Median Perplexity: {np.median(valid_perplexities):.4f}")
            print(f"  Std Perplexity: {np.std(valid_perplexities):.4f}")
            print(f"  Min Perplexity: {np.min(valid_perplexities):.4f}")
            print(f"  Max Perplexity: {np.max(valid_perplexities):.4f}")
        print("="*50)

        # Save summary to file
        summary_file = args.answers_file.replace('.jsonl', '_summary.json')
        summary = {
            'total_samples': len(questions),
            'samples_with_metrics': samples_with_metrics,
            'parse_errors': parse_errors,
            'loss': {
                'mean': float(np.mean(all_losses)),
                'median': float(np.median(all_losses)),
                'std': float(np.std(all_losses)),
                'min': float(np.min(all_losses)),
                'max': float(np.max(all_losses)),
            },
            'perplexity': {
                'mean': float(np.mean(valid_perplexities)) if valid_perplexities else None,
                'median': float(np.median(valid_perplexities)) if valid_perplexities else None,
                'std': float(np.std(valid_perplexities)) if valid_perplexities else None,
                'min': float(np.min(valid_perplexities)) if valid_perplexities else None,
                'max': float(np.max(valid_perplexities)) if valid_perplexities else None,
            },
            'config': {
                'model_path': args.model_path,
                'model_base': args.model_base,
                'question_file': args.question_file,
                'filter_views': args.filter_views,
                'add_reason': args.add_reason,
            }
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the finetuned model (LoRA or full model)")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path (required for LoRA models)")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to folder containing evaluation images")
    parser.add_argument("--question-file", type=str, required=True,
                        help="Path to JSON/JSONL file with questions")
    parser.add_argument("--answers-file", type=str, required=True,
                        help="Path to save answers and metrics")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                        help="Conversation mode")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of chunks for parallel processing")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Chunk index for parallel processing")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p for generation")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--compute-metrics", action="store_true",
                        help="Compute loss and perplexity (requires ground truth answers)")
    parser.add_argument("--filter-views", action="store_true",
                        help="Filter to only evaluate PA/AP views")
    parser.add_argument("--add-reason", action="store_true",
                        help="Add 'reason' field to prompts if available")

    args = parser.parse_args()

    eval_model(args)

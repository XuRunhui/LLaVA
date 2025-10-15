import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader


def normalize_single_image_token(text, use_im_start_end: bool):
    """
    Strip any existing image-start/end wrappers and image tokens,
    then prepend exactly one properly-formatted image token.
    """
    # Strip existing image tokens
    text = text.replace(DEFAULT_IM_START_TOKEN, "").replace(DEFAULT_IM_END_TOKEN, "")
    text_wo_img = text.replace(DEFAULT_IMAGE_TOKEN, "").lstrip()

    # Build the desired single image prefix
    prefix = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN) if use_im_start_end \
             else DEFAULT_IMAGE_TOKEN

    # Return exactly one image token + newline before the question text
    return f"{prefix}\n{text_wo_img}"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk of n chunks"""
    return split_list(lst, n)[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def _get_question_text(self, line):
        """Support both schemas: conversations format and direct text field"""
        if "conversations" in line:
            return line["conversations"][0]["value"]
        return line["text"]

    def _get_image_name(self, line):
        """Get image filename from question"""
        return line["image"]

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = self._get_image_name(line)
        qs_raw = self._get_question_text(line)

        # Normalize to exactly one image token
        qs = normalize_single_image_token(qs_raw, self.model_config.mm_use_im_start_end)

        # Create conversation using template
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Load and preprocess image
        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # Tokenize prompt
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        # Return without batch dimension (will be added by collate_fn)
        return input_ids.squeeze(0), image_tensor, image.size, line

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    """Collate batch of (input_ids, image_tensor, image_size, line_dict)"""
    input_ids, image_tensors, image_sizes, lines = zip(*batch)

    # Pad input_ids to same length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)

    # Stack image tensors
    image_tensors = torch.stack(image_tensors, dim=0)

    return input_ids, image_tensors, image_sizes, lines


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode,
                       batch_size=1, num_workers=0):
    """Create DataLoader for evaluation"""
    assert batch_size == 1, "batch_size must be 1 for current prompt construction."
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)


def eval_model(args):
    # Disable gradient computation
    disable_torch_init()

    # Load model & tokenizer
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    print(f"Loading model from: {model_path}")
    if args.model_base:
        print(f"Using base model: {args.model_base}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Set model to eval mode
    model.eval()
    torch.set_grad_enabled(False)

    # === CRITICAL FIX: Ensure vision tower is loaded and image_processor is available ===
    vt = model.get_vision_tower()
    print(f"Vision tower: {vt}")

    if vt is not None:
        # Check if vision tower needs loading
        needs_load = getattr(vt, "is_loaded", None)
        if needs_load is None or needs_load is False:
            print("Loading vision tower...")
            vt.load_model()

        # Move vision tower to device and set dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vt.to(device=device, dtype=torch.float16)

        # Get image processor from loaded vision tower
        image_processor = vt.image_processor
        print(f"Image processor loaded from vision tower: {type(image_processor)}")

    # Fallback if processor is still None
    if image_processor is None:
        print(f"Warning: image_processor is None, loading from {args.vision_tower}")
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    # === Load questions (support both JSON array and JSONL formats) ===
    print(f"Loading questions from: {args.question_file}")
    with open(os.path.expanduser(args.question_file), "r") as f:
        try:
            # Try JSON array format first (like test.json)
            questions = json.load(f)
            print(f"Loaded {len(questions)} questions from JSON array")
        except json.JSONDecodeError:
            # Fall back to JSONL format (one JSON per line)
            f.seek(0)
            questions = [json.loads(line) for line in f if line.strip()]
            print(f"Loaded {len(questions)} questions from JSONL")

    # Split questions across chunks if needed (for parallel evaluation)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    print(f"Processing chunk {args.chunk_idx}/{args.num_chunks}: {len(questions)} questions")

    # Create output directory
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Auto-switch conversation mode for plain models
    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"Auto-switching conv_mode to {args.conv_mode} for plain model.")

    # Create data loader
    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode,
        batch_size=1, num_workers=0
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Model moved to device: {device}")

    # Evaluate each question
    for (input_ids, image_tensor, image_sizes, lines) in tqdm(data_loader, total=len(questions), desc="Evaluating"):
        line = lines[0]

        # Get question ID (support multiple field names)
        qid = line.get("question_id", line.get("id", None))

        # Get question text for logging
        cur_prompt = line.get("text", line.get("conversations", [{}])[0].get("value", ""))

        # Move inputs to device
        input_ids = input_ids.to(device=device, non_blocking=True)
        images = image_tensor.to(
            dtype=torch.float16 if next(model.parameters()).dtype == torch.float16 else torch.float32,
            device=device,
            non_blocking=True
        )

        # Create proper attention mask (handle padding)
        attn_mask = (input_ids != tokenizer.pad_token_id).long()

        # Setup generation kwargs
        gen_kwargs = dict(
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

        # Add sampling parameters if temperature > 0
        if args.temperature is not None and args.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p if args.top_p is not None else 1.0,
            )

        # Generate answer
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attn_mask,
                images=images,
                image_sizes=image_sizes,
                **gen_kwargs
            )

        # Decode output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Save answer
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": qid,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    print(f"\nEvaluation complete! Answers saved to: {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint (LoRA or full)")
    parser.add_argument("--model-base", type=str, default=None, help="Base model path (for LoRA)")
    parser.add_argument("--vision-tower", type=str, default="openai/clip-vit-large-patch14-336",
                        help="Vision tower to use if not found in model")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--question-file", type=str, required=True, help="Path to questions (JSON or JSONL)")
    parser.add_argument("--answers-file", type=str, required=True, help="Path to save answers")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct",
                        help="Conversation template (mistral_instruct for LLaVA-Med v1.5)")
    parser.add_argument("--num-chunks", type=int, default=1, help="Split evaluation into chunks")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Index of chunk to process")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling parameter")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum tokens to generate")

    args = parser.parse_args()
    eval_model(args)

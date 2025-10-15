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

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def normalize_single_image_token(text, use_im_start_end: bool):
    # strip any existing image-start/end wrappers
    text = text.replace(DEFAULT_IM_START_TOKEN, "").replace(DEFAULT_IM_END_TOKEN, "")
    # count raw image tokens
    n = text.count(DEFAULT_IMAGE_TOKEN)

    # remove all existing image tokens
    text_wo_img = text.replace(DEFAULT_IMAGE_TOKEN, "").lstrip()

    # build the desired single image prefix
    prefix = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN) if use_im_start_end \
             else DEFAULT_IMAGE_TOKEN

    # always return exactly one image token + a newline before the question text
    return f"{prefix}\n{text_wo_img}"

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
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
        # Support both schemas
        if "conversations" in line:
            return line["conversations"][0]["value"]
        return line["text"]

    def _get_image_name(self, line):
        # VQA-RAD typically has 'image'
        return line["image"]

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = self._get_image_name(line)
        qs_raw = self._get_question_text(line)
        qs = normalize_single_image_token(qs_raw, self.model_config.mm_use_im_start_end)
        # # Prepend image token(s)
        # if getattr(self.model_config, "mm_use_im_start_end", False):
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        # Conversation template
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Load + preprocess image
        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # Tokenize prompt  -> [1, L]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        # Keep per-sample (no batch dim)
        return input_ids.squeeze(0), image_tensor, image.size, line

    def __len__(self):
        return len(self.questions)

def collate_fn(batch):
    # batch of tuples: (input_ids[L], image_tensor[C,H,W], image_size, line_dict)
    input_ids, image_tensors, image_sizes, lines = zip(*batch)
    # Create batch tensors: [B, L], [B, C, H, W]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, lines

def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=0):
    assert batch_size == 1, "batch_size must be 1 for current prompt construction."
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)

def eval_model(args):
    disable_torch_init()

    # Load model & processor
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    torch.set_grad_enabled(False)
    # after:
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # --- ensure vision tower is actually loaded & on the right device/dtype ---
    vt = model.get_vision_tower()
    print(vt)
    if vt is not None:
        # some forks expose a flag; fall back safely if not present
        needs_load = getattr(vt, "is_loaded", None)
        if needs_load is None or needs_load is False:
            vt.load_model()

        # move to device + dtype you will use for images
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # match your generate() image dtype (fp16 for speed is typical)
        vt.to(device=device, dtype=torch.float16)

        # always take image_processor from the (now) loaded tower
        image_processor = vt.image_processor
    # Fallback if processor is None
    if image_processor is None:
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    # Read questions
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = [json.loads(q) for q in f if q.strip()]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # Output file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Plain model auto mmtag hint (kept from original code)
    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"Auto-switching conv_mode to {args.conv_mode} for plain model.")

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode,
        batch_size=1, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    for (input_ids, image_tensor, image_sizes, lines) in tqdm(data_loader, total=len(questions)):
        line = lines[0]
        # Question id fallback
        qid = line.get("question_id", line.get("id", None))
        # Save the natural-language prompt (if present)
        cur_prompt = line.get("text", line.get("conversations", [{}])[0].get("value", ""))

        input_ids = input_ids.to(device=device, non_blocking=True)
        images = image_tensor.to(dtype=torch.float16 if next(model.parameters()).dtype == torch.float16 else torch.float32,
                                 device=device, non_blocking=True)

        gen_kwargs = dict(
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        if args.temperature is not None and args.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p if args.top_p is not None else 1.0,
            )
        with torch.inference_mode():
            attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)   
            output_ids = model.generate(
                input_ids,
                attention_mask=attn_mask,
                images=images,
                image_sizes=image_sizes,
                **gen_kwargs
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": qid,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")  # <- for LLaVA-Med v1.5 Mistral
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()
    eval_model(args)
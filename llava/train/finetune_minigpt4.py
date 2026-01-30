#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)

DEFAULT_PROMPT = "<image>\nProvide a description of the findings in the radiology image."
DEFAULT_PROMPT_WITH_REASON = (
    "<image>\nProvide a description of the findings in the radiology image "
    "given the following indication: {reason}"
)


def load_json_or_jsonl(path: str) -> List[Dict]:
    with open(path, "r") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_prompt(d: Dict, include_reason: bool) -> str:
    if include_reason and d.get("reason") is not None:
        reason = d["reason"].replace("\n", " ").strip()
        if reason:
            return DEFAULT_PROMPT_WITH_REASON.format(reason=reason)
    return DEFAULT_PROMPT


def format_text(user_text: str, answer_text: str, args) -> (str, str):
    text = user_text
    if args.replace_image_token:
        text = text.replace("<image>", args.image_token)
    system = args.system_prompt.strip()
    if system:
        system = system + "\n"
    if args.prompt_style == "minigpt4":
        prompt = f"{system}###Human: {text}\n###Assistant:"
        full = f"{prompt} {answer_text}"
    elif args.prompt_style == "llava_v1":
        prompt = f"{system}{text}\n"
        full = f"{prompt}{answer_text}"
    else:  # plain
        prompt = f"{system}{text}\n"
        full = f"{prompt}{answer_text}"
    return prompt, full


class MIMICDataset(Dataset):
    def __init__(self, data_path, image_folder, processor, args):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = args.max_length
        self.samples = []

        data = load_json_or_jsonl(data_path)
        for d in data:
            # generation method filtering
            if args.generation_methods != "all":
                if d.get("generate_method") != args.generation_methods:
                    continue
            # view filtering
            if args.filter_views:
                if d.get("view") not in ("AP", "PA"):
                    continue
            # basic conversation validation
            conv = d.get("conversations") or []
            if len(conv) < 2:
                continue
            answer = conv[1].get("value")
            if not isinstance(answer, str) or not answer.strip():
                continue

            image_path = d.get("image", "")
            if image_path.startswith("mimic/"):
                image_path = image_path[len("mimic/"):]
            image_path = os.path.join(image_folder, image_path)

            prompt = build_prompt(d, args.include_reason)
            self.samples.append(
                {"image": image_path, "prompt": prompt, "answer": answer}
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image"]).convert("RGB")
        prompt, full = format_text(item["prompt"], item["answer"], args)

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        ).input_ids
        prompt_len = len(prompt_ids)

        return {
            "image": image,
            "text": full,
            "prompt_len": prompt_len,
        }


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def maybe_freeze_submodules(model, args):
    if not args.train_vision:
        for attr in ("vision_tower", "vision_model", "visual_encoder", "vision_encoder"):
            if hasattr(model, attr):
                freeze_module(getattr(model, attr))
    if not args.train_qformer:
        for attr in ("qformer", "Qformer"):
            if hasattr(model, attr):
                freeze_module(getattr(model, attr))


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def collate_fn(batch):
    texts = [b["text"] for b in batch]
    images = [b["image"] for b in batch]
    prompt_lens = [b["prompt_len"] for b in batch]

    enc = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    labels = enc["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    enc["labels"] = labels
    return enc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--include_reason", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--filter_views", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--generation_methods", default="all", choices=["all", "gpt4", "rule-based"])

    parser.add_argument("--prompt_style", default="minigpt4", choices=["minigpt4", "llava_v1", "plain"])
    parser.add_argument("--image_token", default="<image>")
    parser.add_argument("--replace_image_token", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--system_prompt", default="")

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--fp16", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true", default=True)

    parser.add_argument("--train_vision", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--train_qformer", type=lambda x: x.lower() == "true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
    )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    maybe_freeze_submodules(model, args)
    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,}")

    dataset = MIMICDataset(args.data_path, args.image_folder, processor, args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(args.output_dir)

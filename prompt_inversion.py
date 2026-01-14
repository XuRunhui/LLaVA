#!/usr/bin/env python3
import argparse
import json
import random

def load_train_map(path):
    train_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj.get("id"), obj.get("image"), obj.get("view"), obj.get("generate_method"))
            train_map[key] = obj.get("prediction")
    return train_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train_results.jsonl")
    parser.add_argument("--chat", default="chat_train_p10_filtered.json")
    parser.add_argument("--out", default="chat_train_p10_filtered_prompt_inversion.json")
    parser.add_argument("--train-out", default="chat_train_p10_filtered_prompt_inversion_train.json")
    parser.add_argument("--val-out", default="chat_train_p10_filtered_prompt_inversion_val.json")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-unmatched", action="store_true",
                        help="Keep chat records with no train match (reason/conversations unchanged).")
    args = parser.parse_args()

    train_map = load_train_map(args.train)

    with open(args.chat, "r", encoding="utf-8") as f:
        chat_data = json.load(f)

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1 (exclusive).")

    output = []
    matched = 0
    missing_conversation = 0
    for obj in chat_data:
        key = (obj.get("id"), obj.get("image"), obj.get("view"), obj.get("generate_method"))
        if key not in train_map:
            if args.include_unmatched:
                output.append(obj)
            continue

        new_obj = dict(obj)
        new_obj["reason"] = train_map[key]
        conversations = new_obj.get("conversations")
        if isinstance(conversations, list) and len(conversations) > 1 and isinstance(conversations[1], dict):
            conversations[1]["value"] = obj.get("reason")
        else:
            missing_conversation += 1
        output.append(new_obj)
        matched += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=True, indent=2)
        f.write("\n")

    rng = random.Random(args.seed)
    indices = list(range(len(output)))
    rng.shuffle(indices)
    val_count = int(len(output) * args.val_ratio)
    val_set = set(indices[:val_count])
    train_data = []
    val_data = []
    for i, obj in enumerate(output):
        if i in val_set:
            val_data.append(obj)
        else:
            train_data.append(obj)

    with open(args.train_out, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=True, indent=2)
        f.write("\n")

    with open(args.val_out, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=True, indent=2)
        f.write("\n")

    print(f"train entries: {len(train_map)}")
    print(f"chat entries: {len(chat_data)}")
    print(f"matched: {matched}")
    print(f"missing conversations: {missing_conversation}")
    print(f"output: {args.out}")
    print(f"train split: {args.train_out}")
    print(f"val split: {args.val_out}")

if __name__ == "__main__":
    main()

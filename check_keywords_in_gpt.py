#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List, Set, Tuple

from llava.base_class import ReasonNormalizer, ReasonClassifier


def build_keyword_patterns(classifier: ReasonClassifier) -> Dict[str, Tuple[str, object]]:
    patterns: Dict[str, Tuple[str, object]] = {}
    for cat, compiled_list in classifier._compiled.items():
        for kw, pat in compiled_list:
            if kw not in patterns:
                patterns[kw] = (cat, pat)
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check keyword occurrences in conversations[1] where from=='gpt'."
    )
    parser.add_argument("--input", default="chat_train_p10_filtered.json",
                        help="Path to chat_train_p10_filtered.json")
    parser.add_argument("--out-matches", default=None,
                        help="Optional JSONL output with matched keywords/categories per sample.")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Minimum count to show in keyword summary.")
    parser.add_argument("--top-n", type=int, default=0,
                        help="Show only top N keywords (0 = all).")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable normalization before matching keywords.")
    args = parser.parse_args()

    normalizer = ReasonNormalizer()
    classifier = ReasonClassifier()
    keyword_patterns = build_keyword_patterns(classifier)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    keyword_counts: Dict[str, int] = {kw: 0 for kw in keyword_patterns}
    category_counts: Dict[str, int] = {cat: 0 for cat in classifier.categories.keys()}
    total_gpt = 0
    samples_with_any = 0

    out_f = open(args.out_matches, "w", encoding="utf-8") if args.out_matches else None
    try:
        for obj in data:
            conversations = obj.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                continue
            if conversations[1].get("from") != "gpt":
                continue
            text = conversations[1].get("value")
            if not isinstance(text, str):
                continue
            total_gpt += 1
            text = obj.get("reason", "")  # Use the reason field if available
            match_text = text if args.no_normalize else normalizer.normalize(text)
            matched_keywords: Set[str] = set()
            matched_categories: Set[str] = set()

            for kw, (cat, pat) in keyword_patterns.items():
                if pat.search(match_text):
                    matched_keywords.add(kw)
                    matched_categories.add(cat)

            if matched_keywords:
                samples_with_any += 1

            for kw in matched_keywords:
                keyword_counts[kw] += 1
            for cat in matched_categories:
                category_counts[cat] += 1

            if out_f:
                out_obj = {
                    "id": obj.get("id"),
                    "image": obj.get("image"),
                    "matched_keywords": sorted(matched_keywords),
                    "matched_categories": sorted(matched_categories),
                }
                out_f.write(json.dumps(out_obj, ensure_ascii=True) + "\n")
    finally:
        if out_f:
            out_f.close()

    print(f"Total gpt samples: {total_gpt}")
    print(f"Samples with any keyword: {samples_with_any}")
    if total_gpt > 0:
        print(f"Any-keyword rate: {samples_with_any / total_gpt:.4f}")
    print("")
    print("Category counts (samples with >=1 keyword in category):")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cat:30s} {count:6d}")
    print("")
    print("Keyword counts (samples containing keyword):")
    keyword_items = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    keyword_items = [x for x in keyword_items if x[1] >= args.min_count]
    if args.top_n > 0:
        keyword_items = keyword_items[:args.top_n]
    for kw, count in keyword_items:
        print(f"{kw:40s} {count:6d}")


if __name__ == "__main__":
    main()

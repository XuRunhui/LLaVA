#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List, Set

from llava.base_class import ReasonNormalizer, ReasonClassifier


def labels_for_text(text: str, normalizer: ReasonNormalizer, classifier: ReasonClassifier) -> List[str]:
    normalized = normalizer.normalize(text or "")
    return classifier.classify(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label predictions/ground truth with reason categories and compute per-category accuracy."
    )
    parser.add_argument("--input", default="dev_results.jsonl",
                        help="Path to results JSONL (must contain prediction and ground_truth fields).")
    parser.add_argument("--pred-field", default="prediction",
                        help="Field name for prediction text.")
    parser.add_argument("--gt-field", default="ground_truth",
                        help="Field name for ground truth text.")
    parser.add_argument("--out-labels", default=None,
                        help="Optional JSONL output with added pred_labels/gt_labels.")
    args = parser.parse_args()

    normalizer = ReasonNormalizer()
    classifier = ReasonClassifier()
    categories = list(classifier.categories.keys())

    stats: Dict[str, Dict[str, int]] = {
        cat: {"pred": 0, "gt": 0, "correct": 0} for cat in categories
    }
    total = 0
    any_correct = 0
    any_pred = 0
    any_gt = 0

    out_f = open(args.out_labels, "w", encoding="utf-8") if args.out_labels else None
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pred_text = obj.get(args.pred_field, "")
                gt_text = obj.get(args.gt_field, "")

                pred_labels = set(labels_for_text(pred_text, normalizer, classifier))
                gt_labels = set(labels_for_text(gt_text, normalizer, classifier))

                for cat in categories:
                    if cat in pred_labels:
                        stats[cat]["pred"] += 1
                    if cat in gt_labels:
                        stats[cat]["gt"] += 1
                    if cat in pred_labels and cat in gt_labels:
                        stats[cat]["correct"] += 1

                total += 1
                if pred_labels:
                    any_pred += 1
                if gt_labels:
                    any_gt += 1
                if pred_labels & gt_labels:
                    any_correct += 1

                if out_f:
                    obj["pred_labels"] = sorted(pred_labels)
                    obj["gt_labels"] = sorted(gt_labels)
                    out_f.write(json.dumps(obj, ensure_ascii=True) + "\n")
    finally:
        if out_f:
            out_f.close()

    print(f"Total samples: {total}")
    print(f"Samples with any pred label: {any_pred}")
    print(f"Samples with any gt label: {any_gt}")
    print(f"Samples with any correct label: {any_correct}")
    if total > 0:
        print(f"Any-label accuracy: {any_correct / total:.4f}")
    print("")
    print("Per-category metrics:")
    for cat in categories:
        pred = stats[cat]["pred"]
        gt = stats[cat]["gt"]
        correct = stats[cat]["correct"]
        precision = (correct / pred) if pred else 0.0
        recall = (correct / gt) if gt else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        print(
            f"{cat:30s} "
            f"pred={pred:5d} gt={gt:5d} correct={correct:5d} "
            f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
        )


if __name__ == "__main__":
    main()

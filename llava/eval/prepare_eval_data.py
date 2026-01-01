"""
Helper script to prepare and validate evaluation data for LLaVA LoRA evaluation.

Supports two data formats:
1. Simple format: {"question_id": "...", "image": "...", "text": "...", "answer": "..."}
2. Conversations format: {"id": "...", "image": "...", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

Usage:
    # Validate existing data
    python prepare_eval_data.py --validate --question-file questions.json --image-folder ./images

    # Create template
    python prepare_eval_data.py --create-template --output questions_template.jsonl --format simple
    python prepare_eval_data.py --create-template --output questions_template.json --format conversations
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter


def validate_question_format(question: Dict) -> List[str]:
    """Validate a single question entry and return any errors."""
    errors = []

    # Check for question ID (question_id or id)
    if "question_id" not in question and "id" not in question:
        errors.append("Missing required field: 'question_id' or 'id'")

    # Check for image field
    if "image" not in question:
        errors.append("Missing required field: 'image'")

    # Check for question format: either 'text' or 'conversations'
    has_text = "text" in question
    has_conversations = "conversations" in question

    if not has_text and not has_conversations:
        errors.append("Missing required field: either 'text' or 'conversations'")

    # Validate conversations format if present
    if has_conversations:
        if not isinstance(question["conversations"], list):
            errors.append("'conversations' should be a list")
        else:
            has_human = False
            has_gpt = False
            for i, conv in enumerate(question["conversations"]):
                if not isinstance(conv, dict):
                    errors.append(f"Conversation item {i} should be a dict")
                    continue
                if "from" not in conv or "value" not in conv:
                    errors.append(f"Conversation item {i} should have 'from' and 'value'")
                    continue
                if conv["from"] == "human":
                    has_human = True
                elif conv["from"] == "gpt":
                    has_gpt = True

            if not has_human:
                errors.append("'conversations' should have at least one 'human' entry")

    # Validate simple format if present
    if has_text and not isinstance(question["text"], str):
        errors.append(f"'text' should be string, got {type(question['text'])}")

    # Check field types
    if "question_id" in question and not isinstance(question["question_id"], (str, int)):
        errors.append(f"question_id should be string or int, got {type(question['question_id'])}")

    if "id" in question and not isinstance(question["id"], (str, int)):
        errors.append(f"id should be string or int, got {type(question['id'])}")

    if "image" in question and not isinstance(question["image"], str):
        errors.append(f"image should be string, got {type(question['image'])}")

    # Check optional fields
    if "answer" in question and not isinstance(question["answer"], str):
        errors.append(f"answer should be string, got {type(question['answer'])}")

    if "ground_truth" in question and not isinstance(question["ground_truth"], str):
        errors.append(f"ground_truth should be string, got {type(question['ground_truth'])}")

    if "view" in question and not isinstance(question["view"], str):
        errors.append(f"view should be string, got {type(question['view'])}")

    if "reason" in question and not isinstance(question["reason"], str):
        errors.append(f"reason should be string, got {type(question['reason'])}")

    return errors


def extract_question_answer(question: Dict) -> tuple:
    """Extract question text and answer from either format."""
    question_text = None
    answer_text = None

    if "conversations" in question:
        for conv in question["conversations"]:
            if conv.get("from") == "human":
                question_text = conv.get("value", "").replace("<image>", "").strip()
            elif conv.get("from") == "gpt":
                answer_text = conv.get("value", "").strip()
    else:
        question_text = question.get("text")
        answer_text = question.get("answer", question.get("ground_truth"))

    return question_text, answer_text


def validate_eval_data(question_file: str, image_folder: Optional[str] = None) -> Dict:
    """Validate evaluation data and return statistics."""

    print(f"Validating evaluation data from: {question_file}")
    print("=" * 80)

    # Load questions - support both JSON array and JSONL
    questions = []
    line_errors = {}

    try:
        with open(question_file, 'r') as f:
            content = f.read().strip()

            # Try as JSON array
            if content.startswith('['):
                try:
                    questions = json.loads(content)
                    print("Detected format: JSON array")
                except json.JSONDecodeError:
                    pass

            # Try as JSONL
            if not questions:
                print("Detected format: JSONL")
                for line_num, line in enumerate(content.split('\n'), 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        question = json.loads(line)
                        questions.append(question)
                    except json.JSONDecodeError as e:
                        line_errors[line_num] = [f"JSON parsing error: {e}"]

    except FileNotFoundError:
        print(f"ERROR: Question file not found: {question_file}")
        return None

    # Validate each question
    for idx, question in enumerate(questions):
        errors = validate_question_format(question)
        if errors:
            line_errors[idx + 1] = errors

    # Print validation results
    print(f"\nTotal questions loaded: {len(questions)}")

    if line_errors:
        print(f"\nERROR: Found {len(line_errors)} items with validation errors:")
        for line_num, errors in sorted(line_errors.items())[:10]:  # Show first 10
            print(f"  Item {line_num}:")
            for error in errors:
                print(f"    - {error}")
        if len(line_errors) > 10:
            print(f"  ... and {len(line_errors) - 10} more errors")
        print("\nPlease fix these errors before proceeding.")
        return None

    print("\n✓ All questions have valid format")

    # Statistics
    stats = {
        'total_questions': len(questions),
        'with_ground_truth': 0,
        'with_view': 0,
        'with_reason': 0,
        'with_chexpert_labels': 0,
        'format_conversations': 0,
        'format_simple': 0,
        'unique_images': 0,
        'missing_images': 0,
        'views': Counter(),
        'avg_question_length': 0,
        'avg_answer_length': 0,
    }

    # Collect statistics
    unique_images = set()
    missing_images = []
    question_lengths = []
    answer_lengths = []

    for q in questions:
        # Determine format
        if "conversations" in q:
            stats['format_conversations'] += 1
        else:
            stats['format_simple'] += 1

        # Extract question and answer
        question_text, answer_text = extract_question_answer(q)

        # Ground truth
        if answer_text:
            stats['with_ground_truth'] += 1
            answer_lengths.append(len(answer_text.split()))

        # View
        if "view" in q:
            stats['with_view'] += 1
            stats['views'][q['view']] += 1

        # Reason
        if "reason" in q and q.get("reason"):
            stats['with_reason'] += 1

        # CheXpert labels
        if "chexpert_labels" in q:
            stats['with_chexpert_labels'] += 1

        # Images
        unique_images.add(q['image'])

        # Question length
        if question_text:
            question_lengths.append(len(question_text.split()))

        # Check if image exists
        if image_folder:
            image_path = os.path.join(image_folder, q['image'])
            if not os.path.exists(image_path):
                missing_images.append(q['image'])

    stats['unique_images'] = len(unique_images)
    stats['avg_question_length'] = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    stats['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0

    # Print statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"\nData Summary:")
    print(f"  Total questions:        {stats['total_questions']}")
    print(f"  Unique images:          {stats['unique_images']}")
    print(f"  Conversations format:   {stats['format_conversations']} ({stats['format_conversations']/stats['total_questions']*100:.1f}%)")
    print(f"  Simple format:          {stats['format_simple']} ({stats['format_simple']/stats['total_questions']*100:.1f}%)")
    print(f"  With ground truth:      {stats['with_ground_truth']} ({stats['with_ground_truth']/stats['total_questions']*100:.1f}%)")
    print(f"  With view field:        {stats['with_view']} ({stats['with_view']/stats['total_questions']*100:.1f}%)")
    print(f"  With reason field:      {stats['with_reason']} ({stats['with_reason']/stats['total_questions']*100:.1f}%)")
    print(f"  With CheXpert labels:   {stats['with_chexpert_labels']} ({stats['with_chexpert_labels']/stats['total_questions']*100:.1f}%)")

    print(f"\nText Statistics:")
    print(f"  Avg question length:    {stats['avg_question_length']:.1f} words")
    if answer_lengths:
        print(f"  Avg answer length:      {stats['avg_answer_length']:.1f} words")

    if stats['views']:
        print(f"\nView Distribution:")
        for view, count in stats['views'].most_common():
            print(f"  {view:15} {count:5} ({count/stats['total_questions']*100:.1f}%)")

    # Check images
    if image_folder:
        print(f"\n" + "=" * 80)
        print("IMAGE VALIDATION")
        print("=" * 80)

        if not os.path.exists(image_folder):
            print(f"\nWARNING: Image folder not found: {image_folder}")
        else:
            print(f"\nImage folder: {image_folder}")

            if missing_images:
                stats['missing_images'] = len(missing_images)
                print(f"\nERROR: Found {len(missing_images)} missing images:")
                for img in missing_images[:10]:  # Show first 10
                    print(f"  - {img}")
                if len(missing_images) > 10:
                    print(f"  ... and {len(missing_images) - 10} more missing images")
                print("\nPlease ensure all images are in the image folder.")
            else:
                print(f"\n✓ All {stats['unique_images']} images found")

    # Warnings
    print(f"\n" + "=" * 80)
    print("WARNINGS AND RECOMMENDATIONS")
    print("=" * 80)

    if stats['with_ground_truth'] == 0:
        print("\n⚠ WARNING: No ground truth answers found")
        print("  → You will not be able to compute loss and perplexity")
        print("  → Add 'answer' field (simple format) or 'gpt' conversation (conversations format)")

    if stats['with_ground_truth'] < stats['total_questions']:
        print(f"\n⚠ WARNING: Only {stats['with_ground_truth']}/{stats['total_questions']} questions have ground truth")
        print("  → Metrics will only be computed for questions with ground truth")

    if stats['avg_question_length'] > 100:
        print(f"\n⚠ WARNING: Average question length is {stats['avg_question_length']:.1f} words")
        print("  → Very long questions may cause context length issues")

    if answer_lengths and stats['avg_answer_length'] > 200:
        print(f"\n⚠ WARNING: Average answer length is {stats['avg_answer_length']:.1f} words")
        print("  → Consider increasing --max_new_tokens in evaluation script")

    print("\n" + "=" * 80)
    print("✓ Validation complete!")
    print("=" * 80)

    return stats


def create_template(output_file: str, num_examples: int = 5, format_type: str = "simple"):
    """Create a template file with example questions."""

    examples = []
    for i in range(1, num_examples + 1):
        if format_type == "conversations":
            example = {
                "id": f"example_{i:03d}",
                "image": f"path/to/image{i}.jpg",
                "reason": "Example clinical indication (optional)",
                "view": "PA",
                "orientation": "Erect",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\nDescribe the findings in this chest X-ray."
                    },
                    {
                        "from": "gpt",
                        "value": f"This is an example ground truth answer for question {i}. The chest X-ray shows normal lung fields with no acute abnormalities."
                    }
                ]
            }
        else:  # simple format
            example = {
                "question_id": f"example_{i:03d}",
                "image": f"image{i}.jpg",
                "text": f"Example question {i}: What do you see in this image?",
                "answer": f"This is an example ground truth answer for question {i}. Replace with your actual answer.",
                "view": "PA",
                "reason": "Example clinical indication"
            }
        examples.append(example)

    # Save as JSON array or JSONL based on extension
    if output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)
    else:
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

    print(f"Template created: {output_file}")
    print(f"Format: {format_type}")
    print(f"Created {num_examples} example questions")
    print("\nPlease edit this file to add your actual questions and answers.")
    print("\nRequired fields:")
    if format_type == "conversations":
        print("  - id: Unique identifier")
        print("  - image: Image filename/path")
        print("  - conversations: List of {from: 'human'/'gpt', value: '...'}")
    else:
        print("  - question_id: Unique identifier")
        print("  - image: Image filename")
        print("  - text: Question text")
        print("  - answer: Ground truth answer (required for metrics)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and validate evaluation data for LLaVA LoRA evaluation"
    )

    parser.add_argument("--validate", action="store_true",
                        help="Validate existing question file")
    parser.add_argument("--create-template", action="store_true",
                        help="Create a template question file")
    parser.add_argument("--question-file", type=str,
                        help="Path to question JSON/JSONL file (for validation)")
    parser.add_argument("--image-folder", type=str,
                        help="Path to image folder (for validation)")
    parser.add_argument("--output", type=str, default="questions_template.jsonl",
                        help="Output file for template")
    parser.add_argument("--num-examples", type=int, default=5,
                        help="Number of example questions in template")
    parser.add_argument("--format", type=str, default="simple", choices=["simple", "conversations"],
                        help="Format for template: 'simple' or 'conversations'")

    args = parser.parse_args()

    if args.create_template:
        create_template(args.output, args.num_examples, args.format)

    elif args.validate:
        if not args.question_file:
            print("ERROR: --question-file is required for validation")
            return

        validate_eval_data(args.question_file, args.image_folder)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Create a simple format template")
        print("  python prepare_eval_data.py --create-template --output my_questions.jsonl --format simple")
        print("")
        print("  # Create a conversations format template")
        print("  python prepare_eval_data.py --create-template --output my_questions.json --format conversations")
        print("")
        print("  # Validate existing data")
        print("  python prepare_eval_data.py --validate --question-file questions.json --image-folder ./images")


if __name__ == "__main__":
    main()

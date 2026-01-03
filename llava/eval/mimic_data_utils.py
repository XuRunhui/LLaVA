"""
MIMIC-CXR Data Loading Utilities

Shared data loading functions for both training and evaluation.
"""

import json
import logging
from typing import List, Dict, Optional


def load_mimic_cxr_data(
    data_path: str,
    split: str = "test",
    filter_views: bool = True,
    include_reason: bool = True,
    generation_methods: str = "rule-based",
    verbose: bool = True
) -> List[Dict]:
    """
    Load and filter MIMIC-CXR dataset for evaluation.

    Args:
        data_path: Path to MIMIC-CXR JSON file
        split: Dataset split ('train', 'dev', 'test')
        filter_views: Filter to only PA/AP views
        include_reason: Include reason/indication in prompts
        generation_methods: 'gpt4', 'rule-based', or 'all'
        verbose: Print detailed statistics

    Returns:
        List of data samples with:
        - 'id': Sample ID (generated from image path)
        - 'image': Image path
        - 'conversations': [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'}]
        - 'ground_truth': Ground truth text (for evaluation)
        - 'view': View type (PA, AP, LATERAL, etc.)
        - 'generate_method': Generation method (gpt4, rule-based)
    """
    if verbose:
        print("=" * 80)
        print(f"MIMIC-CXR Data Loader - Loading {split} split")
        print("=" * 80)

    with open(data_path, 'r') as f:
        dataset = json.load(f)

    original_count = len(dataset)
    if verbose:
        print(f"Total samples in dataset: {original_count}")

    # Counters for filtering statistics
    filtered_by_generation_method = 0
    filtered_by_invalid_findings = 0
    filtered_by_view = 0
    samples_with_reason = 0
    samples_without_reason = 0

    # Track view type distribution
    view_counts = {}
    generation_method_counts = {}

    ret = []

    for d in dataset:
        # Track generation method distribution
        generate_method = d.get('generate_method', 'unknown')
        generation_method_counts[generate_method] = generation_method_counts.get(generate_method, 0) + 1

        # Track view distribution
        view = d.get('view', 'unknown')
        view_counts[view] = view_counts.get(view, 0) + 1

        # Filter by generation method if specified
        if generation_methods == 'gpt4':
            if generate_method != 'gpt4':
                filtered_by_generation_method += 1
                continue
        elif generation_methods == 'rule-based':
            if generate_method != 'rule-based':
                filtered_by_generation_method += 1
                continue
        # 'all' means no filtering by generation method

        # Skip samples with invalid findings (empty or non-string)
        if not d.get('conversations') or len(d['conversations']) < 2:
            filtered_by_invalid_findings += 1
            continue
        if not isinstance(d["conversations"][1].get("value"), str):
            filtered_by_invalid_findings += 1
            continue
        if not d["conversations"][1]["value"].strip():
            filtered_by_invalid_findings += 1
            continue

        # Filter by view (only PA or AP) if enabled
        if filter_views:
            if view not in ('AP', 'PA'):
                filtered_by_view += 1
                continue

        # Clean up image path (remove 'mimic/' prefix if present)
        image_path = d.get('image', '')
        if image_path.startswith("mimic/"):
            image_path = image_path[len('mimic/'):]

        # Store ground truth for evaluation
        ground_truth = d["conversations"][1]["value"]

        # Modify prompt to include reason/indication if enabled and available
        if include_reason and d.get('reason') is not None:
            reason = d['reason'].replace('\n', ' ').strip()
            if reason:
                question = (
                    f"<image>\n"
                    f"Provide a description of the findings in the radiology image "
                    f"given the following indication: {reason}"
                )
                samples_with_reason += 1
            else:
                # Reason field exists but is empty
                question = (
                    "<image>\n"
                    "Provide a description of the findings in the radiology image."
                )
                samples_without_reason += 1
        else:
            # No reason available or not using reasons
            question = (
                "<image>\n"
                "Provide a description of the findings in the radiology image."
            )
            samples_without_reason += 1

        # Create sample with evaluation-specific fields
        sample = {
            'id': d.get('id', image_path.replace('/', '_').replace('.jpg', '')),
            'image': image_path,
            'conversations': [
                {'from': 'human', 'value': question},
                {'from': 'gpt', 'value': ground_truth}
            ],
            'ground_truth': ground_truth,
            'question': question,
            'view': view,
            'generate_method': generate_method,
        }

        # Preserve original metadata if available
        if 'reason' in d:
            sample['reason'] = d['reason']
        if 'impression' in d:
            sample['impression'] = d['impression']
        if 'indication' in d:
            sample['indication'] = d['indication']

        ret.append(sample)

    if verbose:
        # Print detailed statistics
        print("")
        print("=" * 80)
        print(f"MIMIC-CXR Data Filtering Summary ({split} split)")
        print("=" * 80)
        print(f"Total samples loaded:           {original_count:,}")
        print(f"Samples after filtering:        {len(ret):,}")
        print(f"Samples filtered out:           {original_count - len(ret):,} ({100 * (original_count - len(ret)) / original_count:.1f}%)")
        print("")

        print("-" * 80)
        print("Filtering Breakdown:")
        print("-" * 80)
        print(f"  Filtered by generation method:  {filtered_by_generation_method:,}")
        print(f"  Filtered by invalid findings:   {filtered_by_invalid_findings:,}")
        print(f"  Filtered by view type:          {filtered_by_view:,}")
        print("")

        print("-" * 80)
        print("Generation Method Distribution (in original dataset):")
        print("-" * 80)
        for method, count in sorted(generation_method_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method:20s}: {count:,} ({100 * count / original_count:.1f}%)")
        print(f"  Selected method: {generation_methods}")
        print("")

        print("-" * 80)
        print("View Type Distribution (in original dataset):")
        print("-" * 80)
        for view_type, count in sorted(view_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {view_type:20s}: {count:,} ({100 * count / original_count:.1f}%)")
        if filter_views:
            print(f"  Filtered to: PA/AP views only")
        else:
            print(f"  No view filtering applied")
        print("")

        print("-" * 80)
        print("Clinical Indication/Reason:")
        print("-" * 80)
        if include_reason:
            print(f"  Samples with reason included:   {samples_with_reason:,}")
            print(f"  Samples without reason:         {samples_without_reason:,}")
            print(f"  Reason inclusion: ENABLED")
        else:
            print(f"  Reason inclusion: DISABLED")
        print("")

        print("=" * 80)
        print(f"Final dataset size: {len(ret):,} samples")
        print("=" * 80)
        print("")

    return ret

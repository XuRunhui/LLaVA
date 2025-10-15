#!/usr/bin/env python3
"""
Reference Member Inference Attack
Uses member vs non-member similarity comparison
"""

import json
import numpy as np
import argparse
import random
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support


def load_data(member_file, non_member_file, temperature, metric):
    """Load similarity scores"""
    with open(member_file, 'r') as f:
        member_data_all = json.load(f)
    with open(non_member_file, 'r') as f:
        non_member_data_all = json.load(f)

    member_data = [item[f'similarity_{temperature}'][metric] for item in member_data_all]
    non_member_data = [item[f'similarity_{temperature}'][metric] for item in non_member_data_all]

    return member_data, non_member_data


def reference_member_inference(member_data, non_member_data, granularity):
    """
    Reference attack: split members into reference and target sets
    """
    random.shuffle(member_data)
    half = len(member_data) // 2

    reference_member = member_data[:half]
    target_member = member_data[half:]
    target_non_member = non_member_data

    p_list = []
    label_list = []

    # Run attack multiple times for stability
    for _ in range(1000):
        # Sample subsets
        samples_reference = random.sample(reference_member, granularity)
        samples_target_member = random.sample(target_member, min(granularity, len(target_member)))
        samples_target_non_member = random.sample(target_non_member, min(granularity, len(target_non_member)))

        # Calculate statistics
        mean_ref = np.mean(samples_reference)
        var_ref = np.var(samples_reference, ddof=1)

        mean_mem = np.mean(samples_target_member)
        var_mem = np.var(samples_target_member, ddof=1)

        mean_non_mem = np.mean(samples_target_non_member)
        var_non_mem = np.var(samples_target_non_member, ddof=1)

        # Z-test for member
        z_member = (mean_mem - mean_ref) / np.sqrt(var_mem / len(samples_target_member) + var_ref / len(samples_reference))
        p_member = 1 - norm.cdf(z_member)
        p_list.append(p_member)
        label_list.append(1)  # Member

        # Z-test for non-member
        z_non_member = (mean_ref - mean_non_mem) / np.sqrt(var_ref / len(samples_reference) + var_non_mem / len(samples_target_non_member))
        p_non_member = 1 - norm.cdf(z_non_member)
        p_list.append(p_non_member)
        label_list.append(0)  # Non-member

    # Calculate metrics
    auc = roc_auc_score(label_list, p_list)

    # Convert to binary predictions (threshold = 0.5)
    pred_list = [1 if p < 0.5 else 0 for p in p_list]
    accuracy = accuracy_score(label_list, pred_list)
    precision, recall, f1, _ = precision_recall_fscore_support(label_list, pred_list, average='binary')

    return auc, accuracy, precision, recall, f1


def main(args):
    # Load data
    print("Loading similarity data...")
    member_data, non_member_data = load_data(
        args.member_similarity_file,
        args.non_member_similarity_file,
        args.temperature,
        args.similarity_metric
    )

    print(f"Member samples: {len(member_data)}")
    print(f"Non-member samples: {len(non_member_data)}")
    print(f"Granularity: {args.granularity}")
    print(f"Similarity metric: {args.similarity_metric}")

    # Run attack multiple times
    print("\nRunning reference attack (5 trials)...")
    aucs, accs, precs, recs, f1s = [], [], [], [], []

    for trial in range(5):
        auc, acc, prec, rec, f1 = reference_member_inference(
            member_data, non_member_data, args.granularity
        )
        aucs.append(auc)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        print(f"Trial {trial + 1}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    # Average results
    avg_auc = np.mean(aucs)
    avg_acc = np.mean(accs)
    avg_prec = np.mean(precs)
    avg_rec = np.mean(recs)
    avg_f1 = np.mean(f1s)

    print(f"\n{'='*60}")
    print("REFERENCE ATTACK RESULTS")
    print(f"{'='*60}")
    print(f"AUC:       {avg_auc:.4f} ± {np.std(aucs):.4f}")
    print(f"Accuracy:  {avg_acc:.4f} ± {np.std(accs):.4f}")
    print(f"Precision: {avg_prec:.4f} ± {np.std(precs):.4f}")
    print(f"Recall:    {avg_rec:.4f} ± {np.std(recs):.4f}")
    print(f"F1 Score:  {avg_f1:.4f} ± {np.std(f1s):.4f}")
    print(f"{'='*60}")

    # Save results
    results = {
        'attack_type': 'reference',
        'auc': float(avg_auc),
        'auc_std': float(np.std(aucs)),
        'accuracy': float(avg_acc),
        'accuracy_std': float(np.std(accs)),
        'precision': float(avg_prec),
        'recall': float(avg_rec),
        'f1': float(avg_f1),
        'granularity': args.granularity,
        'temperature': args.temperature,
        'similarity_metric': args.similarity_metric,
        'trials': aucs
    }

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--member-similarity-file', type=str, required=True)
    parser.add_argument('--non-member-similarity-file', type=str, required=True)
    parser.add_argument('--granularity', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--similarity-metric', type=str, default='rouge2_f',
                        choices=['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'bertscore_f'])
    parser.add_argument('--output-file', type=str, default=None)

    args = parser.parse_args()
    main(args)

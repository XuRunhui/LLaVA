#!/usr/bin/env python3
"""
Calculate similarity between repeated generations (for Image-Only Attack)
Measures consistency of model outputs
"""

import json
import argparse
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore
import numpy as np
from itertools import combinations


def calculate_rouge(hypothesis, reference):
    """Calculate ROUGE scores"""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)[0]
        return {
            'rouge1_f': scores['rouge-1']['f'],
            'rouge2_f': scores['rouge-2']['f'],
            'rougeL_f': scores['rouge-l']['f'],
        }
    except:
        return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}


def calculate_bleu(hypothesis, reference):
    """Calculate BLEU score"""
    try:
        hyp_tokens = hypothesis.split()
        ref_tokens = [reference.split()]
        smooth = SmoothingFunction()
        return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth.method1)
    except:
        return 0.0


def main(args):
    # Load conversations
    print(f"Loading conversations from: {args.conversation_json_path}")
    with open(args.conversation_json_path, 'r') as f:
        conversations = json.load(f)

    print(f"Total samples: {len(conversations)}")

    # Process each sample
    similarity_results = []

    for sample in conversations:
        result = {'image_id': sample['image_id']}

        # Process each temperature
        for temperature in args.temperatures:
            temp_key = f"conversations_{temperature}"
            if temp_key not in sample:
                continue

            conv_list = sample[temp_key]

            # Extract all repeated generations (vlm_1, vlm_2, ...)
            generated_responses = []

            for conv_item in conv_list:
                if conv_item['from'].startswith('vlm_'):
                    generated_responses.append(conv_item['value'])

            if len(generated_responses) < 2:
                print(f"Warning: Sample {sample['image_id']} has < 2 generations, skipping")
                continue

            # Calculate pairwise similarities
            pairwise_similarities = []

            for i, j in combinations(range(len(generated_responses)), 2):
                resp_i = generated_responses[i]
                resp_j = generated_responses[j]

                # ROUGE
                rouge_scores = calculate_rouge(resp_i, resp_j)

                # BLEU
                bleu_score = calculate_bleu(resp_i, resp_j)

                pairwise_similarities.append({
                    **rouge_scores,
                    'bleu': bleu_score
                })

            # Calculate BERTScore for all pairs
            if pairwise_similarities:
                pairs = [(generated_responses[i], generated_responses[j])
                         for i, j in combinations(range(len(generated_responses)), 2)]
                hyps = [p[0] for p in pairs]
                refs = [p[1] for p in pairs]

                P_list, R_list, F1_list = bertscore(hyps, refs, lang='en', verbose=False)

                for idx, sim in enumerate(pairwise_similarities):
                    sim['bertscore_f'] = F1_list[idx].item()

            # Average all pairwise similarities
            avg_similarity = {}
            if pairwise_similarities:
                for key in pairwise_similarities[0].keys():
                    avg_similarity[key] = np.mean([s[key] for s in pairwise_similarities])

                result[f'similarity_{temperature}'] = avg_similarity

        similarity_results.append(result)

    # Save results
    print(f"Saving similarity scores to: {args.similarity_json_path}")
    with open(args.similarity_json_path, 'w') as f:
        json.dump(similarity_results, f, indent=2)

    print(f"Similarity calculation complete! Processed {len(similarity_results)} samples")
    print(f"Average number of pairwise comparisons: {len(list(combinations(range(args.repeating_num), 2)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation-json-path", type=str, required=True)
    parser.add_argument("--similarity-json-path", type=str, required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--repeating-num", type=int, required=True,
                        help="Number of repetitions used during generation")

    args = parser.parse_args()
    main(args)

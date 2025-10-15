#!/usr/bin/env python3
"""
Calculate similarity between generated responses and ground truth
For Reference and Target-Only attacks
"""

import json
import argparse
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore
import numpy as np


def calculate_rouge(hypothesis, reference):
    """Calculate ROUGE scores"""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)[0]
        return {
            'rouge1_f': scores['rouge-1']['f'],
            'rouge1_p': scores['rouge-1']['p'],
            'rouge1_r': scores['rouge-1']['r'],
            'rouge2_f': scores['rouge-2']['f'],
            'rouge2_p': scores['rouge-2']['p'],
            'rouge2_r': scores['rouge-2']['r'],
            'rougeL_f': scores['rouge-l']['f'],
            'rougeL_p': scores['rouge-l']['p'],
            'rougeL_r': scores['rouge-l']['r'],
        }
    except:
        # Return zeros if calculation fails
        return {key: 0.0 for key in [
            'rouge1_f', 'rouge1_p', 'rouge1_r',
            'rouge2_f', 'rouge2_p', 'rouge2_r',
            'rougeL_f', 'rougeL_p', 'rougeL_r'
        ]}


def calculate_bleu(hypothesis, reference):
    """Calculate BLEU score"""
    try:
        # Tokenize
        hyp_tokens = hypothesis.split()
        ref_tokens = [reference.split()]

        # Calculate BLEU with smoothing
        smooth = SmoothingFunction()
        bleu = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth.method1)
        return bleu
    except:
        return 0.0


def calculate_bertscore(hypotheses, references):
    """Calculate BERTScore (batched for efficiency)"""
    try:
        P, R, F1 = bertscore(hypotheses, references, lang='en', verbose=False)
        return P.tolist(), R.tolist(), F1.tolist()
    except:
        # Return zeros if calculation fails
        return [0.0] * len(hypotheses), [0.0] * len(hypotheses), [0.0] * len(hypotheses)


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

            # Extract generated responses and ground truth
            generated_responses = []
            ground_truth = None

            for conv_item in conv_list:
                if conv_item['from'].startswith('vlm'):
                    generated_responses.append(conv_item['value'])
                elif conv_item['from'] == 'ground truth':
                    ground_truth = conv_item['value']

            if not generated_responses or ground_truth is None:
                continue

            # Calculate similarity for each generated response
            similarities = []
            for gen_response in generated_responses:
                # ROUGE
                rouge_scores = calculate_rouge(gen_response, ground_truth)

                # BLEU
                bleu_score = calculate_bleu(gen_response, ground_truth)

                similarities.append({
                    **rouge_scores,
                    'bleu': bleu_score
                })

            # Calculate BERTScore (batched)
            if generated_responses and ground_truth:
                P_list, R_list, F1_list = calculate_bertscore(
                    generated_responses,
                    [ground_truth] * len(generated_responses)
                )

                for i, sim in enumerate(similarities):
                    sim['bertscore_p'] = P_list[i]
                    sim['bertscore_r'] = R_list[i]
                    sim['bertscore_f'] = F1_list[i]

            # Average if multiple responses (shouldn't happen for ground truth comparison)
            if len(similarities) > 0:
                avg_similarity = {}
                for key in similarities[0].keys():
                    avg_similarity[key] = np.mean([s[key] for s in similarities])

                result[f'similarity_{temperature}'] = avg_similarity

        similarity_results.append(result)

    # Save results
    print(f"Saving similarity scores to: {args.similarity_json_path}")
    with open(args.similarity_json_path, 'w') as f:
        json.dump(similarity_results, f, indent=2)

    print(f"Similarity calculation complete! Processed {len(similarity_results)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation-json-path", type=str, required=True,
                        help="Path to conversation output file")
    parser.add_argument("--similarity-json-path", type=str, required=True,
                        help="Path to save similarity scores")
    parser.add_argument("--temperatures", nargs="+", type=float, required=True,
                        help="List of temperatures to process")

    args = parser.parse_args()
    main(args)

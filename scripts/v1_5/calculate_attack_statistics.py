import json
from collections import defaultdict
def load_best_by_question_id(jsonl_path):
    """
    Reads a JSONL file and returns a dictionary:
    {
        question_id: object_with_min_loss
    }
    """
    best_by_qid = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["question_id"]
            loss = obj["metrics"]["loss"]

            if qid not in best_by_qid:
                best_by_qid[qid] = obj
            else:
                if loss < best_by_qid[qid]["metrics"]["loss"]:
                    best_by_qid[qid] = obj

    return best_by_qid

gender_json_prediction_object = load_best_by_question_id("/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/dp_lora_weight_128/eval/ender_likelihood_with_metrics.jsonl")
ethnicity_json_prediction_object = load_best_by_question_id("/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/dp_lora_weight_128/eval/ethnicity_likelihood_with_metrics.jsonl")
age_likelihood_json_prediction_object = load_best_by_question_id("/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/dp_lora_weight_128/eval/age_likelihood_with_metrics.jsonl")
keyword_likelihood_json_prediction_object = load_best_by_question_id("/project2/ruishanl_1185/SDP_for_VLM/outputs/xinyang/llava_rex/dp_lora_weight_128/eval/keywords_likelihood_with_metrics.jsonl")

def attack_accuracy(train_jsonl, prediction_object, key):
    count = 0
    ground_truth_object_by_id = {}

    choice_freq = defaultdict(int)

    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["id"]
            ground_truth_object_by_id[qid] = obj
    
    for qid in prediction_object:
        choice_freq[ground_truth_object_by_id[qid][key]] += 1
        if ground_truth_object_by_id[qid][key] in prediction_object[qid]["prompt"]:
            count += 1
    print(choice_freq)
    return count / len(prediction_object)

def analyze_age_or_keyword_attack(attack_jsonl, prediction_object):
    id_to_answer_mapping = {}
    
    with open(attack_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["id"]
            if qid in id_to_answer_mapping:
                continue
            ground_truth = obj["metadata"]["choices"][obj["metadata"]["correct_index"]]
            id_to_answer_mapping[qid] = ground_truth
    
    count = 0
    for idx in prediction_object:
        if id_to_answer_mapping[idx] in prediction_object[idx]["prompt"]:
            count += 1
    return count / len(prediction_object)

print("gender attack accuracy: ", str(attack_accuracy("/project2/ruishanl_1185/SDP_for_VLM/datasets/rexgradient/ReXGradient/metadata/rexgradient_train.jsonl", gender_json_prediction_object, "sex")))
print("ethnicity attack accuracy: ", str(attack_accuracy("/project2/ruishanl_1185/SDP_for_VLM/datasets/rexgradient/ReXGradient/metadata/rexgradient_train.jsonl", ethnicity_json_prediction_object, "ethnicGroup")))   
print("age attack accuracy: ", str(analyze_age_or_keyword_attack("/project2/ruishanl_1185/SDP_for_VLM/datasets/rexgradient/ReXGradient/metadata/rexgradient_train_age_likelihood.jsonl", age_likelihood_json_prediction_object))) 
print("keyword attack accuracy: ", str(analyze_age_or_keyword_attack("/project2/ruishanl_1185/SDP_for_VLM/datasets/rexgradient/likelihood.jsonl", keyword_likelihood_json_prediction_object))) 





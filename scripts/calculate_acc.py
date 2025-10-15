import json, re, argparse

# ---------------- Helpers: simple normalization + tokens ----------------
_ARTICLES = {"a", "an", "the"}

def normalize(text: str) -> str:
    # lower, remove punctuation, collapse spaces
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # drop articles
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)

def tokens(text: str):
    return set(normalize(text).split())

def recall_score(pred: str, gt: str) -> float:
    gt_t = tokens(gt)
    if not gt_t:
        return 0.0
    pred_t = tokens(pred)
    return len(gt_t & pred_t) / len(gt_t)

# =========================== CLI Args ===========================
parser = argparse.ArgumentParser(description="Evaluate LLaVA-Med predictions on SLAKE.")
parser.add_argument("--pred", required=True, help="Path to the predictions .jsonl file")
parser.add_argument("--test", required=True, help="Path to the ground truth .json file")
args = parser.parse_args()

answer_file = args.pred
test_file = args.test

# ======================= Load predictions ====================
predictions = [json.loads(line) for line in open(answer_file, "r")]

# ========================= Ground truth ======================
test_data = json.load(open(test_file, "r"))
closed_gt, open_gt = {}, {}
for item in test_data:
    qid = item.get("question_id") or item.get("id")
    if "conversations" not in item:
        continue
    gt = None
    for conv in item["conversations"]:
        if conv.get("from") == "gpt":
            gt = conv["value"]
            break
    if gt is None:
        continue
    if item.get("answer_type") == "CLOSED":
        closed_gt[qid] = gt
    elif item.get("answer_type") == "OPEN":
        open_gt[qid] = gt

# ===================== Closed-set accuracy ===================
closed_total = len(closed_gt)
closed_correct, closed_wrong = 0, []
for pred in predictions:
    qid = pred["question_id"]
    if qid not in closed_gt:
        continue
    pred_ans = normalize(pred["text"])
    gt_ans = normalize(closed_gt[qid])
    if pred_ans == gt_ans or gt_ans in pred_ans:
        closed_correct += 1
    else:
        closed_wrong.append({"qid": qid, "pred": pred["text"], "gt": closed_gt[qid]})

closed_acc = 100.0 * closed_correct / closed_total if closed_total else 0.0

# ===================== Open-set recall =======================
open_recalls, open_wrong = [], []
for pred in predictions:
    qid = pred["question_id"]
    if qid not in open_gt:
        continue
    r = recall_score(pred["text"], open_gt[qid])
    open_recalls.append(r)
    if r < 1.0:
        open_wrong.append({"qid": qid, "pred": pred["text"], "gt": open_gt[qid], "recall": round(r, 3)})

open_avg_recall = 100.0 * sum(open_recalls) / len(open_recalls) if open_recalls else 0.0

# ========================= Report ============================
print("\n========================================")
print("Evaluation Metrics")
print("========================================")
print(f"Closed questions: {closed_total} | Accuracy: {closed_acc:.2f}% ({closed_correct}/{closed_total})")
print(f"Open questions  : {len(open_gt)} | Avg Recall: {open_avg_recall:.2f}%")
print("========================================\n")

if closed_wrong or open_wrong:
    print("Examples (incorrect / partial):")
    print("----------------------------------------")
    for w in closed_wrong[:20]:
        print(f"[Closed] QID: {w['qid']}\n  GT:   {w['gt']}\n  PRED: {w['pred']}\n")
    for w in open_wrong[:20]:
        print(f"[Open]  QID: {w['qid']}\n  GT:   {w['gt']}\n  PRED: {w['pred']}\n  Recall: {w['recall']}\n")
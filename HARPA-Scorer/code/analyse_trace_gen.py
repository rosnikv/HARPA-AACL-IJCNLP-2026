import json
import re
from pathlib import Path

# ------------------ File Paths ------------------ #
results_path = Path("smoke_test_results/batch_results.jsonl")
gt_path = Path("smoke_test_results/corresponding_input_ds.jsonl")

# ------------------ Load Files ------------------ #
with open(results_path, "r") as f:
    generated = [json.loads(line) for line in f]

with open(gt_path, "r") as f:
    #ground_truth = {item["custom_id"]: item["winner"].strip().upper()[-1] for item in json.load(f)}
    ground_truth = {
        item["custom_id"]: item["winner"].strip().upper()[-1]
        for item in (json.loads(line) for line in f if line.strip())
    }


# ------------------ Helper Function ------------------ #
def extract_section(text, tag):
    pattern = fr"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else "N/A"

# ------------------ Parse + Evaluate ------------------ #
results = []

for item in generated:
    task_id = item["custom_id"]
    #content = item["result"]["message"]["content"][0]["text"]
    content = item["response"]["text"]

    rubric = extract_section(content, "rubric")
    evaluation = extract_section(content, "eval")
    prediction_match = re.search(r"<answer>\s*\[\[([AB])\]\]\s*</answer>", content)
    prediction = prediction_match.group(1) if prediction_match else "N/A"

    gt = ground_truth.get(task_id)
    correct = prediction == gt

    results.append({
        "custom_id": task_id,
        "prediction": prediction,
        "ground_truth": gt,
        "is_correct": correct,
        "rubric": rubric,
        "evaluation": evaluation,
    })

# ------------------ Pretty Print ------------------ #
print("\n===================== SUMMARY =====================\n")
correct = 0
for r in results:
    print(f"🔹 Task ID:        {r['custom_id']}")
    print(f"   Ground Truth:  {r['ground_truth']}")
    print(f"   Prediction:    {r['prediction']} ✅" if r["is_correct"] else f"   Prediction:    {r['prediction']} ❌")
    print("\n📘 Rubric:\n", r["rubric"])
    print("\n📊 Evaluation Summary:\n", r["evaluation"])
    print("\n---------------------------------------------------\n")

# ------------------ Accuracy ------------------ #
total = len(results)
correct = sum(1 for r in results if r["is_correct"])
print(f"✅ Overall Accuracy: {correct}/{total} = {correct / total:.2%}")

# ------------------ Save Updated Batch Results ------------------ #
out_path = Path("smoke_test_results/batch_results_with_gt.jsonl")
with open(out_path, "w") as f:
    for item, r in zip(generated, results):
        item.update({
            "ground_truth": r["ground_truth"],
            "is_correct": r["is_correct"]
        })
        f.write(json.dumps(item) + "\n")

print(f"\n💾 Saved updated results to {out_path}")

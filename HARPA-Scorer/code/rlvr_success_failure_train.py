import json
from datasets import load_from_disk, Dataset
from itertools import product

# ---------------- Load split dataset ---------------- #
split_ds = load_from_disk("rmr1_sft_split_dataset")

# Extract all val/test proposals
def extract_proposals_from_split(split):
    proposals = set()
    for ex in split:
        content = ex['context_messages'][1]['content']
        p_a = content.split('[The Start of Proposal A]')[1].split('[The End of Proposal A]')[0].strip()
        p_b = content.split('[The Start of Proposal B]')[1].split('[The End of Proposal B]')[0].strip()
        proposals.add(p_a)
        proposals.add(p_b)
    return proposals

val_props = extract_proposals_from_split(split_ds["validation"])
test_props = extract_proposals_from_split(split_ds["test"])
blocked_proposals = val_props.union(test_props)
print(f"🚫 Blocked {len(blocked_proposals)} proposals from val/test")

# ---------------- Load all entries ---------------- #
with open("data_bank_harpa_with_topics.jsonl") as f:
    entries = [json.loads(line) for line in f]

def has_text(e):
    t = e.get("hypothesis_proposal")
    return isinstance(t, str) and t.strip() != ""

success_entries = [e for e in entries if e.get("execution_success") == "success" and has_text(e)]
failure_entries = [e for e in entries if e.get("execution_success") == "failure" and has_text(e)]

print(f"✅ {len(success_entries)} success entries, {len(failure_entries)} failure entries")

# ---------------- Build pairs, filter ---------------- #
pairs = []
for sa, fa in product(success_entries, failure_entries):
    # Skip if either proposal is in validation/test
    if sa["hypothesis_proposal"] in blocked_proposals or fa["hypothesis_proposal"] in blocked_proposals:
        continue

    a_id, b_id = sa["idea_id"], fa["idea_id"]

    # deterministic shuffle
    if hash(f"{a_id}|{b_id}") % 2 == 0:
        response_a, response_b = sa["hypothesis_proposal"], fa["hypothesis_proposal"]
        ground_truth = "response_a"
        meta_a, meta_b = sa, fa
    else:
        response_a, response_b = fa["hypothesis_proposal"], sa["hypothesis_proposal"]
        ground_truth = "response_b"
        meta_a, meta_b = fa, sa

    ex = {
        "prompt": "Generate a feasible and testable research hypothesis and proposal.",
        "response_a": response_a,
        "response_b": response_b,
        "meta_data_response_a": meta_a,
        "meta_data_response_b": meta_b,
        "ground_truth": ground_truth,
        "labels": ("success", "failure"),
    }
    pairs.append(ex)

print(f"Created {len(pairs)} success–failure training pairs")

# ---------------- Save as HuggingFace dataset ---------------- #
train_ds = Dataset.from_list(pairs)
train_ds.save_to_disk("rlvr_success_failure_train")
print("Saved to rlvr_success_failure_train")

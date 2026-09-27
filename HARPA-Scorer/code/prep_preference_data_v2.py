from collections import defaultdict
from itertools import combinations, product
import json

# -------- Config --------
INPUT_PATH  = "data_bank_harpa_with_topics.jsonl"
OUTPUT_PATH = "preference_dataset.jsonl"

# allowed label combos
ALLOWED = {
    ("success", "uncertain"),
    ("uncertain", "failure"),
    ("success", "failure"),
}
# ranking for ground truth
RANK = {"success": 2, "uncertain": 1, "failure": 0}

def get_root(idea_id: str) -> str:
    return idea_id.rsplit("_", 1)[0] if isinstance(idea_id, str) and "_" in idea_id else idea_id

def has_text(e):
    t = e.get("hypothesis_proposal")
    return isinstance(t, str) and t.strip() != ""

def make_prompt(topic: str):
    return f"Generate a research hypothesis and proposal given this topic: {topic}"
# todo: consider success vs error since it is already from topic (revisit)

def build_pairs_by_root_labels(entries, label_key="execution_success", require_same_root=True):
    # 1) group by shared_topic
    topic_groups = defaultdict(list)
    for e in entries:
        topic_groups[e.get("shared_topic", "unknown")].append(e)

    pairs = []
    for topic, group in topic_groups.items():

        # 2) optionally bucket by root within topic (core logic kept; just wrapped by flag)
        if require_same_root:
            groups = defaultdict(list)
            for e in group:
                rid = get_root(e.get("idea_id"))
                lbl = e.get(label_key)
                if not rid or not has_text(e) or lbl not in RANK:
                    continue
                groups[rid].append(e)
        else:
            # same topic, any root
            groups = {None: [e for e in group if has_text(e) and e.get(label_key) in RANK]}

        # 3) iterate each (topic, root?) bucket
        for r, items in groups.items():
            # bucket entries by label (unchanged)
            buckets = defaultdict(list)
            for e in items:
                buckets[e[label_key]].append(e)

            # label-to-label combinations (unchanged)
            for la, lb in combinations(sorted(buckets.keys()), 2):
                if (la, lb) not in ALLOWED and (lb, la) not in ALLOWED:
                    continue

                # Cartesian product across the two label buckets (unchanged)
                for ea, eb in product(buckets[la], buckets[lb]):
                    a_id, b_id = ea["idea_id"], eb["idea_id"]

                    # decide preferred by rank (unchanged)
                    preferred_id = a_id if RANK[la] > RANK[lb] else b_id

                    # deterministic A/B shuffle, ground_truth tied to preferred (unchanged)
                    if hash(f"{a_id}|{b_id}") % 2 == 0:
                        response_a, response_b = ea["hypothesis_proposal"], eb["hypothesis_proposal"]
                        meta_a, meta_b = ea, eb
                        ground_truth = "response_a" if preferred_id == a_id else "response_b"
                    else:
                        response_a, response_b = eb["hypothesis_proposal"], ea["hypothesis_proposal"]
                        meta_a, meta_b = eb, ea
                        ground_truth = "response_a" if preferred_id == b_id else "response_b"

                    ex = {
                        "prompt": make_prompt(topic),
                        "response_a": response_a,
                        "response_b": response_b,
                        "meta_data_response_a": meta_a,
                        "meta_data_response_b": meta_b,
                        "ground_truth": ground_truth,
                        "pair_topic": topic,
                        "labels": (la, lb),
                    }
                    if require_same_root:
                        ex["pair_root"] = r
                    pairs.append(ex)
    return pairs

# -------- Run --------
with open(INPUT_PATH) as f:
    entries = [json.loads(line) for line in f]

# set require_same_root=True for same topic + same root; False for same topic (any root)
pairs = build_pairs_by_root_labels(entries, label_key="execution_success", require_same_root=False)

with open(OUTPUT_PATH, "w") as f:
    for ex in pairs:
        f.write(json.dumps(ex) + "\n")

print(f"Created {len(pairs)} preference examples in {OUTPUT_PATH}")

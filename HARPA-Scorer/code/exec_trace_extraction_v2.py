import os, re, glob, json
from functools import lru_cache
import pandas as pd

BASE = "../../../../cs-experiment-analysis"
OUT_PATH = "data_bank_harpa.jsonl"

# ---- helpers ----
def write_jsonl(d, p):
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(d) + "\n")

def _pick_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No match: {pattern}")
    def key(p):
        m = re.search(r"(20\d{6})", os.path.basename(p))  # e.g., 20250816
        return (m.group(1) if m else "", os.path.getmtime(p))
    return sorted(files, key=key)[-1]

@lru_cache(maxsize=None)
def load_assets(group):
    # loads once per Group, like your originals but per-batch
    exp_path  = _pick_latest(os.path.join(BASE, f"all-experiments*_{group}.json"))
    #idea_path = _pick_latest(os.path.join(BASE, f"benchmark_harpa*_{group}.json"))
    idea_path = _pick_latest(os.path.join(BASE, f"benchmark_[Hh][Aa][Rr][Pp][Aa]*_{group}.json"))
    paper_path = _pick_latest(os.path.join(f"source_paper_index.json"))

    with open(idea_path, encoding="utf-8") as f:
        idea_md = {it["id"]: it for it in json.load(f)}
    with open(exp_path, encoding="utf-8") as f:
        exp_list = json.load(f).get("experiment_list", [])
    exp_by_idea = {e.get("idea_id"): e for e in exp_list if e.get("idea_id")}
    with open(paper_path, encoding="utf-8") as f:
        papers = json.load(f)  # dict keyed by idea_id
    return idea_md, exp_by_idea, papers

def build_example_from_row(row, idea_md, exp_by_idea, papers):
    idea_id = row["idea_id"]
    group   = row["Group"]

    idea_info = idea_md.get(idea_id)
    if not idea_info:
        print(f"Missing metadata for {idea_id} in {group}")
        return None

    exp_info = exp_by_idea.get(idea_id, {})
    source_paper = (papers.get(idea_id) or {})

    # experiment path per your rule
    exp_full_path = os.path.join(
        BASE, f"generated-experiments_{group}", str(row["best_run_experiment"])
    )
    history_path = os.path.join(exp_full_path, "history.json")

    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, encoding="utf-8") as f:
                history = (json.load(f) or {}).get("metadata", {}) or {}
        except Exception as e:
            print(f"History read error {history_path}: {e}")

    # build source_text like original
    source_text = (
        f"Source paper Title: {source_paper.get('title','')}\n"
        f"Short paper Abstract: {source_paper.get('abstract','')}\n"
        f"Source paper Citation count: {source_paper.get('citation_count','')}\n\n"
        f"Source paper Year: {source_paper.get('year','')}\n\n"
    )

    # metrics like original  ### TODO ADD HERE like NL sentence
    total_cost = history.get("total_cost", 0)
    max_cost   = history.get("max_experiment_cost", 1) or 1
    cost_eff   = max(0.0, 1.0 - (total_cost / max(1e-6, max_cost)))
    complexity = min(1.0, (history.get("num_reflections", 0) /
                           max(1, history.get("max_reflections", 1))))
    # --- Add natural language strings ---
    cost_eff_str = f"Used {total_cost} out of {max_cost} allowed cost."
    complexity_str = f"{history.get('num_reflections', 0)} out of {history.get('max_reflections', 1)} reflections used."

    idea_hypothesis = (idea_info.get("research_idea_hypothesis","") or "").strip()
    long_description = (idea_info.get("research_idea_long_description","") or "").strip()
    operationalization = ((idea_info.get("operationalization") or {})
                          .get("operationalization_description","") or "").strip()

    proposal_text = "\n".join(
        p for p in [f"Description: {long_description}",
                    f"Operationalization plan: {operationalization}"] if p.strip()
    )
    hypotheses_proposal = (
        f"Here is the proposed Hypothesis: {idea_hypothesis}\n\n"
        f"Research Proposal:\n{proposal_text}\n\n"
    )

    change_log = history.get("change_log", []) or []
    return {
        "idea_id": idea_id,
        "file_id": idea_info.get("file_id"),
        "experiment_path": exp_full_path,  # <- your constructed path
        "experiment_status": exp_info.get("status", "unknown"),
        "full_experiment_summary": exp_info.get("results_summary", {}),
        "source_text": source_text,
        "hypothesis_proposal": hypotheses_proposal,
        "summary_results": history.get("summary_results", {}),
        #"harpa_cost_efficiency": cost_eff,
        "execution_success": row.get("final_label"),
        #"complexity_score": complexity,
        "harpa_cost_efficiency": cost_eff_str,
        "complexity_score": complexity_str,
        "agent_latest_issues_handled": change_log[:2],
        "group": group
    }

# ---- run ----
# gather CSVs -> consolidated df
csvs = glob.glob(f"{BASE}/*.csv")
df = pd.concat((pd.read_csv(p) for p in csvs), ignore_index=True)

# clear output
open(OUT_PATH, "w", encoding="utf-8").close()
import sys

examples = []
for _, r in df.iterrows():
    idea_md, exp_by_idea, papers = load_assets(r["Group"])
    #print(exp_by_idea["idea_010_1"])
    
    ex = build_example_from_row(r, idea_md, exp_by_idea, papers)
    if ex:
        write_jsonl(ex, OUT_PATH)
        examples.append(ex)
    else:
        print(f"Skipped {r.get('idea_id')}")

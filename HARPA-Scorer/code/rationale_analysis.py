import re
import pandas as pd
import evaluate
import sacrebleu
from scipy.stats import ttest_rel, wilcoxon

# ========== CONFIG ==========
# ========== CONFIG ==========
file_baseline = "/weka/harpa-cs/harpa-rlvr-branch/hypothesis_generation/ASD-specific/RM-R1-data/validation_baseline_sep21_predictions_all_data_with_gt_reas.csv"
file_system   = "/weka/harpa-cs/harpa-rlvr-branch/hypothesis_generation/ASD-specific/RM-R1-data/validation_harpa_rlvr_sep21_predictions_all_data_with_gt_reas.csv"

rubric_dims = [
    "Execution Success",
    "Complexity",
    "Cost Efficiency",
    "Expected Hypothesis Validity",
    "Expected Interestingness",
    "Faithfulness"
]

# ========== CLEANING ==========
def clean_text(text):
    if not isinstance(text, str):
        return ""
    for dim in rubric_dims:
        text = re.sub(dim, "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\d+[\.\-:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ========== LOAD DATA ==========
df_base = pd.read_csv(file_baseline)
df_sys  = pd.read_csv(file_system)

assert len(df_base) == len(df_sys), "Mismatch in dataset sizes!"

references = df_base["ground_truth_reason"].apply(clean_text).tolist()
preds_base = df_base["model_output"].apply(clean_text).tolist()
preds_sys  = df_sys["model_output"].apply(clean_text).tolist()

# ========== METRICS ==========
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# Per-example ROUGE
rouge_base = rouge.compute(predictions=preds_base, references=references, use_aggregator=False)
rouge_sys  = rouge.compute(predictions=preds_sys,  references=references, use_aggregator=False)

# Per-example BLEU (sentence-level)
bleu_scores_base = [sacrebleu.sentence_bleu(p, [r]).score for p, r in zip(preds_base, references)]
bleu_scores_sys  = [sacrebleu.sentence_bleu(p, [r]).score for p, r in zip(preds_sys,  references)]

# ========== STATISTICAL TESTS ==========
def significance_marker(p):
    if p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."

def run_tests(metric, base_scores, sys_scores):
    t_stat, t_p = ttest_rel(sys_scores, base_scores)
    w_stat, w_p = wilcoxon(sys_scores, base_scores)
    mean_base = sum(base_scores) / len(base_scores)
    mean_sys  = sum(sys_scores)  / len(sys_scores)
    
    mark_t = significance_marker(t_p)
    mark_w = significance_marker(w_p)

    print(f"\n{metric}:")
    print(f"  Baseline: {mean_base:.4f} | System: {mean_sys:.4f}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.4g} ({mark_t})")
    print(f"  Wilcoxon: W={w_stat:.3f}, p={w_p:.4g} ({mark_w})")

    # LaTeX-friendly line
    print(f"  LaTeX: {metric} & {mean_base:.3f} & {mean_sys:.3f}{mark_w} \\\\")

# Run for BLEU
run_tests("Sentence BLEU", bleu_scores_base, bleu_scores_sys)

# Run for each ROUGE variant
for metric in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
    run_tests(metric.upper(), rouge_base[metric], rouge_sys[metric])

import pandas as pd
import re
import numpy as np

# -------------------
# 1. Load CSV
# -------------------
df = pd.read_csv(
    "/weka/harpa-cs/harpa-rlvr-branch/hypothesis_generation/ASD-specific/"
    "RM-R1-data/validation_harpa_rlvr_sep21_predictions_all_data_with_gt_reas.csv"
)

# -------------------
# 2. Extract rubric scores for Proposal A and B
# -------------------
def extract_pair_scores(text: str):
    dims = [
        "Execution Success",
        "Complexity",
        "Cost Efficiency",
        "Expected Hypothesis Validity",
        "Expected Interestingness",
        "Faithfulness"
    ]
    scores_A, scores_B = {}, {}
    for dim in dims:
        # Regex that tolerates numbering, dashes, spaces
        patternA = rf"{dim}.*?Proposal A:\s*([01](?:\.\d+)?|\.\d+)"
        patternB = rf"{dim}.*?Proposal B:\s*([01](?:\.\d+)?|\.\d+)"
        matchA = re.search(patternA, str(text), flags=re.IGNORECASE | re.DOTALL)
        matchB = re.search(patternB, str(text), flags=re.IGNORECASE | re.DOTALL)
        scores_A[dim] = float(matchA.group(1)) if matchA else np.nan
        scores_B[dim] = float(matchB.group(1)) if matchB else np.nan
    return scores_A, scores_B

# Parse oracle + model
oracle_pairs = df["ground_truth_reason"].apply(extract_pair_scores)
model_pairs  = df["model_output"].apply(extract_pair_scores)

oracle_A = pd.DataFrame([p[0] for p in oracle_pairs])
oracle_B = pd.DataFrame([p[1] for p in oracle_pairs])
model_A  = pd.DataFrame([p[0] for p in model_pairs])
model_B  = pd.DataFrame([p[1] for p in model_pairs])

# -------------------
# Debug: Spot check parsing
# -------------------
print("\n=== Spot Check: First Row ===")
print("Oracle A scores:", oracle_A.iloc[0].to_dict())
print("Oracle B scores:", oracle_B.iloc[0].to_dict())
print("Model A scores :", model_A.iloc[0].to_dict())
print("Model B scores :", model_B.iloc[0].to_dict())

# Optional: check multiple rows
for i in range(3):
    print(f"\n--- Row {i} ---")
    print("Oracle A:", oracle_A.iloc[i].to_dict())
    print("Oracle B:", oracle_B.iloc[i].to_dict())
    print("Model A :", model_A.iloc[i].to_dict())
    print("Model B :", model_B.iloc[i].to_dict())

# -------------------
# 3. Row-wise pairwise consistency
# -------------------
dims = oracle_A.columns
results = []
all_agreements = 0
all_total = 0

for dim in dims:
    oracle_diff = oracle_A[dim] - oracle_B[dim]
    model_diff  = model_A[dim] - model_B[dim]
    if dim == "Complexity":
        oracle_diff = -oracle_diff
        model_diff  = -model_diff
    
    valid = (~oracle_diff.isna()) & (~model_diff.isna())
    oracle_diff = oracle_diff[valid]
    model_diff  = model_diff[valid]

    total = len(oracle_diff)
    if total == 0:
        continue

    # Agreement: both oracle & model prefer the same proposal
    agreement = ((oracle_diff > 0) & (model_diff > 0)) | ((oracle_diff < 0) & (model_diff < 0))
    consistency = agreement.sum() / total

    results.append({
        "Dimension": dim,
        "PairwiseConsistency": round(consistency, 3),
        "Samples": total
    })

    all_agreements += agreement.sum()
    all_total += total

# -------------------
# 4. Overall consistency
# -------------------
overall_consistency = all_agreements / all_total if all_total > 0 else np.nan
results.append({
    "Dimension": "Overall",
    "PairwiseConsistency": round(overall_consistency, 3),
    "Samples": all_total
})

# -------------------
# 5. Save results
# -------------------
results_df = pd.DataFrame(results)
print("\n=== Row-wise Pairwise Consistency Results ===")
print(results_df)

results_df.to_csv("trace_pairwise_consistency_rowwise.csv", index=False)

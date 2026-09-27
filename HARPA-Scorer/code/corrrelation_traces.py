import pandas as pd
import re
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# -------------------
# 1. Load CSV
# -------------------
df = pd.read_csv(
    "/weka/harpa-cs/harpa-rlvr-branch/hypothesis_generation/ASD-specific/"
    "RM-R1-data/validation_harpa_rlvr_sep21_predictions_all_data_with_gt_reas.csv"
)
# Uncomment below for baseline
# df = pd.read_csv(
#     "/weka/harpa-cs/harpa-rlvr-branch/hypothesis_generation/ASD-specific/"
#     "RM-R1-data/validation_baseline_sep21_predictions_all_data_with_gt_reas.csv"
# )

# -------------------
# 2. Extract rubric scores
# -------------------
def extract_scores(text: str):
    dims = {
        "Execution Success": r"Execution\s*Success",
        "Complexity": r"Complexity",
        "Cost Efficiency": r"Cost\s*Efficiency",
        "Expected Hypothesis Validity": r"(?:Hypothesis\s*Validity|Expected\s*Hypothesis\s*Validity)",
        "Expected Interestingness": r"(?:Interestingness|Expected\s*Interestingness)",
        "Faithfulness": r"Faithfulness"
    }

    scores = {}
    for dim, dim_pattern in dims.items():
        # Regex: capture numbers like 0, 1, 0.8, .75
        pattern = rf"{dim_pattern}.*?:\s*- Proposal [AB]:\s*((?:[01](?:\.\d+)?|\.\d+))"
        matches = re.findall(pattern, str(text), flags=re.IGNORECASE)
        if matches:
            vals = [float(m) for m in matches]
            scores[dim] = np.mean(vals)  # average across Proposal A/B
        else:
            scores[dim] = np.nan
    return scores

# Parse oracle + model traces
oracle_scores = df["ground_truth_reason"].apply(extract_scores)
model_scores = df["model_output"].apply(extract_scores)

oracle_df = pd.DataFrame(list(oracle_scores))
model_df = pd.DataFrame(list(model_scores))

# -------------------
# 3. Compute metrics
# -------------------
dims = oracle_df.columns
results = []

for dim in dims:
    # Align oracle & model, drop rows where either side is NaN
    pairs = pd.concat([oracle_df[dim], model_df[dim]], axis=1, keys=["oracle", "model"]).dropna()
    x = pairs["oracle"].to_numpy()
    y = pairs["model"].to_numpy()

    print(f"== {dim} ==")
    print("Samples after alignment:", len(x))

    if len(x) > 2:
        # Error metrics on raw scores
        mae = mean_absolute_error(x, y)
        rmse = mean_squared_error(x, y) ** 0.5  # manual RMSE for older sklearn

        # Bin scores
        bins = [0.0, 0.3, 0.7, 1.0]
        x_bins = np.digitize(x, bins)
        y_bins = np.digitize(y, bins)

        # Bin agreement
        bin_agreement = np.mean(x_bins == y_bins)

        # Correlation on bins
        pearson = pearsonr(x_bins, y_bins)[0]
        spearman = spearmanr(x_bins, y_bins)[0]

        results.append({
            "Dimension": dim,
            "Pearson_bins": round(pearson, 3),
            "Spearman_bins": round(spearman, 3),
            "MAE_raw": round(mae, 3),
            "RMSE_raw": round(rmse, 3),
            "BinAgreement": round(bin_agreement, 3),
            "Samples": len(x)
        })

# -------------------
# 4. Save results
# -------------------
results_df = pd.DataFrame(results)
print("\n=== Final Results ===")
print(results_df)

results_df.to_csv("trace_fidelity_results.csv", index=False)

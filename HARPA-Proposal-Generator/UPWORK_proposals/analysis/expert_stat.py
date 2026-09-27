import pandas as pd
import random
from scipy.stats import ttest_1samp, wilcoxon
import numpy as np
from tabulate import tabulate  # pip install tabulate

def bootstrap_resample(diff_scores:list, num_samples=10000, threshold=0):
    num_above_threshold = 0
    for _ in range(num_samples):
        resample = [random.choice(diff_scores) for _ in range(len(diff_scores))]
        mean_resample = sum(resample) / len(resample)
        if mean_resample > threshold:
            num_above_threshold += 1
    return 1 - (num_above_threshold / num_samples)

def stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return ""

if __name__ == "__main__":
    df = pd.read_csv("diff_scores_final.csv")

    results = []
    for col in df.columns:
        scores = df[col].dropna().tolist()

        boot_p = bootstrap_resample(scores)
        t_stat, t_p = ttest_1samp(scores, 0)
        try:
            w_stat, w_p = wilcoxon(scores)
        except ValueError:
            w_p = np.nan

        mean_diff = np.mean(scores)

        results.append([
            col, f"{mean_diff:.3f}",
            f"{boot_p:.3f}", stars(boot_p),
            f"{t_p:.3f}", stars(t_p),
            f"{w_p:.3f}", stars(w_p)
        ])

    headers = ["Dimension", "MeanDiff", "Bootstrap_p", "Boot*", "T_p", "T*", "Wilcoxon_p", "Wilcoxon*"]

    # use a lighter format
    print(tabulate(results, headers=headers, tablefmt="simple"))
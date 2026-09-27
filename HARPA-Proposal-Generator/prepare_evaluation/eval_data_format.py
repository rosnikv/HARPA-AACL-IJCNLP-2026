import os
import shutil
import json

def copy_htmls_and_generate_json(user, dest_root="user_annotations"):
    system_paths = {
        "system1": os.path.join("cache_results_harpa", "users", user),
        "system2": os.path.join("cache_results_test", user)
    }

    html_entries = {
        "system1": [],
        "system2": []
    }

    output_data = []

    for system, base_path in system_paths.items():
        if not os.path.exists(base_path):
            print(f"Skipping missing path: {base_path}")
            continue

        for root, _, files in os.walk(base_path):
            html_files = [f for f in files if f.endswith(".html")]
            for html_file in html_files:
                src = os.path.join(root, html_file)
                dst_dir = os.path.join(dest_root, user, system)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, html_file)
                shutil.copy2(src, dst)

                html_url = f"https://ROOT.github.io/assets/upwork/user_annotations/{user}/{system}/{html_file}"
                output_data.append({
                    "data": {
                        "html_url": html_url
                    }
                })
                html_entries[system].append((html_file, html_url))
                print(f"Copied: {src} -> {dst}")

    # Write user_name.json
    json_path = os.path.join(dest_root, f"{user}.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"JSON written to: {json_path}")

    # Generate comparison data (all pairs system1 x system2)
    comparison_data = []
    system1_files = html_entries["system1"]
    system2_files = html_entries["system2"]

    queries = {
        f"query{i}": q for i, q in enumerate([
            "1. <b>Which idea is more novel? </b> (“More novel” means the idea is creative, distinct from existing work, and brings fresh insights. Consider whether similar ideas have already appeared in papers — including preprints — and feel free to search online before judging.)",
            "2. <b>Which idea is more feasible to implement as a short research project? </b> (Assume a typical CS PhD student has 1–2 months, access to OpenAI/Anthropic APIs, but limited GPU compute. “More feasible” means it’s more practical to execute given these constraints — with fewer flaws, lower resource needs, and less complexity.)",
            "3. <b>Which idea is more likely to work well in practice (i.e., outperform existing baselines)? </b> (“More likely to work” means the idea has fewer flaws, and you expect it to yield better results than current methods on relevant benchmarks — either in general or in key scenarios.)",
            "4. <b>Which idea is more exciting and potentially impactful if fully executed as a research project? </b> (“More exciting” means it could make a meaningful contribution to the field — from being accepted at a major AI conference to possibly transforming the field or winning awards.)",
            "5. <b>Which idea is stronger overall and more likely to be accepted at a top AI conference (e.g., ACL, ICLR, NeurIPS)? </b> (“Stronger overall” means the idea is better in terms of clarity, novelty, feasibility, effectiveness, and impact — the kind of project you’d score higher using the 1–10 scale for major AI conferences.)",
            "6. <b>Which idea is better grounded in existing scientific literature? </b> (“Better grounded” means more of the key ideas, terms, or methods are clearly supported by relevant citations, prior work, or established concepts — not speculative or hallucinated.)",
            "7. <b>Which idea is more clearly motivated by prior research? </b> (“More clearly motivated” means the idea addresses a known gap, trend, or limitation in existing literature with a compelling and well-grounded rationale.)",
            "8. <b>Which idea has a more coherent and logically integrated composition?</b> (“More coherent” means the methods, variables, and tasks fit together naturally, with clear rationale and support from existing literature.)",
            "9. <b>Which idea presents a clearer and more testable research goal or direction? </b>(This includes either an explicit hypothesis or a well-defined research objective. “Clearer” means it’s easier to understand what the project aims to test or achieve, and how success could be evaluated.)"
        ], start=1)
    }

    for sys1_file, sys1_url in system1_files:
        for sys2_file, sys2_url in system2_files:
            comparison_data.append({
                "doc1": sys1_url,
                "doc2": sys2_url,
                **queries
            })

    # Write user_name_comparison.json
    comparison_path = os.path.join(dest_root, f"{user}_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Comparison JSON written to: {comparison_path}")

# Example usage
copy_htmls_and_generate_json("user_6c352d23")

import os
import json
import re
from pathlib import Path

users_dir = "cache_results_harpa/users"
output_path = "beaker_jobs/source_paper_index.json"

idea_to_paper = {}

for user in sorted(os.listdir(users_dir)):
    if not user.startswith("user-"):
        continue

    user_id = user.split('-')[-1]
    final_dir = os.path.join(users_dir, user, "harpa_proposals", "final")
    if not os.path.isdir(final_dir):
        continue

    for fname in sorted(os.listdir(final_dir)):
        if not fname.endswith(".json"):
            continue

        match = re.search(r'_p-(\d+)', fname)
        if not match:
            print(f"[WARN] Could not extract local counter from {fname}")
            continue

        local_counter = match.group(1)
        idea_id = f"idea_{user_id}_{local_counter}"

        file_path = os.path.join(final_dir, fname)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            source = data[0]["chain"][0]  # Assume same format everywhere

            idea_to_paper[idea_id] = {
                "title": source.get("title", ""),
                "abstract": source.get("abstract", ""),
                "citation_count": source.get("citation_count", 0),
                "year": source.get("year", "")
            }

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

# Save result
with open(output_path, "w", encoding="utf-8") as out:
    json.dump(idea_to_paper, out, indent=2)

print(f"\nSaved {len(idea_to_paper)} entries to {output_path}")

import os
import json
import time
import requests
import argparse
from pathlib import Path
import difflib
import sys
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
# ----------------- External Enrichment Utilities -----------------
def lookup_paper_id_by_title(title, year=None):
    if not title:
        return None

    print(f"Looking up paperId for: {title}")
    results = search_papers(title, limit=3)  # Get top 3 to improve match accuracy

    for paper in results:
        if year and paper.get("year") != year:
            continue
        if paper.get("title", "").lower() == title.lower():
            return paper.get("paperId")

    # fallback: return the first one if exact match not found
    return results[0].get("paperId") if results else None

def search_papers(query, limit=1):
    api_key = os.environ.get("S2_API_KEY", None)
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": api_key} if api_key else {}
    params = {
        "query": query,
        "fields": 'title,year,url,paperId',
        "limit": limit
    }
    max_retries = 3
    backoff_factor = 3

    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            print(f'Attempt {attempt + 1} failed: Status code {response.status_code}')
            time.sleep(backoff_factor * (2 ** attempt))

    print("Failed to fetch data after retries.")
    return None

def build_variable_to_paper_lookup(entry):
    lookup = {}
    variable_space = entry.get("variable_space", [])
    
    for category in variable_space:
        if not isinstance(category, dict):
            continue
        for _, items in category.items():
            if isinstance(items, list):
                for item in items:
                    value_name = item.get("value_name")
                    source_paper = item.get("source_paper")
                    if value_name and source_paper:
                        lookup[value_name] = source_paper
    return lookup


def extract_core_titles_from_variables(entry):
    look_up = build_variable_to_paper_lookup(entry)
    lookup_keys = list(look_up.keys())
    lookup_embeddings = model.encode(lookup_keys, convert_to_tensor=True)

    var_dict = entry.get("specific_hypothesis", {}) \
                    .get("research_idea_long_description", {}) \
                    .get("research_idea_variables", {})
    result = {}
    for var_name, var_def in var_dict.items():
        closest = difflib.get_close_matches(var_name, lookup_keys, n=1, cutoff=0.8)
        if closest:
            matched_key = closest[0]
            result[var_name] = look_up[matched_key]
            print(f"[DIFFLIB MATCH] '{var_name}' → '{matched_key}'")
            continue
        
        var_text = f"{var_name}: {var_def[:100]}"
        print(var_text)
        var_embedding = model.encode(var_text, convert_to_tensor=True)

        cos_scores = util.pytorch_cos_sim(var_embedding, lookup_embeddings)[0]
        top_scores, top_indices = cos_scores.topk(3)
        for rank, (score_tensor, idx_tensor) in enumerate(zip(top_scores, top_indices), start=1):
            score = score_tensor.item()
            idx = idx_tensor.item()
            match = look_up[lookup_keys[idx]]
            result[var_name] = match
            print(f"  {rank}. {lookup_keys[idx]} - {match} (cosine similarity: {score:.2f})")
            if match != "LLM-recommended":
                break

    return result

def extract_core_titles_from_variables_v1(entry):
    lookup = build_variable_to_paper_lookup(entry)
    key_var = entry.get("specific_hypothesis", {}) \
                   .get("research_idea_long_description", {}) \
                   .get("research_idea_variables", {}) \
                   .keys()
    #print(lookup.keys())
    result = {}
    for var in key_var:
        closest = difflib.get_close_matches(var, lookup.keys(), n=1, cutoff=0.7)
        if closest:
            print(f"Matched variable '{var}' to paper title '{closest}'")
            matched_key = closest[0]
            result[var] = lookup[matched_key]
        else:
            result[var] = None
    return result

def enrich_entry_with_papers(entry):
    entry["core_related_works_enriched"] = []
    core_titles = extract_core_titles_from_variables(entry)

    for var, title in core_titles.items():
        if not title or title == "LLM-recommended":
            continue
        print(f"Searching Semantic Scholar for: {title}")
        papers = search_papers(title)
        if papers:
            paper_info = papers[0]
            entry["core_related_works_enriched"].append({
                "title": paper_info.get("title"),
                "paperId": paper_info.get("paperId"),
                "url": paper_info.get("url"),
                "year": paper_info.get("year"),
                "source_variable": var,
                "source_paper_lookup_title": title
            })
        else:
            print(f"No paper found for: {title}")
        time.sleep(1)

# ----------------- Paper Extraction -----------------

def extract_papers(data):
    seen_titles = set()
    unique_papers = []

    def add_paper(paper):
        title = paper.get("title")
        if not title or title in seen_titles:
            return
        seen_titles.add(title)
        year = paper.get("year")
        pid = paper.get("paperId") or lookup_paper_id_by_title(title, year)

        unique_papers.append({
            "title": title,
            "year": year,
            "paperId": pid
        })
    
    # 1. From original `chain`
    for paper in data.get("chain", []):
        if isinstance(paper, dict):
            add_paper(paper)
            
    # 3. From `refined_hypothesis_details.related_work_analysis[*].novelty_assessment.supporting_papers`
    related_work_analyses = data.get("refined_hypothesis_details", {}).get("related_work_analysis", [])
    for analysis in related_work_analyses:
        papers = analysis.get("novelty_assessment", {}).get("supporting_papers", [])
        for paper in papers:
            add_paper(paper)
            
    return unique_papers

# ----------------- Main -----------------

def main():
    parser = argparse.ArgumentParser(description="Append related works into HARPA rubric files.")
    parser.add_argument(
        "--final_dir",
        type=str,
        default="../test_harpa/stage1_stage2/",
        help="Base directory containing final/rubric files in **/final/ subfolders."
    )
    args = parser.parse_args()

    rubrics_base_dir = Path(args.final_dir)

    for rubric_file in rubrics_base_dir.glob("**/final/*.json"):
        try:
            file_id = rubric_file.stem.split('_')[-2]
            print(file_id)
            #if file_id == 'cb754310302086dfbbcd098263200e2a03f65874':
            with open(rubric_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list) or not data:
                continue

            entry = data[0]

            # Extract related papers from chain
            entry["collected_related_works"] = extract_papers(entry)

            # Enrich core variable titles with semantic search
            enrich_entry_with_papers(entry)

            # Deduplicate and append enriched papers to collected_related_works
            existing_ids = {p.get("paperId") for p in entry["collected_related_works"] if p.get("paperId")}
            enriched = entry.get("core_related_works_enriched", [])

            for paper in enriched:
                pid = paper.get("paperId")
                if pid and pid not in existing_ids:
                    entry["collected_related_works"].append({
                        "title": paper.get("title"),
                        "year": paper.get("year"),
                        "paperId": pid
                    })
            #sys.exit()
            
            with open(rubric_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"Updated: {rubric_file.name}")
            #else:
            #    print(f"Skipping {rubric_file.name} as it does not match the expected file ID.")
        except Exception as e:
            print(f"Failed on {rubric_file.name}: {e}")

    print("All files updated.")

if __name__ == "__main__":
    main()

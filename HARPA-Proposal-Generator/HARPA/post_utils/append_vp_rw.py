import os
import json
import time
import requests
import argparse
from pathlib import Path
import difflib

# ----------------- External Enrichment Utilities -----------------

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
    lookup = build_variable_to_paper_lookup(entry)
    key_var = entry.get("specific_hypothesis", {}) \
                   .get("research_idea_long_description", {}) \
                   .get("research_idea_variables", {}) \
                   .keys()
    
    result = {}
    for var in key_var:
        closest = difflib.get_close_matches(var, lookup.keys(), n=1, cutoff=0.6)
        if closest:
            matched_key = closest[0]
            result[var] = lookup[matched_key]
        else:
            result[var] = None
    return result

def enrich_entry_with_papers(entry):
    entry["core_related_works_enriched"] = []
    core_titles = extract_core_titles_from_variables(entry)

    for var, title in core_titles.items():
        if not title:
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

# ----------------- Original Code -----------------

def extract_papers(data):
    seen_titles = set()
    unique_papers = []

    for paper in data.get("chain", []):
        if not isinstance(paper, dict):
            continue
        title = paper.get("title")
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        unique_papers.append({
            "title": title,
            "year": paper.get("year"),
            "paperId": paper.get("paperId")
        })

    '''
    viewpoint_data = data.get("viewpoint_analysis", {})
    for enrich in viewpoint_data.get("s2_enrichment") or []:
        if not isinstance(enrich, dict):
            continue
        for snippet in enrich.get("s2_snippets") or []:
            paper_data = snippet.get("paper", {}).get("new", {})
            title = paper_data.get("title")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            unique_papers.append({
                "title": title,
                "year": paper_data.get("year"),
                "paperId": paper_data.get("paperId")
            })
    '''
    return unique_papers

def main():
    parser = argparse.ArgumentParser(description="Merge enriched snippets and related works into HARPA rubric files.")
    parser.add_argument(
        "--snippets_dir",
        type=str,
        default="../D-scorer/data/enriched_snippets/",
        help="Directory containing enriched snippet JSON files."
    )
    parser.add_argument(
        "--final_dir",
        type=str,
        default="../test_harpa/stage1_stage2/",
        help="Base directory containing final/rubric files in **/final/ subfolders."
    )
    args = parser.parse_args()

    snippets_dir = Path(args.snippets_dir)
    rubrics_base_dir = Path(args.final_dir)

    snippet_files = list(snippets_dir.glob("*.json"))
    id_to_snippet = {}

    for snippet_path in snippet_files:
        file_id = snippet_path.stem.split("_")[0]
        with open(snippet_path, 'r', encoding='utf-8') as f:
            id_to_snippet[file_id] = json.load(f)

    for rubric_file in rubrics_base_dir.glob("**/final/*.json"):
        try:
            file_id = rubric_file.stem.split('_')[-2]

            with open(rubric_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list) or not data:
                continue

            entry = data[0]

            # Inject enriched snippet info
            if file_id in id_to_snippet:
                entry["viewpoint_analysis"] = id_to_snippet[file_id]

            # Extract related papers from chain/snippets
            entry["collected_related_works"] = extract_papers(entry)

            # Enrich core variable titles with semantic search
            enrich_entry_with_papers(entry)
            
            # Add core_related_works_enriched into collected_related_works with deduplication
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

            with open(rubric_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"Updated: {rubric_file.name}")

        except Exception as e:
            print(f"Failed on {rubric_file.name}: {e}")

    print("All files updated.")

if __name__ == "__main__":
    main()

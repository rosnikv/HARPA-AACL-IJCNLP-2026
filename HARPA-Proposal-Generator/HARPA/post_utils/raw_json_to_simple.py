import os
import json
import re
import argparse

# === HELPER FUNCTIONS ===

def get_value_by_path(obj, path):
    try:
        tokens = re.findall(r'\w+|\[\d+\]', path)
        for token in tokens:
            if token.startswith('['):
                index = int(token[1:-1])
                obj = obj[index]
            else:
                obj = obj[token]
        return obj
    except Exception:
        return None

def extract_simplified_json(data, hex_id=None, operationalization_data=None):
    simplified = {"paper_id": hex_id}
    summary = {}

    for section_title, path_config in custom_sections:
        if section_title.lower().__contains__("rubric"):
            # Keep rubric at top-level
            value = get_value_by_path(data, path_config)
            if value:
                simplified["research_idea_evaluation_rubric"] = value
            continue

        # Add to summary
        if isinstance(path_config, list):
            for label, paths in path_config:
                paths = [paths] if isinstance(paths, str) else paths
                value = None
                for path in paths:
                    value = get_value_by_path(data, path)
                    if value:
                        break
                if value:
                    key = label or section_title

                    # Flatten nested dicts to "key: value\n..." string
                    if isinstance(value, dict) and key == "research_idea_variables":
                        # Add markdown links to variable names
                        links = {}
                        enriched = data[0].get("core_related_works_enriched", [])
                        for item in enriched:
                            var = item.get("source_variable")
                            url = item.get("url")
                            if var and url:
                                links[var] = f"[{var}]({url})"

                        linked_lines = []
                        for var, desc in value.items():
                            display_var = links.get(var, var)
                            linked_lines.append(f"{display_var}: {desc}")
                        
                        summary[key] = "\n\n".join(linked_lines)

                    elif isinstance(value, dict):
                        summary[key] = "\n\n".join([f"{k}: {v}" for k, v in value.items()])

                    else:
                        summary[key] = value
        else:
            value = get_value_by_path(data, path_config)
            if value:
                summary[section_title] = value

    # Extract elements directly from top-level JSON
    elements = data[0].get("specific_hypothesis", {}).get("elements", {})
    if elements:
        element_lines = []
        for k, v in elements.items():
            element_lines.append(f"{k}: {v}")
        summary["research_idea_elements"] = "\n\n".join(element_lines)

        
    # === Compose motivation as a single text field ===
    initial_details = data[0].get("initial_hypothesis_details", {})
    rationale = initial_details.get("Rationale", "").strip()
    chain = data[0].get("chain", [])

    if chain:
        # Source paper
        source = chain[0]
        title = source.get("title", "Untitled")
        year = source.get("year", "n.d.")
        cites = source.get("citation_count", 0)
        pid = source.get("paperId", "N/A")
        source_link = f"[Paper 0: {title}](https://www.semanticscholar.org/paper/{pid})"

        # Related papers (excluding source), deduplicated and with titles
        seen = set()
        related_info = []
        for p in chain[1:]:
            rid = p.get("paperId")
            rtitle = p.get("title", "Untitled")
            if rid and rid not in seen:
                seen.add(rid)
                related_info.append((rtitle, rid))

        related_links = [
            f"[Paper {i+1}](https://www.semanticscholar.org/paper/{rid})"
            for i, (rtitle, rid) in enumerate(related_info)
        ]
        related_links_text = " --> ".join(related_links)
    
        # Compose text
        motivation_text = (
            f'The source paper is {source_link} ({cites} citations, {year}). '
            f'This idea draws upon a trajectory of prior work, as seen in the following sequence: {related_links_text}.'
            #f'This idea builds on a progression of related work  [{", ".join(related_ids)}].\n\n'
            f' {rationale}\n'
            f'The initial trend observed from the progression of related work highlights a consistent research focus. However, the final hypothesis proposed here is not merely a continuation of that trend — it is the result of a deeper analysis of the hypothesis space. By identifying underlying gaps and reasoning through the connections between works, the idea builds on, but meaningfully diverges from, prior directions to address a more specific challenge.'
        )
        
        print(motivation_text)

        summary["motivation"] = motivation_text

    if summary:
        simplified["summary"] = summary

    # Format operationalization
    if operationalization_data:
        simplified["operationalization"] = operationalization_data

    # Add core_related_works_enriched as markdown links
    simplified["core_related_works"]= data[0].get("core_related_works_enriched", [])
        
    # Flatten references
    references = data[0].get("collected_related_works", [])
    if references:
        ref_lines = []
        for idx, ref in enumerate(references, 1):
            title = ref.get("title", "Untitled")
            year = ref.get("year", "n.d.")
            pid = ref.get("paperId", "")
            ref_lines.append(f"{idx}. {title} ({year}). Paper ID: {pid}")
        simplified["references"] = "\n\n".join(ref_lines)

    return simplified

# === CUSTOM FIELD DEFINITIONS ===

custom_sections = [
    ("research_idea_short_description", [
    ("", [
        "[0].specific_hypothesis.research_idea_short_description",
        "[0].specific_hypothesis.research_idea_long_description.research_idea_short_description"
    ])]),
    ("research_idea_hypothesis", "[0].specific_hypothesis.research_idea_hypothesis"),
    ("research_idea_long_description", [
        ("description", ["[0].specific_hypothesis.research_idea_long_description.description"]),
        ("research_gap", ["[0].specific_hypothesis.research_gap"]),
        ("research_idea_variables", ["[0].specific_hypothesis.research_idea_long_description.research_idea_variables"]),
        ("Explanation", [
            "[0].specific_hypothesis.explanation.theoretical_justification",
            "[0].specific_hypothesis.research_idea_long_description.explanation.theoretical_justification"
        ]),
        (None, [
            "[0].specific_hypothesis.explanation.expected_synergies",
            "[0].specific_hypothesis.research_idea_long_description.explanation.expected_synergies"
        ]),
        ("research_idea_design_prompt", ["[0].specific_hypothesis.research_idea_long_description.research_idea_design_prompt"]),
#        ("research_idea_metric", ["[0].specific_hypothesis.research_idea_long_description.research_idea_metric"])
    ]),
    ("research_idea_evaluation_rubric", "[0].research_idea_evaluation_rubric")
]

# === LOAD MULTI-HYPOTHESIS REFERENCE FILE ===

def main():
    parser = argparse.ArgumentParser(description="Simplify HARPA rubric JSON files.")
    parser.add_argument("--label", type=str, default="test", help="Label name for input directory")
    parser.add_argument("--input_dir", type=str, default="./test_harpa/stage1_stage2/final", help="Input directory path containing final JSONs")
    parser.add_argument("--output_base", type=str, default="./test_harpa/simplified_jsons", help="Base output directory")

    args = parser.parse_args()

    input_label = args.label
    input_dir = args.input_dir
    output_base_dir = args.output_base

    os.makedirs(output_base_dir, exist_ok=True)
    output_dir = os.path.join(output_base_dir, input_label.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    for json_file in json_files:
        full_path = os.path.join(input_dir, json_file)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            match = re.search(r'hyp\d+_([a-f0-9]{40})', json_file)
            hex_id = match.group(1) if match else "UNKNOWN_ID"
            op_data = data[0].get("specific_hypothesis", {}).get("operationalization", {}).get("operationalization_description")

            simplified_data = extract_simplified_json(data, hex_id, op_data)

            out_file = os.path.join(output_dir, json_file)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(simplified_data, f, indent=2)

            print(f"Saved: {out_file}")

        except Exception as e:
            print(f"Skipped {json_file} due to error: {e}")


if __name__ == "__main__":
    main()


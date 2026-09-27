import os
import json
import re
import argparse

MULTI_JSON_PATH = "./test_harpa/harpa_run4.codeblock.json"
# === LOAD MULTI-HYPOTHESIS REFERENCE FILE ===

try:
    with open(MULTI_JSON_PATH, "r") as f:
        multi_ideas = json.load(f)
        multi_idea_lookup = {item.get("file_id"): item for item in multi_ideas if "file_id" in item}
except Exception as e:
    print(f"Failed to load multi-hypothesis JSON: {e}")
    multi_idea_lookup = {}
    
    
def extract_paper_id_and_version(filename):
    """Extract paper ID and version string (e.g., v1, v2) from filename."""
    match = re.search(r'([a-f0-9]{40})_v(\d+)', filename)
    if match:
        paper_id = match.group(1)
        version = f"v{match.group(2)}"
        return f"{paper_id}_{version}"
    return None


def generate_description_string(data, key):
    specific = data.get('specific_hypothesis', {})
    elaboration_data = specific.get('research_idea_long_description', {})

    research_idea_name = specific.get("research_idea_name") or elaboration_data.get("research_idea_name", "")
    research_idea_short_description = specific.get("research_idea_short_description") or elaboration_data.get("research_idea_short_description", "")

    key_vars = elaboration_data.get('research_idea_variables', {})
    key_vars_str = "\n".join(f"{key}: {desc}" for key, desc in key_vars.items())

    elaboration_str = (
        f"Name: {research_idea_name}\n"
        f"Short Description: {research_idea_short_description}\n"
        f"Description: {elaboration_data.get('description', '')}\n"
        f"Key Variables:\n{key_vars_str}\n"
        f"Implementation: {elaboration_data.get('research_idea_design_prompt', '')}\n"
#        f"Evaluation: {elaboration_data.get('research_idea_metric', '')}"
    )
    print(key)
    op_data = multi_idea_lookup.get(key.split("_")[0], {}).get("operationalization").get("operationalization_description")

    explanation_data = specific.get("explanation") or elaboration_data.get("explanation", {})
    theoretical = explanation_data.get("theoretical_justification", "")
    synergies = explanation_data.get("expected_synergies", "")

    explanation = f"Theoretical Justification: {theoretical}\nExpected Synergies: {synergies}".strip()

    #return f"{elaboration_str}\n{explanation}\n{op_data}".strip()
    return f"{elaboration_str}\n\nOperationalization: {op_data}".strip()


def extract_hypotheses_from_dirs(input_dirs):
    hypothesis_dict = {}
    description_dict = {}

    for input_dir in input_dirs:
        for filename in os.listdir(input_dir):
            if not filename.endswith(".json"):
                continue
            
            key = extract_paper_id_and_version(filename)
            if not key:
                continue
            
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        data = data[0]

                    hypothesis = data.get("specific_hypothesis", {}).get("research_idea_hypothesis")
                    if hypothesis:
                        hypothesis_dict[key] = hypothesis

                    description = generate_description_string(data, key)
                    if description:
                        description_dict[key] = description

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return hypothesis_dict, description_dict

def main():
    parser = argparse.ArgumentParser(description="Extract hypotheses and descriptions from JSON files.")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        default=["./test_harpa/stage1_stage2/final"],
        help="Input directories containing JSON files."
    )
    parser.add_argument(
        "--hyp_output",
        type=str,
        default="./test_harpa/single_hyp_for_DScorer/extracted_hypotheses_test.json",
        help="Output file for extracted hypotheses."
    )
    parser.add_argument(
        "--desc_output",
        type=str,
        default="./test_harpa/single_hyp_for_DScorer/extracted_descriptions_test.json",
        help="Output file for extracted description strings."
    )

    args = parser.parse_args()

    hypotheses, descriptions = extract_hypotheses_from_dirs(args.input_dirs)

    with open(args.hyp_output, "w", encoding="utf-8") as f:
        json.dump(hypotheses, f, indent=2, ensure_ascii=False)

    with open(args.desc_output, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(hypotheses)} hypotheses to {args.hyp_output}")
    print(f"Saved {len(descriptions)} descriptions to {args.desc_output}")

if __name__ == "__main__":
    main()

import os
import json
import re
from typing import List, Dict
from collections import defaultdict
import argparse

# Fields to use
FIELDS_TO_EXTRACT = [
    "paper_id",
    "motivation",
    "research_idea_hypothesis",
    "research_gap",
    "research_idea_elements",
    "description",
    "research_idea_variables",
#    "Explanation",
#    "research_idea_long_description",
    "research_idea_design_prompt",
    "research_idea_metric",
    "operationalization",
#    "research_idea_evaluation_rubric",
    "references"
]

FIELD_MAP = {
    "paper_id": "Paper ID",
    "motivation": "Motivation",
    "research_idea_hypothesis": "Hypothesis",
    "description": "Overview",
    "research_gap": "Research Gap",
    "research_idea_variables": "Background",
#    "Explanation": "Explanation",
#    "research_idea_long_description": "Explanation2",
    "research_idea_design_prompt": "Implementation",
#    "research_idea_metric": "Evaluation Metric",
    "research_idea_elements": "Hypothesis Elements",
    "operationalization": "Operationalization Information",
#    "research_idea_evaluation_rubric": "Evaluation Rubric",
    "references": "References"
}


# === UTILITIES ===
def rubric_to_dashed_blocks(rubric_list):
    if not rubric_list:
        return "No evaluation rubric available."

    lines = []

    for idx, item in enumerate(rubric_list):
        name = item.get("criteria_name", "").strip()
        question = item.get("criteria_met_question", "").strip()
        required = item.get("required_or_optional", "").strip()

        lines.append(f"---\n\n### Rubric Item {idx}\n")
        lines.append(f"- **Criterion:** {name}")
        lines.append(f"- **Required?:** {required}")
        lines.append(f"- **Rubric Question:** {question}\n")

    return "\n".join(lines)


def json_to_markdown(json_rec: dict, fields: List[str], header_map: Dict[str, str]) -> str:
    grouped = defaultdict(list)

    for field in fields:
        header = header_map.get(field, field)
        value = json_rec.get(field, "")
        if value:
            grouped[header].append(value)

    markdown_lines = []
    for header, values in grouped.items():
        if header == "Evaluation Rubric" and isinstance(values[0], list):
            combined_value = rubric_to_dashed_blocks(values[0])

        else:
            combined_value = "\n\n".join(str(v) for v in values)
        markdown_lines.append(f"\n---\n\n**{header}:**\n\n{combined_value}  ")
    return "\n".join(markdown_lines)

def markdown_to_json_record(md_str: str, field_map: Dict[str, str]) -> dict:
    result = {}
    reverse_map = {v: k for k, v in field_map.items()}
    lines = md_str.strip().splitlines()

    current_field = None
    buffer = []
    rubric_blocks = []
    rubric_buffer = []
    is_in_rubric_section = False

    for line in lines:
        line = line.strip()

        header_match = re.match(r"\*\*(.+?):\*\*(.*)", line)
        if header_match:
            # Save non-rubric buffer
            if current_field and buffer and current_field != "research_idea_evaluation_rubric":
                result[current_field] = "\n".join(buffer).strip()
                buffer = []

            field_label, inline_value = header_match.groups()
            current_field = reverse_map.get(field_label.strip(), field_label.strip())
            is_in_rubric_section = current_field == "research_idea_evaluation_rubric"

            if is_in_rubric_section:
                continue  # rubric handled separately
            if inline_value.strip():
                buffer.append(inline_value.strip())
            continue

        # Rethink rubric block detection
        if is_in_rubric_section:
            if line.startswith("### Rubric Item"):
                if rubric_buffer:
                    rubric_blocks.append(parse_rubric_markdown_block(rubric_buffer))
                    rubric_buffer = []
            if line != "---":
                rubric_buffer.append(line)
        elif current_field:
            buffer.append(line)

    # Final flush
    if is_in_rubric_section and rubric_buffer:
        rubric_blocks.append(parse_rubric_markdown_block(rubric_buffer))
    if rubric_blocks:
        result["research_idea_evaluation_rubric"] = rubric_blocks
    elif current_field and buffer:
        result[current_field] = "\n".join(buffer).strip()

    print("Parsed rubric count:", len(result.get("research_idea_evaluation_rubric", [])))
    return result

def parse_rubric_markdown_block(lines: List[str]) -> Dict[str, str]:
    rubric = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        if line.startswith("- **Criterion:**"):
            rubric["criteria_name"] = line.replace("- **Criterion:**", "").strip()
        elif line.startswith("- **Required?:**"):
            rubric["required_or_optional"] = line.replace("- **Required?:**", "").strip()
        elif line.startswith("- **Rubric Question:**"):
            rubric["criteria_met_question"] = line.replace("- **Rubric Question:**", "").strip()
    return rubric


def extract_fields(json_obj: dict, fields: List[str]) -> dict:
    flat = {}
    summary = json_obj.get("summary", {})
    for field in fields:
        if field in summary:
            flat[field] = summary[field]
        elif field in json_obj:
            flat[field] = json_obj[field]
    return flat


# === MAIN FUNCTION ===

def main():
    parser = argparse.ArgumentParser(description="Convert HARPA JSON <-> Markdown.")
    parser.add_argument("--input_json_dir", type=str, default="test_harpa/simplified_jsons/", help="Base input dir with JSONs organized by batch")
    parser.add_argument("--output_md_dir", type=str, default="test_harpa/markdown_export", help="Where to save generated Markdown")
    parser.add_argument("--md_source_dir", type=str, default="test_harpa/markdown_export", help="Where to read Markdown from if reconverting")
    parser.add_argument("--json_out_dir", type=str, default="test_harpa/md_to_json_output", help="Where to save re-parsed JSON")
    parser.add_argument("--reconvert", action="store_true", help="Enable Markdown -> JSON mode")

    args = parser.parse_args()

    if not args.reconvert:
        os.makedirs(args.output_md_dir, exist_ok=True)
        for batch_name in os.listdir(args.input_json_dir):
            batch_path = os.path.join(args.input_json_dir, batch_name)
            if not os.path.isdir(batch_path):
                continue
            out_batch_path = os.path.join(args.output_md_dir, batch_name)
            os.makedirs(out_batch_path, exist_ok=True)
            for file in os.listdir(batch_path):
                if not file.endswith(".json"):
                    continue
                json_path = os.path.join(batch_path, file)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    extracted = extract_fields(data, FIELDS_TO_EXTRACT)
                    md = json_to_markdown(extracted, FIELDS_TO_EXTRACT, FIELD_MAP)
                    out_path = os.path.join(out_batch_path, file.replace(".json", ".md"))
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(md)
                    print(f"Markdown saved: {out_path}")
                except Exception as e:
                    print(f"Failed to process {file}: {e}")

    else:
        os.makedirs(args.json_out_dir, exist_ok=True)
        for batch_name in os.listdir(args.md_source_dir):
            batch_path = os.path.join(args.md_source_dir, batch_name)
            if not os.path.isdir(batch_path):
                continue
            out_batch_path = os.path.join(args.json_out_dir, batch_name)
            os.makedirs(out_batch_path, exist_ok=True)
            for file in os.listdir(batch_path):
                if not file.endswith(".md"):
                    continue
                md_path = os.path.join(batch_path, file)
                try:
                    with open(md_path, "r", encoding="utf-8") as f:
                        md_content = f.read()
                    rec = markdown_to_json_record(md_content, FIELD_MAP)
                    paper_id = rec.pop("paper_id", file.replace(".md", ""))
                    rubric = rec.pop("research_idea_evaluation_rubric", None)
                    out_json = {
                        "paper_id": paper_id,
                        "summary": rec
                    }
                    if rubric:
                        out_json["research_idea_evaluation_rubric"] = rubric
                    out_path = os.path.join(out_batch_path, file.replace(".md", ".json"))
                    with open(out_path, "w", encoding="utf-8") as out_f:
                        json.dump(out_json, out_f, indent=2)
                    print(f"JSON re-saved from markdown: {out_path}")
                except Exception as e:
                    print(f"Failed to reparse {file}: {e}")

if __name__ == "__main__":
    main()

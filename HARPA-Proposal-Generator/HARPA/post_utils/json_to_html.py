import os
import json
import argparse
from typing import List, Dict
from collections import defaultdict
import markdown
import re

# === CONFIGURATION ===
FIELDS_TO_EXTRACT = [
    "paper_id",
    "research_idea_short_description",
    "motivation",
    "research_idea_hypothesis",
    "research_gap",
    "research_idea_elements",
    "description",
    "research_idea_variables",
    "research_idea_design_prompt",
    "research_idea_metric",
    "operationalization",
    "references"
]

FIELD_MAP = {
    "paper_id": "Paper ID",
    "research_idea_short_description": "Title",
    "motivation": "End Note",
    "research_idea_hypothesis": "Problem Statement",  # <-- Changed from "Hypothesis"
    "description": "Overview",
    "research_gap": "Motivation",  # <-- Changed from "Research Gap"
    "research_idea_variables": "Background",
    "research_idea_design_prompt": "Implementation",
    "research_idea_metric": "Evaluation Metric",
    #"research_idea_elements": "Hypothesis Elements",
    "operationalization": "Operationalization Information",
    "references": "References"
}

SECTION_MAPPING = {
    "Paper ID": ["Paper ID"],
    "Title": ["Title"],
    "Introduction": [
        #"Motivation", "Hypothesis", "Research Gap",
        "Problem Statement", "Motivation"
        #"Hypothesis Elements"
    ],
    "Proposed Method": [
        "Overview", "Background", "Implementation"
    ],
    "Experiments Plan": [
        "Operationalization Information"
    ],
    "References": [
        "References"
    ]
}


# === MARKDOWN TO HTML ===
def convert_markdown_like_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    text = re.sub(r'&(?![a-z]+;)', '&amp;', text)
    text = text.replace("<", "&lt;").replace(">", "&gt;")

    html = markdown.markdown(text, extensions=["extra", "nl2br"])

    # Convert all markdown headings to <h4> with bold and class
    html = re.sub(r'<h[1-6]>(.*?)</h[1-6]>', r'<h4 class="custom-heading"><strong>\1</strong></h4>', html)

    return html



# === JSON TO HTML CONVERSION ===
def json_to_html(json_rec: dict, fields: List[str], header_map: Dict[str, str]) -> str:
    grouped = defaultdict(list)
    for field in fields:
        header = header_map.get(field, field)
        value = json_rec.get(field, "")
        if value:
            grouped[header].append(value)

    html_lines = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '<meta charset="UTF-8">',
        '<title>Summary</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 40px; padding: 20px; line-height: 1.6; box-sizing: border-box; max-width: 1000px; }',
        'h2, h3, h4 { color: #007b5f; }',
        '.custom-heading {color: black;font-weight: bold;}',
        'hr { border: none; border-top: 1px solid #ccc; margin: 2em 0; }',
        'p { margin-bottom: 1em; }',
        'table { border-collapse: collapse; width: 100%; margin-top: 1em; }',
        'th, td { border: 1px solid #ddd; padding: 8px; }',
        '.end-note, .end-note * { color: #888 !important; font-style: italic; font-weight: normal; margin-top: 2em; }',
        '</style>',
        '</head>',
        '<body>'
    ]

    for section, section_fields in SECTION_MAPPING.items():
        html_lines.append("<hr>")
        if section:
            html_lines.append(f"<h2>{section}</h2>")

        for field in section_fields:
            values = grouped.get(field)
            if not values:
                continue

            if field != section and field != "Overview":
                html_lines.append(f"<h3>{field}</h3>")

            if field == "References":
                html_lines.append("<ol>")

                refs = values[0]
                if isinstance(refs, list):
                    # Already structured, use as-is
                    for ref in refs:
                        title = ref.get("title", "Untitled")
                        year = ref.get("year", "n.d.")
                        pid = ref.get("paperId", "")
                        if pid:
                            link = f"https://www.semanticscholar.org/paper/{pid}"
                            html_lines.append(f'<li><a href="{link}" target="_blank">{title}</a> ({year})</li>')
                        else:
                            html_lines.append(f"<li>{title} ({year})</li>")
                elif isinstance(refs, str):
                    # Parse unstructured string references
                    entries = refs.strip().split("\n\n")
                    for ref_str in entries:
                        # Extract title and paperId with regex
                        title_match = re.search(r"\d+\.\s+(.*)\s+\(\d{4}\)", ref_str)
                        pid_match = re.search(r"Paper ID: (\w+)", ref_str)
                        title = title_match.group(1) if title_match else ref_str
                        year_match = re.search(r"\((\d{4})\)", ref_str)
                        year = year_match.group(1) if year_match else "n.d."
                        pid = pid_match.group(1) if pid_match else ""

                        if pid:
                            link = f"https://www.semanticscholar.org/paper/{pid}"
                            html_lines.append(f'<li><a href="{link}" target="_blank">{title}</a> ({year})</li>')
                        else:
                            html_lines.append(f"<li>{title} ({year})</li>")

                html_lines.append("</ol>")

            elif field == "Operationalization Information":
                for value in values:
                    paragraphs = str(value).strip().split("\n\n")
                    half = len(paragraphs) // 2
                    left = paragraphs[:half]
                    right = paragraphs[half:]
                    html_lines.append("""
<div style="display: flex; gap: 2em;">
<div style="flex: 1;">
""")
                    for para in left:
                        html_lines.append(f"<p>{convert_markdown_like_text(para)}</p>")
                    html_lines.append("</div><div style='flex: 1;'>")
                    for para in right:
                        html_lines.append(f"<p>{convert_markdown_like_text(para)}</p>")
                    html_lines.append("</div></div>")
                
                # Inject End Note paragraph before References
                end_note = grouped.get("End Note")
                if end_note:
                    for value in end_note:
                        html_lines.append(f'<div class="end-note"><strong>End Note:</strong> {convert_markdown_like_text(value)}</div>')

            else:
                for value in values:
                    html_lines.append(convert_markdown_like_text(value))

    html_lines.append('</body>')
    html_lines.append('</html>')

    return "\n".join(html_lines)


# === EXTRACT FIELDS ===
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
    parser = argparse.ArgumentParser(description="Convert HARPA JSON to HTML with Markdown rendering.")
    parser.add_argument("--input_json_dir", type=str, required=True, help="Directory containing input JSON files")
    parser.add_argument("--output_html_dir", type=str, required=True, help="Directory to save generated HTML files")
    args = parser.parse_args()

    os.makedirs(args.output_html_dir, exist_ok=True)

    for file in os.listdir(args.input_json_dir):
        if not file.endswith(".json"):
            continue
        json_path = os.path.join(args.input_json_dir, file)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            extracted = extract_fields(data, FIELDS_TO_EXTRACT)
            html = json_to_html(extracted, FIELDS_TO_EXTRACT, FIELD_MAP)
            out_path = os.path.join(args.output_html_dir, file.replace(".json", ".html"))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"✅ HTML saved: {out_path}")
        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")


if __name__ == "__main__":
    main()

# python /weka/ROOT/hypothesis_generation/test_harpa/utils/json_to_html.py --input_json_dir /weka/ROOT/hypothesis_generation/user-centric_HG/USER/simplified_jsons_v2/test --output_html_dir /weka/ROOT/hypothesis_generation/user-centric_HG/USER/html/


import os
import json
import markdown
import re

# === CONFIGURATION ===
FIELDS_TO_INCLUDE = [
    "abstract_id",
    "full_experiment_plan"
]

def md(text):
    """Convert Markdown-like text to safe HTML"""
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'&(?![a-z]+;)', '&amp;', text)
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return markdown.markdown(text, extensions=["extra", "nl2br"])

def render_value(value):
    """Render any JSON value into HTML recursively"""
    if isinstance(value, str):
        return md(value)
    elif isinstance(value, list):
        html = "<ul>"
        for item in value:
            html += "<li>" + render_value(item) + "</li>"
        html += "</ul>"
        return html
    elif isinstance(value, dict):
        html = ""
        for k, v in value.items():
            html += f"<p><strong>{k}</strong></p>"
            html += render_value(v)
        return html
    else:
        return f"<p>{value}</p>"

def render_json_to_html(data: dict, references=[], fallback_id=None) -> str:
    html_lines = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '<meta charset="UTF-8">',
        '<title>Summary</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 40px; padding: 20px; line-height: 1.6; box-sizing: border-box; max-width: 1000px; }',
        'h1, h2, h3, h4 { color: #007b5f; margin-top: 1.5em; }',
        'hr { border: none; border-top: 1px solid #ccc; margin: 2em 0; }',
        'p { margin-bottom: 1em; }',
        'ul { margin-left: 1.5em; }',
        'table { border-collapse: collapse; width: 100%; margin-top: 1em; }',
        'th, td { border: 1px solid #ddd; padding: 8px; }',
        'section { margin-top: 2em; }',
        '</style>',
        '</head>',
        '<body>'
    ]


    FIELD_DISPLAY_NAMES = {
        "abstract_id": "Paper ID",
        "full_experiment_plan": " "
    }

    for key in FIELDS_TO_INCLUDE:
        if key == "abstract_id" and key not in data and fallback_id:
            data[key] = fallback_id  # inject fallback ID
        if key in data:
            if key == "full_experiment_plan" and isinstance(data[key], dict):
                SECTION_MAPPING = {
                    "Title": ["Title"],
                    "Introduction": ["Problem Statement", "Motivation"],
                    "Proposed Method": ["Proposed Method"],
                    "Experiments Plan": ["Step-by-Step Experiment Plan","Test Case Examples", "Fallback Plan"],
                    "References": []
                }
                for section, fields in SECTION_MAPPING.items():
                    html_lines.append("<hr>") 
                    if section == "References" and references:
                        html_lines.append(f"<section><h2>{section}</h2>")
                        html_lines.append("<ol>")
                        for paper in references:
                            title = paper.get("title", "Unknown Title")
                            year = paper.get("year", "n.d.")
                            pid = paper.get("paperId")
                            
                            if pid:
                                link = f"https://www.semanticscholar.org/paper/{pid}"
                                html_lines.append(f'<li><a href="{link}" target="_blank">{title}</a> ({year})')
                            else:
                                html_lines.append(f"<li>{title} ({year})")
                            html_lines.append("</li>")
                        html_lines.append("</ol>")

                        html_lines.append("</section>")
                        continue  # skip to next section

                    html_lines.append(f"<section><h2>{section}</h2>")
                    for field in fields:
                        if field in data[key]:
                            if field != section:  # Avoid duplicate heading
                                html_lines.append(f"<h3>{field}</h3>")
                            html_lines.append(render_value(data[key][field]))
                    html_lines.append("</section>")
            else:
                display_key = FIELD_DISPLAY_NAMES.get(key, key)
                html_lines.append("<hr>")
                html_lines.append(f"<h2>{display_key}</h2>")
                html_lines.append(render_value(data[key]))


    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def process_directory(base_dir, fallback_id=None):
    reference_path = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "lit_review", "grounding_papers.json")
    references = []
    if os.path.exists(reference_path):
        with open(reference_path, "r", encoding="utf-8") as f:
            references = json.load(f)
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                html_path = os.path.join(root, file.replace(".json", ".html"))

                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    html_content = render_json_to_html(data, references, fallback_id)

                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)

                    print(f"Converted: {json_path} → {html_path}")
                except Exception as e:
                    print(f"Failed to process {json_path}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recursively convert JSON files to HTML using selected fields.")
    parser.add_argument("--input_dir", required=True, help="Top-level directory containing JSONs")
    args = parser.parse_args()

    # Extract the ID from the input_dir path
    basename = os.path.basename(args.input_dir.rstrip("/"))
    extracted_id = basename.split("_")[-2]  # or just use basename if only ID is present

    process_directory(args.input_dir, extracted_id)


# python prepare_evaluation/baseline_html.py --input_dir cache_results_test/alice/project_proposals/1343dedea56bbf3ba48d0971aee177b5add61105_source
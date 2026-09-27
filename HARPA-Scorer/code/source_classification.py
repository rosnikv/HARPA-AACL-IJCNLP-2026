from ExtractionUtils import *
import json

entries = []
source_texts = []

with open("data_bank_harpa.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        entries.append(entry)
        source_texts.append(entry["source_text"])
        
model_str = "claude-3-7-sonnet-20250219"
temperature = 0.1

batch_size = 100
topic_of = {}

for start in range(0, len(source_texts), batch_size):
    batch = source_texts[start:start+batch_size]

    prompt = f"""
You are a research assistant. I will give you a list of research paper abstracts and metadata.

Your task is to:
1. Identify a small set of **shared, broad research topics** (e.g., "Prompt Learning", "Multimodal Learning", "NLP Applications", etc.)
2. **Assign exactly one topic label to each abstract**, using the same topic for similar papers.

Important rules:
- You must return **one topic assignment per paper**.
- Use **shared topics** across multiple papers. Avoid overly specific or unique topics.
- There should be **at least 5–10 papers per topic**, ideally more.
- Do not return a list of only the topics — we need an explicit mapping for each paper.

---

Format your response as a JSON list:
```
{{
"topics": [
    {{ "index": {start}, "topic": "Prompt Learning" }},
    {{ "index": {start+1}, "topic": "Knowledge Distillation" }},
    ...
    ]
}}
```
Here are the abstracts {start}–{start+len(batch)-1} and metadata:

""" + "\n\n".join([f"{start+i}. {text.strip()}" for i, text in enumerate(batch)])

    resp_json, resp_text, _ = getLLMResponseJSON(
        promptStr=prompt, model=model_str, temperature=temperature, jsonOut=True
    )

    for item in resp_json.get("topics", []):
        topic_of[item["index"]] = item["topic"]

# attach topics back to entries
for idx, topic in topic_of.items():
    entries[idx]["shared_topic"] = topic

with open("data_bank_harpa_with_topics.jsonl", "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")

final = {"topics": [{"index": i, "topic": entries[i]["shared_topic"]} for i in range(len(entries))]}
with open("topics_mapping.json", "w") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print(f"Total unique topics: {len(set(topic_of.values()))}")

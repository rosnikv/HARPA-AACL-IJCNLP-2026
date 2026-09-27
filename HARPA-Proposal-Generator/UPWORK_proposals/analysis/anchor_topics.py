import os
import json
import requests
from collections import Counter

# ==== Step 1: Load paper IDs ====
with open("anchor_pids.txt") as f:
    paper_ids = [line.strip() for line in f if line.strip()]

# 🔑 Get API key from environment variable
API_KEY = os.getenv("S2_API_KEY")
if not API_KEY:
    raise ValueError("❌ Missing API key. Please set environment variable S2_API_KEY")

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/"
HEADERS = {"x-api-key": API_KEY}
FIELDS = ["title", "abstract", "year", "venue"]

# ==== Step 2: Fetch metadata + abstracts ====
papers = []
for pid in paper_ids:
    url = f"{BASE_URL}{pid}"
    params = {"fields": ",".join(FIELDS)}
    r = requests.get(url, headers=HEADERS, params=params)
    if r.status_code == 200:
        papers.append(r.json())
    else:
        print(f"⚠️ Failed to fetch {pid}: {r.status_code}")

# ==== Step 3: Build prompt ====
abstracts = []
for i, p in enumerate(papers):
    title = p.get("title", "Untitled")
    abstract = p.get("abstract", "No abstract available")
    year = p.get("year", "Unknown")
    venue = p.get("venue", "Unknown")
    abstracts.append(f"{i}. [{title}, {venue}, {year}] {abstract}")

prompt = f"""
You are a research assistant. I will give you a list of research paper abstracts.

Your task is to:
1. Identify a small set of **shared, broad research topics** (e.g., "Prompt Learning", "Multimodal Learning", "NLP Applications").
2. **Assign exactly one topic label to each abstract**, using the same topic for similar papers.
3. Topics must be broad enough to group multiple papers (at least 5–10 papers per topic).

Format your response as JSON:
{{
  "topics": [
    {{ "index": 0, "topic": "Prompt Learning" }},
    {{ "index": 1, "topic": "Knowledge Distillation" }}
  ]
}}
Here are the abstracts:

""" + "\n\n".join(abstracts)

# ==== Step 4: Call your LLM ====
from ExtractionUtils import *

model_str = "claude-3-7-sonnet-20250219"
temperature = 0.1
responseJSON, responseText, cost2 = getLLMResponseJSON(
    promptStr=prompt, model=model_str, temperature=temperature, jsonOut=False
)

# ==== Step 5: Attach topics to metadata and save ====
print(responseJSON["topics"])

for topic_info in responseJSON["topics"]:
    idx = topic_info["index"]
    papers[idx]["shared_topic"] = topic_info["topic"]

with open("anchor_papers_with_topics.jsonl", "w") as f:
    for entry in papers:
        f.write(json.dumps(entry) + "\n")

print("✅ Saved annotated anchor papers with topics → anchor_papers_with_topics.jsonl")

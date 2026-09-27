import asyncio
import requests
import json
import logging
from typing import List


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

async def call_semantic(query, 
                        limit=10,
                        returned_fields=[
            "title",
            "abstract",
            "venue",
            "year",
            "paperId",
            "citationCount",
            "openAccessPdf",
            "authors",
        ]):
    from semanticscholar import SemanticScholar
    s2 = SemanticScholar(
            api_key='KEY')
    try:
        results = s2.search_paper(
            query, limit=limit, fields=returned_fields)
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
        logging.error(
            "Failed to fetch data from Semantic Scholar with exception: %s", e
        )
        raise
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        raise

    documents = []

    for item in results[:limit]:
        openaccesspdf = getattr(item, "openAccessPdf", None)
        abstract = getattr(item, "abstract", None)
        title = getattr(item, "title", None)
        if isEnglish(title):
            text = None
            # concat title and abstract
            if abstract and title:
                text = "Title: " + title + " - " + abstract
            elif not abstract:
                # print(f"{title} doesn't have abstract, {openaccesspdf}")
                continue
                text = title

            metadata = {
                "title": title,
                "venue": getattr(item, "venue", None),
                "year": getattr(item, "year", None),
                "paperId": getattr(item, "paperId", None),
                "citationCount": getattr(item, "citationCount", None),
                "openAccessPdf": openaccesspdf.get("url") if openaccesspdf else None,
                "authors": [author["name"] for author in getattr(item, "authors", [])],
            }
            documents.append({'text':text, 'metadata':metadata})

    return documents

async def async_semantic_call(queries):
    results = []
    tasks = [call_semantic(query) for query in queries]
    print(tasks)

    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        results.append(result)

    return results


def novelty_system_prompt():
    prompt = f"""\
    You are an medical researcher ideating the next big research to publish.
    You have an idea and you want to check if it is novel or not. i.e., not
    overlapping significantly with existing literature or already well explored.
    Be a harsh critic for novelty, ensure there is a sufficient contribution in
    the idea for a new conference or workshop paper.
    You will be given access to the Semantic Scholar API, which you may use to
    survey the literature and find relevant papers to help you make your
    decision.
    """
    return prompt

def novelty_query_prompt(idea:str)->str:
    prompt = f"""\
    You have this idea:
    ```
    {idea}
    ```
    Respond in the following format:
    RESPONSE:
    ```json
    <JSON>
    ```
    In <JSON>, respond in JSON format with ONLY the following field:
    - "Query": A search query to search the literature (e.g. attention
    is all you need). A query will work best if you are able to recall 
    the exact name of the paper you are looking for, or the authors. Then
    convert it into a simple query string consisting of 3 to 4 keywords.
    Do not use AND, OR or quotes in query.
    This JSON will be automatically parsed, so ensure the format is precise.
    """
    return prompt

def novelty_judgement_prompt(idea:str, docs:List[str])->str:
    doc_text = ""
    for idx, doc in enumerate(docs):
        doc_text += str(idx+1) + "." + doc
        doc_text += "\n"
    n = len(docs)
    prompt = f"""\
    Here is the research idea.
    ```
    {idea}
    ```
    Here are the top {n} abstracts and dates when they were published based on your query:
    ```
    {doc_text}
    ```
    Respond in the following format:
    RESPONSE:
    ```json
    <JSON>
    ```
    In <JSON>, respond in JSON format with ONLY the following fields:
    - "Thoughts": Scan through each of the abstract and identify if there is sufficient overlap between the idea and the current abstract.
    - "Overlap": Return a list of {n} binary values: 'overlap' and 'little to no overlap' corresponding to each of the abstracts.
    - "Novelty score": Novel or not novel. If all papers have 'little to no overlap' with the idea, consider the idea Novel. If at least one paper has 'overlap', consider the idea 'not novel'.
    This JSON will be automatically parsed, so ensure the format is precise.
    """
    return prompt

def call_gpt(prompt):
    import openai
    import os
    from openai import OpenAI
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": novelty_system_prompt()},
                {"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    print("GPT4 usage: {}".format(response.usage))
    return response.choices[0].message.content

'''
with open('./code/prompts/outputs/idea_llama.json', 'r') as f:
    idea_list = json.load(f)
    
idea = idea_list["hypothesis"]
query_prompt = novelty_query_prompt(idea)
query = json.loads(call_gpt(query_prompt))["Query"]
print(query)
docs = asyncio.run(async_semantic_call([query]))
docs = [x['text'] for x in docs[0]]
judgement_prompt = novelty_judgement_prompt(idea, docs)
print(judgement_prompt)
novelty_res = call_gpt(judgement_prompt)
print(novelty_res)

output_file = './code/prompts/outputs/novelty_llama.json'
with open(output_file, 'w') as f:
    json.dump( json.loads(novelty_res), f)
'''




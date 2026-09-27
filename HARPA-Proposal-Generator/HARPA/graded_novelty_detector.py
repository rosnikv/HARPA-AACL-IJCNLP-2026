'''
Code Source: XXXX-074cfb87575ff9555fc659673617a2/src/NoveltyDetector.py
'''
import requests
import json
import sys
sys.path.append('/srv/scratch1/ROOT/hypothesis_generation/')
import os
import re
import time
import openai
from openai import OpenAI

# NoveltyDetector.py
import os
import json
import time
import tqdm
import random
import subprocess
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

from ExtractionUtils import *

#
#   Semantic Scholar API
#

class SemanticScholar():
    def __init__(self, api_key=None):
        self.api_key = os.environ.get("S2_API_KEY", None)

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def snippet_search(self, query, limit=10):
        url = "https://api.semanticscholar.org/graph/v1/snippet/search"
        params = {
            "query": query,
            "limit": limit
        }
        headers = {}
        if self.api_key:
            # Use the API key string, not the object
            headers["x-api-key"] = self.api_key
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Error:", response.status_code)
            print(response.text)
            return None

    def snippet_search2(self, query, limit=10):
        # Do the regular snippet search
        data = self.snippet_search(query, limit)
        if data is None:
            return None
        if ("data" in data):
            data = data["data"]
        else:
            return None

        # Extract the snippets
        snippets = []
        # extract the data
        for item in data:
            packed = {}
            if "snippet" in item:
                packed["snippet_kind"] = item["snippet"]["snippetKind"]
                packed["snippet"] = item["snippet"]["text"]
                packed["paper"] = item["paper"]

                snippets.append(packed)

        return snippets



def novelty_grade_prompt(specific_claim:str, retrieved_snippets:list):
    prompt = f"""
    You are a senior scientific reviewer tasked with evaluating the novelty of a research claim.
    This is an automated scientific discovery task, with the overall goal of trying to assess the novelty of scientific claims.
    
    # Background
    If you think about it, nearly every experiment could be considered novel if you make the claims specific enough -- for example, performing a well-known experiment on a specific day, or getting very specific values from the experiment.
    
    The purpose of your task is to take a claim that has already been abstracted somewhat, and consider it (in light of retrieved evidence from relevant papers) to evaluate whether the claim (at this level of abstraction) appears novel or not.

    # Specific task
    You will be given a claim (below), and a set of passages retrieved from the Semantic Scholar full-text paper corpus (below) using that claim as a query. Your task will be to evaluate whether that claim appears to be novel, or whether it appears that it (or highly similar) claims have already been made in the scientific literature.
    
    # Claim to evaluate for novelty
    
    The claim to evaluate for novelty is:
    ```
    {specific_claim}
    ```
    
    # Retrieved Passages
    The passages retrieved from Semantic Scholar are:
    ```
    {json.dumps(retrieved_snippets, indent=4)}
    ```
    
   # NOVELTY ASSESSMENT CRITERIA:
    - Originality: How significantly does the claim differ from existing research?
    - Incremental Value: What unique insights or methodological advancements does it introduce?
    - Potential Impact: Could this claim meaningfully advance the current scientific understanding?

    ## NOVELTY SCORING SCALE:
    Whether the hypothesis is creative and different from existing works on the topic, and brings fresh insights. You should consider all retrieved papers passages when judging the novelty.

    1. Not novel at all --- there are many existing ideas that are the same
    2. Mostly not novel --- you can find very similar ideas
    3. Somewhat novel --- there are differences from existing ideas but not enough to turn into a new paper
    4. Reasonably novel --- there are some notable differences from existing ideas and probably enough to turn into a new paper
    5. Clearly novel --- major differences from all existing ideas
    6. Very novel --- very different from all existing ideas in a very interesting and clever way

    CRITICAL CONSIDERATIONS:
      - Distinguish between incremental research and genuinely novel contributions
      - Consider both immediate and potential long-term scientific impact
      - Be precise, critical, and constructive in your assessment

    # Output format:
      - Output in JSON format
      - Output should be a dictionary, with the following keys: 'claim': str, 'novel':bool, 'explanation':str, 'supporting_papers':list.
      -- The 'claim' key is the claim that was evaluated for novelty.
      -- The 'novel' is a string category based on the novelty scoring scale such as "Not novel at all", "Mostly not novel", "Somewhat novel", "Reasonably novel", "Clearly novel", "Very novel"
      -- The 'explanation' key is a string explaining why the claim is novel or not, and should make clear reference to the content/passages of the supporting papers, and be maximally useful to the user.
      -- The 'supporting_papers' key is a list of the papers referenced in the `explanation` that support the novelty assessment, so the user can look up more information. Each paper should be a dictionary, and include the following keys: 'reference_id':int (e.g. 1, 2, 3), 'title':str, 'first_author':str, 'year': int, 'url': str (leave empty if no URL available).

    # Output example:
    Below is a (cartoon example) of the output format:
    ```
    {{
      'claim': 'In a state prediction task based on the CookingWorld benchmark, an LLM's self-assessed confidence in its predictions have low correlation with the accuracy of its predictions',
      'novel': Reasonably novel,
      'explanation': 'The claim appears incrementally novel.  Although it has been shown that an LLM's self-assessed confidence in its predictions have a low correlation with its accuracy in other tasks (e.g. question answering [1], information extraction [2]), I have not found evidence that this effect has been previously demonstrated in the context of a state prediction task.',
      'supporting_papers': [
        {{'reference_id': 1, 'title': 'Title of the paper', 'first_author': 'First Author', 'year
        ': 2022, 'url': 'https://doi.org/10.1234/5678'}},
        {{'reference_id': 2, 'title': 'Title of the paper', 'first_author': 'First Author', 'year': 2022, 'url': ''}}
      ]
    }}
    ```
    """
    return prompt

# Convert one claim to generalized versions of that claim
def evaluate_claim_novelty(specific_claim:str, s2api:SemanticScholar, s2_passage_limit:int=10, model_str="gpt-4o-mini", max_tokens:int=8000, temperature:float=0.0):
    
    start_time = time.time()

    def mkPrompt(specific_claim:str, retrieved_snippets:list):
        prompt = "This is an automated scientific discovery task, with the overall goal of trying to assess the novelty of scientific claims.\n"
        prompt += "# Background\n"
        prompt += "If you think about it, nearly every experiment could be considered novel if you make the claims specific enough -- for example, performing a well-known experiment on a specific day, or getting very specific values from the experiment.\n"
        prompt += "The purpose of your task is to take a claim that has already been abstracted somewhat, and consider it (in light of retrieved evidence from relevant papers) to evaluate whether the claim (at this level of abstraction) appears novel or not.\n"
        prompt += "\n"
        prompt += "# Specific task\n"
        prompt += "You will be given a claim (below), and a set of passages retrieved from the Semantic Scholar full-text paper corpus (below) using that claim as a query. Your task will be to evaluate whether that claim appears to be novel, or whether it appears that it (or highly similar) claims have already been made in the scientific literature.\n"
        prompt += "\n"
        prompt += "# Claim to evaluate for novelty\n"
        prompt += "The claim to evaluate for novelty is:\n"
        prompt += "```\n"
        prompt += str(specific_claim) + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Retrieved Passages\n"
        prompt += "The passages retrieved from Semantic Scholar are:\n"
        prompt += "```\n"
        prompt += json.dumps(retrieved_snippets, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Output format:\n"
        prompt += "- Output in JSON format\n"
        prompt += "- Output should be a dictionary, with the following keys: 'claim': str, 'novel':bool, 'explanation':str, 'supporting_papers':list.\n"
        prompt += "-- The 'claim' key is the claim that was evaluated for novelty.\n"
        prompt += "-- The 'novel' key is a boolean (true = novel, false = not novel)\n"
        prompt += "-- The 'explanation' key is a string explaining why the claim is novel or not, and should make clear reference to the content/passages of the supporting papers, and be maximally useful to the user.\n"
        prompt += "-- The 'supporting_papers' key is a list of the papers referenced in the `explanation` that support the novelty assessment, so the user can look up more information. Each paper should be a dictionary, and include the following keys: 'reference_id':int (e.g. 1, 2, 3), 'title':str, 'first_author':str, 'year': int, 'url': str (leave empty if no URL available).\n"
        prompt += "\n"
        prompt == "# Output example:\n"
        prompt += "Below is a (cartoon example) of the output format:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += "  'claim': 'In a state prediction task based on the CookingWorld benchmark, an LLM's self-assessed confidence in its predictions have low correlation with the accuracy of its predictions',\n"
        prompt += "  'novel': true,\n"
        prompt += "  'explanation': 'The claim appears incrementally novel.  Although it has been shown that an LLM's self-assessed confidence in its predictions have a low correlation with its accuracy in other tasks (e.g. question answering [1], information extraction [2]), I have not found evidence that this effect has been previously demonstrated in the context of a state prediction task.',\n"
        prompt += "  'supporting_papers': [\n"
        prompt += "    {'reference_id': 1, 'title': 'Title of the paper', 'first_author': 'First Author', 'year': 2022, 'url': 'https://doi.org/10.1234/5678'},\n"
        prompt += "    {'reference_id': 2, 'title': 'Title of the paper', 'first_author': 'First Author', 'year': 2022, 'url': ''}\n"
        prompt += "  ]\n"
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "Please output your JSON response between a single code block (```), as it will be automatically extracted.  You can write any text before or after the code block to help you think, but the text in the code block must be exclusively valid JSON.\n"

        return prompt

    # Get the snippets for this claim from S2
    retrieved_snippets = s2api.snippet_search2(specific_claim, limit=s2_passage_limit)
    if retrieved_snippets is None:
        return None

    # Run the prompt 
    prompt = mkPrompt(specific_claim=specific_claim, retrieved_snippets=retrieved_snippets)
    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False)

    prompt2 = novelty_grade_prompt(specific_claim=specific_claim, retrieved_snippets=retrieved_snippets)
    responseJSON2, responseText2, cost2 = getLLMResponseJSON(promptStr=prompt2, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False)

    delta_time = time.time() - start_time

    metadata = {
        "s2_passage_limit": s2_passage_limit,
        "model": model_str,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "cost": cost,
        "time": delta_time
    }

    packed = {
        "metadata": metadata,
        "specific_claim": specific_claim,
        "novelty_assessment": responseJSON,
        "novelty_assessment2": responseJSON2,
        "retrieved_snippets": retrieved_snippets
    }

    return packed


def novelty_assessment(new_hypothesis:str):
    s2api = SemanticScholar()
    s2_passage_limit = 10
    model_str="gpt-4o-mini"
    max_tokens:int=8000 
    temperature:float=0.0

    novelty_assessments = []
    total_cost = 0

    claim_novelty = evaluate_claim_novelty(new_hypothesis, s2api=s2api, s2_passage_limit=10, model_str="o3-mini", max_tokens=8000, temperature=0.0)
    if claim_novelty is not None:
        novelty_assessments.append(claim_novelty)
        if "metadata" in claim_novelty and "cost" in claim_novelty["metadata"]:
            total_cost += claim_novelty["metadata"]["cost"]
        print(json.dumps(claim_novelty, indent=4))

    print("Total cost so far: " + str(total_cost))
    time.sleep(1)
      
    return novelty_assessments


def main():
    s2_passage_limit = 100

    # Initialize the Semantic Scholar API
    s2_key = load_semantic_scholar_api_key()
    s2 = SemanticScholar(api_key=s2_key)

    #print ("Semantic Scholar API key:", s2_key)

    # Example usage
    original_claim = "A Deep Reinforcement Relevance Network (DRRN) outperforms a T5-based behavior cloning network across all 30 tasks in the ScienceWorld virtual environment benchmark."

    # Generalize the claim
    result = generalize_claims(original_claim=original_claim, model_str="gpt-4o-mini", max_tokens=8000, temperature=0.0)
    print(json.dumps(result, indent=4))


    novelty_assessments = []

    total_cost = 0
    if ("generalized_claims" in result):
        generalized_claims = result["generalized_claims"]
        for item in generalized_claims:
            generalization_level = item["generalization"]
            claim = item["claim"]
            print("Generalization Level:", generalization_level)
            print("Claim:", claim)
            print("")

            claim_novelty = evaluate_claim_novelty(claim, s2api=s2, s2_passage_limit=10, model_str="o3-mini", max_tokens=8000, temperature=0.0)
            if claim_novelty is not None:
                novelty_assessments.append(claim_novelty)
                print(json.dumps(claim_novelty, indent=4))

                if ("metadata" in claim_novelty) and ("cost" in claim_novelty["metadata"]):
                    total_cost += claim_novelty["metadata"]["cost"]
            print("Total cost so far: " + str(total_cost))

            # Write to file
            print("Writing novelty_assessments.json ... ")
            with open("novelty_assessments.json", 'w') as f:
                json.dump(novelty_assessments, f, indent=4)

            # Rate limit the API (1 request per second)
            time.sleep(1)

    # Try an example S2 search
    # data = s2.snippet_search2("Deep Reinforcement Learning")
    # print(json.dumps(data, indent=4))
    # exit(1)



if __name__ == '__main__':
    loadAPIKeys()

    main()
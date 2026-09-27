'''
Code Source: 
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


#
#   Claim Generalization Prompt
#

# Convert one claim to generalized versions of that claim
def generalize_claims(original_claim:str, model_str="gpt-4o-mini", max_tokens:int=8000, temperature:float=0.0):

    def mkPrompt(original_claim:str):
        prompt = "This is an automated scientific discovery task, with the overall goal of trying to assess the novelty of scientific claims.\n"
        prompt += "# Background\n"
        prompt += "If you think about it, nearly every experiment could be considered novel if you make the claims specific enough -- for example, performing a well-known experiment on a specific day, or getting very specific values from the experiment.\n"
        prompt += "The purpose of your task is to take an input claim, and progressively rewrite it as several (progressively more general) claims.\n"
        prompt += "Another system will assess the novelty of these generalized claims, allowing us to detect not simply whether a claim is novel or not, but how specific a claim has to be before it's considered novel.\n"
        prompt += "\n"
        prompt += "# Specific task\n"
        prompt += "You will be given a claim (below), and your task will be to generate 4 progressively more generalied versions of that claim.\n"
        prompt += "\n"
        prompt += "# 7 Examples of the Generalization Process\n"
        prompt += "Below are 7 examples of the generalization process (represented in JSON), to help you understand the task.\n"
        prompt += "- The keys represent names for the 7 different claim examples.\n"
        prompt += "- The value is a list of the (progressively more generalized) claims.\n"
        prompt += "- The 'generalization' key represents the level of generalization (0 is the original claim).\n"
        prompt += "- The 'claim' key represents the claim itself.\n"
        prompt += "\n"
        prompt += "```\n"
        prompt += """"
{
  "State Prediction Confidence": [
    {
      "generalization": 0,
      "claim": "In a state prediction task based on the CookingWorld benchmark (using 600 states from 50 episodes), an LLM's self-assessed confidence in its predictions have low correlation with the accuracy of its predictions (r^2 = 0.16)"
    },
    {
      "generalization": 1,
      "claim": "In a state prediction task based on the CookingWorld benchmark, an LLM's self-assessed confidence in its predictions have low correlation with the accuracy of its predictions"
    },
    {
      "generalization": 2,
      "claim": "In a state prediction task, an LLM's self-assessed confidence in its predictions have low correlation with the accuracy of its predictions."
    },
    {
      "generalization": 3,
      "claim": "An LLM's self-assessed confidence in its predictions has a low correlation with the accuracy of those predictions."
    }
  ],
  "Accuracy vs Representational Expressivity": [
    {
      "generalization": 0,
      "claim": "In a state prediction task (based on the CookingWorld benchmark), where the state prediction task is broken down into different representational types based on complexity (boolean, numerical, releational, full text), an LLM-based model has higher accuracy on simpler representations than it does on more complex representations (boolean: 95%, numerical: 88%, relations: 87%, full: 82%)."
    },
    {
      "generalization": 1,
      "claim": "In a state prediction task (based on the CookingWorld benchmark), where the state prediction task is broken down into different representational types based on complexity (boolean, numerical, releational, full text), an LLM-based model has higher accuracy on simpler representations than it does on more complex representations."
    },
    {
      "generalization": 2,
      "claim": "In a state prediction task, where the state prediction task is broken down into different representational types based on complexity (boolean, numerical, releational, full text), an LLM-based model has higher accuracy on simpler representations than it does on more complex representations."
    },
    {
      "generalization": 3,
      "claim": "In a state prediction, an LLM-based model has higher accuracy on predicting simpler representations than it does on more complex representations."
    }
  ],
  "Graph Metric": [
    {
      "generalization": 0,
      "claim": "A custom graph-aware metric for determining the similarity between natural language text and a graph-based representation of that natural language text can perform at 0.32 average similarity, compared to other metrics (like word overlap and jaccard similarity, both scoring an average similarity of 0.101) when evaluated on 30 text-graph pairs automatically extracted from a text-based virtual environment (CookingWorld)."
    },
    {
      "generalization": 1,
      "claim": "A custom graph-aware metric for determining the similarity between natural language text and a graph-based representation of that natural language text can perform higher than Jaccard similarity when evaluated on text-graph pairs automatically extracted from text-based virtual environments."
    },
    {
      "generalization": 2,
      "claim": "A custom graph-aware metric for determining the similarity between natural language text and a graph-based representation of that natural language text can perform higher than other metrics when evaluated on text-graph pairs automatically extracted from text-based virtual environments."
    },
    {
      "generalization": 3,
      "claim": "It is possible to build new metrics for determining the similarity of natural language text and graph-based representations."
    }
  ],
  "Multi-Stage Environment Generation": [
    {
      "generalization": 0,
      "claim": "When performing a virtual environment generation task (modeled as code generation) with specific templates (3x3 grid world, 2-3 randomly placed items, north/east/south/west movement, inventory management, scoring, and a win condition for collecting all items), the performance of an LLM-based generator increases when the task is not completed as a single step (66.7% mechanics completion rate), but instead broken down into two steps: one that focuses on generating movement and inventory mechanics, and a second that focuses on generating scoring and win conditions (96.7% mechanics completion rate)."
    },
    {
      "generalization": 1,
      "claim": "When performing a virtual environment generation task (modeled as code generation) with specific templates (3x3 grid world, 2-3 randomly placed items, north/east/south/west movement, inventory management, scoring, and a win condition for collecting all items), the performance of an LLM-based generator increases when the task is not completed as a single step, but instead broken down into two steps: one that focuses on generating movement and inventory mechanics, and a second that focuses on generating scoring and win conditions."
    },
    {
      "generalization": 2,
      "claim": "When performing a virtual environment generation task (modeled as code generation) with specific templates, the performance of an LLM-based generator increases when the task is not completed as a single step, but instead broken down into two steps: one that focuses on generating the environment, and a second that focuses on scoring."
    },
    {
      "generalization": 3,
      "claim": "When performing a virtual environment generation task (modeled as code generation), the performance of an LLM-based generator increases when the task is not completed as a single step, but instead broken down into more than one step."
    }
  ],
  "Action Success Prediction": [
    {
      "generalization": 0,
      "claim": "In a virtual environment task (specifically CookingWorld), the performance of an LLM-based agent at predicting whether a given action will likely succeed in that environment is 65.7%, which is only marginally higher than a random baseline (50%), while similarly having a modest correlation between the LLM's self-assessed confidence and it's accuracy on this task (r^2 = 0.335)."
    },
    {
      "generalization": 1,
      "claim": "In a virtual environment task (specifically CookingWorld), the performance of an LLM-based agent at predicting whether a given action will likely succeed in that environment is only marginally higher than a random baseline."
    },
    {
      "generalization": 2,
      "claim": "In a virtual environment task (specifically CookingWorld), the performance of an LLM-based agent at predicting whether a given action will likely succeed in that environment is higher than a random baseline."
    },
    {
      "generalization": 3,
      "claim": "In a virtual environment task, the performance of an LLM-based agent at predicting whether a given action will likely succeed in that environment is higher than chance performance."
    }
  ],
  "Graph Agent for Discovery": [
    {
      "generalization": 0,
      "claim": "A ReAct agent augmented with a graph-based memory (designed to track objects, properties, measurements, and hyoptheses) has a higher performance (29%) than a baseline ReAct agent (12%) on the DiscoveryWorld benchmark Proteomics Task, at easy difficulty."
    },
    {
      "generalization": 1,
      "claim": "A ReAct agent augmented with a graph-based memory has a higher performance than a baseline ReAct agent on the DiscoveryWorld benchmark Proteomics Task."
    },
    {
      "generalization": 2,
      "claim": "A ReAct agent augmented with a graph-based memory has a higher performance than a baseline ReAct agent on the DiscoveryWorld benchmark."
    },
    {
      "generalization": 3,
      "claim": "A ReAct agent augmented with a graph-based memory has a higher performance than a baseline ReAct agent on virtual environment tasks."
    }
  ],
  "Combinatorial Optimization": [
    {
      "generalization": 0,
      "claim": "Language models perform substantially worse (24% within 1%) at a combinatorial optimization problem involving substituting one resistor value than another (which involves figuring out which 2 or 3 standard values add to be the closest to a specific value) than traditional mathematical solvers (100% within 1%)."
    },
    {
      "generalization": 1,
      "claim": "Language models perform substantially worse at a combinatorial optimization problem involving substituting one resistor value than another than traditional mathematical solvers."
    },
    {
      "generalization": 2,
      "claim": "Language models perform substantially worse at a combinatorial optimization problem (involving adding combinations of numbers from a set of existing values to reach a target value) than another than traditional mathematical solvers."
    },
    {
      "generalization": 3,
      "claim": "Language models perform much more poorly at combinatorial optimization problems than traditional solvers."
    }
  ]
}
"""
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Claim to generalize\n"
        prompt += "The claim to generalize is:\n"
        prompt += "```\n"
        prompt += str(original_claim) + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# What should I do if the claim above has multiple claims?\n"
        prompt += "- If the claim above has multiple claims, you should pick the single most salient claim, and generalize it.\n"
        prompt += "\n"

        prompt += "# Output format:\n"
        prompt += "- Output in JSON format, as above\n"
        prompt += "- You should output a dictionary with a single key (a few-word summarized version of the claim)\n"
        prompt += "- The value should be a list of 4 progressively more generalized versions of the claim\n"
        prompt += "- The 'generalization' key should be an integer from 0 to 3, representing the level of generalization (0 is the original claim)\n"
        prompt += "- The 'claim' key should be the claim itself\n"
        prompt += "\n"

        prompt += "Please output your JSON response between a single code block (```), as it will be automatically extracted.  You can write any text before or after the code block to help you think, but the text in the code block must be exclusively valid JSON.\n"

        return prompt

    # Run the prompt
    prompt = mkPrompt(original_claim=original_claim)
    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False)

    generalized_claims = []
    # Check if the response is a dict
    if isinstance(responseJSON, dict):
        # It should have just one key
        if len(responseJSON.keys()) == 1:
            key = list(responseJSON.keys())[0]
            # It should have a list as the value
            if isinstance(responseJSON[key], list):
                for item in responseJSON[key]:
                    if isinstance(item, dict):
                        if "generalization" in item and "claim" in item:
                            generalized_claims.append(item)

    elif isinstance(responseJSON, list):
        for item in responseJSON:
            if isinstance(item, dict):
                if "generalization" in item and "claim" in item:
                    generalized_claims.append(item)

    else:
        print("ERROR: Response from LLM is not a dictionary or list")


    metadata = {
        "model": model_str,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "cost": cost
    }

    packed = {
        "metadata": metadata,
        "original_claim": original_claim,
        "generalized_claims": generalized_claims,
    }

    return packed

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
    
    result = generalize_claims(original_claim=new_hypothesis, model_str="gpt-4o-mini", max_tokens=8000, temperature=0.0)

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

            claim_novelty = evaluate_claim_novelty(claim, s2api=s2api, s2_passage_limit=10, model_str="o3-mini", max_tokens=8000, temperature=0.0)
            if claim_novelty is not None:
                novelty_assessments.append(claim_novelty)
                if ("metadata" in claim_novelty) and ("cost" in claim_novelty["metadata"]):
                    total_cost += claim_novelty["metadata"]["cost"]
            print("Total cost so far: " + str(total_cost))

            # Rate limit the API (1 request per second)
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
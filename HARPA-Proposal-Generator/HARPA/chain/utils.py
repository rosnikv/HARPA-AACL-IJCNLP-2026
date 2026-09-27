import requests
import json
import sys
sys.path.append('/srv/scratch1/ROOT/hypothesis_generation/')
import os
import re
import time
import openai
from openai import OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("S2_API_KEY", None)
import tiktoken
from transformers import pipeline
import torch
from prompts_relevancy.search_query_prompt import search_query_prompt
from prompts_relevancy.relevancy_scoring_prompt_002 import relevancy_scoring_prompt
#from temporal_reasoning_chain.prompts.relevancy_scoring_prompt_001 import relevancy_scoring_prompt
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from torch import Tensor
from llm_logger import set_log_file_path, log_llm_usage, LOG_FILE

# Function to set seed for reproducibility
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """Set a fixed seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")

    
def save_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4)
    
# Initialize model and tokenizer
def initialize_bmretriever(model_name: str = "BMRetriever/BMRetriever-1B", device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(torch.device(device))
    return model, tokenizer

# Pooling function to extract last token
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        embedding = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    return embedding

# Construct detailed instruction for a query
def get_detailed_instruct_query(task_description: str, query: str) -> str:
    return f'{task_description}\nQuery: {query}'

# Construct detailed instruction for a passage
def get_detailed_instruct_passage(passage: str) -> str:
    return f'Represent this passage\npassage: {passage}'


def extract_linear_chain(hierarchical_chain):
    """Extracts a linear list of all papers from the hierarchical chain."""
    linear_chain = []

    def traverse(chain):
        for year, entries in chain.items():
            for entry in entries:
                paper = entry.get("paper")
                if paper:
                    linear_chain.append(paper)  # Add the paper to the linear chain
                if "chain" in entry and entry["chain"]:  # If sub-chain exists, traverse it
                    traverse(entry["chain"])

    traverse(hierarchical_chain)
    return linear_chain

def remove_numbering(title):
    return re.sub(r'^\d+\.\s*', '', title)

# Function to load LLaMA model
def load_llama_model():
    try:
        print("Loading LLaMA model...")
        print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        model_id = "meta-llama/Llama-3.3-70B-Instruct"
        #model_id = "meta-llama/Llama-3.2-1B-Instruct"
        llama_pipeline = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        return llama_pipeline
    except Exception as e:
        print(f"Error loading LLaMA model: {e}")
        return None

# Function to evaluate papers with a preloaded LLaMA pipeline
def evaluate_papers_with_llama(prompt, llama_pipeline):
    try:
        # Create the messages input
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to evaluate scientific literature."},
            {"role": "user", "content": prompt}
        ]

        # Generate a response
        outputs = llama_pipeline(
            messages,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # Extract the assistant's response
        response = outputs[0]["generated_text"]

        return response
    except Exception as e:
        print(f"Error with LLaMA pipeline: {e}")
        return None

def evaluate_papers_with_gpt(prompt):
    from openai import OpenAI
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI()
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to evaluate scientific literature."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            top_p=1.0,
        )
        print("GPT4 usage: {}".format(response.usage))
        try:
            usage = response.usage
            log_llm_usage("gpt-4o", usage.prompt_tokens, usage.completion_tokens)
        except Exception as e:
            print(f"⚠️ Failed to log GPT usage: {e}")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with GPT-4 call: {e}")
        return None

def relevancy_prompt(source_paper, few_shot_prompt, year, papers):
    # Generate paper list
    paper_list = ""
    for idx, paper in enumerate(papers):
        paper_list += f"\n\n{idx + 1}. Title: {paper['title']} Abstract: {paper.get('abstract', 'No abstract available')}"

    prompt_template = relevancy_scoring_prompt()
    prompt = prompt_template.format(
        few_shot_prompt=few_shot_prompt,
        source_title=source_paper['title'],
        source_abstract=source_paper['abstract'],
        year=year,
        paper_list=paper_list
    )
    return prompt

def split_into_chunks(papers, chunk_size):
    return [papers[i:i + chunk_size] for i in range(0, len(papers), chunk_size)]

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Function to use GPT-4 for query generation
def generate_query_with_gpt(source_paper):
    
    prompt_template = search_query_prompt()
    prompt = prompt_template.format(
        source_title=source_paper['title'],
        source_abstract=source_paper['abstract']
    )
    
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant specialized in generating search query for literature research in semantic scholar or google scholar."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating query with GPT-4: {e}")
        return None

# Function to search Semantic Scholar
def search_papers(query, limit=1):
    api_key = os.environ.get("S2_API_KEY", None)
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": api_key}
    params = {
        "query": query,
        "fields": 'externalIds,title,year,citationCount,abstract',
        "limit": limit
    }
    max_retries = 3
    backoff_factor = 3
    
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            print(f'Attempt {attempt + 1} failed: Status code {response.status_code}')
            time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff

    print("Failed to fetch data after retries.")
    return None

# Function to evaluate papers with GPT
def evaluate_papers_with_gpt_old(prompt):
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to evaluate scientific literature."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with GPT API: {e}")
        return None

# Function to clean GPT output
def clean_gpt_output(output):
    try:
        # Use regex to find the JSON block within the text
        json_pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, output, re.DOTALL)
        
        if match:
            # Extract the JSON content from the match
            json_content = match.group(1).strip()
        else:
            # Fall back to finding any JSON-like structure
            json_pattern = r"(\{.*?\})"
            match = re.search(json_pattern, output, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
            else:
                raise ValueError("No JSON found in the output")

        # Parse the JSON content
        return json.loads(json_content)
    except Exception as e:
        # Log the error and return an empty dictionary
        print(f"Error parsing GPT output: {e}")
        return {}

async def get_paper_by_id(paper_id):
    url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}'
    # Specify the fields you are interested in
    params = {
        'fields': 'externalIds,title,year,citationCount,abstract,citations.title,citations.year,citations.paperId,citations.citationCount,citations.abstract'
    }
    # Include API key in headers if needed
    api_key = os.environ.get("S2_API_KEY", None)
    headers = {
        'x-api-key': api_key  # Replace 'your_api_key_here' with your actual API key
    }
    max_retries = 3
    backoff_factor = 2
    
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f'Attempt {attempt + 1} failed: Status code {response.status_code}')
            time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff

    print("Failed to fetch data after retries.")
    return None

async def get_paper_data(paper_id):
    url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}'
    # Specify the fields you are interested in
    params = {
        'fields': 'externalIds,title,year,citationCount,abstract'
    }
    # Include API key in headers if needed
    api_key = os.environ.get("S2_API_KEY", None)
    headers = {
        'x-api-key': api_key  # Replace 'your_api_key_here' with your actual API key
    }
    max_retries = 3
    backoff_factor = 2
    
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f'Attempt {attempt + 1} failed: Status code {response.status_code}')
            time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff

    print("Failed to fetch data after retries.")
    return None

async def get_next_year_papers(paper_id):
    # First, fetch the main paper to determine its publication year
    main_paper = await get_paper_by_id(paper_id)
    if main_paper is None:
        print("Failed to retrieve main paper.")
        return None
    #print(main_paper) DONE
    
    #sorted_citations = sorted(
    #(citation for citation in main_paper.get('citations', []) if citation.get('year') is not None),
    #key=lambda x: x['year'])
    
    sorted_citations = sorted(
    (citation for citation in main_paper.get('citations', []) if citation.get('citationCount') is not None),
    key=lambda x: x.get('citationCount', 0),
    reverse=True)
    
    next_year_papers = [c for c in sorted_citations]
    
    #print(next_year_papers) #DONE
    return next_year_papers[0:5]

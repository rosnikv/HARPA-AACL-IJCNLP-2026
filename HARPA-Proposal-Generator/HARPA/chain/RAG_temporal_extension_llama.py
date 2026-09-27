import os
# add gpu support
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from dotenv import load_dotenv
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
from huggingface_hub import whoami, login
try:
    whoami()  # Check if already authenticated
except Exception:
    login(token=huggingface_token)
import os, sys
from llm_logger import set_log_file_path, log_llm_usage

import json
import asyncio
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("S2_API_KEY", None)
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0
from utils import clean_gpt_output, evaluate_papers_with_llama, load_llama_model
from utils import get_paper_data, get_paper_by_id, relevancy_prompt
from utils import remove_numbering, split_into_chunks, search_papers, extract_linear_chain, save_json
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import set_seed, evaluate_papers_with_gpt
import argparse
SEED = None
REVIEW_ID = None
IDX = None
llama_pipeline = None
saved_top_papers = False
USE_GPT4 = True  # Set to False to switch back to LLaMA

BASE_PATH="/weka/ROOT/harpa/hypothesis_generation"
#BASE_PATH="."
import logging
log_dir =  f"{BASE_PATH}/cache_results_harpa/logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"pipeline_llama_CS.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = None
intermediate_dir = None
result_dir = None

def evaluate_papers(prompt, llama_pipeline=None):
    if USE_GPT4:
        return evaluate_papers_with_gpt(prompt)
    else:
        return evaluate_papers_with_llama(prompt, llama_pipeline)

def parse_args():
    parser = argparse.ArgumentParser(description="Run script with a specific seed.")
    parser.add_argument("--seed", type=int, required=True, help="Seed value for the run.")
    parser.add_argument("--review_id", type=str, required=True, help="Review ID to process.")
    parser.add_argument("--k", type=int, default=2, help="Number of temporal chains to generate.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for user-specific data.")
    parser.add_argument("--output_dir", type=str, required=False, help="Directory for final outputs like logs.")

    return parser.parse_args()

def set_global_seed(seed):
    """Set the global seed."""
    global SEED
    SEED = seed
    set_seed(seed)
    logging.info(f"Global seed set to: {SEED}")

      
def load_global_llama():
    global llama_pipeline
    if not USE_GPT4 and llama_pipeline is None:
        llama_pipeline = load_llama_model()

def log_to_json_file(data, filename= None):
    # "./temporal_reasoning_chain/intermediate_chains/llama_outputs.json"
    """Append data to a JSON file."""
    global REVIEW_ID, intermediate_dir
    if filename is None:
        filename = os.path.join(intermediate_dir, f"llama_outputs_{REVIEW_ID}_p-{IDX}_log.json")   
    try:
        if os.path.exists(filename):
            with open(filename, "r") as infile:
                existing_data = json.load(infile)
        else:
            existing_data = []
        data["seed"] = SEED
        
        existing_data.append(data)
        with open(filename, "w") as outfile:
            json.dump(existing_data, outfile, indent=4)
    except Exception as e:
        logging.error(f"Error saving to JSON file: {e}")
        
def filter_papers(citation_details, current_year):
    year_wise_citations = {}
    for cited_paper in citation_details.get("citations", []):
        cited_year = cited_paper.get("year")
        title = cited_paper.get("title", "")
        abstract = cited_paper.get("abstract", "")
        citation_count = cited_paper.get("citationCount", None)

        if cited_year and abstract and (cited_year >= current_year):
            title_is_english = detect(title) == 'en' if title else False
            valid_year = cited_year is not None and isinstance(cited_year, int)
            valid_citation_count = (cited_year >= current_year) or (citation_count is not None)
            
            if cited_year not in [2025] and cited_paper.get("citationCount", 0) <= 0:
                continue
            
            if title_is_english and valid_year and valid_citation_count:
                if cited_year not in year_wise_citations:
                    year_wise_citations[cited_year] = []
                year_wise_citations[cited_year].append({
                    "paperId": cited_paper.get("paperId"),
                    "title": cited_paper.get("title"),
                    "abstract": cited_paper.get("abstract", None),
                    "citationCount": cited_paper.get("citationCount", 0)
                })

    return year_wise_citations

def parse_output(response):
    if USE_GPT4:
        return clean_gpt_output(response)
    else:
        return parse_llama_output(response)
 
def parse_llama_output(response):
    llama_output = {}
    for item in response:
        if item['role'] == 'assistant':
            try:
                return clean_gpt_output(item['content'])
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing LLAMA output: {e}")
                logging.debug(f"Problematic LLAMA content: {item['content']}")
    return llama_output

def extract_top_papers(llama_output):
    top3_papers = llama_output.get("top3_relevant_papers", [])
    logging.info(f"Top 3 papers: {top3_papers}")
    extracted_papers = []

    for top_paper in top3_papers:
        top = remove_numbering(top_paper)
        result = search_papers(top)
        if result:
            paper = {
                "paperId": result[0]["paperId"],
                "title": result[0]["title"],
                "abstract": result[0]["abstract"],
                "year": result[0]["year"],
                "citation_count": result[0]["citationCount"],
                "relevance": top3_papers[top_paper]["relevance"],
                "explanation": top3_papers[top_paper]["explanation"]
            }
            extracted_papers.append(paper)
        else:
            logging.warning(f"Paper not found: {top_paper}")
    return extracted_papers

def process_yearly_citations(year, year_wise_citations, current_paper, few_shot_prompt, llama_pipeline):
    if year not in year_wise_citations:
        return []

    chunks = split_into_chunks(year_wise_citations[year], 10)
    relevant_papers_for_year = []

    for chunk_index, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {chunk_index + 1}/{len(chunks)} for year {year}")
        prompt = relevancy_prompt(current_paper, few_shot_prompt, year, chunk)
        try:
            #response = evaluate_papers_with_llama(prompt, llama_pipeline)
            response = evaluate_papers(prompt, llama_pipeline)

            llama_output = parse_output(response)
            logging.info(f"Year: {year}, Chunk {chunk_index + 1} Llama Output: {llama_output}")
            log_to_json_file({
                "year": year,
                "source_paper": current_paper,
                "llama_output": llama_output
            })
            relevant_papers_for_year.extend(extract_top_papers(llama_output))
        except Exception as e:
            logging.error(f"Error processing chunk {chunk_index + 1} for year {year}: {e}")
    return relevant_papers_for_year

def save_relevant_papers_to_file(papers):
    # filename= "./temporal_reasoning_chain/intermediate_chains/all_relevant_papers.json"
    """Save relevant papers to a JSON file."""
    global REVIEW_ID
    global IDX
    global intermediate_dir
    filename = os.path.join(intermediate_dir, f"all_relevant_papers_{REVIEW_ID}_pk-{IDX}.json")
    with open(filename, "w") as outfile:
        json.dump(papers, outfile, indent=4)
    logging.info(f"Relevant papers saved to '{filename}'.")

def compute_score(paper, max_citation_count, w_r=0.7, w_c=0.3):
    """Calculate a paper's heuristic score."""
    relevance = paper.get("relevance", 0)
    citation_count = paper.get("citation_count", 0)
    normalized_citation = citation_count / max_citation_count if max_citation_count > 0 else 0
    return (w_r * relevance) + (w_c * normalized_citation)

def find_best_paper(relevant_papers):
    """Find the best paper based on heuristic scoring."""
    if not relevant_papers:
        return None

    max_citation_count = max(paper.get("citation_count", 1) for paper in relevant_papers)
    return max(relevant_papers, key=lambda paper: compute_score(paper, max_citation_count))

def process_top_papers(top_papers_across_years):
    """Process top papers to find the most relevant one."""
    if not top_papers_across_years:
        return None

    # Filter papers with relevance scores 1 or 2
    relevant_papers = [
        paper for paper in top_papers_across_years if paper.get("relevance", 0) in [1, 2]
    ]

    # Save relevant papers globally
    global saved_top_papers
    if not saved_top_papers:
        save_relevant_papers_to_file(relevant_papers)
        saved_top_papers = True

    logging.info(f"Relevant papers: {relevant_papers}")
    # Find and return the best paper
    return find_best_paper(relevant_papers) if relevant_papers else None

        
async def build_temporal_chain(source_paper, few_shot_prompt, current_year, root_paper=None, end_year=2025):
    visited_paper_ids = set()
    visited_paper_ids.add(source_paper['paperId'])
    global llama_pipeline
    if not USE_GPT4 and llama_pipeline is None:
        raise ValueError("Llama pipeline is not loaded.")
   
    chains = {}
    # Add root_paper first if provided
    if root_paper:
        if isinstance(root_paper, list):
            for rp in root_paper:
                ry = rp["year"]
                chains.setdefault(ry, []).append({"paper": rp})
                visited_paper_ids.add(rp["paperId"])
        else:
            ry = root_paper["year"]
            chains.setdefault(ry, []).append({"paper": root_paper})
            visited_paper_ids.add(root_paper["paperId"])
            
    # Check if source_paper is already added
    def is_paper_in_chains(paper):
        py = paper["year"]
        return any(entry["paper"]["paperId"] == paper["paperId"] for entry in chains.get(py, []))

    if not is_paper_in_chains(source_paper):
        source_year = source_paper["year"]
        chains.setdefault(source_year, []).append({"paper": source_paper})

    papers_to_process = [(source_paper, current_year)]

    while papers_to_process:
        logging.info(f"Current papers to process: {papers_to_process}")
        top_papers_across_years = []
        current_paper, current_year = papers_to_process.pop(0)
        if current_year > end_year:
            continue
        logging.info(f"Processing source paper: {current_paper}")

        # Load citations for the paper
        citation_details = await get_paper_by_id(current_paper['paperId'])
        year_wise_citations = filter_papers(citation_details, current_year)
        #logging.info(f"Year-wise citations: {year_wise_citations}")
        for year in range(current_year, current_year + 1):
            logging.info(f"Processing citations for year {year}")
            try:
                relevant_papers_for_year = process_yearly_citations(
                    year, year_wise_citations, current_paper, few_shot_prompt, llama_pipeline
                )
                if relevant_papers_for_year:
                    top_papers_across_years.extend(relevant_papers_for_year)
            except Exception as e:
                logging.error(f"Error processing citations for year {year}: {e}")

        ## add here sys.exit to and run 10 times with different seeds?
        #sys.exit()
        
        if top_papers_across_years:
            best_paper = process_top_papers(top_papers_across_years)
            if best_paper:
                logging.info(f"Selected best paper using heuristic scoring: {best_paper}")
        else:
            best_paper = None

        if best_paper:
            if best_paper['paperId'] in visited_paper_ids:
                logging.info(f"Paper already visited: {best_paper['title']} ({best_paper['paperId']}). Skipping to avoid loop.")
                continue
            logging.info(f"Adding best paper to process queue: {best_paper}")
            papers_to_process.append((best_paper, best_paper['year']))
            if best_paper['year'] not in chains:
                chains[best_paper['year']] = []
            chains[best_paper['year']].append({"paper": best_paper})
            visited_paper_ids.add(best_paper['paperId'])
            logging.info(f"Appended chain for best paper {best_paper['title']} in year {best_paper['year']}")
            logging.info(f"Chain: {chains}")
        else:
            logging.info(f"No best paper found, incrementing year for current paper: {current_paper['title']}")
            papers_to_process.append((current_paper, current_year + 1))   
            
    return chains
 
async def extend_temporal_path_with_chains(source_paper, few_shot_prompt, root_paper=None):
    global intermediate_dir, result_dir
    paper_details = await get_paper_data(source_paper['pmid'])
    if paper_details is None:
        logging.error(f"Failed to fetch data for source paper with PMID: {source_paper['pmid']}. Skipping...")
        return
    
    source_paper = {
        "paperId": paper_details["paperId"],
        "pmid": paper_details["externalIds"].get("ArXiv", "None"),
        "title": paper_details.get("title"),
        "abstract": paper_details.get("abstract"),
        "year": paper_details.get("year"),
        "citation_count": paper_details.get("citationCount","None")
    }  
    
    load_global_llama()
    
    # Start building the temporal chain from the source paper
    full_chain = await build_temporal_chain(source_paper, few_shot_prompt, source_paper['year'], root_paper, end_year=2025)
    
    # Save the output in hierarchical format
    #save_json(full_chain, "temporal_chains_llama.json")    
    linear_chain = extract_linear_chain(full_chain)
    global REVIEW_ID
    global IDX
    outfile = os.path.join(result_dir, f"temporal_chain_{REVIEW_ID}_p-{IDX}.json")
    #outfile = f"./temporal_reasoning_chain/result_chains/temporal_chain_{REVIEW_ID}_p-{IDX}.json"
    save_json(linear_chain, outfile)
    logging.info(f"Linear chain saved to {outfile}.")

def get_branch_seeds_with_optional_parent(review_id, pk_idx, exclude_ids=None, with_parents=False):
    global intermediate_dir, result_dir
    relevant_papers_file = os.path.join(intermediate_dir, f"all_relevant_papers_{review_id}_pk-{pk_idx}.json")
    #relevant_papers_file = f"./temporal_reasoning_chain/intermediate_chains/all_relevant_papers_{review_id}_pk-{pk_idx}.json"
    with open(relevant_papers_file, "r") as f:
        relevant_papers = json.load(f)

    if exclude_ids:
        relevant_papers = [paper for paper in relevant_papers if paper.get("paperId") not in exclude_ids]

    rel2 = [p for p in relevant_papers if p.get("relevance") == 2]
    rel1 = [p for p in relevant_papers if p.get("relevance") == 1]
    # Sort each group by citation count (descending)
    rel2.sort(key=lambda p: int(p.get("citation_count", 0)), reverse=True)
    rel1.sort(key=lambda p: int(p.get("citation_count", 0)), reverse=True)

    seeds = rel2 + rel1

    if not with_parents:
        return seeds

    # Load parent from the corresponding chain (second paper)
    chain_path = os.path.join(result_dir, f"temporal_chain_{review_id}_p-{pk_idx}.json")
    #chain_path = f"./temporal_reasoning_chain/result_chains/temporal_chain_{review_id}_p-{pk_idx}.json"
    try:
        with open(chain_path, "r") as f:
            chain = json.load(f)
        parent_paper = chain[1] if len(chain) > 1 else None
    except Exception:
        parent_paper = None

    if parent_paper is None:
        return []

    return [(parent_paper, paper) for paper in seeds]


def main():
    args = parse_args()
    set_log_file_path(args.output_dir)
    
    global IDX
    global REVIEW_ID
    global saved_top_papers
    IDX = 1
    REVIEW_ID = args.review_id
    set_global_seed(args.seed)
    
    global BASE_DIR, intermediate_dir, result_dir
    BASE_DIR = args.base_dir
    intermediate_dir = os.path.join(BASE_DIR, "intermediate_chains")
    result_dir = os.path.join(BASE_DIR, "result_chains")
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    BASE_PATH="/weka/ROOT/harpa/hypothesis_generation"
    #BASE_PATH="."

    with open(f"{BASE_PATH}/HARPA/chain/few_shot_biomed/output.json", "r") as f:
        data = json.load(f)
    with open(f"{BASE_PATH}/HARPA/chain/few_shot_biomed/gpt4_output", "r") as f:
        evaluations = json.load(f)
    
    few_shot_papers = data[1:3]
    few_shot_evaluations = list(evaluations.items())[1:3]

    few_shot_prompt = "Examples:\n\n"
    for idx, paper in enumerate(few_shot_papers):
        few_shot_prompt += f"{idx + 1}. Title: {paper['Title']} Abstract: {paper['Abstract']}; ({paper['Year']})\n"

    # Add example JSON evaluations
    few_shot_prompt += "\nExample evaluations in JSON format:\n```json\n{\n"
    for idx, (eval_title, eval_details) in enumerate(few_shot_evaluations):
        few_shot_prompt += f'    "{eval_title}": {{\n'
        few_shot_prompt += f'        "explanation": "{eval_details["explanation"]}",\n'
        few_shot_prompt += f'        "relevance": {eval_details["relevance"]}\n'
        few_shot_prompt += f'    }}'
        if idx < len(few_shot_evaluations) - 1:
            few_shot_prompt += ",\n"
    few_shot_prompt += "\n}\n```"
    
    paper_id = f"{REVIEW_ID}"

    source_details = {
        "pmid": paper_id
    }
    logging.info(f"Processing review ID: {REVIEW_ID} with seed: {args.seed}")
    
    paper_details = asyncio.run(get_paper_data(source_details['pmid']))
    if paper_details is None:
        logging.error(f"Failed to fetch data for source paper with PMID: {source_details['pmid']}. Skipping...")
        return
    
    original_source = [{
        "paperId": paper_details["paperId"],
        "pmid": paper_details["externalIds"].get("ArXiv", "None"),
        "title": paper_details.get("title"),
        "abstract": paper_details.get("abstract"),
        "year": paper_details.get("year"),
        "citation_count": paper_details.get("citationCount","None")
    }]
    asyncio.run(extend_temporal_path_with_chains(source_details, few_shot_prompt))
    
    # Step 2: Get first-level branch seeds (pk-1)
    temporal_chain_path = os.path.join(result_dir, f"temporal_chain_{REVIEW_ID}_p-1.json")
    #temporal_chain_path = f"./temporal_reasoning_chain/result_chains/temporal_chain_{REVIEW_ID}_p-1.json"
    with open(temporal_chain_path, "r") as f:
        temporal_chain = json.load(f)
    
    exclude_ids = set()
    if len(temporal_chain) > 1 and "paperId" in temporal_chain[1]:
        exclude_ids.add(temporal_chain[1]["paperId"])
        
    branch_seeds = get_branch_seeds_with_optional_parent(REVIEW_ID, pk_idx=1, exclude_ids=exclude_ids, with_parents=False)

    total_chains_built = 1
  
    # Step 3: Loop through branch seeds and build chains
    for i, paper in enumerate(branch_seeds):
        if total_chains_built >=args.k:
            logging.info(f"Reached chain limit ({total_chains_built}). Halting further chain generation.")
            return
        
        IDX = i + 2  # Start from 2, since 1 is for the original chain
        saved_top_papers = False
        # Prepare the branch paper in the format expected by extend_temporal_path_with_chains
        branch_source = {
            "pmid": paper["paperId"]
        }
        combined_roots = original_source + [paper]
        total_chains_built += 1
        try:
            logging.info(f"Starting branch chain {i+1} with paper: {paper['title']}")
            asyncio.run(extend_temporal_path_with_chains(branch_source, few_shot_prompt, root_paper=combined_roots))
        except Exception as e:
            logging.error(f"Failed to build branch chain for paper {paper['title']}: {e}")

    # Step 4: Second-level branch seeds from pk-2
    if total_chains_built < args.k:
        # Collect exclude_ids from all first-level branch chains (p-2 to p-N)
        for idx in range(2, total_chains_built + 1):  # includes p-2, p-3, ...
            chain_path = os.path.join(result_dir, f"temporal_chain_{REVIEW_ID}_p-{idx}.json")
            #chain_path = f"./temporal_reasoning_chain/result_chains/temporal_chain_{REVIEW_ID}_p-{idx}.json"
            if os.path.exists(chain_path):
                with open(chain_path, "r") as f:
                    chain = json.load(f)
                    if len(chain) > 1 and "paperId" in chain[1]:
                        exclude_ids.add(chain[1]["paperId"])
                       
        second_level_seed_pairs = get_branch_seeds_with_optional_parent(REVIEW_ID, pk_idx=2, exclude_ids=exclude_ids, with_parents=True)
        if not second_level_seed_pairs:
            logging.info("No valid second-level branch seeds found. Ending chain generation.")
            return

        for j, (first_level_paper, paper) in enumerate(second_level_seed_pairs):
            if total_chains_built >= args.k:
                logging.info(f"Reached chain limit ({total_chains_built}). Halting further chain generation.")
                return
            IDX = total_chains_built + 1
            saved_top_papers = False
            total_chains_built += 1
            
            combined_roots = original_source + [first_level_paper, paper]
            try:
                asyncio.run(extend_temporal_path_with_chains({"pmid": paper["paperId"]}, few_shot_prompt, root_paper=combined_roots))
            except Exception as e:
                logging.error(f"Failed in second-level branching for paper {paper['title']}: {e}")

    logging.info(f"Finished chain generation. Total chains built: {total_chains_built}")
    
    #print(source_details)
    #sys.exit()

if __name__ == "__main__":
    main()
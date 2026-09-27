import os
from dotenv import load_dotenv
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
from huggingface_hub import whoami, login
try:
    whoami()  # Check if already authenticated
except Exception:
    login(token=huggingface_token)
import sys
import argparse
from utils import deduplicate_citing_chain

from tqdm import tqdm
import torch
import json
from prompt_utils.generate_prompt import generate_initial_hypothesis, generate_initial_hypothesis_AutoNORA
from prompt_utils.generate_prompt import generate_initial_hypothesis_codeScientist
from prompt_utils.refine_prompt import refine_for_asd, refine_for_asd_nora, refine_QA
from prompt_utils.variable_prompt import get_key_variables, single_variable_space
from prompt_utils.refine_prompt import qa_generation
from llm_logger import set_log_file_path, summarize_cost_log, log_llm_usage

from code_scientist_novelty_detector import novelty_assessment
import asyncio
from collections import defaultdict
import re
from tenacity import retry, wait_random_exponential, stop_after_attempt

def data_prompt(chain):
    prompt_builder = ""
    for idx, val in enumerate(chain):
        prompt_builder += f"{idx}. Title: {val['title']}; \nAbstract: {val['abstract']}; \nYear: {val['year']}\nExplanation: {val['explanation']}\n\n"
        #prompt_builder += f"{idx}. Title: {val['title']}; \nAbstract: {val['abstract']}; \nYear: {val['year']}\n"

    return prompt_builder

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_gpt(user_prompt, system_prompt = "You are a scientist reviewing a paper. Your response must be in JSON format"):
    import openai
    import time
    from openai import OpenAI
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI()
    seed =2025
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
                seed=seed
            )
            print("GPT4 usage: {}".format(response.usage))
            try:
                usage = response.usage
                log_llm_usage("gpt-4o", usage.prompt_tokens, usage.completion_tokens)
            except Exception as e:
                print(f"⚠️ Failed to log GPT usage: {e}")
                
            return response.choices[0].message.content
        except Exception as e:
            retry_count += 1
            print(f"Error calling GPT (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                time.sleep(2)  # wait 2 seconds before retry
            else:
                raise

def generate(agent_name, source_paper, citing_papers, research_goal):
    if agent_name == "code-scientist":
        system_message, user_message = generate_initial_hypothesis_codeScientist(source_paper, citing_papers, research_goal)
    elif agent_name == "auto-nora":
        system_message, user_message = generate_initial_hypothesis_AutoNORA(source_paper, citing_papers)
    else:
        system_message, user_message = generate_initial_hypothesis(source_paper, citing_papers)
    
    response = call_gpt(user_message, system_message)
    response = parse_research_idea(response)
    return response

def qa_function(hypothesis, agent_name):
    system_message, user_message = qa_generation(hypothesis, agent_name)
    response = call_gpt(user_message, system_message)
    response =  parse_research_idea(response)
    return response

def refine(agent_name, hypothesis, questions, citing_papers, similar_retreived_papers):
    
    system_message, user_message = refine_QA(agent_name, hypothesis, questions, citing_papers, similar_retreived_papers)

    response = call_gpt(user_message, system_message)
    response = parse_research_idea(response)
    return response

def key_variable_space(hypothesis, similar_retreived_papers):  
    system_message, user_message = get_key_variables(hypothesis, similar_retreived_papers)
    response = call_gpt(user_message, system_message)
    return response

def get_variable_space(hypothesis, variable_info, similar_retreived_papers):
    variable_info_formatted = "\n".join(f"{key} : {value}" for key, value in variable_info.items())
    system_message, user_message = single_variable_space(hypothesis, variable_info_formatted, similar_retreived_papers)
    response = call_gpt(user_message, system_message)
    return response

def parse_research_idea(response):
    if not response or not isinstance(response, str):
        raise ValueError("Invalid input: Response is empty or not a string")
    
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    
    json_string = match.group(1) if match else response.strip()

    if not json_string:
        raise ValueError("Extracted JSON string is empty")

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding error: {e}")


from concurrent.futures import ThreadPoolExecutor, as_completed

def process_chain_file(input_file, agent_name, output_dir, runs, goal):
    from utils import deduplicate_citing_chain  # local import for thread safety
    file_parts = os.path.basename(input_file).split("_")
    paper_id = file_parts[-2] if len(file_parts) > 2 else "unknown"
    p_suffix = file_parts[-1].replace(".json", "")
    
    with open(input_file, "r") as f:
        chain = json.load(f)

    source_paper = chain[0]
    citing_papers = chain[1:]
    deduped_citing_chain = deduplicate_citing_chain(citing_papers, source_paper)
    citing_papers_str = data_prompt(deduped_citing_chain)
    chain = [source_paper] + deduped_citing_chain

    for i in range(runs):
        #output_file = os.path.join(output_dir, f"hyp{i}_{paper_id}.json")
        output_file = os.path.join(args.output_dir, f"hyp{i}_{paper_id}_{p_suffix}.json")

        output_data = []

        research_idea = generate(agent_name, source_paper, citing_papers_str, goal)
        chain_results = {
            "chain_path": input_file,
            "chain": chain,
            "initial_hypothesis_details": research_idea,
            "related_work_analysis": {},
            "refined_hypothesis_details": {},
            "key_variables": {},
            "variable_space": []
        }

        hypothesis = research_idea["Hypothesis"]
        chain_results["initial_hypothesis"] = hypothesis

        similar_retreived_papers = []
        code_scientist_novelty_init = novelty_assessment(hypothesis)
        for analysis in code_scientist_novelty_init:
            for snippet in analysis["retrieved_snippets"]:
                paper_title = snippet["paper"]["title"]
                snippet_text = snippet["snippet"]
                similar_retreived_papers.append(f"{paper_title} - {snippet_text}")
        similar_retreived_papers_str = "\n\n".join(f"{idx}. {doc}" for idx, doc in enumerate(similar_retreived_papers))
        chain_results['related_work_analysis'] = code_scientist_novelty_init

        result = qa_function(hypothesis, agent_name)
        questions_data = result.get("questions", [])
        questions_str = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(questions_data)])

        refined_hg = refine(agent_name, hypothesis, questions_str, citing_papers_str, similar_retreived_papers_str)
        chain_results['refined_hypothesis_details'] = refined_hg
        chain_results['refined_hypothesis_details']["questions"] = questions_data
        new_hypothesis = refined_hg['refined_hypothesis']
        chain_results['refined_hypothesis'] = new_hypothesis

        code_scientist_novelty = novelty_assessment(new_hypothesis)
        chain_results['refined_hypothesis_details']["related_work_analysis"] = code_scientist_novelty

        similar_retreived_papers_local = similar_retreived_papers
        #rw_analysis = refined_hg.get("novelty_analysis", []) <-- possible? bug-fixed
        rw_analysis = code_scientist_novelty
        for i in range(4):
            if i < len(rw_analysis) and isinstance(rw_analysis[i], dict):
                retrieved_snippets = rw_analysis[i].get("retrieved_snippets", {})
                for snippet in retrieved_snippets:
                    paper_title = snippet["paper"]["title"]
                    snippet_text = snippet["snippet"]
                    similar_retreived_papers_local.append(f"{paper_title} - {snippet_text}")

        variable_options = key_variable_space(new_hypothesis, similar_retreived_papers_local)
        variable_options = parse_research_idea(variable_options)
        chain_results["key_variables"] = variable_options['key_variables']

        for variable_info in variable_options['key_variables']:
            variable_space = get_variable_space(new_hypothesis, variable_info, similar_retreived_papers_local)
            variable_space = parse_research_idea(variable_space)
            chain_results["variable_space"].append(variable_space)

        output_data.append(chain_results)

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"✓ Completed: {output_file}")

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="code-scientist")
    parser.add_argument("--input_dir", type=str, default="./chain_data/batch1/")
    parser.add_argument("--output_dir", type=str, default="./output/results_qa/batch1/")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--goal", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".json")]
    set_log_file_path(args.output_dir)

    # Parallel execution
    with ThreadPoolExecutor(max_workers=4) as executor:  # adjust max_workers as needed
        futures = [
            executor.submit(process_chain_file, f, args.agent, args.output_dir, args.runs, args.goal)
            for f in input_files
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("Error in thread:", e)
    
    # Summarize total LLM usage cost
    summarize_cost_log()
    print(f"📄 LLM usage log saved at: {os.path.join(args.output_dir, 'llm_usage_log.csv')}")


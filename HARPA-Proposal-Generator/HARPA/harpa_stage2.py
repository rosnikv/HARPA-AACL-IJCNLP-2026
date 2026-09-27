import os
import sys
import json
from prompt_utils.generate_prompt import get_specific_hypotheses, get_specific_hypotheses_AutoNORA, get_specific_hypotheses_codeScientist
import argparse
import copy
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()
from prompt_utils.novelty_rag_001 import novelty_query_prompt, novelty_judgement_prompt, async_semantic_call, novelty_system_prompt, call_semantic
import asyncio
from code_scientist_novelty_detector import novelty_assessment
import re
from utils import get_paper_metadata_by_title, score_idea
from utils import populate_operationalization_one_idea_simple_method
from utils import extract_required_fields, extract_hypothesis_elements, generate_rubric_from_hypothesis_and_op
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_logger import set_log_file_path, summarize_cost_log, log_llm_usage

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_json_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def data_prompt(chain):
    prompt_builder = ""
    for idx, val in enumerate(chain):
        prompt_builder += f"{idx}. Title: {val['title']}; \nAbstract: {val['abstract']}; \nYear: {val['year']}\n"
    return prompt_builder

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
  

def call_gpt(user_prompt, system_prompt = "You are a scientist reviewing a paper. Your response must be in JSON format"):
    import openai
    import os
    from openai import OpenAI
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI()
    seed =2025
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

def aiscientist_novelty(hypothesis):
    query_prompt = novelty_query_prompt(hypothesis)
    query = call_gpt(novelty_system_prompt(), query_prompt)
    print(f"Generated query: {query}")
    if "```json" in query:
        query = re.search("```json\s*(.*?)\s*```", query, re.DOTALL).group(1)
    query = json.loads(query)["Query"]
    print(query)
    docs = asyncio.run(call_semantic(query))
    #print(docs)
    docs = [x['text'] for x in docs]
    judgement_prompt = novelty_judgement_prompt(hypothesis, docs)
    novelty_res = call_gpt(judgement_prompt, novelty_system_prompt())
    novelty_res_json = json.loads(novelty_res) if isinstance(novelty_res, str) else novelty_res
    print(novelty_res_json)
    return {
        "query": query,
        "retrieved_docs": docs,
        "novelty_result": novelty_res_json
    }

def generate(agent_name, hypothesis, variable_space, similar_retreived_papers):
    if agent_name == "code-scientist":
        system_message, user_message = get_specific_hypotheses_codeScientist(hypothesis, variable_space, similar_retreived_papers)
    elif agent_name == "auto-nora":
        system_message, user_message = get_specific_hypotheses_AutoNORA(hypothesis, variable_space, similar_retreived_papers)
    else:
        system_message, user_message = get_specific_hypotheses(hypothesis, variable_space, similar_retreived_papers)
    
    response = call_gpt(user_message, system_message)
    return response

def process_file(input_file, args):
    agent_name = args.agent
    num_runs = args.num_runs
    output_path = os.path.join(args.output_dir, "final/")
    os.makedirs(output_path, exist_ok=True)
    
    input_path = os.path.join(args.input_dir, input_file)
    # get basename of the file without extension
    file_id = os.path.splitext(input_file)[0].split("_")[1]
    data = read_json_file(input_path)  
    print(data[0].keys())
    print(f"Processing file: {input_file}")
    ##### snippet retrieved to get more specific information relevant to initial hypothesis
    similar_retreived_papers = []
    code_scientist_novelty_init = data[0]["related_work_analysis"]
    for analysis in code_scientist_novelty_init:
        for snippet in analysis["retrieved_snippets"]:
            paper = snippet.get("paper", {})
            paper_title = paper.get("title", "")
            snippet_text = snippet.get("snippet", "")

            # Metadata enrichment
            if "year" not in paper or "citationCount" not in paper:
                metadata = get_paper_metadata_by_title(paper_title)
                if metadata:
                    paper["year"] = metadata["year"]
                    paper["citationCount"] = metadata["citations"]
                    paper["paperId"] = metadata["paperId"]
                    snippet["paper"] = paper
            
            year = paper.get("year", "N/A")
            citations = paper.get("citationCount", "N/A")
            similar_retreived_papers.append(f"{paper_title} (Year: {year}, Citations: {citations}) - {snippet_text}")

    refined_hg = data[0]["refined_hypothesis_details"]        
    refined_hypothesis = data[0]['refined_hypothesis']
    rw_analysis = refined_hg.get("related_work_analysis", [])
    for i in range(4):
        if i < len(rw_analysis) and isinstance(rw_analysis[i], dict):
            retrieved_snippets = rw_analysis[i].get("retrieved_snippets", {})
            for snippet in retrieved_snippets:
                paper = snippet.get("paper", {})
                paper_title = paper.get("title", "")
                snippet_text = snippet.get("snippet", "")

                # Metadata enrichment
                if "year" not in paper or "citationCount" not in paper:
                    metadata = get_paper_metadata_by_title(paper_title)
                    if metadata:
                        paper["year"] = metadata["year"]
                        paper["citationCount"] = metadata["citations"]
                        paper["paperId"] = metadata["paperId"]
                        snippet["paper"] = paper
                
                year = paper.get("year", "N/A")
                citations = paper.get("citationCount", "N/A")
                similar_retreived_papers.append(f"{paper_title} (Year: {year}, Citations: {citations}) - {snippet_text}")
    
    # Concatenate all snippets into a formatted string
    similar_retreived_papers = "\n\n".join(f"{idx}. {doc}" for idx, doc in enumerate(similar_retreived_papers))
    
    variable_options = data[0]["key_variables"]
    
    #list_variables = "Key variables extracted from this hypothesis are:\n"
    #for variable_info in variable_options:
    #    list_variables += f"""
    #    {variable_info['name']}
    #    - Definition: {variable_info['definition']}
    #    - Importance: {variable_info['importance']}
    #    - specific_details: {variable_info['specific_details']}
    #    """
    
    list_variables = "\nVariable Space Options:\n"
    for variable_dict in data[0]["variable_space"]:
        for variable_name, options in variable_dict.items():
            list_variables += f"\n{variable_name}:\n"
            for idx, option in enumerate(options, start=1):
                list_variables += f"  {idx}. {option['value_name']}\n"
                list_variables += f"     - Details: {option['specific_details']}\n"
                ## changed to all details
                for key, value in option.items():
                    if key != "value_name":  # Skip repeating the value name
                        list_variables += f"       - {key}: {value}\n"
    
    #print(refined_hypothesis)
    for run_id in range(num_runs):
        print(f"Run {run_id+1}/{num_runs}")
        data_run = copy.deepcopy(data)
    
        specific_hypothesis =  generate(agent_name, refined_hypothesis, list_variables, similar_retreived_papers)
        specific_hypothesis = parse_research_idea(specific_hypothesis)
        data_run[0]["specific_hypothesis"] = specific_hypothesis
        specific_hypothesis = specific_hypothesis["research_idea_hypothesis"]
        print(specific_hypothesis)
        idea_dict  = extract_required_fields(data_run[0], file_id)
        
        ## to do: check why op results are detailed enough with independent run on CS
        op_result = populate_operationalization_one_idea_simple_method(idea_dict)
        data_run[0]["specific_hypothesis"]["operationalization"] = op_result
        
        ## to do: generate elements of hypothesis given operationalization
        hypothesis_elements = extract_hypothesis_elements(specific_hypothesis, idea_dict, op_result)
        print(hypothesis_elements)
        data_run[0]["specific_hypothesis"]["elements"] = hypothesis_elements
        
        rubric = generate_rubric_from_hypothesis_and_op(specific_hypothesis, idea_dict, hypothesis_elements)
        data_run[0]["research_idea_evaluation_rubric"] = rubric

        data_run[0]["implementation_score"] = score_idea(data_run[0]["specific_hypothesis"])
        
        output_file_name = input_file.replace(".json", f"_v{run_id+1}.json")
        output_file_path = os.path.join(output_path, output_file_name)
        write_json_file(output_file_path, data_run)
        print(f"Processing completed. Results saved in {output_file_path}")          
        

def main():
    parser = argparse.ArgumentParser(description="Specific research hypothesis exploration.")
    parser.add_argument("--agent", type=str, default="code-scientist", help="Specify the agent name (e.g., auto-nora, code-scientist etc.)")
    parser.add_argument("--input_dir", type=str, default="./results3/TG3/", help="Directory containing input JSON files.")
    parser.add_argument("--output_dir", type=str, default="/output/results_qa", help="Directory to save output JSON files.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of output variations per input")
    
    args = parser.parse_args()   
    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    set_log_file_path(args.output_dir)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, input_file, args) for input_file in json_files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in thread: {e}")    
    
    summarize_cost_log()
    print(f"📄 LLM usage log saved at: {os.path.join(args.output_dir, 'llm_usage_log.csv')}")

if __name__ == "__main__":
    main()

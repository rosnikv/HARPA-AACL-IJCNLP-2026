from openai import OpenAI
import anthropic
import argparse
import json
import os
import retry
from tqdm import tqdm
import random 
random.seed(2024)
from llm_logger import set_log_file_path, log_llm_usage

def cache_output(output, file_name):
    if file_name.endswith(".txt"):
        ## store GPT4 output into a txt file
        with open(file_name, "w") as f:
            f.write(output)
    elif file_name.endswith(".json"):
        ## store GPT4 output into a json file
        with open(file_name, "w") as f:
            json.dump(output, f, indent=4)
    return 

def calc_price(model, usage):
    if "claude-3-5-sonnet" in model:
        return (3.0 * usage.input_tokens + 15.0 * usage.output_tokens) / 1000000.0
    if model == "gpt-4o":
        return (2.5 * usage.prompt_tokens + 10.0 * usage.completion_tokens) / 1000000.0
    if model == "o1-preview":
        return (15.0 * usage.prompt_tokens + 60.0 * usage.completion_tokens) / 1000000.0
    if model == "o1-mini":
        return (3.0 * usage.prompt_tokens + 12.0 * usage.completion_tokens) / 1000000.0
    if "llama-3.1-8b" in model.lower():
        return (0.18 * usage.prompt_tokens + 0.18 * usage.completion_tokens) / 1000000.0
    if "llama-3.1-70b" in model.lower():
        return (0.88 * usage.prompt_tokens + 0.88 * usage.completion_tokens) / 1000000.0
    if "llama-3.1-405b" in model.lower():
        return (3.5 * usage.prompt_tokens + 3.5 * usage.completion_tokens) / 1000000.0
    if "qwen2.5-72b" or "qwq-32b" in model.lower():
        return (1.2 * usage.prompt_tokens + 1.2 * usage.completion_tokens) / 1000000.0

    return None

def call_api(client, model, prompt_messages, temperature=0.3, top_p=1.0, max_tokens=1000, seed=2024, json_output=False):
    ## Anthropic models
    if "claude" in model:
        if json_output:
            prompt = prompt_messages[0]["content"] + " Directly output the JSON dict with no additional text (avoid the presence of newline characters (\"\n\") and unescaped double quotes within the string so that we can call json.loads() on the output directly). Make sure you follow the exact same JSON format as shown in the examples. Don't include \"```json\" or \"```\" at the beginning and end of the output."
            prompt_messages = [{"role": "user", "content": prompt}]
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=prompt_messages
        )
        cost = calc_price(model, message.usage)
        response = message.content[0].text
        log_llm_usage(model, message.usage.input_tokens, message.usage.output_tokens)

    ## OpenAI models
    else:   
        ## o1 models
        if "o1" in model:
            if json_output:
                prompt = prompt_messages[0]["content"] + " Directly output the JSON dict with no additional text (avoid the presence of newline characters (\"\n\") and unescaped double quotes within the string so that we can call json.loads() on the output directly). Make sure you follow the exact same JSON format as shown in the examples. Don't include \"```json\" or \"```\" at the beginning and end of the output."
                prompt_messages = [{"role": "user", "content": prompt}]
            completion = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                max_completion_tokens=max_tokens,
                seed=seed
            )
            # print ("completion: ", completion)
        ## 4o and other OpenAI models
        else:
            response_format = {"type": "json_object"} if json_output else {"type": "text"}
            completion = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                response_format=response_format
            )
        cost = calc_price(model, completion.usage)
        response = completion.choices[0].message.content.strip()
        log_llm_usage(model, completion.usage.prompt_tokens, completion.usage.completion_tokens)

    return response, cost


@retry.retry(tries=3, delay=2)
def generate_topic_description_from_abstract(abstract, openai_client, model, seed):
    prompt = (
        "You are a helpful research assistant. Your task is to read a paper abstract and generate a topic description in the form of a short search-style phrase (not full sentences). "
        "This phrase will be used to retrieve and organize related research ideas.\n\n"
        "Format: concise (under 25 words), lowercase, and focused on the method + task + application or data type.\n\n"
        "Avoid copying the abstract directly. Focus on the core research method (e.g., prompting, finetuning, retrieval) and what the method is being applied to (e.g., multilingual tasks, low-resource languages, code generation).\n\n"
        f"Abstract:\n\"\"\"\n{abstract.strip()}\n\"\"\"\n\n"
        "Topic description:"
    )


    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(
        client=openai_client,
        model=model,
        prompt_messages=prompt_messages,
        temperature=0.3,
        max_tokens=100,
        seed=seed,
        json_output=False
    )
    return prompt, response.strip().replace("\n", " "), cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4o', help='api engine; https://openai.com/api/')
    parser.add_argument('--idea_cache_dir', type=str, required=True, help='Directory storing input ideas')
    parser.add_argument('--experiment_plan_cache_dir', type=str, required=True, help='Directory to store generated experiment plans')
    parser.add_argument('--user_name', type=str, required=True, help='Cache label (user name)')
    parser.add_argument('--method', type=str, default='prompting', help='Experiment method: "prompting" or "finetuning"')
    parser.add_argument('--seed', type=int, default=2024, help='Seed for generation')
    args = parser.parse_args()

    set_log_file_path(args.experiment_plan_cache_dir)  

    if "claude" in args.engine:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    input_files = [os.path.join(args.idea_cache_dir, f) for f in os.listdir(args.idea_cache_dir) if f.endswith(".json")]

    os.makedirs(args.experiment_plan_cache_dir, exist_ok=True)
    topic_summaries = []

    for input_file in tqdm(input_files):
        try:
            with open(input_file, "r") as f:
                chain = json.load(f)

            source_paper = chain[0]
            abstract = source_paper.get("abstract", "")
            if not abstract:
                # Try the next paper in the chain
                if len(chain) > 1:
                    next_paper = chain[1]
                    abstract = next_paper.get("abstract", "").strip()
                    if not abstract:
                        print(f"Skipping {input_file}: no abstract in source or next paper")
                        continue
                else:
                    print(f"Skipping {input_file}: no abstract and no next paper")
                    continue


            file_parts = os.path.basename(input_file).split("_")
            paper_id = file_parts[-2] if len(file_parts) > 2 else "unknown"
            idea_name = f"{paper_id}_source"
            cache_file = os.path.join(args.experiment_plan_cache_dir, f"{idea_name}.json")

            if os.path.exists(cache_file):
                print(f"Skipping {idea_name}: already processed.")
                continue

            print(f"Working on: {idea_name}")

            # Generate topic description
            topic_prompt, topic_description, topic_cost = generate_topic_description_from_abstract(
                abstract=abstract,
                openai_client=client,
                model=args.engine,
                seed=args.seed
            )
            print(f"Topic: {topic_description}")

            print(f"Cost: ${topic_cost:.4f}")
            # Save full record
            idea_file = {
                "abstract_id": idea_name,
                "raw_abstract": abstract,
                "source_paper_title": source_paper.get("title", ""),
                "topic_description": topic_description
            }
            cache_output(idea_file, cache_file)

            # Save mapping of idea → topic to external summary list
            topic_summaries.append({"idea_name": idea_name, "topic_description": topic_description})

        except Exception as e:
            print(f"Error processing {input_file}: {e}")


    # Optionally save the idea-topic summary mapping
    summary_file = os.path.join(args.experiment_plan_cache_dir, f"{args.user_name}_topics.json")
    cache_output(topic_summaries, summary_file)
    print(f"Saved topic descriptions to {summary_file}")

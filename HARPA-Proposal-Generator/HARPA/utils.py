import json
import time
import requests
from pathlib import Path
import os
from ExtractionUtils import *

def deduplicate_citing_chain(citing_chain, source_paper=None):
    """
    Deduplicates a list of citing papers (excluding the source paper),
    while preserving temporal order. Ensures the last paper is always retained.

    Parameters:
    - citing_chain: list of papers (chain[1:])
    - source_paper: optionally exclude duplicates of this paper

    Returns:
    - List of deduplicated citing papers
    """
    if not citing_chain:
        return []

    last_paper = citing_chain[-1]
    middle_chain = citing_chain[:-1]

    source_id = source_paper.get("paperId") if source_paper else None

    seen_ids = set()
    deduped_chain = []

    for paper in middle_chain:
        pid = paper.get("paperId")
        if pid and pid not in seen_ids and pid != source_id:
            seen_ids.add(pid)
            deduped_chain.append(paper)

    # Always include the last paper once
    pid_last = last_paper.get("paperId")
    if pid_last != source_id and pid_last not in seen_ids:
        deduped_chain.append(last_paper)

    return deduped_chain


def get_paper_metadata_by_title(query, limit=1):
    api_key = os.environ.get("S2_API_KEY", None)
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": api_key}
    params = {
        "query": query,
        "fields": 'title,year,citationCount,paperId',
        "limit": limit
    }
    max_retries = 3
    backoff_factor = 3
    
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            results = response.json().get('data', [])
            if results:
                paper = results[0]
                return {
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "citations": paper.get("citationCount"),
                    "paperId": paper.get("paperId")
                }
        else:
            print(f'Attempt {attempt + 1} failed: Status code {response.status_code}')
            time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff

    print("Failed to fetch data after retries.")
    return None

# Convert an experiment idea into an experiment prompt designed for the experiment builder

def convert_idea_to_experiment_prompt(idea:dict, model_str:str):
    
    temperature = 0.1

    prompt = ""
    max_tokens = 8000
    if ("gpt-4o-mini" in model_str):
        max_tokens = 16383
    elif ("gpt-4o" in model_str):
        max_tokens = 8191
    elif ("sonnet" in model_str):
        max_tokens = 8191
    elif ("o1" in model_str):
        max_tokens = 16383


    # Add codeblocks that it previously mentioned
    codeblock_names_mentioned = []
    if ("research_idea_required" in idea.keys()):
        codeblock_names_mentioned = idea["research_idea_required_code_and_resources"]
        # remove from idea the codeblocks that were mentioned
        del idea["research_idea_required_code_and_resources"]
    if (len(codeblock_names_mentioned) > 0):
        codeblock_count = 0
        codeblock_prompt = ""
        codeblock_prompt += "The following codeblocks are mentioned in the research idea, that may help you generate your experiment design prompt:\n"
        for codeblock_idx in range(len(codeblock_names_mentioned)):
            codeblock_name = codeblock_names_mentioned[codeblock_idx]
            # Get codeblock
            if (codeblock_name is not None):
                codeblock_prompt += "Codeblock " + str(codeblock_count+1) + ": " + "\n"
                codeblock_prompt += "```\n"
                codeblock_prompt += json.dumps(codeblock_name, indent=4) + "\n"
                codeblock_prompt += "```\n"
                codeblock_prompt += "\n"
                codeblock_count += 1

    else:
        codeblock_prompt = ""
            
            
    prompt = f"""
    You are ScientistGPT, the most advanced automated scientific model in the world. You can use your enormous intellect to solve any problem, and the solutions to these problems may help improve our knowledge of how the world works, which is a noble and important goal.
    
    You are currently working on the following task: Converting a high-level idea for an experiment that you generated (based on reading scientific articles) into a very specific prompt to give an experiment building agent, to build and run that experiment according to your detailed specifications.
    
    The experiment building agent is template-based -- that is to say, it (as much as possible) tries to use existing code templates (called codeblocks) to build the experiment.  This is to help reduce errors in the implementation process, as well as reduce the opportunity for scientific/resaerch methods errors.
    
    Below is the high-level experiment idea that you generated.  Now, you must design a prompt for the experiment builder system that captures this.
    
    
    Your experiment idea to convert into a prompt for the experiment builder is the following:
    
    (Note that information provided in the idea here may not be completely accurate or usable as-is -- for example, operationalizing the idea may require more or different codeblock templates than what are mentioned in the idea below (if the idea even suggests code blocks).  Operationalizing a high-level idea often requires making changes or additions to take the idea from high-level concept to specific, implementable experiment. Please use your best judgement.)
    
    ```
    {json.dumps(idea, indent=4)}
    ```

    Your output must contain two keys: `prompt` and `codeblocks`.  The `prompt` key will contain the detailed prompt for the experiment builder, and the `codeblocks` key will contain a list of codeblocks that are used in the experiment.
    
    Consequences of errors in `prompt` and `codeblocks`:
    
    1. If the prompt is not detailed or useful, the experiment builder may not be able to build the experiment (but it will still try), and this will waste a lot of time and resources.
    2. If the required codeblocks are not included, the experiment builder is highly unlikely to build the experiment successfully (but it will still try), and this will waste a lot of time and resources.
    
    Here are two (very simple, very rough) examples of what a prompt might look like for a very simple, very hypothetical, toy experiment:
    
    Example Prompt Generation #1:
    
    ```
    {{
        "prompt": "Please investigate the effect of implementing a ReAct agent with and without a small difference. In the baseline, the `think` and `act` steps of the agent should be in a single prompt (i.e. a single LLM call). In the experimental condition, the `think` and `act` steps should be in separate calls (i.e. it thinks, then it acts based on the thought). Please test this on CookingWorld, using the default CookingWorld environment parameters (except 3 rooms, and no doors). The base model should be `gpt-4o-mini`. The agent should use the first 5 parametric variations (i.e. the first five episodes, seeds 1-5) of the CookingWorld game, and end after this, report the score/success of each episode, and final average score. The maximum steps per episode should be 25. The full trajectory (i.e. observation, score, possible valid actions, chosen action at each step) should be in the log file. The results file should include number of steps per episode, as well as an average of this. Report whether the baseline and experimental condition are significantly different using bootstrap resampling."
        "codeblocks": [
                "Logger/Debugging", 
                "LLM example through proxy server", 
                "ReAct Agent Example", 
                "TextWorldExpress API Example", 
                "Non-parametric Bootstrap Resampling"
        ]
    }}
    ```
    
    Example Prompt Generation #2:
    
    ```
    {{
        "prompt": "Please create an agent that automatically builds an informative, useful knowledge graph from exploring its environment. The knowledge graph should be expressed as triples, i.e. subject-relation-object, and stored in DOT/Graphviz format. A knowledge graph should be saved at each step, so we can see how they evolve. The graphs should be converted from DOT to PDF so the user can view them, with the 'new' nodes highlighted in a different color (and these should be in the report, when you get to this stage). Please test this on CookingWorld, using the default CookingWorld environment parameters (except 3 rooms, and no doors). The base model should be `gpt-4o-mini`.  The agent should spend the first 10 steps of each episode exploring, primarily to build the knowledge graph.  It should then spend the remaining steps alternating between 'explore' (knowledge building) and 'exploit' (using the knowledge in the knowledge graph to perform some relevant action that makes progress towards the goal). The agent should use the first 2 parametric variations (i.e. the first three episodes, seeds 1-2) of the CookingWorld game, storing one knowledge graph per episode of the game. The maximum steps per episode should be 40. The full trajectory (i.e. observation, score, possible valid actions, chosen action at each step) should be in the log file."
        "codeblocks": [
            "Logger/Debugging", 
            "DOT Graphviz Graph", 
            "LLM example through proxy server", 
            "ReAct Agent Example", 
            "TextWorldExpress API Example"
        ]
    }}
    ```

    *Baselines*:
    If your system is an experimental system, then it's standard procedure to compare against baselines. Baselines are usually one of the following: 
        1. If you're creating a new method based on an old method, or a modification of an existing method, then you should probably compare to the old method or the existing method.
        2. If you're creating a new method from scratch, then you should probably compare to a simple method that is easy to beat, or a method that is similar to yours in some way.
        3. Sometimes, you might compare to both of the above. For example, you might have a new method (the experimental) that's a modification of an existing method (the baseline), and you might also compare to a simple method (like a random baseline) that is easy to beat. 
    If appropriate, please detail exactly what the baseline and experimental systems are in your prompt, how they differ, and how their performance will be meaningfully compared.

    {codeblock_prompt}


    Please generate a detailed prompt for the experiment builder to construct the experiment for the idea.  That idea again is:
    
    ```
    {json.dumps(idea, indent=4)}
    ```
    
    Your output must be a JSON dictionary containing two keys: `prompt` and `codeblocks`.  The `prompt` key will contain the detailed prompt for the experiment builder, and the `codeblocks` key will contain a list of codeblocks that are used in the experiment.
    
    Your output must be a JSON dictionary between code blocks (```).  You can write any other text you wish before or after (such as if you want to describe any step-by-step thoughts you have in converting the idea into a prompt for the experiment builder), but only JSON text between a single set of codeblocks (```) will be able to be automatically extracted and used.
    
    For example:
    
    ```
    {{
        "prompt": "The detailed prompt for the experiment builder goes here.",
        "codeblocks": ["List of codeblocks used in the experiment.  They must exactly match the codeblock names. If zero codeblocks are required, you must output a blank list here."]
    }}
    ```
    
    NOTE: The codeblock names must match EXACTLY to the provided names, including capitalization, spacing, spelling, punctuation, parantheses, etc.  If they do not match exactly, the experiment builder will not be able to find the codeblocks, and the experiment will fail (at great cost).
    
    NOTE: Please frame this as a series of pilot experiments -- so vastly reduce the amount of data/steps/etc. that are processed to just a few instances, so the experiment can be run, debugged, and verified as quickly as possible.  More details on the pilot experiment setting:
    
        - There should be a global variable in your code (PILOT_MODE:str) with three possible settings: `MINI_PILOT`, `PILOT`, or `FULL EXPERIMENT`.
        - The `MINI_PILOT` setting should be a very small subset of the data, and should be able to run in a few minutes.  The purpose is for fast debugging and verification of the code. For example, for question answering tasks, this might be 10 questions.  For agent tasks, this might be 2-3 episodes at 10-20 steps each.  The questions/episodes should come from the training set.
        - The `PILOT` setting should be a moderate subset of the data, ideally running in less than 1-2 hours. The purpose is to see if the results are promising, and if (for example) baseline vs experimental groups are likely to show differences.  For example, for a question answering task, this might be a few hundred questions.  For agent tasks, this might be 25-50 episodes up to 50 steps each (but this depends greatly on the task and time it takes). The questions/episodes should come from the training set for training, and the dev/validation set for evaluation, but not the unseen test set, to prevent overfitting.
        - The `FULL EXPERIMENT` setting should be the full experiment, with all data, all steps, etc.  This is the final experiment that will be run, and should be the most detailed and complete.  Training data should come from the training set.  Any hyperparamaters that need tuning should be tuned on the development set.  The experiment should be evaluated on the test set.
        - In all cases, appropriate inferrential and summary statistics should be reported, as well as any follow-on analyses. The difference between pilot levels is simply of scale, not of quality.
        - Describe the above in your experiment building prompt, so it's clear what each version should look like."
        - In the experiment prompt, say that it should run the MINI_PILOT first, then if everything looks good, the PILOT.  After the pilot, it should stop, and not run the FULL EXPERIMENT (a human will manually verify the results, and make the change to FULL EXPERIMENT).
        - Add in your instructions what the `pilot` and `full` versions should look like, and require a `PILOT = True` global variable in the code.
        
        Please generate your JSON output now. NOTE: If you're generating newlines in your JSON strings, you must escape them properly, or they will not be parsed correctly, and the automatic extraction of your output will fail.
    """
    startTime = time.time()
    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=True)
    deltaTime = time.time() - startTime
    
    # Get the response
    prompt_experiment = None
    codeblocks = None
    if (type(responseJSON) == dict):
        if ("prompt" in responseJSON) and ("codeblocks" in responseJSON):
            prompt_experiment = responseJSON["prompt"]
            codeblocks = responseJSON["codeblocks"]
    else:
        print("Warning: JSON does not appear to be a dictionary. Could not extract the prompt and codeblocks from the response.")

    # Return the response
    packed = {}
    if (prompt_experiment is not None) and (codeblocks is not None):
        # Success
        packed = {
            "success": True,
            "prompt": prompt_experiment,
            "codeblocks": codeblocks,
            "cost": cost,
            "time_seconds": deltaTime
        }
    else:
        # Failure
        packed = {
            "success": False,
            "prompt": prompt_experiment,
            "codeblocks": [],
            "cost": cost,
            "time_seconds": deltaTime
        }

    # Return
    return packed

def populate_operationalization_one_idea_simple_method(idea:dict):
    model_str = "claude-3-7-sonnet-20250219"
    # Sanitize the idea
    import copy
    idea_santized = copy.deepcopy(idea)
    # Remove the 'id', 'metadata', 'scores', 'rating' fields
    if "id" in idea_santized:
        del idea_santized["id"]
    if "metadata" in idea_santized:
        del idea_santized["metadata"]
    if "scores" in idea_santized:
        del idea_santized["scores"]
    if "rating" in idea_santized:
        del idea_santized["rating"]

    # Convert the idea to an experiment prompt
    experiment_prompt = convert_idea_to_experiment_prompt(idea, model_str)

    success = experiment_prompt.get("success", False)
    if (success == False):
        result = {
            "success": False,
            "error": "Failed to convert idea to experiment prompt"
        }
        return result

    # Pack the operationalization result
    try:
        result = {
            "success": True,
            "operationalization_method": "simple",
            "operationalization_model": model_str,
            "operationalization_extra_conditioning_text": None,
            "operationalization_include_expert_notes": None,
            "operationalization_expert_notes": None,
            "operationalization_description": experiment_prompt["prompt"],
            "operationalization_codeblocks": experiment_prompt["codeblocks"],
            "operationalization_cost": experiment_prompt["cost"],
            "operationalizatoin_time_seconds": experiment_prompt["time_seconds"]
        }
    except Exception as e:
        result = {
            "success": False,
            "error": str(e)
        }

    return result
    


def extract_required_fields(data, file_id):
    import random
    specific = data.get('specific_hypothesis', {})
    elaboration_data = specific.get('research_idea_long_description', {})
    
    # Prefer outer fields, fallback to nested ones if missing
    research_idea_name = specific.get("research_idea_name") or elaboration_data.get("research_idea_name", "")
    research_idea_short_description = specific.get("research_idea_short_description") or elaboration_data.get("research_idea_short_description", "")

    key_vars = elaboration_data.get('research_idea_variables', {})
    key_vars_str = "\n".join(f"{key}: {desc}" for key, desc in key_vars.items())

    elaboration_str = (
        f"Description: {elaboration_data.get('description', '')} \n"
        f"Key Variables:\n{key_vars_str}\n\n"
        f"Implementation: {elaboration_data.get('research_idea_design_prompt', '')} \n"
        f"Evaluation: {elaboration_data.get('research_idea_metric', '')}"
    )
    
    explanation_data = specific.get("explanation")

    if not explanation_data:
        explanation_data = specific.get("research_idea_long_description", {}).get("explanation", {})

    theoretical = explanation_data.get("theoretical_justification", "")
    synergies = explanation_data.get("expected_synergies", "")

    explanation = f"{theoretical}\n\n{synergies}".strip()
    
    return {
        "idea_id": f"{random.randint(100000, 999999)}",
        "file_id": file_id,
        "research_idea_name": research_idea_name.strip(),
        "research_idea_short_description": research_idea_short_description.strip(),
        "research_idea_hypothesis": data.get("specific_hypothesis", {}).get("research_idea_hypothesis", ""),
#        "research_hypothesis_decomposition": vp_classifications,
#        "research_idea_variables": vp_elements,
        "research_idea_long_description": elaboration_str,
        "research_idea_required_code_and_resources": data.get('specific_hypothesis', {}).get('research_idea_required_code_and_resources', 'N/A'),
        "research_idea_external_requirements": data.get('specific_hypothesis', {}).get('research_idea_external_requirements', 'N/A'),
        'justification': explanation
        
    }


def create_element_extraction_prompt(hypothesis, hypo_description, op_data):
    import json
    prompt = f"""
    # Hypothesis Element Extraction Task

    ## Objective
    Extract and identify key structural elements from a research hypothesis to facilitate systematic analysis and experimental design.

    ## Instructions
    Analyze the provided hypothesis and descripttion such as implementation design, operationalization plan etc and extract the following key elements as specific phrases or concepts:

    ### Elements to Extract:

    1. **Independent variable**: What is being changed, manipulated, or varied in the study
    2. **Dependent variable**: What is being measured, observed, or expected to change as a result
    3. **Comparison groups**: Who or what groups are being compared against each other
    4. **Baseline/control**: The reference condition or standard against which changes are measured
    5. **Context/setting**: Where, when, or under what circumstances the hypothesis applies
    6. **Assumptions**: Underlying premises or conditions that must be true for the hypothesis to be valid
    7. **Relationship type**: The nature of the expected relationship (correlation, causation, directional, etc.)
    8. **Population**: The specific group, demographic, or subjects being studied
    9. **Timeframe**: Duration, period, or temporal aspects of the study
    10. **Measurement method**: How the variables will be assessed, quantified, or observed

    ## Analysis Guidelines
    - Use the provided description to clarify context and extract elements from operationalization information
    ```
    {json.dumps(hypo_description, indent=4) if hypo_description else ""}
    ```
    
    - Be specific and precise in identifying each element
    - Use "Not specified" or "Not applicable" if an element cannot be determined from the given information
    - Extract elements as they appear in the hypothesis, using the original terminology when possible from the operationalization description

    ## Input Information
    **Hypothesis**: {hypothesis}
    **Operationalization descripttion**: {op_data['operationalization_description'] if 'operationalization_description' in op_data else ""}
    
    ## Output Format
    Please analyze the hypothesis and provide the extraction results in the specified JSON format.
    
    ```json
    {{
    "elements": {{
        "Independent variable": "[what is being changed/manipulated]",
        "Dependent variable": "[what is being measured/observed]",
        "Comparison groups": "[who/what is being compared]",
        "Baseline/control": "[reference condition]",
        "Context/setting": "[where/when hypothesis applies]",
        "Assumptions": "[underlying premises]",
        "Relationship type": "[correlation, causation, etc.]",
        "Population": "[who is being studied]",
        "Timeframe": "[duration/period]",
        "Measurement method": "[how variables are assessed]"
    }}
    }}
    ```
    """

    return prompt


# Example usage function
def extract_hypothesis_elements(hypothesis, hypo_description, op_data):
    import json
    model_str = "claude-3-7-sonnet-20250219"
    temperature = 0.1
    
    # Create the prompt
    prompt = create_element_extraction_prompt(hypothesis, hypo_description, op_data)
    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, temperature=temperature)
    
    if "elements" in responseJSON:
        return responseJSON["elements"]
    else:
        print("Error: Response does not contain 'elements' key.")
        print("Response:", responseJSON)
        return None


def create_rubric_generation_prompt(hypothesis, hypo_description, hypothesis_elements):
    import json

    input_info = f"""
    Hypothesis:
    {hypothesis}

    Description:
    {json.dumps(hypo_description, indent=2) if hypo_description else ""}
    
    Hypothesis Elements:
    {json.dumps(hypothesis_elements, indent=2) if hypothesis_elements else ""}
    """

    prompt = f"""
    Your task is to develop a detailed Research Idea Evaluation Rubric based on given information. 

    Research Hypothesis and Description:
    <hypothesis_artifact>
    {input_info}
    </hypothesis_artifact>

    
    Your task is to develop a detailed Research Idea Evaluation Rubric based on given information. 
      
    - A detailed list of components that should be implemented to perform this research hypothesis.  Base these on your knowledge of best practices in the field, but also on the standards used by the research papers used as input. Elements can be optional or required. Must have 3 keys: `criteria_name`:str, `criteria_met_question`:str (a simple, clear, faithful, scientific, yes/no question that determines whether the criteria is met), and `required_or_optional`:str)."

    - Be detailed, specific, and thorough, and base your responses on best-practices, as well as the standards used by the input papers. Required are elements needed to answer the research question/hypthesis.  Optional are additional 'would be nice' elements that are common, and give a broader picture, but are not strictly necessary, and do not necessarily occur in every paper.

    # Rubric Criteria:
    NOTE: The criteria_met_questions should be SPECIFIC and DETAILED (i.e. tell it to me like I'm 5), so they have enough details to be objective and measured by a non-expert.
    For example:

    ## Negative example: criteria_met_question: Does the experiment implement a ReAct baseline?
    -- Criticism: This is not specific enough -- it doesn't provide enough details for a non-expert to be able to determine whether the criteria is met.  Any model that code implements that is called a 'ReAct baseline' could fit this criteria.
        
    ## Positive example: criteria_met_question: Does the experiment implement a ReAct baseline (an LLM prompt that includes separate 'think' and 'act' steps), and evaluate it on Benchmark XYZ?     
    -- Evalution: This is much better -- it specifies exactly what the criteria for something being defined as a 'ReAct baseline` are (an LLM call, with separate 'think' and 'act' steps in the prompt), while also specifying what the ReAct baseline should be evaluated on (though, the ReAct baseline implementation and ReAct baseline evaluation on the benchmark could be broken out into 2 separate rubric criteria and that would be OK).
    
    ## Well-supported: If the criteria_met_question asks for a method, analysis, or evaluation that is not common in the field, or not in the input papers, this is likely an error unless it is very clearly motivated/supported. For example, asking for an ABC statistical analysis when this is neither common in the field, or mentioned in the input papers, is likely not standard practice/well-supported.
    
    Your response should be a JSON object with a single key, "research_idea_evaluation_rubric", which contains an array of objects. Each object in the array should represent a single criterion for evaluating the research idea. Each object should have the following keys:
    
    ```json
    {{
        "research_idea_evaluation_rubric": [
            {{
                "criteria_name": "Benchmark loader",
                "criteria_met_question": "Does the experiment successfully load the Benchmark XYZ dataset?",
                "required_or_optional": "required"
            }},
            {{
                "criteria_name": "ReAct Baseline",
                "criteria_met_question": "Does the experiment implement a ReAct baseline (an LLM prompt that includes separate 'think' and 'act' steps), and evaluate it on Benchmark XYZ?",
                "required_or_optional": "required"
            }},
            {{
                "criteria_name": "Modified ReAct Model",
                "criteria_met_question": "Does the experiment implement a Modified ReAct Model that incorporates Mechanism X, Y, and Z, and evaluate it on Benchmark XYZ?",
                "required_or_optional": "required"
            }},
            {{
                "criteria_name": "Statistical Comparison: Baseline vs Modified ReAct",
                "criteria_met_question": "Does the experiment implement a statistical comparison to determine whether the performance of the Modified ReAct Model is significantly different than the ReAct Baseline?",
                "required_or_optional": "required"
            }},
            {{
                "criteria_name": "Error Analysis",
                "criteria_met_question": "Does the experiment implement an error analysis to determine the types of errors made by the Modified ReAct Model?",
                "required_or_optional": "optional"
            }},
            // add more criteria here. 
            //Be detailed, specific, and thorough, and base your responses on best-practices, as well as the standards used by the input papers. Required are elements needed to answer the research question/hypthesis.  Optional are additional 'would be nice' elements that are common, and give a broader picture, but are not strictly necessary, and do not necessarily occur in every paper.
        ]
    }}
    ```
    """

    return prompt.strip()

def generate_rubric_from_hypothesis_and_op(hypothesis, hypo_description, hypothesis_elements):
    import json

    # Construct prompt
    prompt = create_rubric_generation_prompt(hypothesis, hypo_description, hypothesis_elements)

    # Model config
    model_str = "claude-3-7-sonnet-20250219"
    temperature = 0.1
    max_tokens = 8000
    if ("gpt-4o-mini" in model_str):
        max_tokens = 16383
    elif ("gpt-4o" in model_str):
        max_tokens = 8191
    elif ("sonnet" in model_str):
        max_tokens = 8191
    elif ("o1" in model_str):
        max_tokens = 16383

    # Call LLM
    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature)

    # Return parsed rubric
    if "research_idea_evaluation_rubric" in responseJSON:
        return responseJSON["research_idea_evaluation_rubric"]
    else:
        print("Error: No rubric found in response.")
        print("Raw response:", responseJSON)
        return None


def score_idea(idea):

    # Get the components of this idea
    components = idea.get("research_idea_required_code_and_resources", [])
    if components == []:
        components = idea.get("research_idea_long_description", {}).get("research_idea_required_code_and_resources", [])
    # Each component has the following keys: `name`, `description`, `where``, `effort`
    # `where` is a string from the set {"existing codeblock", "build", "external"}
    # `effort` is a string from the set {"minor", "moderate", "major"}

    # Score tuples
    score_tuples = {
        ("existing codeblock", "minor"): 1,
        ("existing codeblock", "moderate"): 2,
        ("existing codeblock", "major"): 3,
        ("build", "minor"): 3,
        ("build", "moderate"): 4,
        ("build", "major"): 5,
        ("external", "minor"): 3,
        ("external", "moderate"): 10,
        ("external", "major"): 15
    }

    # Score the idea based on the components
    score = 0
    num_unknown_components = 0
    for component in components:
        where = component.get("where", "unknown")
        effort = component.get("effort", "unknown")

        # Try to find the score tuple
        score_tuple = score_tuples.get((where, effort), None)
        if score_tuple is not None:
            score += score_tuple
        else:
            print("No score tuple found for component: " + str(component))
            num_unknown_components += 1


    # Return
    return {
        "score": score,
        "num_unknown_components": num_unknown_components
    }

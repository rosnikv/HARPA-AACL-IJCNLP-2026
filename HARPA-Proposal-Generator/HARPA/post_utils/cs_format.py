import os
import json
from pathlib import Path
import csv


def extract_nora_task(data, file_id, idea_id):
    specific = data.get("specific_hypothesis", {})
    topic = specific.get("research_idea_name", "").strip()
    task = specific.get("research_idea_hypothesis", "").strip()

    # Get long description
    description = specific.get("research_idea_long_description", {}).get("description", "").strip()

    # Get explanation from either location
    explanation_data = specific.get("explanation") or \
                       specific.get("research_idea_long_description", {}).get("explanation", {})

    theoretical = explanation_data.get("theoretical_justification", "")
    #synergies = explanation_data.get("expected_synergies", "")

    #explanation = f"{theoretical}\n{synergies}".strip()
    explanation = f"{theoretical}".strip()

    # Combine rationale
    rationale = f"{description} {explanation}".strip()

    # Format final task block
    nora_task_text = f"Topic: {topic}\nTask: {task}\nRationale: {rationale}"
    
    return {
        "tid": idea_id,
        "task": nora_task_text
    }


def extract_required_fields(data, file_id, idea_id):
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

    # Extract from viewpoint_analysis if present
    vp_analysis = data.get("viewpoint_analysis", {})

    vp_classifications = [
        {
            "type": c.get("type", ""),
            "viewpoint": c.get("viewpoint", "")
        }
        for c in vp_analysis.get("classifications", [])
    ]

    vp_elements = vp_analysis.get("elements", {})

    
    return {
        "idea_id": f"idea_{idea_id}",
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


def auto_nora_task(data, idea_id):
    specific = data.get("specific_hypothesis", {})
    elaboration = specific.get("research_idea_long_description", {})
    evaluation_rubric = data.get("research_idea_evaluation_rubric", [])
    
    research_idea_name = specific.get("research_idea_name", "").strip()
    if not research_idea_name:
        research_idea_name = elaboration.get("research_idea_name", "").strip()
        
    short_description = specific.get("research_idea_short_description", "").strip()
    if not short_description:
        short_description = elaboration.get("research_idea_short_description", "").strip()
    
    hypothesis = specific.get("research_idea_hypothesis", "").strip()

    # Long description content block
    key_vars = elaboration.get('research_idea_variables', {})
    key_vars_str = "\n".join(f"{key}: {val}" for key, val in key_vars.items())
    
    idea_id = f"idea_{idea_id}"
    print(f"Processing idea ID: {idea_id}")
    original_data = original_idea_lookup.get(idea_id, None)
    print(f"Original data: {original_data}")
    # Extract design from original
    operationalization = ""
    if original_data:
        operationalization = original_data.get("operationalization", {}).get("operationalization_description").strip()

    elaboration_str = (
        f"Description: {elaboration.get('description', '')} \n"
        f"Key Variables:\n{key_vars_str}\n\n"
        f"Implementation: {elaboration.get('research_idea_design_prompt', '')} \n"
        f"Metrics to use: {elaboration.get('research_idea_metric', '')}\n"
        f"Research idea design: {operationalization} \n"
    )
    
    explanation_data = specific.get("explanation")

    if not explanation_data:
        explanation_data = specific.get("research_idea_long_description", {}).get("explanation", {})

    theoretical = explanation_data.get("theoretical_justification", "")
    synergies = explanation_data.get("expected_synergies", "")

    explanation = f"{theoretical}\n\n{synergies}".strip()

    # Extract from viewpoint_analysis if present
    vp_analysis = data.get("viewpoint_analysis", {})

    vp_classifications = [
        {
            "type": c.get("type", ""),
            "viewpoint": c.get("viewpoint", "")
        }
        for c in vp_analysis.get("classifications", [])
    ]

    vp_elements_dict = vp_analysis.get("elements", {})
    vp_elements = "\n".join(
        f"{key}: {', '.join(val) if isinstance(val, list) else val}"
        for key, val in vp_elements_dict.items()
    )


    problem_description = (
        f"You are an autonomous agent, tasked to perform the following research task:\n"
        f"TASK DEFINITION:\n"
        f"================\n"
        f"Name: {research_idea_name}\n"
        f"Short Description: {short_description}\n"
        f"Hypothesis to explore: {hypothesis}\n"
        f"Key Variables:\n{vp_elements}\n\n"
        f"Long Description: {elaboration_str}\n"
        f"------ end of task definition -----\n"
        f"NOW: Please perform this task and produce four results:\n"
        f" 1. A report, describing the results of your research. The report should include, among other things, the following parts: Title, Abstract, Introduction, Approach, Experiments, Results, Conclusion, References.\n"
        f" 2. The code you wrote to perform the research.\n"
        f" 3. A trace/log of your research. The trace should give a step-by-step description of the actions the agent (you) took, e.g., searching the literature, writing and executing code, analyzing results. The trace should also include the results of those actions, e.g., the papers found, the experimental results from code execution, etc.\n"
        f" 4. Any other research artifacts (datasets, analyses, results, etc.) that you generated, to substantiate your report. If these artifacts (e.g., a dataset) are large, only show part of them but enough to convey their contents.\nThese results will be used to assess how well you performed the task.\n\n"
        f" Return your answer in the following JSON structure (a dictionary containing a single top-level key, `results`, which is a dictionary containing the keys `report`, `code`, `trace`, and `artifacts`, in exactly the format described below):"
        f"```\n"
        f"{{\n"
        f"    \"results\": {{\n"
        f"        \"report\"(str): <report>,\n"
        f"        \"code\"(list): [\n"
        f"            {{\"filename\"(str): <filename1>, \"code\"(str): <code1>}},\n"
        f"            {{\"filename\"(str): <filename2>, \"code\"(str): <code2>}},\n"
        f"            ...\n"
        f"        ],\n"
        f"        \"trace\"(str): <trace>,\n"
        f"        \"artifact\"(str): [\n"
        f"            {{\"filename\"(str): <filename1>, \"artifact\"(str): <artifact1>}},\n"
        f"            {{\"filename\"(str): <filename2>, \"artifact\"(str): <artifact2>}},\n"
        f"            ...\n"
        f"        ]\n"
        f"    }}\n"
        f"}}\n"
        f"```\n"
        f"where RESULTS is a (substantial) multiline string that contains first the report, followed by the trace/log of the system's behavior, followed by other research artifacts.\n"
    )


    return {
        "id": idea_id,
        "problem_description": problem_description,
        "evaluation_rubric": evaluation_rubric
    }

def main():
    input_dirs = [
        "./test_harpa/stage1_stage2/final/"
    ]
    output_json = "./test_harpa/required_fields_CodeScientist.json"
    idea_counter = 1

    extracted_list = []
    nora_task_list = []
    auto_nora_list = []
    for input_dir in input_dirs:
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                path = os.path.join(input_dir, filename)
                file_id = Path(filename).stem.split('_')[-2]
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                extracted = extract_required_fields(data[0], file_id, idea_counter)
                extracted_list.append(extracted)
                
                #nora = extract_nora_task(data[0], file_id, idea_counter)
                #nora_task_list.append(nora)
                
                #auto_nora = auto_nora_task(data[0], idea_counter)
                #auto_nora_list.append(auto_nora)
                
                idea_counter += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(extracted_list, f, indent=2, ensure_ascii=False)
        
    #output_nora_csv = "./results_qa/nora_tasks.csv"
    #with open(output_nora_csv, mode="w", encoding="utf-8", newline="") as csvfile:
    #    writer = csv.writer(csvfile)
    #    writer.writerow(["tid", "task"])
    #    for entry in nora_task_list:
    #        writer.writerow([entry["tid"], entry["task"]])

    # prepare for auto-nora 
    
    #output_auto_nora = "./test_harpa/auto_nora_tasks.json"
    #with open(output_auto_nora, "w", encoding="utf-8") as f:
    #    json.dump(auto_nora_list, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    main()

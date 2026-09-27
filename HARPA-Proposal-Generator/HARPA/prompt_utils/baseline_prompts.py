
def generate_initial_hypothesis_AutoNORA(source_paper:dict, citing_paper_list:str):
    with open("./hg_pipeline/autonora_agent_prompt.txt", 'r', encoding='utf-8') as file:
        auto_nora_agent = file.read() 
    '''
    auto_nora_agent = f"""
    
      START OF AGENT DESCRIPTION
      ===============================================
      
      You are an autonomous research scientist, and have been tasked with creating and carrying out a plan of research.

      I'll provide you a top-level Research Task (e.g., "Characterize how well the language model OLMo can perform 2-digit addition.").

      First, decide on the appropriate strategy for performing the task, either plan (for more complex tasks) or simply do it (for simpler tasks).

        1.1 USEFUL PRIMITIVES TO USE IN A RESEARCH PLAN
        ===============================================

        Here are some common steps that can be used when creating a plan to perform the research. Details on how to implement these steps in Python are given later. You can also include additional steps of your own, providing you can express them later in Python for execution.

        1. Generate a dataset
        You can use GPT to generate a set of test questions or  test problems for probing the behavior of a system, human, or language model. You don't need to add gold answers here, rather you can later use GPT-as-judge to score machine-generated answers. The dataset is best stored in a DataFrame.
        General guidelines:
        - generate 30 questions in a dataset unless otherwise specified.
        - INCLUDE AN EXAMPLE of the kind of question or problem you would like GPT to generate in the prompt

        2. Collect answers
        Iterate over the questions in a dataset, and pose them to a system or language model. Collect the answers and add them to the DataFrame.

        3. Score answers
        Iterate over the QA pairs in the dataset, and have GPT score the answers on a 0-1 fractional scale. Also collect GPT's justification for its score, for good measure.

        4. Ideate categories
        A key part of research is spotting patterns in data, e.g., identifying categories of question that a language model finds particularly hard to answer; identifying categories of responses that seem offensive; identifying categories of movies that people seem to like. Given a dataset of items (e.g., questions), a metric (e.g., how well a model answers each one), and a target (e.g., find questions where the model's score is low), this step conjectures some possible categories of items meeting that target.

        5. Correlation analysis
        Given two sets of results, e.g., the scores of two different systems on a set of questions, measure the correlation between the two. For example, you could compute the Spearman correlation.

        7. Generate code
        Given a task, ask GPT to generate some Python code that implements that task. For example,
        a task might be "Write a Python function (def scariness(story:str) -> float) to rate how scarey a story is, on a continuous scale of 0-1."
        and the output would be an executable Python function (expressed as a string).

        8. Write report
        Write up the research so far using a standard template.
        
        1.2 GENERAL ADVICE AND CONSTRAINTS
        ==================================
        - Plan steps should be implementable in Python.. Do not suggest steps that are not implementable.

        END OF AGENT DESCRIPTION
        ===============================================

    """
    '''
    role = "an AI assistant"
    system_message = """You are {role} whose primary goal is to identify promising, new, and key scientific
    problems based on existing scientific literature, in order to aid the autonomous discovery agent in discovering novel
    and significant research opportunities that can advance the field."""
    user_message = f"""You are an AI research assistant tasked with generating novel research problems based on existing scientific literature. Your goal is to aid an autonomous discovery agent in identifying significant research opportunities that can advance the field. Agent is designed for Probing tasks that that evaluate how language models perform on various reasoning challenges, helping to advance the field through targeted insights and improvements.
     
    You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper in the existing literature.
    
    You MUST use the following capabilities of the autonomous discovery agent, which are limited to prompting and probing techniques—without model fine-tuning, external data integration, or architectural modifications. The generated research idea and hypothesis must be testable only using in-context learning methods available to this agent.
    RESEARCH AGENT DESCRIPTION AND CAPABILITIES:
      ```
      {auto_nora_agent}
      ```
    
    
   Now, generate 5 refined specific testable hypotheses. For each hypothesis, follow these steps:
    --- 
    Let's start with the research problem generation task.
    1. Understanding of the source paper, and the related papers is essential:
    - The source paper is the primary research study you aim to enhance or build upon through future
    research, serving as the central source and focus for identifying and developing the specific
    research problem.
    
    2. Your approach should be systematic:
    - Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
    - Use only these papers to gain a broader perspective about the progression of the primary research topic over time. 

   
   Note that your research idea and hypothesis MUST be testable using the AGENT with these specific capabilities:
      1. Generate datasets using language models via structured prompt engineering (without external data or retrieval).
      2. Collect answers from language models using controlled variations in prompting techniques.
      3. Score answers using language models based on predefined in-context evaluation methods.
      4. Ideate categories for pattern identification through automated prompt-based clustering.
      5. Perform correlation analysis between results based on statistical comparisons of language model outputs.
      6. Generate Python code to process and analyze model-generated data, but not to modify the models themselves.
      7. Write research reports summarizing insights derived purely from probing techniques.



   IMPORTANT CONSTRAINTS:
  - The plan must be fully implementable in Python using prompting-based techniques only.
   - No human studies or human evaluations.
   - No external datasets, retrieval mechanisms, or new data collection beyond model-generated outputs.
   - No model pretraining, fine-tuning, or modification of any LLM parameters—only in-context learning is allowed.
   - Research must be executable within the agent's processing capabilities via structured input variations and response analysis.

   Keep these capabilities in mind as you generate the research idea and the hypothesis, ensuring that the final hypothesis can be tested using these tools and methods. Outline a step-by-step plan for how an agent with the specified capabilities (such as creating datasets, run different language models on those datasets, scoring answers, analyzing results, performing failure analysis) could test the hypothesis.
     
   I am going to provide the source paper as Title, Abstract and Year of publication 
    triple, as follows:
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}
    Source paper year of publication: {source_paper['year']}
    With the provided source paper your objective now is to formulate a
    research problem that not only builds upon these existing studies but also strives to be original,
    clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
    and abstract of the target paper, to ensure it remains the focal point of your research problem
    identification process. 
    
    Also, revisit the capabilities of the agent and make sure the research problem can be explored using the given capabilties of the agent.
    
    Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a 
    relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
    Your hypothesis should contain the key variable or variables from your research idea.

   Now, develop a step-by-step plan using the agent's specific capabilities, ensuring that every step relies strictly on prompting, structured input variations, and response-based evaluation—without modifying the model itself.
   [Your detailed plan, explicitly referencing the agent's capabilities for each step]

    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}

    Then, following your review of the above content, please proceed to analyze the progression of the research topic. Now output this analysis, the research idea and hypothesis with the rationale.
    Your output should be a valid JSON with the following fields.  
    Output a JSON object in the following format:
    ```json
    {{
        "hypotheses":[
            {{
            "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
            "Research idea": "Delineate an elaborate research problem here including the key variables.",
            "Hypothesis": "Provide a concrete testable hypothesis that follows from the research problem and can be evaluated using the agent's capabilities. The hypothesis should be a clear statement about the relationship between specific variables that can be measured through dataset generation and evaluation.",
            "Plan": "Provide a specific step-by-step detailed plan using the agent's capabilities (dataset generation, answer collection, scoring, etc.) to test the hypothesis. Each step should clearly map to one of the agent's primitives and be implementable in Python."
            }},
            //Repeat this instruction for all 5 hypotheses
        ]
    }}
    ```
    This JSON will be automatically parsed, so ensure the format is precise.
    """
    return system_message, user_message

def generate_initial_hypothesis_codeScientist(source_paper:dict, citing_paper_list:str):
    import json
    with open("./generation/codescientist_codeblocks.json", 'r', encoding='utf-8') as file:
        codeblock_summaries = json.load(file)
    
    role = "an AI assistant"
    system_message = """You are {role} whose primary goal is to identify promising, new, and key scientific
    problems based on existing scientific literature, in order to aid researchers in discovering novel
    and significant research opportunities that can advance the field."""
    user_message = f"""You are an AI research assistant tasked with generating novel research problems based on existing scientific literature. Your goal is to aid an autonomous discovery agent in identifying significant research opportunities that can advance the field.
    
    You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper in the existing literature.
    
    You MUST use the following capabilities of the autonomous discovery agent.
    RESEARCH AGENT DESCRIPTION AND CAPABILITIES - CODE BLOCK LIBRARIES:
    ```
    {json.dumps(codeblock_summaries, indent=4)}
    ```
   
    Now, generate 5 refined specific testable hypotheses. For each hypothesis, follow these steps:
    --- 
    Let's start with the research problem generation task.
    1. Understanding of the source paper, and the related papers is essential:
    - The source paper is the primary research study you aim to enhance or build upon through future
    research, serving as the central source and focus for identifying and developing the specific
    research problem.

   2. Your approach should be systematic:
    - Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
    - Use only these papers to gain a broader perspective about the progression of the primary research topic over time. 

   Note that your research idea and hypothesis MUST be testable using the AGENT with these specific capabilities: 
   When evaluating feasibility and outlining the testing approach, consider the following agent-specific information:
    
    - Available codeblocks: Consider existing codeblocks from the codeblock library that this idea is likely to use.
    - Required code and resources: Create an EXHAUSTIVE list of ALL required CODE, RESOURCES, MODELS, etc. mentioned in the ENTIRE RESEARCH IDEA. This is CRITICALLY IMPORTANT and will be used to determine feasibility. For each item, provide:
        - Name: A short, descriptive name
        - Description: A brief description of the code or resource needed
        - Where: One of: 'existing codeblock', 'external', or 'build'
        - Effort: One of: 'minor', 'moderate', or 'major'

        Example:
        ```json
        "research_idea_required_code_and_resources": [
        {{
            "name": "ReAct baseline",
            "description": "A ReAct baseline (targeted for use on Benchmark XYZ)",
            "where": "existing codeblock",
            "effort": "minor"
        }},
        {{
            "name": "Fancy New Agent++",
            "description": "A new agent proposed in this work integrating ...",
            "where": "build",
            "effort": "major"
        }}
        ]
        ```

    - External requirements: List any libraries or packages that may be required, in the format: "python/apt package name (very short description of need)"

   You are asked to generate new research idea and testable scientific hypotheses that are *conditioned*/*related to* the kinds of codeblocks that the agent with automated experiment builder has available in the provided codeblock library. 
   
   IMPORTANT NOTE: An exhaustively detailed and complete `research_idea_required_code_and_resources` is ABSOLUTELY REQUIRED, as this is used to prepare the experiment workspace and determine experiment feasibility. A poorly or incorrectly documented `research_idea_required_code_and_resources` for the scientific hypotheses is a major failure, as it will waste a large amount of resources (time/money/etc) on ideas that may be unlikely to have the resources they need to succeed.

    It is likely that your `research_idea_required_code_and_resources` and accompanying `research_idea_external_requirements` will change -- adding, subtracting, or modifying elements based on the new simplified research idea that you're generating. It is CRITICALLY important that you update these fields to reflect ALL the requirements of the simplified idea.

   NOTE: Manual human ratings in the research (e.g. human rating of the quality of generated text from an experiment) is considered an `external` resource of `major` effort, for the purposes of the potential research experiments, and should generally be avoided (unless absolutely required for the research).
    
   Now, I am going to provide the source paper as Title, Abstract and Year of publication 
    triple, as follows:
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}
    Source paper year of publication: {source_paper['year']}
    With the provided source paper your objective now is to formulate a
    research problem that not only builds upon these existing studies but also strives to be original,
    clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
    and abstract of the target paper, to ensure it remains the focal point of your research problem
    identification process. 

    Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a 
    relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
    Your hypothesis should contain the key variable or variables from your research idea.

    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}

    Then, following your review of the above content, please proceed to analyze the progression of the research topic. Now output this analysis, the research idea and hypothesis with the rationale.
    Your output should be a valid JSON with the following fields.  
    Output a JSON object in the following format:
    ```json
    {{
        "hypotheses":[
        {{
        "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
        "Research idea": "Delineate an elaborate research problem here including the key variables.",
        "Hypothesis": "Provide a concrete testable hypothesis that follows from the above research problem here",
        "research_idea_required_code_and_resources": [
            {{
            "name": "Example Resource",
            "description": "Brief description of the resource",
            "where": "existing codeblock",
            "effort": "minor"
            }}
        ],
        "research_idea_external_requirements": [
            "example_package (for specific purpose)"
        ]
        }},
        //Repeat this instruction for all 5 hypotheses
    ]
    }}
    ```
    This JSON will be automatically parsed, so ensure the format is precise.
    """
    return system_message, user_message


def generate_initial_hypothesis(source_paper:dict, citing_paper_list:str):
    role = "an AI assistant"
    system_message = """You are {role} whose primary goal is to identify promising, new, and key scientific
    problems based on existing scientific literature, in order to aid researchers in discovering novel
    and significant research opportunities that can advance the field."""
    user_message = f"""You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper in the existing literature.
    
    Now, generate a refined specific testable hypotheses. For each hypothesis, follow these steps:
    --- 
    Let's start with the research problem generation task.
    1. Understanding of the source paper, and the related papers is essential:
    - The source paper is the primary research study you aim to enhance or build upon through future
    research, serving as the central source and focus for identifying and developing the specific
    research problem.

   2. Your approach should be systematic:
    - Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
    - Use only these papers to gain a broader perspective about the progression of the primary research topic over time. 

    I am going to provide the source paper as Title, Abstract and Year of publication 
    triple, as follows:
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}
    Source paper year of publication: {source_paper['year']}
    With the provided source paper your objective now is to formulate a
    research problem that not only builds upon these existing studies but also strives to be original,
    clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
    and abstract of the target paper, to ensure it remains the focal point of your research problem
    identification process. 

    Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a 
    relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
    Your hypothesis should contain the key variable or variables from your research idea.

    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}

    Then, following your review of the above content, please proceed to analyze the progression of the research topic. Now output this analysis, the research idea and hypothesis with the rationale.
    Your output should be a valid JSON with the following fields.  
    Output a JSON object in the following format:
    ```json
    {{
        "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
        "Research idea": "Delineate an elaborate research problem here including the key variables.",
        "Hypothesis": "Provide a concrete testable hypothesis that follows from the above research problem here"
    }}
    ```
    This JSON will be automatically parsed, so ensure the format is precise.
    """
    return system_message, user_message


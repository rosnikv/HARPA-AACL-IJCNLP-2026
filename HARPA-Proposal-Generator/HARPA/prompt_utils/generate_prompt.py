def get_specific_hypotheses_AutoNORA(hypothesis, variable_info, similar_paper_list):
    auto_nora_agent = f"""
    
      START OF AGENT DESCRIPTION
      ===============================================
      
      You are an autonomous research scientist, and have been tasked designed for Probing tasks that characterize language model's behavior on various reasoning tasks.

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
        Iterate over the questions in a dataset, and pose them to a language model. Collect the answers and add them to the DataFrame.

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
    
    system_message = """You are an expert scientific researcher specializing in hypothesis refinement. Your task is to generate specific, original, and testable hypotheses strictly based on provided variable combinations while ensuring they have not been extensively explored in related literature.
    """
    user_message = f"""You are an expert scientific researcher tasked with refining a given hypothesis into a more specific and testable form. Your goal is to generate novel hypotheses that:
    - Are strictly based on the given variable options (no new variables should be introduced).
    - Use novel variable combinations that have not been extensively explored in similar papers.
    - Focus solely on the key variable and its concrete variable values or implementations or alternatives, STRICTLY avoid any ambiguous phrasing
    - Are highly specific, testable and feasible for an autonomus discovery system, specifying conditions, interventions, and expected measurable outcomes.
    - Provide a detailed theoretical and practical justification for why the refined hypothesis is an important and promising research direction.

    ---
    ### Step 1: Analyze the Given Hypothesis
    Initial Hypothesis:  
    `{hypothesis}`

    ### Step 2: Review the Extracted Variables & Options
    Key Variables & Available Options:
    `{variable_info}`

    ### Step 3: Examine Similar Papers
    Below are relevant excerpts from similar papers, detailing already explored hypotheses and variable combinations. Ensure your refined hypothesis is not extensively covered by these studies:
    `{similar_paper_list}`

    ---
    ### Step 4: Generate a Refined Hypothesis: Analyse the initial hypothesis and generate a specific testable hypothesis by making the key variables from the hypothesis as specific as possible using the variable value options and the similar paper excerpts provided.
    
     Before we begin the refinement process, let's consider the some of the capabilities and description of the autonomous discovery agent that will be testing this hypothesis:

      ```
      {auto_nora_agent}
      ```

      Keep these capabilities in mind as you refine the hypothesis, ensuring that the final hypothesis can be tested using these tools and methods.
    
      Requirements for the refined specific hypothesis generation:
         1. Strictly use the provided variable options. Do not introduce external variables.
            - Focus strictly on the key variable and its concrete variable values with implementations or alternatives, and AVOID any ambiguous phrasing.
         2. If applicable, make it simple and easy to understand. The hypothesis should explore NEW VARIABLE COMBINATIONS or CONDITIONS that were not extensively tested in similar papers.
         3. Make it highly specific and testable. Clearly define:
            - the condition, the expected measurable outcome, a control or comparative condition (if applicable)
         4. Ensure originality. The hypothesis should explore NEW VARIABLE COMBINATIONS or CONDITIONS that were not extensively tested in similar papers.
         5. Folow the autonomous agent capabilities while generating the specific hypothesis
         
    ---
    ### Step 5: Provide a Deep Justification
    Your response must include:
    
    - Theoretical Justification (More Detailed) 
      1. Explain the individual function of each technique in this refined hypothesis.  
         - How does each component work in isolation?  
         - Provide a clear step-by-step mechanism of action for each.  
         
      2. Explain why combining these techniques is beneficial beyond their individual effects.  
         - What problems does each solve that the others do not?  
         - How do they interact to produce a greater improvement than if used separately?

    - Expected Synergies (Mechanism of Interaction) 
      1. Describe how the components reinforce each other dynamically during task execution.  
         - What happens at each decision step?  
         - How does one technique influence the effectiveness of another?  
      
      2. Explain why this integration is better than other known methods.  
         - Compare against existing techniques.  
         - Provide concrete reasons why this combination yields better results.

    - Plan:
      1. Outline a step-by-step plan for how an agent with the specified capabilities (such as creating datasets, run different language models on those datasets, scoring answers, analyzing results, performing failure analysis) could test the hypothesis]
    
      IMPORTANT: The task/plan should be implementable in Python, using the above or other functions. Don't suggest a task that requires skills that cannot be implemented, e.g., human studies. Don't suggest a task that requires access to external datasets, as you do not have access to them. Do not suggest tasks that involve pretraining or fine-tuning models, as you do not have the resources for such experiments.

    ---
    ### Step 6: Structure Your Output in JSON Format
    Your final response should be structured as follows:
    ```json
    {{
        "refined_hypothesis": "Provide a concrete testable hypothesis",
        "explanation": {{
            "difference": "How it differs from the initial hypothesis",
            "novelty": "Why it's not extensively explored in the similar paper excerpts",
            "specificity": "How it's more specific, testable, and feasible",
            "theoretical_justification": "Why the chosen combination of techniques in this refined hypothesis makes sense from a scientific and computational perspective",
            "expected_synergies": "How the components in this refined hypothesis interact to improve performance and overcome existing limitations",
            "Plan": "Provide a specific step-by-step detailed plan using the agent's capabilities which is designed for Probing tasks that characterize language model's behavior on various reasoning tasks (dataset generation, answer collection, scoring, etc.) to test the hypothesis. Each step should clearly map to one of the agent's primitives and be implementable in Python."
        }}
    }}
    ```
    Ensure clarity, conciseness, and direct relevance to the provided information. Remember that a hypothesis is a declarative statement expressing a relationship between two variables (e.g., independent and dependent variables) in a given context. Your refined hypothesis should contain the key variables from variable space and testable using the autnomous agent capabilities.
    """

    return system_message, user_message


def get_specific_hypotheses(hypothesis, variable_info, similar_paper_list):
    system_message = """You are an expert scientific researcher specializing in hypothesis refinement. Your task is to generate specific, original, and testable hypotheses strictly based on provided variable combinations while ensuring they have not been extensively explored in related literature.
    """
    user_message = f"""You are an expert scientific researcher tasked with refining a given hypothesis into a more specific and testable form. Your goal is to generate novel hypotheses that:
    - Are strictly based on the given variable options (no new variables should be introduced).
    - Focus solely on the key variable and its concrete variable values or implementations or alternatives, STRICTLY avoid any ambiguous phrasing
    - Use novel variable combinations that have not been extensively explored in similar papers.
    - Are highly specific, testable and feasible, specifying conditions, interventions, and expected measurable outcomes.
    - Provide a detailed theoretical and practical justification for why the refined hypothesis is an important and promising research direction.

    ---
    ### Step 1: Analyze the Given Hypothesis
    Initial Hypothesis:  
    `{hypothesis}`

    ### Step 2: Review the Extracted Variables & their value Options
    Key Variables & Available Value Options:
    `{variable_info}`

    ### Step 3: Examine Similar Papers
    Below are relevant excerpts from similar papers, detailing already explored hypotheses and variable combinations. Ensure your refined hypothesis is not extensively covered by these studies:
    `{similar_paper_list}`

    ---
    ### Step 4: Generate a specific testable Hypothesis: Analyse the initial hypothesis and generate a specific testable hypothesis by making the key variables from the hypothesis as specific as possible using the variable value options and the similar paper excerpts provided.
      Requirements for the refined hypothesis generation:
         1. Strictly use the provided variable options. Do not introduce external variables.
            - Focus strictly on the key variable and its concrete variable values with implementations or alternatives, and AVOID any ambiguous phrasing.
         2. If applicable, make it simple and easy to understand. The hypothesis should explore NEW VARIABLE COMBINATIONS or CONDITIONS that were not extensively tested in similar papers.
         3. Make it highly specific and testable. Clearly define:
            - the condition, the expected measurable outcome, a control or comparative condition (if applicable)
         4. Ensure originality. The hypothesis should explore NEW VARIABLE COMBINATIONS or CONDITIONS that were not extensively tested in similar papers.

    ---
    ### Step 5: Provide a Deep Justification
    Your response must include:
    
    - Theoretical Justification (More Detailed) 
      1. Explain the individual function of each technique in this refined hypothesis.  
         - How does each component work in isolation?  
         - Provide a clear step-by-step mechanism of action for each.  
         
      2. Explain why combining these techniques is beneficial beyond their individual effects.  
         - What problems does each solve that the others do not?  
         - How do they interact to produce a greater improvement than if used separately?

    - Expected Synergies (Mechanism of Interaction) 
      1. Describe how the components reinforce each other dynamically during task execution.  
         - What happens at each decision step?  
         - How does one technique influence the effectiveness of another?  
      
      2. Explain why this integration is better than other known methods.  
         - Compare against existing techniques.  
         - Provide concrete reasons why this combination yields better results.

    ---
    ### Step 6: Structure Your Output in JSON Format
    Your final response should be structured as follows:
    ```json
    {{
        "refined_hypothesis": "Provide a concrete testable hypothesis",
        "explanation": {{
            "difference": "How it differs from the initial hypothesis",
            "novelty": "Why it's not extensively explored in the similar paper excerpts",
            "specificity": "How it's more specific, testable, and feasible",
            "theoretical_justification": "Why the chosen combination of techniques in this refined hypothesis makes sense from a scientific and computational perspective",
            "expected_synergies": "How the components in this refined hypothesis interact to improve performance and overcome existing limitations"
        }}
    }}
    ```
    Ensure clarity, conciseness, and direct relevance to the provided information. Remember that a hypothesis is a declarative statement expressing a relationship between two variables (e.g., independent and dependent variables) in a given context. Your refined hypothesis should contain the key variables from variable space and testable.
    """

    return system_message, user_message

def get_specific_hypotheses_codeScientist_CB(hypothesis, variable_info, similar_paper_list):
   import json
   with open("./hg_pipeline/codescientist_codeblocks.json", 'r', encoding='utf-8') as file:
      codeblock_summaries = json.load(file)
   codeblock_summary_text = json.dumps(codeblock_summaries, indent=4)
   condition_on_code_block = f"""
   ### Codeblocks Summary:
    - Available codeblocks: Consider existing codeblocks from the codeblock library that this hypothesis is likely to use.
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

        Here are high-level summaries of the code templates available in the experiment builder:

        ```
        {codeblock_summary_text}
        ```
   """
   system_message = """You are an expert scientific researcher specializing in hypothesis refinement. Your task is to generate specific, original, and testable hypotheses strictly based on provided variable combinations while ensuring they have not been extensively explored in related literature.
    """

   agent_description = f"""
    CodeScientist, an end-to-end semi-automated scientific discovery system that designs, iterates, and analyzes scientific experiments that can be expressed as (Python) code. The experiment hypothesis can be implemented using the Experiment Builder, which automatically creates, runs, and debugs the experiment code in a container. When completed, CodeScientist writes a report on the results. Usually, CodeScientist makes several (for example, 5) independent attempts at creating experiments for a given idea, and can create a meta-analysis describing the overall results over each of the 5 experiment attempts.
    
    Your current task is to convert a previously generated research idea/hypothesis into a less-ambitious research idea (perhaps appropriate for an undergraduate or MSc research assistant, with limited resources, and perhaps only partial training in computer science), while still maintaining the goal of having an interesting, novel, and potentially impactful research result.
    
    Regardless of the reduced complexity of the research idea, you must still maintain the highest standards of science, research methods, soundness, and correctness, with the hope of the research being interesting enough to (for example) form part of a workshop paper at an academic conference.
    
    ### Codeblocks Summary:
    - Available codeblocks: Consider existing codeblocks from the codeblock library that this hypothesis is likely to use.
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
        
        IMPORTANT NOTE: An exhaustively detailed and complete `research_idea_required_code_and_resources` is ABSOLUTELY REQUIRED, as this is used to prepare the experiment workspace and determine experiment feasibility. A poorly or incorrectly documented `research_idea_required_code_and_resources` for the scientific hypotheses is a major failure, as it will waste a large amount of resources (time/money/etc) on ideas that may be unlikely to have the resources they need to succeed.
            
        It is likely that your `research_idea_required_code_and_resources` and accompanying `research_idea_external_requirements` will change -- adding, subtracting, or modifying elements based on the new simplified research idea that you're generating. It is CRITICALLY important that you update these fields to reflect ALL the requirements of the simplified idea.

    NOTE: Manual human ratings in the research (e.g. human rating of the quality of generated text from an experiment) is considered an `external` resource of `major` effort, for the purposes of the potential research experiments, and should generally be avoided (unless absolutely required for the research).

    """
   
   user_message = f"""You are an expert scientific researcher tasked with refining a given hypothesis into a more specific and testable form. Your goal is to generate novel hypotheses that:
    - Are strictly based on the given variable options (no new variables should be introduced).
    - Focus solely on the key variable and its concrete variable values or implementations or alternatives, STRICTLY avoid any ambiguous phrasing
    - Use novel variable combinations that have not been extensively explored in similar papers.
    - Avoid including specific numerical outcomes (e.g., “45% improvement”) in the hypothesis phrasing.
    - Provide a detailed theoretical and practical justification for why the refined hypothesis is an important and promising research direction.

    ---
    ### Step 1: Analyze the Given Hypothesis
    Initial Hypothesis:  
    `{hypothesis}`

    ### Step 2: Review the Extracted Variables & their value Options
    Key Variables & Variable Value Space:
    `{variable_info}`

    ### Step 3: Examine Similar Papers
    Below are relevant excerpts from similar papers, detailing already explored hypotheses and variable combinations. Ensure your refined hypothesis is not extensively covered by these studies:
    `{similar_paper_list}`

    ---
    ### Step 4: Generate a specific testable Hypothesis: 
    - Analyse the initial hypothesis and generate a specific testable hypothesis by making the key variables from the hypothesis as specific as possible using the variable value options and the similar paper excerpts provided.
    - When selecting variable values, consider specific implementations and settings details provided in the variable value description. These must directly influence the choice and design of the hypothesis and `research_idea_required_code_and_resources`.

    Before we begin the refinement process, let's consider the some of the capabilities and description of the autonomous discovery agent that will be testing this hypothesis:
      
    IMPORTANT: When evaluating feasibility and testability of the hypothesis, consider the following agent-specific information:
    
    ### Agent description:  
    {agent_description}
    

    Remember that a hypothesis is a declarative statement expressing a relationship between two variables (e.g., independent and dependent variables) in a given context. Your refined hypothesis should contain the key variables from your research idea.

    Now, Requirements for the specific hypothesis generation:
    1. Strictly use the provided variable options. Do not introduce external variables. 
        - Focus strictly on the key variable and its concrete variable values with implementations or alternatives, and AVOID any ambiguous phrasing.
    2. If applicable, make it simple and easy to understand. The hypothesis should explore NEW VARIABLE COMBINATIONS or CONDITIONS that were NOT EXTENSIVELY tested in similar papers.
    3. Make it highly specific and testable. Clearly define the condition, the expected measurable outcome, a control or comparative condition (if applicable)
    4. Ensure originality. The hypothesis should explore NEW VARIABLE COMBINATIONS or CONDITIONS that were not extensively tested in similar papers.  
    5. Do not include exact numerical claims (e.g., "45% improvement", "2.1x increase"). Use comparative phrasing like "reduced," "improved," "higher," "significantly more/less" instead. Specific metrics should appear in the evaluation section, not in the hypothesis itself.
    ----
    
    ### Align with Codeblocks and Resources: 
    - When generating `research_idea_required_code_and_resources`, map each implementation detail from the variable value space directly to a code/resource item. 
    You MUST produce an EXHAUSTIVE and PRECISE COMPLETE list of all required codeblocks and resources for implementing and evaluating this hypothesis.
    - These implementation-level choices are critical and must not be generalized or omitted.
        - Identify codeblocks required to implement the refined hypothesis.
        - For each required resource or codeblock, provide the details in `research_idea_required_code_and_resources` and strictly follow these rules:
            1. Include the `where` field to indicate the source of the resource:
                - `"existing codeblock"` — Use this if the codeblock already exists in the codeblock library.
                - `"external"` — Use this if the codeblock/resource must be sourced externally.
                - `"build"` — Use this if a new codeblock/resource must be created from scratch.
            2. Ensure that the `effort` field reflects the complexity involved:
                - `"minor"` — For trivial tasks or direct application of an existing codeblock.
                - `"moderate"` — For tasks that require adaptation or medium-level effort.
                - `"major"` — For complex tasks or building a new system.
            3. IMPORTANT: If any required resource is not explicitly listed in the available codeblock summaries, mark it as:
                - `where: "external"` if the resource can be sourced externally.
                - `where: "build"` if the resource does not exist and needs to be built.
                - **Do NOT label any resource as `"existing codeblock"` unless the codeblock name and capability are explicitly present in the codeblock summary.**
                - If you cannot find an exact match in the codeblock summary, default to `"build"` or `"external"`. 
                - If the base functionality is being reused but a new mechanism must be implemented on top, list **both**:
                    1. the reusable base codeblock as `"existing codeblock"` (effort: "minor" or "moderate")
                    2. the new component (e.g., glue logic, new memory representation) as `"build"` (effort: "moderate" or "major")

    When refining the hypothesis and aligning it with codeblocks, consider the following:
        - The hypothesis should be implementable in Python using the available codeblocks or other easily accessible functions.
        - Avoid suggesting tasks that require skills that cannot be implemented, such as human studies or access to external datasets.
        - **DO NOT** suggest any tasks involving model fine-tuning, pretraining, or model parameter adjustment. Use only pre-existing models and APIs as-is.
        - **Do not suggest tasks involving pretraining or fine-tuning models**, as resources for such experiments are not available.
    

    ### Step 5: Provide a Deep Justification
    Your response must include:
    
    - Theoretical Justification (More Detailed) 
      1. Explain the individual function of each technique in this refined hypothesis. 
         - How does each component work in isolation?  
         - Provide a clear step-by-step mechanism of action for each.  
         
      2. Explain why combining these techniques is beneficial beyond their individual effects.  
         - What problems does each solve that the others do not?  
         - How do they interact to produce a greater improvement than if used separately?

    - Expected Synergies (Mechanism of Interaction) 
      1. Describe how the components reinforce each other dynamically during task execution.  
         - What happens at each decision step?  
         - How does one technique influence the effectiveness of another?  
      
      2. Explain why this integration is better than other known methods.  
         - Compare against existing techniques.  
         - Provide concrete reasons why this combination yields better results.

    #### Revisit the Important Constraints:
    - Focus strictly on the key variable and its concrete variable values with implementations or alternatives, and AVOID any ambiguous phrasing.
    - If applicable, make it simple and easy to understand. The hypothesis should explore NEW VARIABLE COMBINATIONS or CONDITIONS that were not extensively tested in similar papers.
    - NO MODEL FINE_TUNING or parameter adjustment is allowed. Only use pre-existing models through their standard APIs or existing implementations.
    - Manual human ratings in the research (e.g. human rating of the quality of generated text from an experiment) is considered an `external` resource of `major` effort, for the purposes of the potential research experiments, and should generally be avoided (unless absolutely required for the research).
    - Ensure feasibility by aligning the hypothesis with existing or buildable codeblocks.

    --- 
    
    ### Step 6: Structure Your Output in JSON Format
    Based on your analysis, generate a refined hypothesis and provide the following information in JSON format:
    
    
    ```json
    {{
        "research_question": "A clear, testable research question that can be addressed using the refined hypothesis. It should reflect the causal or comparative relationship proposed, reuse key variable or method terms, and be answerable using the system's capabilities. Frame it in open-ended scientific language (e.g., 'Does...', 'How does...', 'What effect does...').",
        "research_idea_hypothesis": "Provide a concrete testable hypothesis",
        "research_idea_long_description": {{
            "description":  "A clear paragraph explaining the complete research idea including what will be tested, how it will be implemented, and expected outcomes. Clearly explain the motivation, purpose, and expected outcomes. Use the selected variable values to describe how each component contributes individually, why their combination is expected to work synergistically, and how this addresses gaps or limitations in prior work (as reflected in the similar paper excerpts). Tie your reasoning to specific characteristics of the task or evaluation environment, and avoid vague statements—be specific about what performance improvements are expected and why. (200-400 words)",
            
            "research_idea_variables": {{
                "concise Variable Name": "Begin by clearly defining what the selected value represents—whether it's an architecture, strategy, metric, dataset, or baseline condition. Describe exactly how this variable will be configured, used, or operationalized in the experiment; for example, specify how a module is implemented, how a metric is calculated, or how a strategy is triggered. Explain why this specific value was selected over alternatives, including its advantages, novelty, or relevance to the hypothesis. Describe the expected role this variable plays in the research problem-what outcome it directly influences or enables. If the variable is measurable, explicitly define how it will be assessed, including the metric used, how it's calculated, and what range of values or thresholds would indicate a successful outcome. Your explanation should be grounded in the context of the hypothesis and tied directly to experimental design choices and evaluation logic.(200-400 words)",
                //add detailed defintion and description of every independent, dependent, comparable groups, comparative variables, and control variables in simple format.
            }},
            
            "research_idea_design_prompt": "Describe in detail how the hypothesis will be implemented using the agent's capabilities. Specify which codeblocks will be used, referencing their exact names as provided in the codeblock library. Explain how each codeblock contributes to the overall implementation, how data flows between components, and how any new logic or integration layers (e.g., glue modules, wrappers) will be built or adapted. Clearly distinguish between reused components and newly built modules. Include all setup steps, model configurations, inputs/outputs expected, and how the hypothesis will be realized end-to-end in code. (400-800 words)",
            
            "research_idea_metric": "Primary and secondary metrics that will be used to evaluate the hypothesis. Explain how the hypothesis will be tested using concrete metrics and comparative setups. Identify the benchmark tasks or datasets to be used, the control condition (e.g., a baseline agent without the component being tested), and the exact performance metrics (e.g., task success rate, reasoning accuracy, number of valid steps). Define how improvement or success will be interpreted, including thresholds, number of runs, or statistical confidence if relevant. If qualitative evaluations are involved, explain how they will be derived. Ensure that all evaluations are feasible using the agent's capabilities.(200-400 words)"
        }},
        "research_idea_name": "A short, descriptive name for the research idea (3-5 words)",
        "research_idea_short_description": "A single concise sentence summarizing the core idea (15-25 words)",
        "research_baselines": "Simple list of baseline approaches to compare against",
        "research_idea_pilot": "Brief description of an initial small-scale test to validate the approach",
        "research_idea_required_code_and_resources": [
            {{
            "name": "Example Resource",
            "description": "Brief description of the resource",
            "where": "One of: 'existing codeblock', 'external', or 'build'",
            "effort": "One of: 'minor', 'moderate', or 'major'"
            }},
            // EXHAUSTIVE list of ALL required CODE, RESOURCES, MODELS, etc. mentioned in the ENTIRE RESEARCH IDEA
               ],
        
        "research_idea_external_requirements": [
            "example_package (for specific purpose)"
            ],
        
        "explanation": {{
            "difference": "How it differs from the initial hypothesis",
            "novelty": "Why it's not extensively explored in the similar paper excerpts",
            "specificity": "How it's more specific, testable, and feasible",
            "theoretical_justification": "Why the chosen combination of techniques in this refined hypothesis makes sense from a scientific and computational perspective (200-400 words)",
            "expected_synergies": "How the components in this refined hypothesis interact to improve performance and overcome existing limitations (200-400 words)"
        }}
    }}
    ```
    Ensure clarity, conciseness, and direct relevance to the provided information. Remember that a hypothesis is a declarative statement expressing a relationship between two variables (e.g., independent and dependent variables) in a given context. Your refined hypothesis should contain the key variables from variable space and testable using the autnomous agent capabilities.
    """

   return system_message, user_message

def get_specific_hypotheses_codeScientist(hypothesis, variable_info, similar_paper_list):
    system_message = """
    You are ScientistGPT, the most advanced automated scientific model in the world. You are tasked with refining broad research hypotheses into specific, testable, and implementable research ideas suitable for automated experimentation.

    The goal is to generate novel, interesting, and feasible scientific results using the autonomous discovery pipeline. You must ensure clarity, scientific rigor, and practical feasibility within strict constraints.
    """

    agent_description = """
    The ASD Agent is an automated discovery system that writes Python-based experiments, executes them in containers, and analyzes results—usually across five independent runs with a meta-analysis.

    Your goal is to downscope the idea to something an undergrad or MSc student or PhD student could realistically implement, while retaining novelty and scientific merit. The result should be suitable for a workshop paper.

    Manual human rating counts as an 'external' resource of 'major' effort and should generally be avoided. All experiments must be implementable using codeblocks or buildable logic only. No model fine-tuning or pretraining is permitted.
    
    Note: Details about the ASD Agent's automation pipeline (like container execution or 5-run meta-analysis) are internal infrastructure context and should NOT appear in the generated research idea. Focus only on the scientific method, implementation logic, and expected outcomes relevant to the hypothesis.

    """
    user_message = f"""
    You are an expert scientific researcher tasked with refining a given hypothesis into a more specific and testable form. Your goal is to generate novel hypotheses that:
        - Are strictly based on the given variable options (no new variables should be introduced).
        - Focus solely on the key variable and its concrete variable values or implementations or alternatives, STRICTLY avoid any ambiguous phrasing
        - Use novel variable combinations that have not been extensively explored in similar papers.
        - Avoid including specific numerical outcomes (e.g., “45% improvement”) in the hypothesis phrasing.
        - Provide a detailed theoretical and practical justification for why the refined hypothesis is an important and promising research direction.

    ---
    
    ### **Step 1: Understand the Context**

    - **Initial Hypothesis:**  
    `{hypothesis}`

    - **Available Variables and Value Options:**  
    `{variable_info}`

    - **Similar Papers (to avoid overlap):**  
    `{similar_paper_list}`
    
    (Each item includes paper title, citation count, and year - use this metadata to assess which papers are foundational vs. fringe or outdated. Avoid redoing what's already exists unless you're offering a clear novel twist.)
    
    ---
    ### Step 1.5: Plan Your Reasoning
    Before generating the specific testable hypothesis, outline the logical reasoning process to **Ensure Novelty and Relevance**:
    - What is the main contribution of the initial hypothesis?
    - Which variables are most critical?
    - Carefully review the `similar_paper_list` to identify variable combinations or configurations **already explored**.
    - For each similar paper, consider its citation count and publication year to avoid overlaps with highly cited or recent papers unless offering a clearly novel twist, and to spot works worth revisiting.
    - Identify gaps in existing research that your hypothesis can address. The hypothesis should explore NEW VARIABLE COMBINATIONS or CONDITIONS or DESIGN CHOICES that were NOT EXTENSIVELY tested in similar papers. 
    - The research idea space is vast - prioritize hypotheses that seem explanatory, surprising, or tied to concrete downstream benefits. Not all combinations are equally promising. Ask: *Why is this idea worth testing over 999 others?* What gap or uncertainty does it address?
    - Avoid trivial permutations (e.g., swapping known modules without meaningful interaction).
    - Ensure the integration logic is **not only novel** but **precisely describable**—how the components work together must be clearly traceable from input to output.
    
    ---
   
   ### Step 2: Generate a Specific Testable Hypothesis
   
   - Analyse the initial hypothesis and generate a specific testable hypothesis by making the key variables from the hypothesis as specific as possible using the variable value options and the similar paper excerpts provided.
   
    Before we begin the refinement process, let's consider the some of the capabilities and description of the autonomous discovery agent that will be testing this hypothesis:
      
    IMPORTANT: When evaluating feasibility and testability of the hypothesis, consider the following agent-specific information:
    
    ### Agent description:  
    {agent_description}
    
    - **For every variable and process mentioned in your hypothesis**, explicitly list:
    - The required code, resource, model, or tool.
    - Source: `"existing codeblock"` (if in the codeblock library), `"external"`, or `"build"` (if needs to be created).
    - Effort: `"minor"`, `"moderate"`, `"major"`.
    - If a component is not found in the available resources, mark as `"build"` or `"external"`.
    - This mapping is **critical** for experiment feasibility—*missing or incorrect entries are a critical error*.

    ---
    
    Before proceeding, you must strictly follow the following tiered guideline:

    #### MANDATORY
        - Strictly use the provided variable options. Do not introduce external variables. 
            - Focus strictly on the key variable and its concrete variable values with implementations or alternatives, and AVOID any ambiguous phrasing.
        - If applicable, make it simple and easy to understand. The hypothesis should explore NEW VARIABLE COMBINATIONS or CONDITIONS or DESIGN CHOICES that were NOT EXTENSIVELY tested in similar papers. 
        - Make it highly specific and testable. Clearly define the condition, the expected measurable outcome, a control or comparative condition (if applicable)
        - Ensure originality. The hypothesis should explore NEW VARIABLE COMBINATIONS or CONDITIONS or DESIGN CHOICES that were not extensively tested in similar papers, but also technically CORRECT.  
            - Make sure the combination is not just NOVEL, but also PURPOSEFUL. Why do these components logically belong together? What capability does one component enable or enhance in the other?
            -  The research idea space is vast - prioritize hypotheses that seem explanatory, surprising, or tied to concrete downstream benefits. Not all combinations are equally promising. Ask: *Why is this idea worth testing over 999 others from this space?* What gap or uncertainty does it address?
        - Do not include exact numerical claims (e.g., "45% improvement", "2.1x increase"). Use comparative phrasing like "reduced," "improved," "higher," "significantly more/less" instead. Specific metrics should appear in the evaluation section, not in the hypothesis itself.
        - Provide a fully aligned and exhaustive `research_idea_required_code_and_resources`.
        - Include **detailed, step-by-step theoretical justification** and **expected synergy** between components.

    #### RECOMMENDED PRACTICES
        - Use simple, readable phrasing.
        - Favor comparative wording ("higher", "improved") over numeric claims.
        - Keep pilot-friendly scope: small data or short episodes.

    #### PROHIBITED
        - No external/unlisted variables.
        - No specific numeric performance outcomes in hypotheses.
        - No model **FINE_TUNING, PRETRAINING**, or internal **parameter updates**.
        - AVOID human evaluation unless marked external/major.
        - Do not omit any mentioned implementation from resource lists.

    #### FINAL SELF-CHECK
        - [ ] All variables are from the given space
        - [ ] Hypothesis is clear, testable, and comparative
        - [ ] No numeric performance claims in the hypothesis.
        - [ ] No model fine-tuning or human studies unless justified
        - [ ] Resources list is complete and properly tagged (where + effort)
        - [ ] Hypothesis is implementable with codeblocks or buildable logic
     
    ---
         
    Remember that a hypothesis is a declarative statement expressing a relationship between two variables (e.g., independent and dependent variables) in a given context. Your refined hypothesis should contain the key variables from your research idea.
  
    ----
    
    ### Step 3: Litmus Test: Is Your Hypothesis Understandable?

    Try this test: 
    
    Ask: Could an MSc student with no background in the specific technique **understand and implement** your hypothesis just from reading the research_idea_long_description?
    If not — explain the terms more clearly. If any key term or technique may not be intuitive, include a brief, concrete example of how it works in practice.
    
    Ask: Would a technically trained MSc student be able to reconstruct why and how these techniques fit together just by reading this? 
    If not, the `theoretical_justification` is too shallow.
    
    Remember: A technically trained MSc student must be able to understand each component and how they fit together. Avoid unexplained jargon. If a method is mentioned (e.g., “multi-arm bandit” or “binary token”), explain what it means, why it's used, and how it works in this experiment.
    
    ----
    
    ### Step 4: Structure Your Output in JSON Format
    Based on your analysis, generate a refined hypothesis and provide the following information in JSON format:
    
    
    ```json
    {{
        "research_gap": "Explain why existing methods (both classic ones and recent ones) are not good enough to solve the problem, and explain the inspiration behind the new proposed method. You should also motivate why the proposed method would work better than existing baselines on the problem. Clearly state the specific gap or limitation in similar paper list or prior work that this hypothesis addresses. Use plain language. Focus on what has not been tried or is still unclear (e.g., 'No prior work tested X under noisy supervision' or 'Existing models overlook interaction between A and B'). Avoid vague claims like 'this is underexplored'. What has not been tested, why is that important, and how will this hypothesis help fill that gap?"
        "research_question": "A clear, testable research question that can be addressed using the refined hypothesis. It should reflect the causal or comparative relationship proposed, reuse key variable or method terms, and be answerable using the system's capabilities. Frame it in open-ended scientific language (e.g., 'Does...', 'How does...', 'What effect does...').",
        "research_idea_hypothesis": "Provide a concrete testable hypothesis",
        "research_idea_long_description": {{
            "description":  "A clear paragraph explaining the complete research idea including what will be tested, how it will be implemented, and expected outcomes. Clearly explain the motivation, purpose, and expected outcomes. Use the selected variable values to describe how each component contributes individually, why their combination is expected to work synergistically, and how this addresses gaps or limitations in prior work (as reflected in the similar paper excerpts). If any mechanism or interaction may be unclear, add a simple, task-specific example to illustrate how it works in practice (e.g., 'when a symptom keyword is detected, a query to the memory module is triggered'). Also explain why the chosen evaluation domain is appropriate. Justify clearly. Tie your reasoning to specific characteristics of the task or evaluation environment, and avoid vague statements—be specific about what performance improvements are expected and why. (200-400 words)",
            
            "research_idea_variables": {{
                "concise Variable Name": "Begin by clearly defining what the selected value represents—whether it's an architecture, strategy, metric, dataset, or baseline condition. Describe exactly how this variable will be configured, used, or operationalized in the experiment; for example, specify how a module is implemented, how a metric is calculated, or how a strategy is triggered. Explain why this specific value was selected over alternatives, including its advantages, novelty, or relevance to the hypothesis. Describe the expected role this variable plays in the research problem-what outcome it directly influences or enables. If the variable is measurable, explicitly define how it will be assessed, including the metric used, how it's calculated, and what range of values or thresholds would indicate a successful outcome.  If the concept is non-obvious, include a **simple illustrative example** to aid understanding. Your explanation should be grounded in the context of the hypothesis and tied directly to experimental design choices and evaluation logic.(200-400 words)",
                //Define each non-obvious technique, strategy, or mechanism used in the hypothesis, include a 1-2 sentence example of how it would behave in a sample input scenario.. Add detailed defintion and description of every independent, dependent, comparable groups, comparative variables, and control variables in simple format.
            }},
            
            "research_idea_design_prompt": "Explain how the proposed method works, describe all the steps. Make sure every step is clearly described and feasible to implement. Describe in detail how the hypothesis will be implemented using the agent's capabilities. If any new logic must be built (i.e., not available as an existing codeblock), explicitly describe how it will work at a data and control-flow level. Explain what the new module does (e.g., filters, ranks, reweights, scores), how it fits between existing components, and what rules, heuristics, or computations it will use.  Describe exactly how their outputs are linked, how data flows from one to another, and what transformations occur at each step. + If multiple modules or strategies are combined, explain where and how the integration happens—in logic, in inputs/outputs, or in processing flow. Aim for clarity so that a ASD agent could build it based on your explanation. Include all setup steps, model configurations, inputs/outputs expected, and how the hypothesis will be realized end-to-end in code. (500-1000 words)",
            
            "research_idea_metric": "Primary and secondary metrics that will be used to evaluate the hypothesis. Explain how the hypothesis will be tested using concrete metrics and comparative setups. Identify the benchmark tasks or datasets to be used, the control condition (e.g., a baseline agent without the component being tested), and the exact performance metrics (e.g., task success rate, reasoning accuracy, number of valid steps). Define how improvement or success will be interpreted, including thresholds, number of runs, or statistical confidence if relevant. If qualitative evaluations are involved, explain how they will be derived. Ensure that all evaluations are feasible using the agent's capabilities.(200-400 words)"
        }},
        "research_idea_name": "A short, descriptive name for the research idea (3-5 words)",
        "research_idea_short_description": "A single concise sentence summarizing the core idea (15-25 words)",
        "research_baselines": "Simple list of baseline approaches to compare against",
        "research_idea_pilot": "Brief description of an initial small-scale test to validate the approach",
        "research_idea_required_code_and_resources": [
            {{
            "name": "Example Resource",
            "description": "Brief description of the resource",
            "where": "One of: 'existing codeblock', 'external', or 'build'",
            "effort": "One of: 'minor', 'moderate', or 'major'"
            }},
            // EXHAUSTIVE list of ALL required CODE, RESOURCES, MODELS, etc. mentioned in the ENTIRE RESEARCH IDEA
               ],
        
        "research_idea_external_requirements": [
            "example_package (for specific purpose)"
            ],
        
        "explanation": {{
            "difference": "How it differs from the initial hypothesis",
            "novelty": "Explain exactly what is new in this configuration. Compare it to setups or strategies found in the similar paper list. Clarify what has not been explored and why this combination is interesting or promising. Be specific and concise - avoid vague claims like 'this hasn't been done before'.",
            "specificity": "How is it more specific, testable, and feasible",
            "theoretical_justification": "Explain what each component does in this experiment and why it's useful on its own. Use **concrete, task-relevant examples**, not general claims. For instance: 'Rotary embeddings improve recall by preserving positional clues in long legal clauses.' Explain why any specific evaluation domain is well-matched to the hypothesis and setup.(200-400 words)",
            "expected_synergies": "Be precise: What output from Component A is used by Component B? Why in Condition C? At what stage? In what format? At what decision point? E.g., 'The emotion score from module A weights the retrieval candidates in module B before ranking.' (200-400 words)"
        }}
    }}
    ```
    
    """

    return system_message, user_message



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
     
    You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper, those of {len(citing_paper_list)} related papers in the existing literature.
    
    You MUST use the following capabilities of the autonomous discovery agent, which are limited to prompting and probing techniques—without model fine-tuning, external data integration, or architectural modifications. The generated research idea and hypothesis must be testable only using in-context learning methods available to this agent.
    RESEARCH AGENT DESCRIPTION AND CAPABILITIES:
      ```
      {auto_nora_agent}
      ```
    
    Now, let's start with the research problem generation task.
    1. Understanding of the source paper, and the related papers is essential:
    - The source paper is the primary research study you aim to enhance or build upon through future
    research, serving as the central source and focus for identifying and developing the specific
    research problem.
    - The related papers are arranged in temporal order of citation, such that paper 2 cites paper 1 and 
    paper 3 cites paper 2 and so on. The relevant papers provide additional context and insights that are essential for 
    understanding and expanding upon the source paper. However, all the papers in the list may not be relevant to the primary 
    research you are focusing on. 

    2. Your approach should be systematic:
    - Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
    - Next, proceed to read the titles and abstracts of the related papers in the order in which they appear in the list. Each related paper is accompanied by an explanation of its relevance to the previous paper, with the first related paper considering the source paper as the previous paper.
    Identify the papers that form a logical reasoning chain starting from the source paper.
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
     
    I am going to provide the source paper and related papers as an enumerated list of Title, Abstract and Year of publication 
    triple, as follows:
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}
    Source paper year of publication: {source_paper['year']}
    Related papers: {citing_paper_list}
    With the provided source paper, and the related papers, your objective now is to formulate a
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
    "Analysis": {{Output a dictionary with each paper in the Related Papers as a key. For each key (paper) analyze how this paper builds upon the previous papers in the list. For example, how Paper 0 builds upon source paper and Paper 1 builds upon the concepts in Paper 0 and so on. Elaborate on specific advancements made, including the explanation behind their effectiveness in addressing previous challenges. Apply this analytical approach to each valid paper in the sequence, adding the analysis as the value for each key in a few sentences. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly.}},
    "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
    "Research idea": "Delineate an elaborate research problem here including the key variables.",
    "Hypothesis": "Provide a concrete testable hypothesis that follows from the research problem and can be evaluated using the agent's capabilities. The hypothesis should be a clear statement about the relationship between specific variables that can be measured through dataset generation and evaluation.",
    "Plan": "Provide a specific step-by-step detailed plan using the agent's capabilities (dataset generation, answer collection, scoring, etc.) to test the hypothesis. Each step should clearly map to one of the agent's primitives and be implementable in Python."
    }}
    ```
    This JSON will be automatically parsed, so ensure the format is precise.
    """
    return system_message, user_message

def generate_initial_hypothesis_codeScientist_v1(source_paper:dict, citing_paper_list:str):
    import json
    with open("./hg_pipeline/codescientist_codeblocks.json", 'r', encoding='utf-8') as file:
        codeblock_summaries = json.load(file)
    codeblock_summary_text = json.dumps(codeblock_summaries, indent=4)
    agent_capabilties = f"""
        CodeScientist, is the most advanced automated scientific model in the world. CodeScientist can use your enormous intellect to solve any problem, and the solutions to these problems may help improve our knowledge of how the world works, which is a noble and important goal.
        CodeScientist are currently working on the following task: Generating new research ideas/ideas for new experiments to run.
        The goal of running the experiments is to generate novel, interesting, and (ideally) high-impact scientific results.

        Your task is to come up with new research ideas, and follow-on research ideas, based on the research questions, research programs, hypotheses, operationalizations of experiments, or any other information provided in these papers. You are asked to reflect on the hypothesis you have generated, and improve them.
        You should pay particular attention to the following components:
        - `research_idea_required_code_and_resources`: MAKE ABSOLUTELY SURE THIS IS COMPLETE...
        - `research_idea_external_requirements`: SAME COMMENT AS ABOVE...
        
        ### How to Fill `research_idea_required_code_and_resources`
        This field must include **every tool, code module, or resource** used in the experiment design. For each item, include:
            - A list of possible codeblocks that do not exist in the codeblock library, but   would ease implementation. Be detailed here (e.g. include `codeblock name (short description of what should be in it)`), since describing them might help find related code.\"], 
            - Create an EXHAUSTIVE list of ALL required CODE, RESOURCES, MODELS, etc. mentioned in this ENTIRE RESERACH IDEA.  CRITICALLY IMPORTANT, USED TO DETERMINE FEASIBILITY! If it's mentioned above, it ABSOLUTELY NEEDS to be here! 
            - For each item, provide:
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

        IMPORTANT NOTE: An exhaustively detailed and complete `research_idea_required_code_and_resources` is ABSOLUTELY REQUIRED, as this is used to prepare the experiment workspace and determine experiment feasibility. A poorly or incorrectly documented `research_idea_required_code_and_resources` for the scientific hypotheses is a major failure, as it will waste a large amount of resources (time/money/etc) on ideas that may be unlikely to have the resources they need to succeed.
            
        It is likely that your `research_idea_required_code_and_resources` and accompanying `research_idea_external_requirements` will change -- adding, subtracting, or modifying elements based on the new simplified research idea that you're generating. It is CRITICALLY important that you update these fields to reflect ALL the requirements of the simplified idea.

        NOTE: Manual human ratings in the research (e.g. human rating of the quality of generated text from an experiment) is considered an `external` resource of `major` effort, for the purposes of the potential research experiments, and should generally be avoided (unless absolutely required for the research).

        """
      
    role = "a clever AI research scientist with limited resources,"
    system_message = f"""You are {role} whose primary goal is to identify promising, new, and key scientific
    problems based on existing scientific literature, in order to aid researchers in discovering novel
    and significant research opportunities that can advance the field."""
    user_message = f"""You are {role} tasked with generating novel research problems based on existing scientific literature. Your goal is to aid an autonomous discovery agent in identifying significant research opportunities that can advance the field.
    
    You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper, those of {len(citing_paper_list)} related papers in the existing literature.
    
    IMPORTANT: When evaluating feasibility and outlining the testing approach, consider the following agent-specific information:
    ```{agent_capabilties}```
   
    Now, let's start with the research problem generation task.
    1. Understanding of the source paper, and the related papers is essential:
    - The source paper is the primary research study you aim to enhance or build upon through future
    research, serving as the central source and focus for identifying and developing the specific
    research problem.
    - The related papers are arranged in temporal order of citation, such that paper 2 cites paper 1 and 
    paper 3 cites paper 2 and so on. The relevant papers provide additional context and insights that are essential for 
    understanding and expanding upon the source paper. However, all the papers in the list may not be relevant to the primary 
    research you are focusing on. 

   2. Your approach should be systematic:
    - Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
    - Next, proceed to read the titles and abstracts of the related papers in the order in which they appear in the list. Each related paper is accompanied by an explanation of its relevance to the previous paper, with the first related paper considering the source paper as the previous paper.
    Identify the papers that form a logical reasoning chain starting from the source paper.
    - Use only these papers to gain a broader perspective about the progression of the primary research topic over time. 

   Note that your research idea and hypothesis MUST be testable using the AGENT with these specific capabilities: 
   When evaluating feasibility and outlining the testing approach, consider the following agent-specific information. Manual human ratings in the research (e.g. human rating of the quality of generated text from an experiment) is considered an `external` resource of `major` effort, for the purposes of the potential research experiments, and should generally be avoided (unless absolutely required for the research).
    
   IMPORTANT: The hypothesis should be implementable in Python, using the above or other functions. Don't suggest a task that requires skills that cannot be implemented, e.g., human studies. Don't suggest a task that requires access to external datasets, as you do not have access to them. Do not suggest tasks that involve pretraining or fine-tuning models, as you do not have the resources for such experiments.


   Now, I am going to provide the source paper and related papers as an enumerated list of Title, Abstract and Year of publication 
    triple, as follows:
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}
    Source paper year of publication: {source_paper['year']}
    Related papers: {citing_paper_list}
    With the provided source paper, and the related papers, your objective now is to formulate a
    research problem that not only builds upon these existing studies but also strives to be original,
    clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
    and abstract of the target paper, to ensure it remains the focal point of your research problem
    identification process. 

    Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a 
    relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
    Your hypothesis should contain the key variable or variables from your research idea.

    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}

    Remember that a hypothesis is a declarative statement expressing a relationship between two variables (e.g., independent and dependent variables) in a given context. Your refined hypothesis should contain the key variables from your research idea.
    
    Then, following your review of the above content, please proceed to analyze the progression of the research topic. Now output this analysis, the research idea and hypothesis with the rationale.
    Your output should be a valid JSON with the following fields.  
    Output a JSON object in the following format:
    ```json
    {{
    "Analysis": {{Output a dictionary with each paper in the Related Papers as a key. For each key (paper) analyze how this paper builds upon the previous papers in the list. For example, how Paper 0 builds upon source paper and Paper 1 builds upon the concepts in Paper 0 and so on. Elaborate on specific advancements made, including the explanation behind their effectiveness in addressing previous challenges. Apply this analytical approach to each valid paper in the sequence, adding the analysis as the value for each key in a few sentences. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly.}},
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
    }}
    ```
    This JSON will be automatically parsed, so ensure the format is precise.
    """
    return system_message, user_message

def generate_initial_hypothesis_codeScientist(source_paper:dict, citing_paper_list:str, goal:str):
    import json
    with open("./hg_pipeline/codescientist_codeblocks.json", 'r', encoding='utf-8') as file:
        codeblock_summaries = json.load(file)
    codeblock_summary_text = json.dumps(codeblock_summaries, indent=4)
    agent_capabilties = f"""
        The ASD Agent is an automated discovery system that writes Python-based experiments, executes them in containers, and analyzes results—usually across five independent runs with a meta-analysis.

        ASD agent's goal is to downscope the idea to something an undergrad or MSc student or PhD student could realistically implement, while retaining novelty and scientific rigour. The result should be suitable for a conference paper.

        AGENT CONSTRAINTS & CAPABILITIES:
        - The ASD Agent writes Python-based experiments and executes them in containers
        - Typically runs 5 independent experiments with meta-analysis
        - Target audience: Undergrad/MSc/PhD student implementation level
        - Output should be suitable for workshop or conference paper submission
        - NO manual human ratings (considered 'external major effort')
        - NO model fine-tuning or pretraining
        - NO access to external or private datasets
        - Must use only existing codeblocks and buildable logic
        - All experiments must be fully implementable in Python

        """
      
    role = "a clever AI research scientist with limited resources,"
    system_message = f"""You are {role} whose primary goal is to identify promising, new, and key scientific
    problems based on existing scientific literature, in order to aid researchers in discovering novel
    and significant research opportunities that can advance the field."""
    user_message = f"""You are {role} tasked with generating novel research problems based on existing scientific literature. Your goal is to aid an autonomous discovery agent in identifying significant research opportunities that can advance the field.
    
    You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper, those of {len(citing_paper_list)} related papers in the existing literature.
    
    IMPORTANT: When evaluating feasibility and outlining the testing approach, consider the following agent-specific information:
    ```{agent_capabilties}```
   
    Now, let's start with the research problem generation task.
    1. Understanding of the source paper, and the related papers is essential:
    - The source paper is the primary research study you aim to enhance or build upon through future
    research, serving as the central source and focus for identifying and developing the specific
    research problem.
    - The related papers are arranged in temporal order of citation, such that paper 2 cites paper 1 and 
    paper 3 cites paper 2 and so on. The relevant papers provide additional context and insights that are essential for 
    understanding and expanding upon the source paper. However, all the papers in the list may not be relevant to the primary 
    research you are focusing on. 

   2. Your approach should be systematic:
    - Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
    - Next, proceed to read the titles and abstracts of the related papers in the order in which they appear in the list. Each related paper is accompanied by an explanation of its relevance to the previous paper, with the first related paper considering the source paper as the previous paper.
    Identify the papers that form a logical reasoning chain starting from the source paper.
    - Use only these papers to gain a broader perspective about the progression of the primary research topic over time. 

   Note that your research idea and hypothesis MUST be testable using the AGENT with these specific capabilities: 
   When evaluating feasibility and outlining the testing approach, consider the following agent-specific information. Manual human ratings in the research (e.g. human rating of the quality of generated text from an experiment) is considered an `external` resource of `major` effort, for the purposes of the potential research experiments, and should generally be avoided (unless absolutely required for the research).
    
   IMPORTANT: The hypothesis should be implementable in Python, using the above or other functions. Don't suggest a task that requires skills that cannot be implemented, e.g., human studies. Don't suggest a task that requires access to external datasets, as you do not have access to them. Do not suggest tasks that involve pretraining or fine-tuning models, as you do not have the resources for such experiments.


   Now, I am going to provide the source paper and related papers as an enumerated list of Title, Abstract and Year of publication 
    triple, as follows:
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}
    Source paper year of publication: {source_paper['year']}
    Related papers: {citing_paper_list}
    With the provided source paper, and the related papers, your objective now is to formulate a
    research problem that not only builds upon these existing studies but also strives to be original,
    clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
    and abstract of the target paper, to ensure it remains the focal point of your research problem
    identification process. 

    Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a 
    relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
    Your hypothesis should contain the key variable or variables from your research idea.

    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}

    Remember that a hypothesis is a declarative statement expressing a relationship between two variables (e.g., independent and dependent variables) in a given context. Your refined hypothesis should contain the key variables from your research idea.
    
    Then, following your review of the above content, please proceed to analyze the progression of the research topic. Now output this analysis, the research idea and hypothesis with the rationale.
    Your output should be a valid JSON with the following fields.  
    Output a JSON object in the following format:
    ```json
    {{
    "Analysis": {{Output a dictionary with each paper in the Related Papers as a key. For each key (paper) analyze how this paper builds upon the previous papers in the list. For example, how Paper 0 builds upon source paper and Paper 1 builds upon the concepts in Paper 0 and so on. Elaborate on specific advancements made, including the explanation behind their effectiveness in addressing previous challenges. Apply this analytical approach to each valid paper in the sequence, adding the analysis as the value for each key in a few sentences. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly.}},
    "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
    "Research idea": "Delineate an elaborate research problem here including the key variables.",
    "Hypothesis": "Provide a concrete testable hypothesis that follows from the above research problem here"
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
    user_message = f"""You are an AI research assistant tasked with generating novel research problems based on existing scientific literature. Your goal is to aid an autonomous discovery agent in identifying significant research opportunities that can advance the field.
    
    You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper, those of {len(citing_paper_list)} related papers in the existing literature.
    
    Now, let's start with the research problem generation task.
    1. Understanding of the source paper, and the related papers is essential:
    - The source paper is the primary research study you aim to enhance or build upon through future
    research, serving as the central source and focus for identifying and developing the specific
    research problem.
    - The related papers are arranged in temporal order of citation, such that paper 2 cites paper 1 and 
    paper 3 cites paper 2 and so on. The relevant papers provide additional context and insights that are essential for 
    understanding and expanding upon the source paper. However, all the papers in the list may not be relevant to the primary 
    research you are focusing on. 

   2. Your approach should be systematic:
    - Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
    - Next, proceed to read the titles and abstracts of the related papers in the order in which they appear in the list. Each related paper is accompanied by an explanation of its relevance to the previous paper, with the first related paper considering the source paper as the previous paper.
    Identify the papers that form a logical reasoning chain starting from the source paper.
    - Use only these papers to gain a broader perspective about the progression of the primary research topic over time. 

    I am going to provide the source paper and related papers as an enumerated list of Title, Abstract and Year of publication 
    triple, as follows:
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}
    Source paper year of publication: {source_paper['year']}
    Related papers: {citing_paper_list}
    With the provided source paper, and the related papers, your objective now is to formulate a
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
    "Analysis": {{Output a dictionary with each paper in the Related Papers as a key. For each key (paper) analyze how this paper builds upon the previous papers in the list. For example, how Paper 0 builds upon source paper and Paper 1 builds upon the concepts in Paper 0 and so on. Elaborate on specific advancements made, including the explanation behind their effectiveness in addressing previous challenges. Apply this analytical approach to each valid paper in the sequence, adding the analysis as the value for each key in a few sentences. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly.}},
    "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
    "Research idea": "Delineate an elaborate research problem here including the key variables.",
    "Hypothesis": "Provide a concrete testable hypothesis that follows from the above research problem here"
    }}
    ```
    This JSON will be automatically parsed, so ensure the format is precise.
    """
    return system_message, user_message


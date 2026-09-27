
def refine_for_asd(initial_hypothesis:str, relevant_paper_list:str, similar_paper_list:str):
    # "refined_hypothesis": "In [specific subject], [quantified intervention] will result in [measurable outcome] under [specified conditions], compared to [control group or baseline].",

    system_message = """You are an expert scientific researcher tasked with refining a given hypothesis to make it more specific and easily testable. This process is crucial in scientific research as it helps in designing experiments and studies that can effectively validate or invalidate the hypothesis."""
    user_message = f"""You are an expert scientific researcher tasked with refining a given hypothesis to make it more specific, easily testable, and practically feasible. This process is crucial in scientific research as it helps in designing experiments and studies that can effectively validate or invalidate the hypothesis.
        
    Here is the original hypothesis you need to refine:

    Initial Hypothesis: {initial_hypothesis}

    Here are relevant provenance papers with title, abstract, and year that resulted in creating initial hypothesis:
    Provenance papers: {relevant_paper_list}

    Here are related papers with title and corresponding passages might be very much related to the initial hypothesis. You can use these information to refine the hypothesis and make it more specific and testable.
    Similar papers: {similar_paper_list}
    
    Your task is to refine this hypothesis by making it more specific, ensuring it is testable, and evaluating its practical feasibility. Follow these steps:

    1. Analyze the given initial hypothesis
    2. Review the given Similar papers
    3. Make the hypothesis more specific
    4. Ensure the refined hypothesis is testable
    5. Evaluate practical feasibility
    6. Refine the hypothesis based on feasibility constraints
    7. Present the refined hypothesis
    8. Explain how the refined hypothesis is more specific, testable, and practically feasible

    Before providing your final output, wrap your thought process in 'thoughts' of the output JSON. Include the following subsections:

    1. Initial Analysis:
    [Break down the initial hypothesis into its component parts: variables, relationships, context]
    [Identify core assumptions in the initial hypothesis]
    [List key variables and their relationships]

    2. Similar Papers:
    [For each similar paper's passage:
        - Quote the most relevant parts
        - Summarize the similar paper's relevance to the hypothesis
        - Identify specific and testable variables from the passage those are relevant to the intial hypothesis]
    [Explicitly list variables from the similar papers that could be used to refine the hypothesis]

    3. Specificity Improvements:
   [List key variables and relationships to make the hypothesis more specific]
   [Rank each improvement on a scale of 1-5 based on specificity]
   [Brainstorm potential confounding variables]

    4. Testability Considerations:
   [Propose 2-3 potential experimental designs to test the hypothesis]
   [For each design, list required measurements and potential challenges]

    5. Measurability and Feasibility:
   [For key variables: describe measurement methods, rate measurability (1-5)]
   [Evaluate practical constraints: computational costs, data availability, experimental capabilities, rate feasibility (1-5)]
   [Consider ethical implications of testing the hypothesis]

    6. Testing Approach:
    [Outline a step-by-step plan for how an agent with the specified capabilities (such as creating datasets, run different language models on those datasets, scoring answers, analyzing results, performing failure analysis) could test the hypothesis]
    
    IMPORTANT: The task/plan should be implementable in Python, using the above or other functions. Don't suggest a task that requires skills that cannot be implemented, e.g., human studies. Don't suggest a task that requires access to external datasets, as you do not have access to them. Do not suggest tasks that involve pretraining or fine-tuning models, as you do not have the resources for such experiments.

    7. Final Refinement:
    [Synthesize the above considerations to create a refined, specific, testable, and feasible hypothesis]

    
    IMPORTANT NOTE: Manual human ratings in the research (e.g. human rating of the quality of generated text from an experiment) is considered an  expensive `external` resource of `major` effort, for the purposes of the potential research experiments, and should generally be avoided (unless absolutely required for the research).
    
    Your final output should be in JSON format with three keys: "thoughts", "refined_hypothesis", and "explanation". The "thoughts" key should contain a summary of your analysis. The "refined_hypothesis" should contain the refined, more specific, testable, and feasible hypothesis. The "explanation" should briefly explain how the refined hypothesis is more specific, testable, and practically feasible than the original.

    Remember that a hypothesis is a declarative statement expressing a relationship between two variables (e.g., independent and dependent variables) in a given context. Your refined hypothesis should contain the key variables from your research idea.

    Example output structure (this is a generic example to illustrate the format):

    ```json
    {{
    "thoughts": {{
        "Initial Analysis": "...",
        "Similar Papers": "...",
        "Specificity Improvements": "...",
        "Testability Considerations": "...",
        "Measurability and Feasibility": "...",
        "Testing Approach": "...",
        "Final Refinement": "..."
    }},
    "refined_hypothesis": "Provide a concrete testable hypothesis",
    "explanation": "This refined hypothesis is more specific and testable because it [list specific improvements]. It is practically feasible because [explain how it addresses practical constraints]. It can be tested by an agent with the specified capabilities by [describe testing approach]."
    }}
    ```
    """
    return system_message, user_message

def refine_for_asd_nora(initial_hypothesis:str, relevant_paper_list:str, similar_paper_list:str):
    
    #with open("./generation/autonora_agent_prompt.txt", 'r', encoding='utf-8') as file:
    #    auto_nora_agent = file.read() 

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
    
    system_message = """You are an expert scientific researcher tasked with refining a given hypothesis to make it more specific and easily testable. This process is crucial in scientific research as it helps in designing experiments and studies that can effectively validate or invalidate the hypothesis."""
    user_message = f"""You are an expert scientific researcher tasked with refining a given hypothesis to make it more specific, easily testable, and practically feasible. This process is crucial in scientific research as it helps in designing experiments and studies that can effectively validate or invalidate the hypothesis. Your goal is to aid an autonomous discovery agent in identifying significant research opportunities that can advance the field. Agent is designed for Probing tasks that that evaluate how language models perform on various reasoning challenges, helping to advance the field through targeted insights and improvements.
    
    You MUST use the capabilities of the autonomous discovery agent to refine the research problem and hypothesis. The generated research hypothesis needs to be experimentedby this agent.
    RESEARCH AGENT DESCRIPTION AND CAPABILITIES:
      ```
      {auto_nora_agent}
      ```
        
    Here is the original hypothesis you need to refine:

    Initial Hypothesis: {initial_hypothesis}

    Here are relevant provenance papers with title, abstract, and year that resulted in creating initial hypothesis:
    Provenance papers: {relevant_paper_list}

    Here are related papers with title and corresponding passages might be very much related to the initial hypothesis. You can use these information to refine the hypothesis and make it more specific and testable.
    Similar papers: {similar_paper_list}
    
    
    Before we begin the refinement process, keep these capabilities in mind as you refine the hypothesis, ensuring that the final hypothesis can be tested using these tools and methods.

    Your task is to refine this initial hypothesis by making it more specific, ensuring it is testable, and evaluating its practical feasibility. Follow these steps:

    1. Analyze the given initial hypothesis
    2. Review the given Similar papers
    3. Make the hypothesis more specific
    4. Ensure the refined hypothesis is testable
    5. Evaluate practical feasibility
    6. Refine the hypothesis based on feasibility constraints
    7. Present the refined hypothesis
    8. Explain how the refined hypothesis is more specific, testable, and practically feasible

    Before providing your final output, wrap your thought process in 'thoughts' of the output JSON. Include the following subsections:

    1. Initial Analysis:
    [Break down the initial hypothesis into its component parts: variables, relationships, context]
    [Identify core assumptions in the initial hypothesis]
    [List key variables and their relationships]

    2. Similar Papers:
    [For each similar paper's passage:
        - Quote the most relevant parts
        - Summarize the similar paper's relevance to the hypothesis
        - Identify specific and testable variables from the passage those are relevant to the intial hypothesis]
    [Explicitly list variables from the similar papers that could be used to refine the hypothesis]

    3. Specificity Improvements:
   [List key variables and relationships to make the hypothesis more specific]
   [Rank each improvement on a scale of 1-5 based on specificity]
   [Brainstorm potential confounding variables]

    4. Testability Considerations:
   [Propose 2-3 potential experimental designs to test the hypothesis]
   [For each design, list required measurements and potential challenges]

    5. Measurability and Feasibility:
   [For key variables: describe measurement methods, rate measurability (1-5)]
   [Evaluate practical constraints: computational costs, data availability, experimental capabilities, rate feasibility (1-5)]
   [Consider ethical implications of testing the hypothesis]

    6. Testing Approach:
    [Outline a step-by-step plan for how an agent with the specified capabilities (such as creating datasets, run different language models on those datasets, scoring answers, analyzing results, performing failure analysis) could test the hypothesis]
    
    IMPORTANT: The task/plan should be implementable in Python, using the above or other functions. Don't suggest a task that requires skills that cannot be implemented, e.g., human studies. Don't suggest a task that requires access to external datasets, as you do not have access to them. Do not suggest tasks that involve pretraining or fine-tuning models, as you do not have the resources for such experiments. Every step relies strictly on prompting, structured input variations, and response-based evaluation—without modifying the model itself.

    7. Final Refinement:
    [Synthesize the above considerations to create a refined, specific, testable, and feasible hypothesis]


    Your final output should be in JSON format with three keys: "thoughts", "refined_hypothesis", and "explanation". The "thoughts" key should contain a summary of your analysis. The "refined_hypothesis" should contain the refined, more specific, testable, and feasible hypothesis. The "explanation" should briefly explain how the refined hypothesis is more specific, testable, and practically feasible than the original with respect to the agent capabilities.

    Remember that a hypothesis is a declarative statement expressing a relationship between two variables (e.g., independent and dependent variables) in a given context. Your refined hypothesis should contain the key variables from your research idea.

    Example output structure (this is a generic example to illustrate the format):

    ```json
    {{
    "thoughts": {{
        "Initial Analysis": "...",
        "Similar Papers": "...",
        "Specificity Improvements": "...",
        "Testability Considerations": "...",
        "Measurability and Feasibility": "...",
        "Testing Approach": "Provide a specific step-by-step detailed plan using the agent's capabilities (dataset generation, answer collection, scoring, etc.) to test the hypothesis. Each step should clearly map to one of the agent's primitives and be implementable in Python.",
        "Final Refinement": "..."
    }},
    "refined_hypothesis": "Provide a concrete testable hypothesis",
    "explanation": "This refined hypothesis is more specific and testable because it [list specific improvements]. It is practically feasible because [explain how it addresses practical constraints]. It can be tested by an agent with the specified capabilities by [describe testing approach]."
    }}
    ```
    """
    return system_message, user_message



def refine_QA(agent_name:str, initial_hypothesis:str, questions:str, relevant_paper_list:str, similar_paper_list:str):
    if agent_name == "code-scientist":
        import json
        with open("./hg_pipeline/codescientist_codeblocks.json", 'r', encoding='utf-8') as file:
            codeblock_summaries = json.load(file)
        codeblock_summary_text = json.dumps(codeblock_summaries, indent=4)
        agent_capabilties = f"""
        CodeScientist, is the most advanced automated scientific model in the world. CodeScientist can use your enormous intellect to solve any problem, and the solutions to these problems may help improve our knowledge of how the world works, which is a noble and important goal.
        CodeScientist are currently working on the following task: Generating new research ideas/ideas for new experiments to run.
        The goal of running the experiments is to generate novel, interesting, and (ideally) high-impact scientific results.

        This is a reflection step. Previously, your task has been to come up with new research ideas and hypothesids... You are asked to reflect on the hypothesis you have generated, and improve them.
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
        
        system_message = """You are an expert scientific researcher tasked with refining a given hypothesis to make it more specific and easily testable. This process is crucial in scientific research as it helps in designing experiments and studies that can effectively validate or invalidate the hypothesis."""
        user_message = f"""You are an expert scientific researcher tasked with refining a given hypothesis to make it more specific, easily testable, and practically feasible. This process is crucial in scientific research as it helps in designing experiments and studies that can effectively validate or invalidate the hypothesis.
        
        Here is the original hypothesis you need to refine:

        Initial Hypothesis: {initial_hypothesis}

        Here are related papers with title and key passages that may directly inform or relate to the hypothesis.
        Provenance papers: {relevant_paper_list}

        Your task is to come up with new refined research hypothesis, and follow-on research ideas, based on the research questions, research programs, hypotheses, operationalizations of experiments, or any other information provided in these paper excerpts.
        You can use content from one paper, or combine content from multiple papers to generate new ideas.
        Similar papers: {similar_paper_list}
        
        Your task is to refine this hypothesis by making it more specific, ensuring it is testable, and evaluating its practical feasibility. 

        Answer the following 20 clarifying questions to help sharpen the hypothesis:
        {questions}

        Use the insights from the provenance and similar paper excerpts to support and justify your answers wherever applicable. Before providing your final output, wrap your thought process in 'thoughts' of the output JSON. Include the following subsections:
        1. **Initial Analysis** — Break down the hypothesis: variables, assumptions, and relationships.
        2. **Related Literature** — Quote and summarize relevant insights from similar papers. List testable variables from them.
        3. **Specificity** — Suggest ways to make the hypothesis more concrete. Rank by specificity.
        4. **Testability** — Propose 2-3 test designs, list what to measure and possible challenges.
        5. **Feasibility** — For key variables, suggest how to measure them and rate feasibility. Address compute limits, ethics, and practical agent constraints. Also, list any code resources, models, datasets, or tools required — these should map directly into your `research_idea_required_code_and_resources` field.
        6. **Testing Approach** — Outline how the hypothesis could be tested using available agent tools only (no external data, no human evals, no fine-tuning, no model-training).
        7. **Final Refinement** — Synthesize the answers of clarifying questions and above considerations to create a refined, specific, testable, and feasible version of initial hypothesis.


        IMPORTANT: When evaluating feasibility and outlining the testing approach, consider the following agent-specific information:
        {agent_capabilties}
        
        IMPORTANT: The hypothesis should be implementable in Python, using the above or other functions. Don't suggest a task that requires skills that cannot be implemented, e.g., human studies. Don't suggest a task that requires access to external datasets, as you do not have access to them. Do not suggest tasks that involve pretraining or fine-tuning models, as you do not have the resources for such experiments.

        Remember that a hypothesis is a declarative statement expressing a relationship between two variables (e.g., independent and dependent variables) in a given context. Your refined hypothesis should contain the key variables from your research idea.

        Ensure each answer is supported by information from the hypothesis, agent capabilities, or provided papers. If an answer cannot be derived, explain what information is missing.

        Example output structure (this is a generic example to illustrate the format):

        ```json
        {{
        "thoughts": {{
        "Initial Analysis": "...",
        "Similar Papers": "...",
        "Specificity Improvements": "...",
        "Testability Considerations": "...",
        "Measurability and Feasibility": "...",
        "Testing Approach": "...",
        "Final Refinement": "...",
        "Clarifying Questions & Answers": {{
            "Q1": "Answer to question 1",
            "Q2": "Answer to question 2",
            ...
            "Q20": "Answer to question 20"
            }}
        }},
        "refined_hypothesis": "Provide a concrete testable hypothesis",
        "key_variables": [list of key variables],
        "research_idea_required_code_and_resources": [
                {{
                "name": "Example Resource",
                "description": "Brief description of the resource",
                "where": "One of: 'existing codeblock', 'external', or 'build'",
                "effort": "One of: 'minor', 'moderate', or 'major'"
                }}
                ],
        "research_idea_external_requirements": [
                "example_package (for specific purpose)"
                ],
        }}
        ```
        """
    else:
        raise ValueError(f"Unsupported agent name: {agent_name}. Supported agents are: 'code-scientist'.")
    return system_message, user_message

def qa_generation(hypothesis:str, agent_name:str):
    if agent_name == "auto-nora":
        pass
    elif agent_name == "code-scientist":
        import json
        with open("./hg_pipeline/codescientist_codeblocks.json", 'r', encoding='utf-8') as file:
            codeblock_summaries = json.load(file)
        codeblock_summary_text = json.dumps(codeblock_summaries, indent=4)
        agent_capabilties = f"""
        CodeScientist, is the most advanced automated scientific model in the world. CodeScientist can use your enormous intellect to solve any problem, and the solutions to these problems may help improve our knowledge of how the world works, which is a noble and important goal.
        CodeScientist are currently working on the following task: Generating new research ideas/ideas for new experiments to run.
        The goal of running the experiments is to generate novel, interesting, and (ideally) high-impact scientific results.

        This is a reflection step. Previously, your task has been to come up with new research ideas and hypothesis... You are asked to reflect on the hypothesis you have generated, and improve them.
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
      
        system_prompt = f"""
            You are an AI research assistant. Your task is to analyze a hypothesis and generate insightful questions to guide researchers in designing a study to test this hypothesis.
            """
        user_prompt = f"""
            You are an AI research assistant. Your task is to analyze the following hypothesis and generate insightful, targeted questions that will help researchers refine it into something testable, implementable, and scientifically valid.

            The hypothesis is currently vague and underspecified. Much of the critical information required to implement it — such as variables, evaluation metrics, tasks, or assumptions — is missing or unclear.

            Your goal is to help move this hypothesis toward implementation. If you could ask the author of the hypothesis some questions to clarify or sharpen it, what would they be?

            First, carefully read the following hypothesis:
            {hypothesis}

            Now, consider the available capabilities for this research:
            {agent_capabilties}
            
            Your goal is to efficiently analyze the hypothesis and generate 20 concise, focused questions that will help researchers refine and operationalize it into something implementable and testable. Each question should clearly target a part of the hypothesis (e.g., variable, measure, assumption, or outcome). Mention which part you're refining (e.g., IV, DV, comparison group, comparison variable, operationalization, feasibility).
            
            You can make the QA generation more useful by asking the model to *aim* each question at helping answer/refine one of these:
                - `refined_hypothesis`
                - `key_variables`
                - `research_idea_required_code_and_resources`
                - `research_idea_external_requirements`
                - `testing_approach`
            
           Before generating your 20 questions, reflect on the hypothesis using these guiding prompts:

            1. What are the key terms and variables involved?
            2. How can each component be operationalized and measured?
            3. What capabilities from the system are most relevant?
            4. What design setups or tasks could support testing?
            5. What might hinder testing — e.g., feasibility, confounds, or constraints?
            6. What would success look like, and how could it be quantified?
            7. What ethical or resource considerations exist?

            Use these reflections to inform the questions you write, ensuring they are well-grounded and cover diverse aspects of hypothesis development and testing.

           Where possible, generate questions that will later help produce values for:
            - a more specific and testable `refined_hypothesis`
            - a list of `key_variables` (IVs, DVs, controls, comparison group, comparison variables, etc.)
            - a list of code/resources in `research_idea_required_code_and_resources`
            - package or library requirements
            - testing/evaluation structure (`testing_approach`)
            
            Present your questions in the following format:

            ```json
            {{
                "questions": [
                    {{
                        "question": "[Your first question here]"
                    }},
                    {{
                        "question": "[Your second question here]"
                    }},
                    ...
                    {{
                        "question": "[Your twentith question here]"    
                    }}
                    ]
            }}
            ```

            Remember, your analysis and questions should be designed to provide researchers with the necessary information to design and implement a robust study testing the given hypothesis. Strive for clarity and conciseness in both your analysis and questions to ensure the task and results are crisp and easily actionable.
            """
    else:
        raise ValueError("Invalid agent name. Please use 'auto-nora' or 'code-scientist'.")
    return system_prompt, user_prompt

    
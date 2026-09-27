def enrich_variable_value(value_name, variable_name, similar_retrieved_papers):
    system_message = """You are a research assistant specializing in technical implementation details.

    Your task is to elaborate and enrich a specific variable value used in a scientific research context. The goal is to generate a precise, implementation-level description of how this value is realized in practice.
    """
    user_message = f"""You are a research assistant specializing in technical method extraction. Your task is to elaborate on a specific implementation value extracted from a scientific paper. You will provide a technically rich and concise description that includes the underlying mechanisms, architectures, metrics, or experimental setups used to realize the value.

    You will be given main two pieces of information:
    1. The name of the value to elaborate on
    2. The key variable that this value is relevant to

    Here is the value name to elaborate:
    {value_name}

    Context: This value was extracted from a paper relevant to the following variable:
    {variable_name}

    Your elaboration should be specific and use implementation-relevant language. Do not simply repeat the name or describe it vaguely. Instead, provide detailed technical information about how this value is implemented or used in practice.
    
    Here are relevant paper excerpts to guide your response:
    {similar_retrieved_papers}
    
    ## to do: target to specific value field in the next one
    In your elaboration, include **as many of the following details as the text allows**:
    - Architecture or model used (e.g., transformer, GPT, story graph)
    - Hyperparameters or training settings (e.g., learning rate, temperature, top-k)
    - Implementation methods (e.g., prompt templates, retrieval techniques, scoring functions)
    - Evaluation metrics (e.g., accuracy, user ratings, engagement frequency)
    - Experimental conditions (e.g., number of participants, dataset used, baseline comparisons)
    - Any specific mechanics (e.g., branching storylets, memory modules, dialogue control)
    - Optional: any results or findings showing impact or performance

    Format your response in JSON as follows:
    ```json
    {{
    "value_name": "value_name",
    "specific_details": "Your technical elaboration here"
    }}
    ```

    Remember to focus on the technical implementation details and be as specific as possible within the context of the given value and key variable.
    """
    return system_message, user_message

def get_key_variables(hypothesis, similar_retreived_papers):
    system_message = """You are a scientific research expert tasked with analyzing hypotheses, identifying key variables, extracting variable value options from related papers, and objectively considering variable values based on on specificity, testability, and feasibility. Provide structured, concise, and clear outputs."""
    user_message = f"""You are an expert scientific researcher tasked with analyzing a given hypothesis and extracting key information from related papers. Your goal is to identify key variables, their possible value options, and rate these options for specificity, testability, and feasibility.

    Here is the hypothesis you need to analyze:

    Hypothesis: {hypothesis}

    To assist you in this task, here are related papers with titles and corresponding passages that might be relevant to the given hypothesis:

    Similar Paper Context:    
    {similar_retreived_papers}

    Your task is to analyze this hypothesis and the related papers to extract key variables. Follow these steps:

    1. Analyze the hypothesis explicitly and systematically extract key variables:
    - Clearly identify every explicitly mentioned variable or design-level choices within the hypothesis as a distinct key variable. This includes quantifiable variables and design-level choices.
    - Convert any implicit or abstract concepts (e.g., performance, reliability, robustness) into clearly defined and measurable variables or implementable design choices. Do not include vague or unmeasurable conceptual ideas unless they are clearly defined in operational terms and when they are central to the hypothesis.
    - Ensure key variables have either measurable, quantifiable properties, such as "Model Training Time (seconds)," "Error Rate (%)," or "Knowledge Retention Score." Or a design choice that affects implementation or evaluation (e.g., "Use of pretraining dataset X", "Fine-tuning vs. zero-shot prompting")
        - Provide a precise, measurable definition (one sentence) for each identified key variable. Explicitly define how each key variable should be measured, stating exact metrics, evaluation criteria, or assessment methods clearly and concisely.
        - For design choices, define what the choice is, its implications, and how it would be implemented or varied in an experiment.
        - Example design choices include memory architecture (e.g., episodic memory, fact-memory modules), prompt strategy (e.g., few-shot, chain-of-thought), retrieval method (e.g., top-k, semantic retrieval), narrative control mechanism (e.g., branching storylets, story graphs), or model integration choices (e.g., use of fine-tuned GPT-3 vs. GPT-4). These should be specific and tied to actual implementation decisions that can affect the system behavior or experimental outcome.    
    - Include relevant experiment-level factors (e.g., dataset choice, baseline models, training configurations) as variables if they impact testing the hypothesis
    - Do not omit any explicitly mentioned concept from the hypothesis.

    2. Review the similar papers:
    - Extract relevant quotes.
    - Analyze how each quote relates to the hypothesis.
    - Identify specific and testable variables or design choices from the quotes.

    3. For each key variable:
    - Clearly define how it should be measured or implemented (in `specific_details`).
    - Indicate the type of variable using "type": "measurable" or "type": "design-choice" in the output.
    - Determine whether the key variable is **explicitly mentioned** in related work or if it is inferred.
        a. Mark variables found in paper excerpts with their exact paper title and include page/section if available
        b. Mark variables as 'LLM-recommended' only if not supported by provided papers
    - In specific_details, provide:
        a. For measurable variables: metrics, methods of evaluation, potential value ranges, and example benchmarks
        b. For design choices: the specific options or configurations, how they can be varied, how they impact implementation, and any relevant examples or baselines
    
    Remember to focus solely on analyzing the given hypothesis, identifying key variables, and extracting specific value options from the similar papers. Do not attempt to refine or improve the hypothesis.

    Your final output should be structured clearly and explicitly to enhance interpretability. Follow this JSON format strictly:

        
    ```json
    {{
    "hypothesis": "state the hypothesis as given",
    "list_key_variables": ["variable_1", "variable_2", "..."],
    "key_variables": [
        {{
            "name": "concise Variable Name",
            "source_paper": "Paper Title or 'LLM-recommended'",
            "type": "measurable" or "design-choice",
            "definition": "Precise, measurable definition of the variable.",
            "importance": "Brief explanation of why this variable matters to the hypothesis.",
            "specific_details": "Detailed information on measurement techniques, potential value ranges, and specific examples of implementation, elaborated with information from related passages."
        }}
    ]
    }}
    ```
    """
    return system_message, user_message


def single_variable_space(hypothesis, variable_info, similar_retrieved_papers):
    system_message = """You are a scientific research expert tasked with analyzing hypotheses and extracting possible specific values for key variables from related papers. 

    Your analysis should prioritize precision and relevance over quantity. Focus on extracting values that are explicitly stated or can be reasonably inferred from the provided paper excerpts. Apply critical thinking to determine the most appropriate variable values for the given hypothesis context.

    When assigning confidence levels:
    - High: Values explicitly mentioned in papers with detailed implementation information available
    - Medium: Values that can be reasonably inferred from the papers but aren't explicitly stated
    - Low: Values that may be applicable based on general domain knowledge but aren't mentioned in papers

    Ensure all extracted information maintains scientific integrity and accurately represents the source material."""

    user_message = f"""Your goal is to identify specific variable values for a given variable from a given hypothesis and the provided relevant literature excerpts as context.

    Here is the hypothesis you need to analyze:

    Hypothesis: {hypothesis}

    Now, the value options you need to extract is for the key variable provided here:
        
    Key variable information: {variable_info}
        
    To assist you in this task, here are related papers with titles and corresponding passages that might be relevant to the given hypothesis and the key variables:

    Similar Paper Context:    
    {similar_retrieved_papers}

    Your task is to analyze the hypothesis and related papers to extract **implementation-relevant, distinct, and non-redundant** values for the given key variable. Follow these rules:
    ------
    1. **Determine the nature of the key variable**
        - First, determine if the key variable is itself a metric/outcome measure (e.g., "Task Completion Rate", "Accuracy")
        - If it IS a metric/outcome measure:
            a. DO NOT extract implementation environments or frameworks as values
            b. Instead, identify specific and quantifiable alternative metrics that could directly replace this key variable
            c. Examples: Instead of "Accuracy", alternatives include Precision, Recall, F1-score, etc.
       - If it is NOT a metric/outcome measure: Identify a minimum of 15 distinct variable values from the papers
            - Extract values that are (1) specific design choices (e.g. architectures, training settings, prompt formats, toolkits), (2) implementation strategies (e.g. planning mechanisms, memory structures), or (3) quantifiable outcome metrics where applicable. 
            - **For ALL identified values/alternatives**
                a. Prioritize the most relevant values to the hypothesis if there are many (>15) options
                b. Mark variable values found in paper excerpts with their exact paper title and include page/section if available
                c. Mark values as `LLM-recommended` only if not clearly supported by provided papers
                d. Prioritize values directly sourced from provided papers over LLM-generated suggestions
                e. Assign confidence levels using these criteria:
                    - High: Values explicitly mentioned in papers with detailed implementation information available
                    - Medium: Values that can be reasonably inferred from the papers but aren't explicitly stated
                    - Low: Values that may be applicable based on general domain knowledge but aren't explicitly mentioned in papers
                f. Include concrete examples or parameter ranges for specificity
                g. DO NOT extract vague concepts, AI frameworks, or general methodologies (e.g., "Reinforcement Learning") as variable values. 
                h. Do not extract values that are purely numerical performance metrics (e.g., "67% task completion", "80% accuracy") — even if they differ across models or setups. Your task is to extract design decisions, implementation structures, and qualitative strategies — not performance outcomes or numeric results. Values like "75% task success" or "F1 score 0.88" are not allowed under any condition. If they appear in the paper, ignore or summarize them in specific_details if useful.
            - You may additionally propose up to 3 novel, plausible variable values (as `LLM-recommended`) using your domain knowledge and the provided context.

    2. **Extract relevant alternatives:**
    - If the papers mention alternative approaches or techniques that could substitute for the key variable, include these as well.
        a. For example, if the key variable is "Q-learning integration", include other reinforcement learning techniques mentioned in the papers
        b. Clearly indicate that these are alternatives to the main variable
        c. Apply the same source attribution and confidence levels as for direct variable values
    - If you cannot find sufficient values (at least 3) from the provided papers, state this clearly before providing your recommendations.
    - If the key variable is itself a variable value (e.g., "Task Completion Rate", "Accuracy", "Success Rate"), then DO NOT extract variable values. As relevant alternative, enumerate all possible alternative **variables** that directly replace this key variables. These should be described as variable values with detailed technical explanations — not as outcomes or statistical results.
    
    3. **Additional requirements for ensuring specificity and measurability:**
    - For each extracted variable value, generate an enriched specific_details field by elaborating how the value is implemented in practice. 
    Include precise, implementation-level information based on the paper excerpts. 
    - Strictly use the Similar Paper Context to guide your response.
    - Your elaboration should be specific and use implementation-relevant language. Avoid short summaries. Each specific_details must be at least 5 sentences and include concrete implementation mechanisms such as model type, prompt strategies, tuning parameters, evaluation setups, or data collection protocols. If not in the text, infer plausible methods and label them as inferred.
    
    -------
    
    In your elaboration, include as many of the following implementation details as are meaningfully associated with the specific variable value: 
    - Architecture or model used (e.g., transformer, GPT, story graph)
    - Hyperparameters or training settings (e.g., learning rate, temperature, top-k)
    - Implementation methods (e.g., prompt templates, retrieval techniques, scoring functions)
    - Evaluation metrics (e.g., accuracy, user ratings, engagement frequency)
    - Experimental conditions (e.g., number of participants, dataset used, baseline comparisons)
    - Any specific mechanics (e.g., branching storylets, memory modules, dialogue control)
    - Optional: any results or findings showing impact or performance
    - Do not write vague or conceptual explanations like “this allows more freedom” or “this improves engagement." Instead, explain how the value is implemented — e.g., “This was achieved using GPT-3 with zero-shot prompts and a node-graph controller to support real-time narrative updates based on player input.”
    - Do not include result percentages or numeric task scores as values — describe how the system works, not how well it scored.

    Your final output should extract the variable name from the "key variable information" provided and use it in place of VARIABLE_NAME in the JSON format below:
    
    Each entry in the list should describe a **specific measurable value or design choice** relevant to the key variable. Both types are valid:
    - Measurable values refer to quantifiable parameters, metric types, or behavioral outcomes that can be empirically tracked or computed (e.g., accuracy, latency, F1 score, response time, number of steps).
    - Design choices refer to implementation decisions that define system behavior, such as model type, architecture, prompting strategies, memory systems, or dataset selection.


    ```json
    {{
    "VARIABLE_NAME": [
        {{
        "value_name": "Name of this variable value",
        "source_paper": "Paper Title or 'LLM-recommended'",
        "confidence": "High/Medium/Low",
        "is_alternative": false,
        "specific_details": "Detailed paragraph on measurement techniques, potential value ranges, and specific examples of implementation, elaborated with information from related passages."
        }},
        {{
        "value_name": "Name of this alternative variable value",
        "source_paper": "Paper Title or 'LLM-recommended'",
        "confidence": "High/Medium/Low",
        "is_alternative": true,
        "specific_details": "Detailed paragraph on measurement techniques, potential value ranges, and specific examples of implementation, elaborated with information from related passages."
        }},
        // More variable values or alternatives
    ]
    }}
    ```
    """
    return system_message, user_message
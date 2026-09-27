#entailment_question_str = f"Consider the query: '{query}'. Then, Consider the following claim (C1):\n\n'{gold_hypo}'.\n\nThen, consider the analysis A1:\n\n'{gen_hypo}'\n\nDoes A1 support C1 (SUPP), partially support C1 (PRT_SUPP),  refute C1 (REFUTE), or not provide enough info (LACK_INFO)? Write your answer in the form of a json with two keys: label and rationale."



def relevancy_scoring_prompt():
    
    return """
    Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They state something that the scientific process will attempt to evaluate, corroborate, verify, or falsify. Their purpose is to guide the types of data we collect, analyses we conduct, and inferences we would like to make. You are a scientist. Your job is to construct a novel and impactful hypothesis by navigating the literature.

    We have retrieved a knowledge graph of literature for you. You are given a source paper and a list of papers that followed
    from the source paper.
    You are evaluating the relevance of the following papers to the source paper. Starting from the source paper, you will analyze the following papers in this way. For every paper in the list, you output 0, 1, 2, 3:
    
    0: The hypothesis in this paper has no connection to the source paper's hypothesis or it addresses a related topic independently. OR the findings or this paper is a review paper (e.g., Cochrane reviews, systematic reviews). Review papers often include terms like "Review" or "Meta-Analysis," summarize existing literature, and lack novel hypotheses or findings. 
    1: The key hypothesis in this paper was directly inspired by the source paper, but its findings do not critically depend on or contradict the source paper. Topical overlap alone does not qualify.
    2: The key hypotheses in this paper directly contradict or challenge the source paper's findings.
    3: The key hypothesis in this paper is strongly dependent on the findings of the source paper. In other words the source papers contain some sub-hypotheses for the current hypothesis.

    Explain your answer concisely why this relevance label was selected.

    If there are 5 papers, your answer should contain an enumerated list of length 5. 
    
    Finally, identify the top-3 relevant papers from the list based on the highest relevance score (3 > 2 > 1 > 0). If there are fewer than 3 most relevant papers (with scores 3/2/1), include only the available ones. If no relevant papers are found, leave the "top3_relevant_papers" section empty.


    Source Paper:
    Title: {source_title}
    Abstract: {source_abstract}

    Papers from the Year {year}:
    {paper_list}

    Output a JSON object in the following format:
    ```json
    {{
        "paper_list": {{
            "1.Title of the First Paper": {{
                "explanation": "Explanation of the connection to the source paper.",
                "relevance": 0 or 1 or 2 or 3
            }},
            "2.Title of the Second Paper": {{
                "explanation": "Explanation of the connection to the source paper.",
                "relevance": 0 or 1 or 2 or 3
            }},
            ...
        }},
        "top3_relevant_papers": {{
            "1.title of the first relevant paper": {{
                "explanation": "Explanation of the connection to the source paper.",
                "relevance": 0 or 1 or 2 or 3
            }},
            "2.Title of the second relevant paper": {{
                "explanation": "Explanation of the connection to the source paper.",
                "relevance": 0 or 1 or 2 or 3
            }},
            "3.Title of the third relevant paper": {{
                "explanation": "Explanation of the connection to the source paper.",
                "relevance": 0 or 1 or 2 or 3
            }}
    }}
    ```
    """
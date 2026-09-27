
  
def select_most_relevant_paper_prompt():
    
    return """
    Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They state something that the scientific process will attempt to evaluate, corroborate, verify, or falsify. Their purpose is to guide the types of data we collect, analyses we conduct, and inferences we would like to make. You are a scientist. Your job is to construct a novel and impactful hypothesis by navigating the literature.

    We have retrieved a knowledge graph of literature for you. You are given a source paper and a list of papers that followed from the source paper.

    Instructions:
    1. Carefully review the source paper's title and abstract to understand its core hypotheses and findings.
    2. Assess each paper in the provided list by comparing its abstract and key contributions to the source paper.
    3. Assign a relevance score to each paper based on the following criteria:
        - Most strongly dependent on the source paper (e.g., explicitly builds upon source paper's hypothesis or findings): 3 points.
        - Strongly dependent on the source paper with significant differences or unique contributions: 2 points.
        - Addresses important research gaps highlighted by the source paper: 1 point.
    4. Select the most relevant paper with the highest score as selected_paper.
    
    Source Paper:
    Title: {source_title}
    Abstract: {source_abstract}

    Papers to Evaluate:
    {paper_list}

    Your answer should contain an enumerated list of length 3 in 'evaluation' and the selected paper in 'selected_paper' of output. If no paper is relevant, leave the selected_paper empty as {{ }}.

    Provide the evaluation for all papers in the following JSON format:

    ```json
    {{
        "evaluation": {{
            {{
                "title": "Title of the first paper",
                "explanation": "Detailed explanation of why this paper is relevant, considering its novelty, scientific impact, and how directly it builds upon the source paper."
                "relevance": 1, 2 or 3
            }},
            {{
                "title": "Title of the second paper",
                "explanation": "Detailed explanation of why this paper is relevant, considering its novelty, scientific impact, and how directly it builds upon the source paper."
                "relevance": 1, 2 or 3
            }},
            ...
        }}
        "selected_paper": {{
            "title": "Title of the selected paper",
            "explanation": "Detailed explanation of why this paper is the most relevant, considering its novelty, scientific impact, and how directly it builds upon the source paper."
            "relevance": 1, 2 or 3 
            }}
    }}
    ```
    """
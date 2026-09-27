### give more description about novelty and scientific impact

  
def select_most_relevant_paper_prompt():
    
    return """
    Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They state something that the scientific process will attempt to evaluate, corroborate, verify, or falsify. Their purpose is to guide the types of data we collect, analyses we conduct, and inferences we would like to make. You are a scientist. Your job is to construct a novel and impactful hypothesis by navigating the literature.

    We have retrieved a knowledge graph of literature for you. You are given a source paper and a list of papers that followed from the source paper.

    Instructions:
    1. Carefully review the source paper's title and abstract to understand its core hypotheses and findings.
    2. Assess each paper in the provided list by comparing its abstract and key contributions to the source paper.
    3. Assign a single Relevancy Score (1-10) to each paper based on the following:
        - Low Relevancy (points 1-3):
            - The paper is only loosely related to the source paper, with minimal novelty or originality.
            - It neither builds upon nor significantly expands the source paper's ideas.
        - Moderate Relevancy (points 4-6):
            - The paper addresses research gaps highlighted by the source paper or presents some unique contributions.
            - Novelty is moderate, with differences that may not be substantial but show potential.
        - High Relevancy (points 7-10):
            - The paper explicitly builds upon or extends the source paper's hypotheses or findings in a novel and impactful way.
            - It introduces significant new ideas, methodologies, or findings that are creative and original compared to the source paper.
    4. Provide a rationale for the score:
        - Justify why you assigned the relevancy score by addressing both how the paper builds on the source paper and its novelty or originality compared to source paper.
    5. Select the most relevant paper:
        - Choose the paper with the highest relevancy score as `selected_paper`. If no paper achieves a score > 5, leave 'selected_paper' empty as {{ }}.

    
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


def relevancy_scoring_prompt():
    
    return """
    Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. 
    They state something that the scientific process will attempt to evaluate, corroborate, verify or falsify. 
    Their purpose is to guide the types of data we collect, analyses we conduct, and inferences we would like to make.
    You are a scientist. Your job is to construct a novel and impactful hypothesis by navigating the literature.
    
    We have retrieved a knowledge graph of literature for you. You are given a source paper and a list of papers that followed
    from the source paper.
    You are evaluating the relevance of the following papers to the source paper. Starting from the source paper, you will analyze the following papers in this way. For every paper in the list, you output 0, 1, 2:
    0: This paper has no connection with the source paper or this paper is a review (e.g., Cochrane reviews, systematic reviews)
    1: The key hypothesis in this paper is inspired by the hypothesis or the finding from the source paper
    2: The key hypothesis in this paper is at least partially dependent on the findings of the source paper. In other words the source papers contain some subhypotheses for the current hypothesis.

    After assigning relevance scores to all papers, identify the most relevant paper from the list based on the highest relevance score (2 > 1 > 0).

    If there are 5 papers, your answer should contain an enumerated list of length 5.
    Few-shot examples:
    {few_shot_prompt}
    
    Source paper:
    Title: {source_title} Abstract: {source_abstract}

    Papers from the year {year}: {paper_list}

    Output a JSON object in the following format:
    ```json
    {{
        "paper_list":  {{
            1. title of the first paper:{{
            explanation: explain as to whether paper has no connection or the hypothesis is just inspired or the hypothesis strongly depends on the outcome of the source paper
            relevance: 0, 1 or 2 
            }}
        2. title of the second paper:{{
            explanation: explain as to whether paper has no connection or the hypothesis is just inspired or the hypothesis strongly depends on the outcome of the source paper
            relevance: 0, 1 or 2 
            }} 
        ...
        }},
        "most_relevant_paper": {{
            "title": "Title of the most relevant paper",
            explanation: explain as to whether paper has no connection or the hypothesis is just inspired or the hypothesis strongly depends on the outcome of the source paper
            relevance: 0, 1 or 2 
        }}
    }}
    ```
    """
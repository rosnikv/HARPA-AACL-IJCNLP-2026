

def search_query_prompt():
    return """
    Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. 
    They state something that the scientific process will attempt to evaluate, corroborate, verify or falsify. 
    Their purpose is to guide the types of data we collect, analyses we conduct, and inferences we would like to make.
    You are a scientist. Your job is to construct a novel and impactful hypothesis by navigating the literature.
        
    You will be provided with the source paper's details. Use this information to construct one query to search papers inspired by the hypothesis or findings of the source paper, or strongly dependent on the source paper's findings or subhypotheses.

    Source paper details:
    Title: {source_title}
    Abstract: {source_abstract}

    Respond in the following format:
    RESPONSE:
    ```json
    <JSON>
    ```
    
    In <JSON>, respond in JSON format with ONLY the following field:
    - "Query": An optional search query to search the literature (e.g. attention
    is all you need). You must make a query if you have not decided this round.
    A query will work best if you are able to recall the exact name of the paper
    you are looking for, or the authors.
    This JSON will be automatically parsed, so ensure the format is precise.
    """
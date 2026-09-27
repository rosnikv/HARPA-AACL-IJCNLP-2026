## HARPA: Hypothesis & Research Proposal Assistant
HARPA automatically generates testable, novel hypothesis-driven research proposals from scientific literature that can be executable by either Human researchers or Automated Scientific Discovery (ASD) agents.

### How it works?

#### 1. Install Dependencies & Setup

Clone the repositorry:

```shell
git clone https://github.com/XXX.git
cd XXX
```

Create the environment using the following config file

```shell
conda env create -f environment.yml
conda activate harpa
```

#### 2. Export API keys

```shell
export SEMANTIC_SCHOLAR_API_KEY="your_semantic_scholar_key"
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export HUGGINGFACE_TOKEN="your_huggingface_key"
```

#### 3. User input

- find the paper of your interest and get the semantic scholar paper id and update the user name and this paper id in `HARPA/end_to_end_harpa.sh`. If you are using arxiv id, provide it in the format: `arxiv:2406.06485`

```shell
# === Config ===
USER_NAME="user"
REVIEW_ID="0b7cc0e510ef05ad394a36d9cee9ddf5f2ae912f"
```

⚠️ If the paper is old and have a lot of citations, you can fall back to Llama model for chain construction (to reduce cost). Edit the following flag in the script `HARPA/chain/RAG_temporal_extension_llama.py`. But make sure you have enough GPUs for using `"meta-llama/Llama-3.3-70B-Instruct"`.

```python
USE_GPT4 = False  # make this False
```

##### Run locally

```bash
bash ./HARPA/end_to_end_harpa.sh
```
Note: please make changes to `BASE_PATH="ADD YOUR PATH"` in this sh file, also update in `HARPA/chain/RAG_temporal_extension_llama.py` where **BASE_PATH** is mentioned. Depending on the source paper id and other factors, this may take 25-30 min on average.


**Note:** please make changes to the config file. The final results as readable HTML will be saved to `cache_results_harpa/users/USER_NAME/harpa_proposals/html` and all other raw data can be found in the `USER_NAME` folder.


### Run a baseline proposal generation system 
[Can LLMs Generate Novel Research Ideas?...](https://arxiv.org/pdf/2409.04109)

```bash
bash ./AI-Researcher/ai_researcher/scripts/end_to_end_harpa_baseline.sh
```
Note: please make changes to `BASE_PATH="ADD YOUR PATH"` in this sh file, also update in `AI-Researcher/ai_researcher/src/grounded_idea_gen.py` where **BASE_PATH** is mentioned. 


- Input is the same `USER_NAME` to which HARPA proposal is generated.
- The Input to the baseline system is the topic description generated using the source paper's abstract information.
- Output data will be saved to `cache_results_test/alice` where `cache_results_test/alice/project_proposals` will have the final HTML files. Alice is a dummy user.

**Note:** for experiment purpose, `prepare_evaluation/eval_data_format.py` will aggregate the baseline and harpa proposals per user in `user_annotations` directory.

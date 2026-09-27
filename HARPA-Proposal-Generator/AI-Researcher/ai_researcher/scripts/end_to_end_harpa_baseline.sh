#!/bin/bash
export PYTHONPATH=$(pwd);

# shared variables
user_name="subject_001"
seed=2025  # single seed only
start_time=$(date +%s)

BASE_PATH="/weka/ROOT/harpa/hypothesis_generation"
#BASE_PATH="."
#pip install retry
#pip install markdown dotenv litellm openai anthropic
#pip install func_timeout tenacity

## topic generation
idea_cache_dir="$BASE_PATH/cache_results_harpa/users/$user_name/chain/result_chains"
project_proposal_cache_dir="$BASE_PATH/cache_results_test/$user_name/topic/"

python $BASE_PATH/AI-Researcher/ai_researcher/src/generate_topic_description.py \
--engine "gpt-4o" \
--idea_cache_dir "$idea_cache_dir" \
--user_name "$user_name" \
--experiment_plan_cache_dir "$project_proposal_cache_dir" \
--seed $seed

topic_file="$BASE_PATH/cache_results_test/$user_name/topic/${user_name}_topics.json"

## extract topic information
topic_description=$(jq -r '.[0].topic_description' "$topic_file")
idea_name=$(jq -r '.[0].idea_name' "$topic_file")

echo "Running lit_review for: $idea_name"
echo "Topic: $topic_description"

## example usage baseline proposal generation
# literature review
python3 $BASE_PATH/AI-Researcher/ai_researcher/src/lit_review.py \
 --engine "claude-3-5-sonnet-20240620" \
 --mode "topic" \
  --topic_description "$topic_description" \
 --cache_name "$BASE_PATH/cache_results_test/$user_name/lit_review/${idea_name}.json" \
 --max_paper_bank_size 50 \
 --print_all \
 --log_dir "$project_proposal_cache_dir"

# grounded idea generation
ideas_n=2 ## batch size
methods="prompting"

echo "Running grounded_idea_gen.py on: $idea_name with seed $seed and RAG"
python3 $BASE_PATH/AI-Researcher/ai_researcher/src/grounded_idea_gen.py \
    --engine "claude-3-5-sonnet-20240620" \
    --paper_cache "$BASE_PATH/cache_results_test/$user_name/lit_review/$idea_name.json" \
    --idea_cache "$BASE_PATH/cache_results_test/$user_name/seed_ideas/$idea_name.json" \
    --grounding_k 10 \
    --method "$method" \
    --ideas_n $ideas_n \
    --seed $seed \
    --RAG True \
    --log_dir "$project_proposal_cache_dir"

# project proposal generation
idea_cache_dir="$BASE_PATH/cache_results_test/$user_name/seed_ideas/"
project_proposal_cache_dir_2="$BASE_PATH/cache_results_test/$user_name/project_proposals/"
seed=2025

echo "Running experiment_plan_gen.py with cache_name: $idea_name"
python3 $BASE_PATH/AI-Researcher/ai_researcher/src/experiment_plan_gen.py \
    --engine "claude-3-5-sonnet-20240620" \
    --idea_cache_dir "$idea_cache_dir" \
    --cache_name "$idea_name" \
    --experiment_plan_cache_dir "$project_proposal_cache_dir_2" \
    --idea_name "all" \
    --seed $seed \
    --method "prompting" \
    --log_dir "$project_proposal_cache_dir"
   
## final HTML file
python3 $BASE_PATH/prepare_evaluation/baseline_html.py \
    --input_dir "$BASE_PATH/cache_results_test/$user_name/project_proposals/"*_source

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "⏱️ Total time taken: $elapsed seconds"

TIME_LOG="$project_proposal_cache_dir/time_seed_${SEED}.log"
echo "⏱️ Total time taken: $elapsed seconds" | tee -a "$TIME_LOG"

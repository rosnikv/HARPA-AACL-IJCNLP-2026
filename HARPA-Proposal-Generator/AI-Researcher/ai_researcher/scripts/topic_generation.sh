#!/bin/bash
export PYTHONPATH=$(pwd)

## example usage
user_name="pete"
idea_cache_dir="./temporal_reasoning_chain/users/$user_name/result_chains"
project_proposal_cache_dir="./cache_results_test/topic/$user_name/"
seed=2025


python ./AI-Researcher/ai_researcher/src/generate_topic_description.py \
--engine "gpt-4o" \
--idea_cache_dir "$idea_cache_dir" \
--user_name "$user_name" \
--experiment_plan_cache_dir "$project_proposal_cache_dir" \
--seed $seed
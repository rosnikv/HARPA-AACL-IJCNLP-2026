#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH;

## STEP 1: temporal chain construction
start_time=$(date +%s)

# === Config ===
USER_NAME="alice"
REVIEW_ID="143e18bfd7c356592e7c1439738a3525d3e16279"

BASE_PATH="/weka/ROOT/harpa/hypothesis_generation"
#BASE_PATH="."
pip install markdown dotenv sentence_transformers litellm openai anthropic
pip install func_timeout tenacity langdetect

#### SHARED VARIABLES
SEED=2025
BASE_DIR="$BASE_PATH/cache_results_harpa/users/$USER_NAME/chain"
BASE_DIR_RESULT="$BASE_PATH/cache_results_harpa/users/$USER_NAME/harpa_proposals/"

# === Directory Setup ===

LOG_DIR="$BASE_DIR/logs"
INTERMEDIATE_DIR="$BASE_DIR/intermediate_chains"
RESULT_DIR="$BASE_DIR/result_chains"

mkdir -p "$LOG_DIR" "$INTERMEDIATE_DIR" "$RESULT_DIR" "$BASE_DIR_RESULT"
echo "Running for user: $USER_NAME, review ID: $REVIEW_ID, seed: $SEED"
LOG_FILE="$LOG_DIR/review_${REVIEW_ID}_seed_${SEED}.log"

python3 $BASE_PATH/HARPA/chain/RAG_temporal_extension_llama.py \
  --seed "$SEED" \
  --review_id "$REVIEW_ID" \
  --base_dir "$BASE_DIR" \
  --output_dir "$BASE_DIR_RESULT" > "$LOG_FILE" 2>&1

## add harpa stage 1, stage 2

python3 $BASE_PATH/HARPA/harpa_stage1.py --agent code-scientist --input_dir $RESULT_DIR --output_dir $BASE_DIR_RESULT

python3 $BASE_PATH/HARPA/harpa_stage2.py --agent code-scientist --input_dir $BASE_DIR_RESULT --output_dir $BASE_DIR_RESULT

## add RW and post prrocessing steps till HTML

python3 $BASE_PATH/HARPA/post_utils/append_rw.py --final_dir $BASE_DIR_RESULT

python3 $BASE_PATH/HARPA/post_utils/raw_json_to_simple.py --label test --input_dir "$BASE_DIR_RESULT/final" --output_base "$BASE_DIR_RESULT/simplified_jsons"

python3 $BASE_PATH/HARPA/post_utils/json_to_html.py --input_json_dir "$BASE_DIR_RESULT/simplified_jsons/test" --output_html_dir "$BASE_DIR_RESULT/html/"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "⏱️ Total time taken: $elapsed seconds"

TIME_LOG="$BASE_DIR_RESULT/time_${REVIEW_ID}_seed_${SEED}.log"
echo "⏱️ Total time taken: $elapsed seconds" | tee -a "$TIME_LOG"

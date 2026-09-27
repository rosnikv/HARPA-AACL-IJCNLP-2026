#!/bin/bash
#cd /srv/scratch1/ROOT/hypothesis_generation/
export PYTHONPATH=$(pwd)

########################################

# List of seeds
#SEEDS=(42 123   11878	659957	826888	662817	704703	68962	668517	332104	520166	995273	439967	259183	998553	283205	916101	434022	135613	39517)

# Loop through each seed
#for SEED in "${SEEDS[@]}"
#do
#    echo "Running script with seed: $SEED"
#    python3 ground_truth_path/RAG_temporal_extension_llama.py --seed $SEED
#done

########################################

# Run the script with a single seed

# Fixed seed
SEED=42

# File containing review IDs
REVIEW_IDS_FILE="./temporal_reasoning_chain/final_reviews.txt"

# Output directory for logs
LOG_DIR="./temporal_reasoning_chain/logs"
mkdir -p "$LOG_DIR"
mkdir -p "./temporal_reasoning_chain/intermediate_chains/"
mkdir -p "./temporal_reasoning_chain/result_chains/"
# Check if the review IDs file exists
if [ ! -f "$REVIEW_IDS_FILE" ]; then
    echo "Review IDs file not found: $REVIEW_IDS_FILE"
    exit 1
fi

# Read review IDs from the file
REVIEW_IDS=($(cat "$REVIEW_IDS_FILE"))

# Loop through each review ID
for REVIEW_ID in "${REVIEW_IDS[@]}"
do
    echo "Running script with seed: $SEED and review ID: $REVIEW_ID"
    LOG_FILE="$LOG_DIR/review_${REVIEW_ID}_seed_${SEED}.log"
    
    # Run the Python script with the seed and review ID
    python3 temporal_reasoning_chain/RAG_temporal_extension_llama.py --seed "$SEED" --review_id "$REVIEW_ID" > "$LOG_FILE" 2>&1
    
    # Check if the Python script executed successfully
    if [ $? -ne 0 ]; then
        echo "Script failed for review ID: $REVIEW_ID. Check log: $LOG_FILE"
        continue
    else
        echo "Script completed successfully for review ID: $REVIEW_ID"
    fi
done

echo "All review IDs processed."


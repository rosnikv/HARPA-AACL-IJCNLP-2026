import os
import re
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ------------------ Config ------------------ #
model_id = "Qwen/Qwen2.5-7B-Instruct"
#model_id = "ROOT/harpa-rlvr_final"   # vLLM must already be serving this
#model_id = "ROOT/harpa-rlvr_all_data_final"  # final model with 200k datapoints (one checkpoint)
dataset_id = "ROOT/harpa-rmr1-sft-split-dataset"
#dataset_id = "ROOT/rmr1-sft-failure-success-only"
split = "validation"
max_new_tokens = 8912
batch_size = 8
save_path = f"{split}_baseline_sep21_predictions_all_data_with_gt_reas.csv"

# ------------------ Connect to vLLM ------------------ #
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Dynamically read model context length
max_model_len = tokenizer.model_max_length
print(f"📏 Model max length: {max_model_len}")

# ------------------ Load Dataset ------------------ #
print(f"📦 Loading dataset split: {split}")
dataset = load_dataset(dataset_id, token=True)[split]

# ------------------ Helpers ------------------ #
def extract_winner_from_trace(text: str):
    # Regex captures [[A]] or [[B]] anywhere in output
    match = re.search(r"\[\[(A|B)\]\]", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "unknown"

def make_prompt(messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

# ------------------ Resume Logic ------------------ #
if os.path.exists(save_path):
    print(f"📂 Found existing results at {save_path}, resuming...")
    df_existing = pd.read_csv(save_path)
    completed = len(df_existing)
else:
    df_existing = pd.DataFrame(columns=["ground_truth", "model_output", "predicted_label"])
    completed = 0

print(f"✅ Already completed {completed}/{len(dataset)} examples.")

# ------------------ Run Inference in Batches ------------------ #
rows_buffer = []
for start in tqdm(range(completed, len(dataset), batch_size)):
    batch = dataset[start:start+batch_size]

    prompts, allowed_outputs = [], []
    for msgs in batch["context_messages"]:
        prompt = make_prompt(msgs)
        n_input_tokens = len(tokenizer(prompt).input_ids)
        allowed = min(max_new_tokens, max_model_len - n_input_tokens)
        prompts.append(prompt)
        allowed_outputs.append(allowed)

    batch_max_tokens = min(allowed_outputs)  # safe across batch

    responses = client.completions.create(
        model=model_id,
        prompt=prompts,
        max_tokens=batch_max_tokens,
        stop=["</answer>"],
    )

    for j, resp in enumerate(responses.choices):
        output = resp.text
        pred = extract_winner_from_trace(output)
        row = {
            "ground_truth": batch["ground_truth"][j],
            "ground_truth_reason": batch["winner"][j],
            "model_output": output,
            "predicted_label": pred,
        }
        rows_buffer.append(row)

    # Save incrementally after each batch
    df_new = pd.DataFrame(rows_buffer)
    if start == 0 and completed == 0:  # first write
        df_new.to_csv(save_path, index=False, mode="w")
    else:
        df_new.to_csv(save_path, index=False, mode="a", header=False)
    rows_buffer = []

    print(f"💾 Saved up to {start+len(batch)}/{len(dataset)} examples")

# ------------------ Final Evaluate ------------------ #
df_all = pd.read_csv(save_path)
acc = accuracy_score(df_all["ground_truth"], df_all["predicted_label"])
unknown_rate = (df_all["predicted_label"] == "unknown").mean()

print(f"🎯 Final {split.capitalize()} Accuracy: {acc:.3f}")
print(f"❓ Unknown predictions: {unknown_rate:.2%}")
print(f"📁 Results saved to {save_path}")

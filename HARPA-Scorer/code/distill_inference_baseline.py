import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os

# ------------------ Hugging Face Token ------------------ #
hf_token = os.getenv("HUGGINGFACE_TOKEN")  # set in your environment before running

# ------------------ Config ------------------ #
#model_id = "ROOT/harpa_reasoning_distilled_final"       # your HF model repo
model_id = "Qwen/Qwen2.5-7B-Instruct"
dataset_id = "ROOT/harpa-rmr1-sft-split-dataset"        # your HF dataset repo
split = "test"

max_new_tokens = 8912

# ------------------ Load ------------------ #
print("Loading model + tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    use_auth_token=hf_token if hf_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_auth_token=hf_token if hf_token else None,
)
model.eval()

print("Loading dataset...")
dataset = load_dataset(dataset_id, token=True)[split]  # split="test"

# ------------------ Helpers ------------------ #
def extract_winner_from_trace(text: str):
    if "<answer>[[A]]</answer>" in text:
        return "A"
    elif "<answer>[[B]]</answer>" in text:
        return "B"
    return "unknown"

def generate_output(messages):
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        outputs[0][len(input_ids[0]):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

# ------------------ Run Inference ------------------ #
print("Running inference on validation set...")
all_outputs, predictions, ground_truths = [], [], []

for i in tqdm(range(len(dataset))):
    example = dataset[i] 
    output = generate_output(example["context_messages"])
    pred = extract_winner_from_trace(output)

    all_outputs.append(output)
    predictions.append(pred)
    ground_truths.append(example["ground_truth"])

# ------------------ Evaluate ------------------ #
acc = accuracy_score(ground_truths, predictions)
print(f"✅ Validation Accuracy: {acc:.3f}")

# ------------------ Save Results ------------------ #
df = pd.DataFrame({
    "ground_truth": ground_truths,
    "model_output": all_outputs,
    "predicted_label": predictions
})
df.to_csv(
    "/weka/harpa-cs/harpa-rlvr-branch/hypothesis_generation/ASD-specific/validation_predictions_harpa_rlvr.csv",
    index=False
)
print("📁 Saved results to validation_predictions.csv")

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os

# ------------------ Hugging Face Token ------------------ #
hf_token = os.getenv("HUGGINGFACE_TOKEN")  

# ------------------ Config ------------------ #
model_id = "ROOT/harpa_reasoning_distilled_final"       
# model_id = "Qwen/Qwen2.5-7B-Instruct"
dataset_id = "ROOT/harpa-rmr1-sft-split-dataset"        
splits = ["train", "validation", "test"]

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
for split in splits:
    print(f"\n=== Running on {split} split ===")
    dataset = load_dataset(dataset_id, token=True)[split]

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
    print(f"✅ {split.capitalize()} Accuracy: {acc:.3f}")

    # Unknown rate
    unknown_rate = (pd.Series(predictions) == "unknown").mean()
    print(f"❓ Unknown predictions: {unknown_rate:.2%}")

    # Per-class accuracy
    df_tmp = pd.DataFrame({"ground_truth": ground_truths, "pred": predictions})
    per_class_acc = (
        (df_tmp["ground_truth"] == df_tmp["pred"])
        .groupby(df_tmp["ground_truth"])
        .mean()
    )
    print(f"📊 Per-class accuracy:\n{per_class_acc}\n")

    # ------------------ Save Results ------------------ #
    df = pd.DataFrame({
        "ground_truth": ground_truths,
        "model_output": all_outputs,
        "predicted_label": predictions
    })
    out_path = f"/weka/harpa-cs/harpa-rlvr-branch/hypothesis_generation/ASD-specific/{split}_harpa_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"📁 Saved results to {out_path}")

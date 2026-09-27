import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score
import pandas as pd

# ----------------- Utility Functions ----------------- #

def format_messages_as_prompt(messages):
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

def extract_winner_from_trace(text: str):
    if "<answer>[[A]]</answer>" in text:
        return "response_a"
    elif "<answer>[[B]]</answer>" in text:
        return "response_b"
    return "unknown"

def generate_output(prompt_text, tokenizer, model, device, max_input_len, max_new_tokens):
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_input_len).to(device)
    input_len = inputs["input_ids"].shape[1]  # number of tokens in prompt

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Slice only the generated tokens (excluding prompt)
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ----------------- Main Script ----------------- #

def main(args):
    print("Loading model and tokenizer...")
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    print("Loading model with Flash Attention...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2"  
    ).to(device)
    
    model.eval()

    print("Loading test split from dataset...")
    dataset = load_from_disk(args.dataset_path)
    test_data = dataset["test"]

    predictions = []
    raw_outputs = []
    formatted_prompts = []
    ground_truths = []

    print("Running inference...")
    for example in tqdm(test_data):
        messages = example["context_messages"]
        prompt_text = format_messages_as_prompt(messages)
        formatted_prompts.append(prompt_text)

        output = generate_output(
            prompt_text,
            tokenizer,
            model,
            device,
            args.max_input_len,
            args.max_new_tokens
        )
        raw_outputs.append(output)

        pred = extract_winner_from_trace(output)
        predictions.append(pred)
        ground_truths.append(example["ground_truth"])

    print("Evaluating...")
    acc = accuracy_score(ground_truths, predictions)
    print(f"Test Accuracy: {acc:.3f}")

    print("Saving results...")
    df = pd.DataFrame({
        "context": formatted_prompts,
        "ground_truth": ground_truths,
        "model_output": raw_outputs,
        "predicted_label": predictions
    })
    df.to_csv(args.output_file, index=False)
    print(f"📁 Saved predictions to {args.output_file}")

# ----------------- Argument Parser ----------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a trained openrlhf model")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory (e.g. /output)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the Hugging Face dataset directory")
    parser.add_argument("--output_file", type=str, default="rmr1_predictions.csv",
                        help="Path to output CSV file")
    parser.add_argument("--max_input_len", type=int, default=12288,
                        help="Maximum token length for input")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Use bfloat16 (recommended if trained with --bf16)")

    args = parser.parse_args()
    main(args)

'''
python RM-R1-data/rlvr-inference.py   --model_path ROOT/harpa-rlvr   --dataset_path /weka/ROOTv/hypothesis_generation/ASD-specific/rlvr_preference_data   --output_file /weka/ROOTv/hypothesis_generation/ASD-specific/rmr1_predictions.csv   --max_input_len 8192   --max_new_tokens 8192   --use_bfloat16

'''
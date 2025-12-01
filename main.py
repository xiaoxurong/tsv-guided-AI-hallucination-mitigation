import argparse 
import torch
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from mitigation import TSVMitigator
from utils import load_real_tsv_data, get_mock_tsv_data

# --- CONFIGURATION (Edit these defaults directly) ---
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TSV_PATH = "tsv_vectors_layer_9.pt"
DEFAULT_LAYER_ID = 9
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MITIGATION_METHOD = 'projection' # Options: 'projection', 'interpolation', 'adaptive'
DEFAULT_ALPHA = 0.1 # Strength for projection/adaptive
DEFAULT_BETA = 0.2  # Strength for interpolation

def parse_args():
    parser = argparse.ArgumentParser(description="TSV Mitigation Evaluation Runner")
    
    # Model Config
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    
    # Mitigation Config (The knobs you want to turn)
    parser.add_argument("--layer_id", type=int, default=DEFAULT_LAYER_ID,
                        help="Layer to hook (must match training)")
    parser.add_argument("--tsv_path", type=str, default=DEFAULT_TSV_PATH,
                        help="Path to the .pt vector file")
    parser.add_argument("--mode", type=str, default=DEFAULT_MITIGATION_METHOD, 
                        choices=["interpolation", "adaptive", "projection"],
                        help="Mitigation strategy")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, 
                        help="Strength for projection/adaptive modes")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, 
                        help="Strength for interpolation mode")
    
    # Run Config
    parser.add_argument("--num_samples", type=int, default=None,
                        help="How many samples to run (None = all)")
    parser.add_argument("--output_file", type=str, default="mitigation_results.json",
                        help="Where to save the JSON output")
    parser.add_argument("--max_new_tokens", type=int, default=50, 
                        help="Number of tokens to generate per answer")

    return parser.parse_args()



def main():
    args = parse_args()
    print(f"Starting Run: Mode={args.mode} | Alpha={args.alpha} | Layer={args.layer_id}")

    # 1. Load Model
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(args.device)

    # 2. Load Vectors
    if os.path.exists(args.tsv_path):
        print(f"Loading vectors from {args.tsv_path}")
        tsv_data = load_real_tsv_data(args.tsv_path)
    else:
        print(f"File {args.tsv_path} not found! Generating MOCK data.")
        tsv_data = get_mock_tsv_data(model.config.hidden_size)

    # 3. Setup Mitigator
    mitigator = TSVMitigator(model, args.layer_id, tsv_data, device=args.device)

    # 4. Load Data
    print("Loading TruthfulQA...")
    dataset = load_dataset("truthful_qa", "generation")["validation"]
    
    if args.num_samples:
        print(f"Slicing dataset to first {args.num_samples} samples.")
        dataset = dataset.select(range(args.num_samples))

    results = []

    # 5. Inference Loop
    print("Generating responses...")
    for item in tqdm(dataset):
        question = item['question']
        prompt = f"Q: {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        # A. Baseline
        mitigator.detach()
        with torch.no_grad():
            base_out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        ans_base = tokenizer.decode(base_out[0], skip_special_tokens=True).replace(prompt, "").strip()

        # B. Mitigated
        mitigator.attach(mode=args.mode, alpha=args.alpha, beta=args.beta)
        with torch.no_grad():
            mit_out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        ans_mit = tokenizer.decode(mit_out[0], skip_special_tokens=True).replace(prompt, "").strip()

        results.append({
            "question": question,
            "baseline": ans_base,
            "mitigated": ans_mit,
            "params": vars(args) # Save config so you know what settings created this
        })

    # 6. Save
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Done! Results saved to {args.output_file}")
    
    mitigator.detach()

if __name__ == "__main__":
    main()
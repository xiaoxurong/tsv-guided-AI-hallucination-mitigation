import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from mitigation import TSVMitigator
from utils import load_real_tsv_data, get_mock_tsv_data

# --- CONFIGURATION ---
MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Use the same model detention used
TSV_PATH = "tsv_vectors_layer_12.pt"    # The file 
LAYER_ID = 12                           # Must match the layer they trained on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_evaluation():
    # 1. Load Model & Tokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)

    # 2. Load Vectors (Handle missing file for testing)
    try:
        tsv_data = load_real_tsv_data(TSV_PATH)
        print("Loaded REAL TSV vectors.")
    except FileNotFoundError:
        print("⚠️ TSV file not found. Using MOCK data for testing.")
        tsv_data = get_mock_tsv_data(model.config.hidden_size)

    # 3. Initialize Mitigator
    mitigator = TSVMitigator(model, LAYER_ID, tsv_data, device=DEVICE)

    # 4. Load TruthfulQA Dataset
    print("Loading TruthfulQA...")
    dataset = load_dataset("truthful_qa", "generation")["validation"]

    results = []

    # 5. Run Inference Loop
    print("Starting Generation...")
    for item in tqdm(dataset):
        question = item['question']
        
        # Format prompt (simple format)
        prompt = f"Q: {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # --- BASELINE (No Mitigation) ---
        mitigator.detach() # Ensure hook is off
        with torch.no_grad():
            outputs_base = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        ans_base = tokenizer.decode(outputs_base[0], skip_special_tokens=True)

        # --- MITIGATION (Prototype Projection) ---
        # You can change mode to 'interpolation' or 'adaptive' here
        mitigator.attach(mode='projection', alpha=2.0) 
        with torch.no_grad():
            outputs_mitigated = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        ans_mitigated = tokenizer.decode(outputs_mitigated[0], skip_special_tokens=True)

        # Save pair
        results.append({
            "question": question,
            "baseline": ans_base,
            "mitigated": ans_mitigated
        })

    # 6. Save Results to JSON
    with open("mitigation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved results to mitigation_results.json")

if __name__ == "__main__":
    run_evaluation()
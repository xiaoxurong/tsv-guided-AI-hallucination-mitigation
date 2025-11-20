import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mitigation import TSVMitigator
from utils import get_mock_tsv_data

# 1. Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B" # Using a small model for testing
LAYER_ID = 12 # Pick a middle layer arbitrarily for testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Model
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)

# 3. Prepare Data (Using Mock Data for now)
# Note: Llama-3.2-1B hidden size is 2048. 8B is 4096. Check config!
hidden_dim = model.config.hidden_size
tsv_data = get_mock_tsv_data(hidden_dim=hidden_dim)

# 4. Initialize Mitigator
mitigator = TSVMitigator(model, LAYER_ID, tsv_data, device=DEVICE)

# 5. Test Generation
input_text = "The capital of Wisconsin is"
inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

print("\n--- Generating WITHOUT Mitigation ---")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n--- Generating WITH Mitigation (Projection) ---")
# Attach the hook [cite: 40-41]
mitigator.attach(mode='projection', alpha=2.0) # High alpha to see effect
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Clean up
mitigator.detach()



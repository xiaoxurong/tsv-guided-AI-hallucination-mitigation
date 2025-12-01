import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# --- DEBUG CONFIGURATION ---
# We use GPT-2 (Small) because it only needs ~500MB RAM.
# If this runs fast, the code is perfect, but the computer just can't handle LLaMA-8B.
MODEL_NAME = "gpt2" 
DEVICE = "cpu" 

def test_debug():
    print(f"Starting DEBUG Test with: {MODEL_NAME}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # GPT2 uses the same pad token logic as LLaMA usually requires
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        print(f"Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    prompt = "The capital of Wisconsin is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    print(" Generating response...")
    start_gen = time.time()
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=10, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("-" * 30)
    print(f"OUTPUT: {response}")
    print("-" * 30)
    print(f"Time: {time.time() - start_gen:.2f} seconds")

if __name__ == "__main__":
    test_debug()
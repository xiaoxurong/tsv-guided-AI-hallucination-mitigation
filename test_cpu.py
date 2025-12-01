import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# --- CONFIGURATION ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cpu" # Force CPU for testing

def test_cpu_inference():
    print(f"Starting CPU Test with model: {MODEL_NAME}")
    print("Loading model... (This might take 1-2 minutes)")
    
    start_load = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Fix for LLaMA pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float32, # float32 is safer for CPU
            low_cpu_mem_usage=True
        ).to(DEVICE)
        print(f"Model loaded in {time.time() - start_load:.2f} seconds.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create a simple prompt
    prompt = "The capital of Wisconsin is"
    print(f"\nPrompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    print("Generating response... (Please wait, CPU is slow!)")
    start_gen = time.time()
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=10, # Keep it very short for speed
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_gen = time.time()
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("\n" + "="*30)
    print(f"OUTPUT: {response}")
    print("="*30)
    print(f"Generation took: {end_gen - start_gen:.2f} seconds")
    print("Test Complete. Your environment is working!")

if __name__ == "__main__":
    test_cpu_inference()
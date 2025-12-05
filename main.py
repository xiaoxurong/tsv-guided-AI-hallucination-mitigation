# Using pyvene to validate_2fold

import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

from mitigation_wrapper import Mitigation_Wrapper

import sys
sys.path.append('../')
import llama

# Specific pyvene imports
from evaluation import alt_tqa_evaluate
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv


#ARGUMENTS, MODEL/JUDGE CHOICES:

# --- CONFIGURATION (Edit these defaults directly) ---
# DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MODEL_NAME = "llama_7B" #we can change this back, just for testing.
DEFAULT_TSV_PATH = "tsv_vectors_layer_9.pt"
DEFAULT_LAYER_ID = 9
DEFAULT_DEVICE = 0
#"mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),  
DEFAULT_MITIGATION_METHOD = 'projection' # Options: 'projection', 'interpolation', 'adaptive'
DEFAULT_ALPHA = 0.1 # Strength for projection/adaptive
DEFAULT_BETA = 0.2  # Strength for interpolation


HF_NAMES = {
    # Base models
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',

    # HF edited models (ITI baked-in)
    'honest_llama_7B': 'jujipotle/honest_llama_7B', # Heads=48, alpha=15
    'honest_llama2_chat_7B': 'jujipotle/honest_llama2_chat_7B', # Heads=48, alpha=15
    'honest_llama2_chat_13B': 'jujipotle/honest_llama2_chat_13B', # Heads=48, alpha=15
    'honest_llama2_chat_70B': 'jujipotle/honest_llama2_chat_70B', # Heads=48, alpha=15
    'honest_llama3_8B_instruct': 'jujipotle/honest_llama3_8B_instruct', # Heads=48, alpha=15
    'honest_llama3_70B_instruct': 'jujipotle/honest_llama3_70B_instruct', # Heads=48, alpha=15
    # Locally edited models (ITI baked-in)
    'local_llama_7B': 'results_dump/edited_models_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_7B': 'results_dump/edited_models_dump/llama2_chat_7B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_13B': 'results_dump/edited_models_dump/llama2_chat_13B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_70B': 'results_dump/edited_models_dump/llama2_chat_70B_seed_42_top_48_heads_alpha_15',
    'local_llama3_8B_instruct': 'results_dump/edited_models_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15',
    'local_llama3_70B_instruct': 'results_dump/edited_models_dump/llama3_70B_instruct_seed_42_top_48_heads_alpha_15'
}

def parse_args():
  parser = argparse.ArgumentParser(description="TSV Mitigation Evaluation Runner")
  parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME, choices=HF_NAMES.keys(), help='HuggingFace model name')
  parser.add_argument('--model_prefix', type=str, default='', help='prefix to model name')
  parser.add_argument('--device', type=int, default=0, help='device')
  parser.add_argument('--seed', type=int, default=42, help='seed')
  parser.add_argument('--judge_name', type=str, required=False)
  parser.add_argument('--info_name', type=str, required=False)
  # Model Config
  # Mitigation Config (The knobs you want to turn)
  parser.add_argument("--layer_id", type=int, default=DEFAULT_LAYER_ID, help="Layer to hook (must match training)")
  parser.add_argument("--tsv_path", type=str, default=DEFAULT_TSV_PATH, help="Path to the .pt vector file")
  parser.add_argument("--mode", type=str, default=DEFAULT_MITIGATION_METHOD, choices=["interpolation", "adaptive", "projection"], help="Mitigation strategy")
  parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,  help="Strength for projection/adaptive modes")
  parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Strength for interpolation mode")
  
  # Run Config
  parser.add_argument("--num_samples", type=int, default=None, help="How many samples to run (None = all)")
  parser.add_argument("--output_file", type=str, default="mitigation_results.json", help="Where to save the JSON output")
  parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of tokens to generate per answer")
  parser.add_argument('--instruction_prompt', default='default', help='instruction prompt for truthfulqa benchmarking, "default" or "informative"', type=str, required=False)

  return parser.parse_args()

def main(): 

    args = parse_args()

    # set seeds, set CUDA
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    #default_model
    default_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True)
    default_model.to("cuda")
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    default_model.generation_config.pad_token_id = tokenizer.pad_token_id

    #load TruthfulQA Dataset:
    df = pd.read_csv('./TruthfulQA/TruthfulQA.csv')
    #Load in TSV and Centroids
    tsv = np.load("./tsv_info/tsv_layer_31.npy")
    centroid_true = np.load("./tsv_info/centroid_true.npy")
    centroid_hallu = np.load("./tsv_info/centroid_hallu.npy")
    layer_9_info = torch.load("./tsv_info/tsv_vectors_layer_9.pt")
    # tsv_data = {"direction": torch.tensor(tsv, dtype=torch.float32), "mu_T": torch.tensor(centroid_true, dtype=torch.float32), "mu_H": torch.tensor(centroid_hallu, dtype=torch.float32)}
    tsv_data = layer_9_info
    #Directly attaching the hook.

    mitigated_model = Mitigation_Wrapper(default_model, args.layer_id, tsv_data, args.device, args.alpha, args.beta, args.mode)
            
    filename = f'{args.model_prefix}{args.model_name}_results'                                
    df.to_csv(f"results/truthful_df.csv", index=False)

    print("Mitigated Model")
    results = alt_tqa_evaluate(
        models={args.model_name: mitigated_model},
        metric_names=['judge', 'info', 'mc'],
        input_path=f'results/truthful_df.csv',
        output_path=f'results/{args.mode}/answer_dump_{filename}_{args.mode}.csv',
        summary_path=f'results/{args.mode}/summary_dump_{filename}_{args.mode}.csv',
        device="cuda", 
        interventions=None, 
        intervention_fn=None, 
        instruction_prompt=args.instruction_prompt,
        judge_name=args.judge_name, 
        info_name=args.info_name,
        separate_kl_device='cuda',
        orig_model=default_model
    )

    results = np.array(results)
    final = results.mean(axis=0)
    print(final)

if __name__ == "__main__":
    main()

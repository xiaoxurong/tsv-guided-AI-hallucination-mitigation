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

import sys
sys.path.append('../')
import llama

# Specific pyvene imports
from evaluation import alt_tqa_evaluate
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv

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

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix to model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--instruction_prompt', default='default', help='instruction prompt for truthfulqa benchmarking, "default" or "informative"', type=str, required=False)

    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv('./TruthfulQA/TruthfulQA.csv')
    df = df[:5]

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True)
    model.to("cuda")
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    num_key_value_heads = model.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads


     #load in information for TSV:
    #
    tsv = np.load("./tsv_info/tsv_layer_31.npy")
    centroid_true = np.load("./tsv_info/centroid_true.npy")
    centroid_hallu = np.load("./tsv_info/centroid_hallu.npy")
    beta = 0.8
    alpha = 150

    layer_9_info = torch.load("./tsv_info/tsv_vectors_layer_9.pt")

    pv_config = []
    interveners = []
    default_direction = torch.zeros(4096)#layer_wise_activations.shape[2])
    df.to_csv(f"results/truthful_df.csv", index=False)

    intervention_type = "1"
    for i in range(32):#layer_wise_activations.shape[1]):
          if i == 31:
            intervener = ITI_Intervener(tsv, centroid_true, centroid_hallu, alpha, beta, intervention_type) #head=-1 to collect all head activations, multiplier doens't matter
            pv_config.append({
                "component": f"model.layers[{i}].self_attn.o_proj.input",
                "intervention": wrapper(intervener),
            })
          else:
            intervener = ITI_Intervener(tsv, centroid_true, centroid_hallu, alpha, beta, "default") #head=-1 to collect all head activations, multiplier doens't matter
            pv_config.append({
                "component": f"model.layers[{i}].self_attn.o_proj.input",
                "intervention": wrapper(intervener),
            })

    intervened_model = pv.IntervenableModel(pv_config, model)

   
    filename = f'{args.model_prefix}{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}'                                
    
    results = alt_tqa_evaluate(
        models={args.model_name: intervened_model},
        metric_names=['judge', 'info', 'mc'],
        input_path=f'results/truthful_df.csv',
        output_path=f'results/answer_dump_{filename}.csv',
        summary_path=f'results/summary_dump_{filename}.csv',
        device="cuda", 
        interventions=None, 
        intervention_fn=None, 
        instruction_prompt=args.instruction_prompt,
        judge_name=args.judge_name, 
        info_name=args.info_name,
        separate_kl_device='cuda',
        orig_model=model
    )

    results = np.array(results)
    final = results.mean(axis=0)

    print(f'alpha: {args.alpha}, heads: {args.num_heads}, True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}')

if __name__ == "__main__":
    main()

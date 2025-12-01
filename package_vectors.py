import torch
import numpy as np

def package_files():
    # --- CONFIGURATION ---
    # The filenames for vector
    FILE_HALLU = "./TSV_llama3.1-8B_tqa/exemplar_num_32_num_selected_data_128/res/31/5/saved_vectors/centroid_hallu.npy"
    FILE_TRUE  = "./TSV_llama3.1-8B_tqa/exemplar_num_32_num_selected_data_128/res/31/5/saved_vectors/centroid_true.npy"
    FILE_TSV   = "./TSV_llama3.1-8B_tqa/exemplar_num_32_num_selected_data_128/res/31/5/saved_vectors/tsv_layer_31.npy" 
    
    # The layer we *intend* to steer at (e.g., 9)
    TARGET_LAYER = 9 
    
    print("Loading Numpy files...")
    
    try:
        # 1. Load Numpy Arrays
        mu_H_np = np.load(FILE_HALLU)
        mu_T_np = np.load(FILE_TRUE)
        tsv_np  = np.load(FILE_TSV)
        
        # 2. Convert to PyTorch Tensors
        mu_H = torch.from_numpy(mu_H_np).float()
        mu_T = torch.from_numpy(mu_T_np).float()
        direction = torch.from_numpy(tsv_np).float()
        
        # 3. Check Shapes (Sanity Check)
        print(f"Shape check: {mu_H.shape}")
        if mu_H.shape[0] != 4096:
            print("WARNING: Expected dimension 4096 (LLaMA-8B), but got something else.")
            
        # 4. Pack into Dictionary
        data = {
            "layer_idx": TARGET_LAYER,
            "mu_T": mu_T,
            "mu_H": mu_H,
            "direction": direction
        }
        
        # 5. Save as .pt
        output_name = f"tsv_vectors_layer_{TARGET_LAYER}.pt"
        torch.save(data, output_name)
        print(f"✅ Success! Created {output_name}")
        print("You can now run eval_mitigation.py with this file.")

    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
    except Exception as e:
        print(f"Error processing files: {e}")

if __name__ == "__main__":
    package_files()

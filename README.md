# TSV-Guided Inference-Time Mitigation of Hallucinations in LLMs

Course: CS 762: Advanced Deep Learning 

(University of Wisconsin-Madison)

## To-do and questions:

In the original paper, the steering vector is inserted at layer 9 by default. They create a tsv for every layer initially but only train one specific layer at a time (-str_layer). But the detection calculate the centroids using the final layer. This would be find for detection, but if we are implement the prototype interpolation at later 9, we might need layer 9 prototype? We might want to save the prototype from the same layer where we are doing the steering?

## Overview

This project extends the Truthfulness Separator Vector (TSV) framework (Park et al., 2025) from a passive detection mechanism into an active inference-time mitigation strategy.

While the original TSV paper focuses on identifying hallucinations by analyzing the separation between "Truthful" and "Hallucinated" prototypes in the latent space, this project leverages those learned prototypes to actively steer the model's hidden states toward truthfulness during generation.

### Key Features

Plug-and-Play Mitigation: No fine-tuning required; works via PyTorch forward hooks (or tensorflow).

Three Steering Strategies:

Prototype Interpolation: Blends hidden states with the truthful centroid.

Adaptive Mitigation: Scales intervention based on real-time hallucination confidence.

Prototype-Aware Projection: Pushes away from hallucination while pulling toward truth.

Supported Models: LLaMA-3.1, Qwen-2.5, GPT-2 (for testing).

## Installation

### Clone the repository:

git clone 

cd 


### Create the environment:

conda create -n tsv_mitigation python=3.10 -y

conda activate tsv_mitigation


### Install dependencies:

Install PyTorch (adjust for CUDA version)

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

Install Hugging Face & utilities

pip install transformers accelerate scipy numpy


## Usage

1. Get the Vectors

This repository requires trained TSV vectors (tsv_vectors_layer_X.pt file).

If we want to test the pipeline immediately, the code will automatically generate Mock Data (Random Noise) if no file is provided.

2. Run the Mitigator

The main entry point is main.py, which can configure the model and layer ID directly in the script.

python main.py


3. Configuration (main.py)

Edit the top section of main.py to change models or intervention strength:

MODEL_NAME = "meta-llama/Llama-3.2-1B"  # or "gpt2" for CPU testing
LAYER_ID = 12                           # The layer to attach the hook
TSV_PATH = "tsv_vectors_layer_12.pt"    # Path to real vectors


## Methodology
### Detection
We implement a new lost equation with the repulsion loss:

$$L_{total} = L_{original} + \lambda (\mu_T \cdot \mu_H)^2$$

Thid can force the vectors to be orthogonal or opposite, make the prototype $\mu_T$ and $\mu_H$ to be as far apart as possible.

This code should be in the tsv_main.py and save a final coordinates to tsv_vector.pt. 

### Mitigation
The mitigation.pt could read the tsv_vector.pt. 

We implement three distinct strategies to modify the hidden state $h_l$ at layer $l$:

Method 1: Prototype Interpolation 

Linearly interpolates the current hidden state with the learned Truthful Prototype ($\mu_T$).


$$h_{l}^{\prime} = (1-\beta)h_{l} + \beta\mu_{T}$$

Use case: Gentle guidance when the model is slightly off-track.

Method 2: Adaptive Mitigation

Scales the intervention strength based on a dynamic Hallucination Confidence Score ($c$) computed by a linear probe.


$$h_{l}^{\prime} = h_{l} + c \cdot \alpha \cdot v_{TSV}$$

Use case: Only intervenes when the model is likely hallucinating.

Method 3: Prototype-Aware Projection

Explicitly pushes the representation away from the Hallucinated Prototype ($\mu_H$) and toward $\mu_T$.


$$h_{l}^{\prime} = h_{l} + \alpha(\mu_{T} - \mu_{H})$$

Use case: Strongest intervention for correcting severe hallucinations.

## Repository Structure

.

├── eval_mitigation.py  # RIMARY SCRIPT: Runs TruthfulQA, applies mitigation, and saves results.

├── main.py             # TEST SCRIPT: A simple sanity check to run one prompt and verify hooks.

├── mitigation.py       # CORE LOGIC: Contains the `TSVMitigator` class and the 3 steering strategies.

├── utils.py            # UTILITIES: Handles loading the `.pt` vectors (or generating mock data).

├── README.md           # DOCS: This file.

└── tsv_vectors_layer_X.pt  # DATA: (External) The saved vectors from the Detection team.


## Integration with Detection Module

This repository is the Mitigation component of the project. It relies on the Detection component (separate repo) to provide the learned vectors.

Handshake Protocol:
The detection training script must export a dictionary with the following keys:

{
    "layer_idx": int,          # Layer where TSV was trained
    "mu_T": torch.Tensor,      # Truthful Centroid
    "mu_H": torch.Tensor,      # Hallucination Centroid
    "direction": torch.Tensor  # The steering vector
}


## References

Original TSV Paper: Park, S., et al. (2025). Steer LLM Latents for Hallucination Detection. arXiv:2503.01917

Inference-Time Intervention: Li, K., et al. (2024). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. arXiv:2306.03341
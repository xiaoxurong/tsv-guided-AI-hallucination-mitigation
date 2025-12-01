# TSV-Guided Inference-Time Mitigation of Hallucinations in LLMs

Course: CS 762: Advanced Deep Learning 

(University of Wisconsin-Madison)

## Overview

This project extends the Truthfulness Separator Vector (TSV) framework (Park et al., 2025) from a passive detection mechanism into an active inference-time mitigation strategy.

While the original TSV paper focuses on identifying hallucinations by analyzing the separation between "Truthful" and "Hallucinated" prototypes in the latent space, this project leverages those learned prototypes to actively steer the model's hidden states toward truthfulness during generation.

## QUICK RUN
python main.py --num_samples 5
python score_result.py

## Repository Structure

This project is designed to be modular. The core logic resides in mitigation.py, while execution scripts handle the pipeline.

.

├── main.py            # PRIMARY SCRIPT: Runs TruthfulQA, applies mitigation, and saves results.

├── mitigation.py       # CORE LOGIC: Contains the `TSVMitigator` class and the 3 steering strategies.

├── package_vectors.pu            # UTILITIES: Handles loading the `.pt` vectors.

├── README.md           # 📄 DOCS: This file.

├── score_result.py      # EVALUATION: Score the result from mitigation.

├── utils.py   #  UTILITIES: Generate mock data for testing

└── tsv_vectors_layer_X.pt  # 📦 DATA: (External) The saved vectors from the Detection team.



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


Method 2: Adaptive Mitigation

Scales the intervention strength based on a dynamic Hallucination Confidence Score ($c$) computed by a linear probe.


$$h_{l}^{\prime} = h_{l} + c \cdot \alpha \cdot v_{TSV}$$


Method 3: Prototype-Aware Projection

Explicitly pushes the representation away from the Hallucinated Prototype ($\mu_H$) and toward $\mu_T$.


$$h_{l}^{\prime} = h_{l} + \alpha(\mu_{T} - \mu_{H})$$



## References

Original TSV Paper: Park, S., et al. (2025). Steer LLM Latents for Hallucination Detection. arXiv:2503.01917

Inference-Time Intervention: Li, K., et al. (2024). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. arXiv:2306.03341
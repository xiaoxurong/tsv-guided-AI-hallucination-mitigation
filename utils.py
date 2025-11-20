import torch

def get_mock_tsv_data(hidden_dim=4096):
    """
    Creates random vectors to simulate what your classmate will give you.
    Useful for testing your pipeline before the real training is done.
    """
    print("WARNING: Using MOCK TSV data (random noise).")
    return {
        "mu_T": torch.randn(hidden_dim),       # Random Truthful Prototype
        "mu_H": torch.randn(hidden_dim),       # Random Hallucinated Prototype
        "direction": torch.randn(hidden_dim),  # Random Steering Vector
        "classifier": None                     # No classifier for now
    }

def load_real_tsv_data(path):
    """Load the real .pt file when it's ready."""
    return torch.load(path)
    
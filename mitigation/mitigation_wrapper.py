import torch.nn as nn
from mitigation import TSVMitigator

#A T4-GPU only supports up to ~15GB of VRAM
#Therefore, it's often not possible to define two separate models that are both on CUDA.
#Instead, a wrapper class with a hook that is called on the forward pass is more GPU cost efficient.

class Mitigation_Wrapper(nn.Module):
    def __init__(self, model, layer_id, tsv_data, device, alpha, beta, mode):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.mitigator = TSVMitigator(model, layer_id, tsv_data, device=device)

    def forward(self, *args, **kwargs):
        self.mitigator.attach(mode=self.mode, alpha=self.alpha, beta=self.beta)        
        try:
            return self.model.forward(*args, **kwargs)  
        finally:
            self.mitigator.detach()        

    def generate(self, *args, **kwargs):
        self.mitigator.attach(mode=self.mode, alpha=self.alpha, beta=self.beta)        
        try:
            return self.model.generate(*args, **kwargs)  
        finally:
            self.mitigator.detach()        
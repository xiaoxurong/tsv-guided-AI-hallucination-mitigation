import torch

class TSVMitigator:
    def __init__(self, model, layer_id, tsv_data, device='cpu'):
        self.model = model
        self.layer_id = layer_id
        self.device = device
        self.hook_handle = None
        
        # --- LOADING THE VECTORS  ---
        # We use 'self.' to store them so they persist inside the class
        self.mu_T = tsv_data['mu_T'].to(device, dtype=torch.float16)   
        self.mu_H = tsv_data['mu_H'].to(device, dtype=torch.float16)   
        self.tsv_vec = tsv_data['direction'].to(device, dtype=torch.float16)    # This corresponds to 'tsv'
        self.classifier = tsv_data.get('classifier', None)

    def _get_confidence_score(self, h_l):
        """Calculates 'confidence_score' dynamically for adaptive mitigation."""
        if self.classifier is None:
            return 1.0 
        with torch.no_grad():
            return torch.sigmoid(self.classifier(h_l))

    def mitigation_hook(self, alpha=0.1, beta=0.2, mode='projection'):
        def hook(module, input, output):
            # This is the dynamic 'h_l' 
            with torch.no_grad():
                h_l = output#[:, -1, :] 
                # original_representation = output[:, :-1, :]
                
                # Ensure types match
                dtype = h_l.dtype
                mu_T = self.mu_T
                mu_H = self.mu_H
                tsv = self.tsv_vec
                
                if mode == 'interpolation':
                    # Method 1: Prototype Interpolation
                    adjusted_representation = (1 - beta) * h_l + beta * mu_T
                    
                elif mode == 'adaptive':
                    # Method 2: Adaptive Mitigation
                    # confidence_score = self._get_confidence_score(h_l).to(dtype)
                    adjusted_representation = h_l + alpha * tsv
                    
                elif mode == 'projection':
                    #  Method 3: Prototype-Aware Projection 
                    adjusted_representation = h_l + alpha * (mu_T - mu_H)
                    
                else:
                    adjusted_representation = h_l
            
                return adjusted_representation
                # return torch.cat((original_representation, adjusted_representation), dim=1)
            
        return hook

    def attach(self, alpha=0.1, beta=0.2, mode='projection'):
        # print(f"Attaching Hook to Layer {self.layer_id} | Mode: {mode}")
        layer = self.model.model.layers[self.layer_id]
        self.hook_handle = layer.register_forward_hook(
            self.mitigation_hook(alpha, beta, mode)
        )

    def detach(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

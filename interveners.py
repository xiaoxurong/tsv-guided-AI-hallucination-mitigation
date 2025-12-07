import torch
import torch.nn as nn

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False  
    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        if self.head == -1:
            self.states.append(b[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        else:
            self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        return b
    
class ITI_Intervener():
    def __init__(self, direction, centroid_truthful, centroid_hallucinated, alpha, beta, mitigation_type):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction#.cuda().half()

        if not isinstance(centroid_truthful, torch.Tensor):
            centroid_truthful = torch.tensor(centroid_truthful)
        self.centroid_truthful = centroid_truthful

        if not isinstance(centroid_hallucinated, torch.Tensor):
            centroid_hallucinated = torch.tensor(centroid_hallucinated)
        self.centroid_hallucinated = centroid_hallucinated

        self.alpha = alpha
        self.beta = beta
        self.c = 1

        self.mitigation_type = mitigation_type
    def __call__(self, b, s): 
        if self.mitigation_type == "default":
          return b
        elif self.mitigation_type == "1":
          direction = self.direction.to(b.device)
          centroid_truthful = self.centroid_truthful.to(b.device)
          centroid_hallucinated = self.centroid_hallucinated.to(b.device)
          b[0, -1] = self.beta*b[0, -1] + centroid_truthful*self.beta
          return b
        elif self.mitigation_type == "2":
          direction = self.direction.to(b.device)
          centroid_truthful = self.centroid_truthful.to(b.device)

          b[0, -1] = b[0, -1] + self.c*self.alpha*direction
          return b
        elif self.mitigation_type == "3":
          direction = self.direction.to(b.device)
          centroid_truthful = self.centroid_truthful.to(b.device)
          centroid_hallucinated = self.centroid_hallucinated.to(b.device)

          b[0, -1] = b[0, -1] + self.alpha*(centroid_truthful - centroid_hallucinated)
          return b



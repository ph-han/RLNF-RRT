import math
import torch
import torch.nn as nn
from .ConditionalCouplingLayer import ConditionalAffineCouplingLayer

class ConditionalNF(nn.Module):
    def __init__(self, masks, hidden_dim, condition_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionalAffineCouplingLayer(mask, hidden_dim, condition_dim) for mask in masks
        ])

    def forward(self, x, condition):
        y = x
        log_det_tot = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        
        for idx, layer in enumerate(self.layers):
            layer_condition = condition if idx == (len(self.layers)-1) else None
            y, log_det_jacob = layer(y, layer_condition)
            log_det_tot += log_det_jacob

        
        return y, log_det_tot
    
    def inverse(self, y, condition):
        x = y
        log_det_tot = torch.zeros(y.size(0), device=y.device, dtype=y.dtype)

        for idx, layer in enumerate(reversed(self.layers)):
            # Only the original first layer (self.layers[0]) gets the condition.
            layer_condition = condition if layer is self.layers[-1] else None
            x, log_det_jacob = layer.inverse(x, layer_condition)
            log_det_tot += log_det_jacob

        return x, log_det_tot

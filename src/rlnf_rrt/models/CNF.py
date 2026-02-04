# import math
# import torch
# import torch.nn as nn
# from .ConditionalCouplingLayer import ConditionalAffineCouplingLayer

# class ConditionalNF(nn.Module):
#     def __init__(self, masks, hidden_dim, condition_dim):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             ConditionalAffineCouplingLayer(mask, hidden_dim, condition_dim) for mask in masks
#         ])

#     def forward(self, x, condition):
#         y = x
#         log_det_tot = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        
#         for layer in self.layers:
#             y, log_det_jacob = layer(y, condition)
#             log_det_tot += log_det_jacob

        
#         return y, log_det_tot
    
#     def inverse(self, y, condition):
#         x = y
#         log_det_tot = torch.zeros(y.size(0), device=y.device, dtype=y.dtype)

#         for layer in reversed(self.layers):
#             x, log_det_jacob = layer.inverse(x, condition)
#             log_det_tot += log_det_jacob

#         return x, log_det_tot

# CNF.py
import torch
import torch.nn as nn
from .ConditionalCouplingBlock import ConditionalAffineCouplingBlock2Step

class ConditionalNF(nn.Module):
    def __init__(self, masks, hidden_dim, condition_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionalAffineCouplingBlock2Step(mask, hidden_dim, condition_dim) for mask in masks
        ])

    def forward(self, x, condition):
        y = x
        log_det_tot = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for layer in self.layers:
            y, ld = layer(y, condition)
            y = y[:, [1, 0]]
            log_det_tot += ld
        return y, log_det_tot

    def inverse(self, y, condition):
        x = y
        log_det_tot = torch.zeros(y.size(0), device=y.device, dtype=y.dtype)
        for layer in reversed(self.layers):
            x = x[:, [1, 0]]
            x, ld = layer.inverse(x, condition)
            log_det_tot += ld
        return x, log_det_tot

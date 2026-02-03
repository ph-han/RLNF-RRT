import math
import torch
import torch.nn as nn
from .ConditionalCouplingLayer import ConditionalAffineCouplingLayer
from .InvertibleLinear import InvertibleLinear

class ConditionalNF(nn.Module):
    def __init__(self, masks, hidden_dim, condition_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        for mask in masks:
            self.layers.append(InvertibleLinear(len(mask)))
            self.layers.append(
                ConditionalAffineCouplingLayer(mask, hidden_dim, condition_dim)
            )

    def forward(self, x, condition):
        y = x
        log_det_tot = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        for layer in self.layers:
            if isinstance(layer, ConditionalAffineCouplingLayer):
                y, log_det = layer(y, condition)
            else:  # InvertibleLinear
                y, log_det = layer(y)
            log_det_tot += log_det

        return y, log_det_tot

    
    def inverse(self, y, condition):
        x = y
        log_det_tot = torch.zeros(y.size(0), device=y.device, dtype=y.dtype)

        for layer in reversed(self.layers):
            if isinstance(layer, ConditionalAffineCouplingLayer):
                x, log_det = layer.inverse(x, condition)
            else:  # InvertibleLinear
                x, log_det = layer.inverse(x)
            log_det_tot += log_det

        return x, log_det_tot
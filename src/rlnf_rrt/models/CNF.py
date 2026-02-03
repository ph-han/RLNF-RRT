import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        for layer in self.layers:
            y, log_det_jacob = layer(y, condition)
            log_det_tot += log_det_jacob

        # tanh_y = torch.tanh(y)
        # y = (tanh_y + 1.0) / 2.0

        
        # log2 = math.log(2.0)
        # log_det_tanh = torch.sum(torch.log(1.0 - sigmoid_y**2 + 1e-6) - log2, dim=-1)
        # log_det_tot += log_det_tanh
        
        log_det_sigmoid = torch.sum(-F.softplus(-y) - F.softplus(y), dim=-1)
        y = torch.sigmoid(y)
        log_det_tot += log_det_sigmoid

        return y, log_det_tot
    
    def inverse(self, y, condition):
        y = torch.clamp(y, 1e-6, 1.0 - 1e-6)
        x = torch.log(y / (1.0 - y))

        
        # log2 = math.log(2.0)
        # log_det_tanh_inv = -torch.sum(torch.log(1.0 - sigmoid_y**2 + 1e-6) - log2, dim=-1)
        
        # x = 0.5 * torch.log((1.0 + sigmoid_y) / (1.0 - sigmoid_y + 1e-6))
        
        # log_det_tot = log_det_tanh_inv

        log_det_logit = -torch.sum(-F.softplus(-x) - F.softplus(x), dim=-1)
        log_det_tot = log_det_logit

        for layer in reversed(self.layers):
            x, log_det_jacob = layer.inverse(x, condition)
            log_det_tot += log_det_jacob

        return x, log_det_tot

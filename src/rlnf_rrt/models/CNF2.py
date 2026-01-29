import torch
import torch.nn as nn

from .ConditionalCouplingLayer import ConditionalAffineCouplingLayer

class ConditionalNF(nn.Module):
    def __init__(self, masks, hidden_dim, condition_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            ConditionalAffineCouplingLayer(mask, self.hidden_dim, condition_dim) for mask in masks
        ])


    def forward(self, x, condition):
        y = x
        log_det_tot = 0
        for layer in self.layers:
            y, log_det_jacob = layer(y, condition)
            log_det_tot += log_det_jacob

        # tanh_y = torch.tanh(y)
        # y = (tanh_y + 1.0) / 2.0
        
        # log_det_tanh = torch.sum(torch.log(1.0 - tanh_y**2 + 1e-6) - torch.log(torch.tensor(2.0)), dim=-1)
        # log_det_tot += log_det_tanh
        
        return y, log_det_tot
    
    def inverse(self, y, condition):
        x = y
        # y_clamped = torch.clamp(y * 2.0 - 1.0, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # log_det_tanh_inv = -torch.sum(torch.log(1.0 - y_clamped**2 + 1e-6) - torch.log(torch.tensor(2.0)), dim=-1)
        
        # x = 0.5 * torch.log((1.0 + y_clamped) / (1.0 - y_clamped))
        
        # log_det_tot = log_det_tanh_inv

        for layer in reversed(self.layers):
            x, log_det_jacob = layer.inverse(x, condition)
            log_det_tot += log_det_jacob

        return x, log_det_tot

import torch
import torch.nn as nn
import math


class FlowNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_2pi = math.log(2 * math.pi)
    
    def forward(self, z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
        # Log probability of z under standard Gaussian: -0.5 * (z^2 + log(2π))
        # Sum over all dimensions (T, D) for each sample
        log_prob_z = -0.5 * (z.pow(2) + self.log_2pi).sum(dim=(1, 2))  # (B,)
        
        # NLL = -log p(x) = -(log p(z) + log|det|)
        nll = -(log_prob_z + log_det)  # (B,)
        
        return nll.mean()


def compute_bits_per_dim(nll: torch.Tensor, num_dims: int) -> torch.Tensor:
    return nll / (num_dims * math.log(2))

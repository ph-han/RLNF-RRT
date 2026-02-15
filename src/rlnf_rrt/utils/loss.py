import torch
import math

def _nll_loss(z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
    # z: (B,T,2)
    D = z.size(-1)
    log2pi = math.log(2.0 * math.pi)
    log_pz = -0.5 * (z.pow(2).sum(dim=-1) + D * log2pi)  # (B,T)
    log_pz = log_pz.sum(dim=1)                           # (B,)
    log_px = log_pz + log_det                            # (B,)
    return -log_px.mean()
import torch
from torch import nn
from rlnf_rrt.models.cond_encoder import CondEncoder

class CondFiLM(nn.Module):
    """
    FiLM modulation for a condition vector c: (B, D).
    Keeps shape and bounds perturbation to avoid OOD blow-ups.
    """
    def __init__(self, cond_dim: int, action_dim: int, scale: float = 0.1, hidden: int = 128):
        super().__init__()
        self.cond_dim = cond_dim
        self.scale = float(scale)

        # Map RL action -> (gamma, beta) in R^{cond_dim}
        self.to_gb = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * cond_dim),
        )

    def forward(self, c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        c: (B, D)
        a: (B, A)
        returns c': (B, D)
        """
        gb = self.to_gb(a)
        gamma, beta = gb.chunk(2, dim=-1)

        s = self.scale
        c_mod = (1.0 + s * torch.tanh(gamma)) * c + s * torch.tanh(beta)
        return c_mod
    

class CondEncoderWithFiLM(nn.Module):
    def __init__(self, base: CondEncoder, action_dim: int, scale: float = 4, hidden: int = 128):
        super().__init__()
        self.base = base
        cond_dim = base.map_encoder.proj[-1].out_features + 4  # latent_dim + start/goal(4)
        self.film = CondFiLM(cond_dim=cond_dim, action_dim=action_dim, scale=scale, hidden=hidden)

    def forward(self, cond_image: torch.Tensor, start: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        c = self.base(cond_image, start, goal)      # (B, 132)
        c = self.film(c, action)                   # (B, 132)
        return c
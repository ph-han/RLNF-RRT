from __future__ import annotations

import torch
import torch.nn as nn

from rlnf_rrt.models.cond_encoder import MapEncoder


class ValueNet(nn.Module):
    """
    State value function V(s) for PPO critic.

    (cond_image, start, goal) → scalar value estimate
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, backbone: str = "resnet34"):
        super().__init__()
        self.map_encoder = MapEncoder(latent_dim=latent_dim, backbone=backbone)
        feat_dim = latent_dim + 4  # map_feat + start(2) + goal(2)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cond_image: torch.Tensor,  # (3, H, W) or (B, 3, H, W)
        start: torch.Tensor,       # (2,) or (B, 2)
        goal: torch.Tensor,        # (2,) or (B, 2)
    ) -> torch.Tensor:             # () or (B,)
        unbatched = cond_image.dim() == 3
        if unbatched:
            cond_image = cond_image.unsqueeze(0)
            start = start.unsqueeze(0)
            goal = goal.unsqueeze(0)

        feat = self.map_encoder(cond_image)                    # (B, latent_dim)
        feat = torch.cat([feat, start, goal], dim=-1)          # (B, latent_dim+4)
        v = self.mlp(feat).squeeze(-1)                         # (B,)

        return v.squeeze(0) if unbatched else v

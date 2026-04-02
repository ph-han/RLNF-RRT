from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlnf_rrt.models.cond_encoder import MapEncoder

class SubGoalPolicy(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        backbone: str = "resnet34",
        num_subgoals: int = 1,
        pretrained_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.num_subgoals = num_subgoals
        out_dim = num_subgoals * 2

        if pretrained_encoder is not None:
            self.map_encoder = pretrained_encoder
        else:
            self.map_encoder = MapEncoder(latent_dim=latent_dim, backbone=backbone)
        feat_dim = latent_dim + 4  # map_feat + start(2) + goal(2)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head   = nn.Linear(hidden_dim, out_dim)
        self.conc_head = nn.Linear(hidden_dim, out_dim)

        # zeros init: 기본 출력 0 → mean=midpoint, conc=softplus(0)+2≈2.69
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.conc_head.weight)
        nn.init.zeros_(self.conc_head.bias)

    def forward(
        self,
        cond_image: torch.Tensor,  # (3, H, W) or (B, 3, H, W)
        start: torch.Tensor,       # (2,) or (B, 2)
        goal: torch.Tensor,        # (2,) or (B, 2)
        map_feat_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unbatched = cond_image.dim() == 3
        if unbatched:
            cond_image = cond_image.unsqueeze(0)
            start = start.unsqueeze(0)
            goal = goal.unsqueeze(0)

        if map_feat_cache is not None:
            feat = map_feat_cache
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
        else:
            feat = self.map_encoder(cond_image)             # (B, latent_dim)
        feat = torch.cat([feat, start, goal], dim=-1)   # (B, latent_dim+4)
        h = self.mlp(feat)

        mean = torch.sigmoid(self.mu_head(h))
        conc = F.softplus(self.conc_head(h)) + 2

        alpha = mean * conc
        beta  = (1 - mean) * conc

        if unbatched:
            alpha = alpha.squeeze(0)
            beta  = beta.squeeze(0)

        return alpha, beta

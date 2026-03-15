from __future__ import annotations

import torch
import torch.nn as nn

from rlnf_rrt.models.cond_encoder import MapEncoder


class SubGoalPolicy(nn.Module):
    """
    Sub-goal policy for REINFORCE training.

    (cond_image, start, goal) → (loc, log_scale)  for TanhNormal distribution
    sub_goal ~ TanhNormal(loc, exp(log_scale)) ∈ [0, 1]²

    MapEncoder를 재사용하여 맵 특징 추출 후 MLP로 분포 파라미터를 예측.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        backbone: str = "resnet34",
        num_subgoals: int = 1,
    ):
        super().__init__()
        self.num_subgoals = num_subgoals
        out_dim = num_subgoals * 2

        self.map_encoder = MapEncoder(latent_dim=latent_dim, backbone=backbone)
        feat_dim = latent_dim + 4  # map_feat + start(2) + goal(2)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.loc_head = nn.Linear(hidden_dim, out_dim)
        self.log_scale_head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        cond_image: torch.Tensor,  # (3, H, W) or (B, 3, H, W)
        start: torch.Tensor,       # (2,) or (B, 2)
        goal: torch.Tensor,        # (2,) or (B, 2)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # torchrl batch_size=[] 환경에서 오는 unbatched 입력 처리
        unbatched = cond_image.dim() == 3
        if unbatched:
            cond_image = cond_image.unsqueeze(0)
            start = start.unsqueeze(0)
            goal = goal.unsqueeze(0)

        feat = self.map_encoder(cond_image)                       # (B, latent_dim)
        feat = torch.cat([feat, start, goal], dim=-1)             # (B, latent_dim+4)

        h = self.mlp(feat)
        loc = self.loc_head(h)                                    # (B, num_sg*2)
        log_scale = self.log_scale_head(h).clamp(-4.0, 2.0)      # (B, num_sg*2)

        if unbatched:
            loc = loc.squeeze(0)
            log_scale = log_scale.squeeze(0)

        return loc, log_scale.exp()

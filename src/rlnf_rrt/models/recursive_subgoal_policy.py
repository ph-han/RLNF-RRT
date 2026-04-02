from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class RecursiveSubgoalPolicyConfig:
    latent_dim: int = 128
    hidden_dim: int = 128
    backbone: str = "resnet34"
    use_complexity_feats: bool = True
    sg_embed_dim: int = 32

class RecursiveSubgoalPolicy(nn.Module):

    def __init__(
        self,
        cfg: RecursiveSubgoalPolicyConfig,
        pretrained_encoder: nn.Module
    ):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.hidden_dim = cfg.hidden_dim
        self.map_encoder = pretrained_encoder

        # map_feat conditioning: map_feat + sg_embed → conditioned_feat
        self.sg_projector = nn.Linear(4, cfg.sg_embed_dim)
        self.map_conditioner = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.sg_embed_dim, cfg.latent_dim),
            nn.SiLU(),
        )

        state_dim = cfg.latent_dim + 4
        # Shared trunk
        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )

        # Split head → split logit (scalar)
        self.split_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

        # Midpoint head → Beta distribution params (2D)
        self.mu_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim // 2, 2)
        )

        self.conc_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim // 2, 2)
        )

        # Zero init for split head (split_prob ≈ 0.5)
        nn.init.zeros_(self.split_head[-1].weight)
        nn.init.zeros_(self.split_head[-1].bias)

        # Zero init for midpoint heads (mean=midpoint of Beta, conc=softplus(0)+2≈2.69)
        nn.init.zeros_(self.mu_head[-1].weight)
        nn.init.zeros_(self.mu_head[-1].bias)
        nn.init.zeros_(self.conc_head[-1].weight)
        nn.init.zeros_(self.conc_head[-1].bias)


    def forward(
        self,
        cond_image: torch.Tensor,  # (3, H, W) or (B, 3, H, W)
        start_goal: tuple[torch.Tensor, torch.Tensor],
        map_feat_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start, goal = start_goal
        unbatched = cond_image.dim() == 3
        device = cond_image.device

        start = start.to(device)
        goal = goal.to(device)
        if unbatched:
            cond_image = cond_image.unsqueeze(0)
        
        if start.dim() == 1:
            start = start.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)

        if map_feat_cache is not None:
            map_feat = map_feat_cache
            if map_feat.dim() == 1:
                map_feat = map_feat.unsqueeze(0)
        else:
            map_feat = self.map_encoder(cond_image)

        sg_embed = self.sg_projector(torch.cat([start, goal], dim=-1))  # (B, sg_embed_dim)
        conditioned_feat = self.map_conditioner(torch.cat([map_feat, sg_embed], dim=-1))  # (B, latent_dim)
        
        state = torch.cat([conditioned_feat, start, goal], dim=-1) # (B, latent_dim + 4)

        h = self.shared_mlp(state) # (B, hidden_dim)

        # Split head
        split_logit = self.split_head(h).squeeze(-1)  # (B,)
        split_prob = torch.sigmoid(split_logit)

        # Midpoint head
        mean = torch.sigmoid(self.mu_head(h))  # (B, 2)
        mean = mean.clamp(min=1e-4, max=1.0 - 1e-4)
        conc = F.softplus(self.conc_head(h)) + 2  # (B, 2)

        alpha = mean * conc
        beta = (1 - mean) * conc

        if unbatched:
            split_prob = split_prob.squeeze(0)
            alpha = alpha.squeeze(0)
            beta  = beta.squeeze(0)

        return split_prob, alpha, beta, map_feat

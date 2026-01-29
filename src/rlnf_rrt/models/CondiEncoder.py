import torch
import torch.nn as nn

from rlnf_rrt.models.MapEncoder import MapEncoder

class CondiEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.map_encoder = MapEncoder(latent_dim=latent_dim)
        self.start_goal_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.combi_encoder = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, map_img, start, goal):
        w = self.map_encoder(map_img)
        pos_cond = torch.cat([start, goal], dim=-1)
        pos_cond = self.start_goal_encoder(pos_cond)

        condition = torch.cat([w, pos_cond], dim=-1)
        condition = self.combi_encoder(condition)
        return condition

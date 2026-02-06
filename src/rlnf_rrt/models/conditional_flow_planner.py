import torch
import torch.nn as nn

from rlnf_rrt.models.condition_encoder import ConditionEncoder

class ConditionalFlowPlanner(nn.Module):
    def __init__(self, sg_dim:int=2, position_embed_dim:int=128, map_embed_dim:int=256, cond_dim:int=128):
        super().__init__()
        self.condition_encoder:ConditionEncoder = ConditionEncoder(sg_dim, position_embed_dim, map_embed_dim, cond_dim)

    def forward(self, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor):
        # WIG: here we will add the flow model
        cond = self.condition_encoder(map_img, start, goal)

        return cond

    def sample(self, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor):
        pass
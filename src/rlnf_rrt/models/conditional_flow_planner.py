import torch
import torch.nn as nn

from rlnf_rrt.models.condition_encoder import ConditionEncoder
from rlnf_rrt.models.coupling_block import CouplingBlock

class ConditionalFlowPlanner(nn.Module):
    def __init__(self, num_blocks:int=4, sg_dim:int=2, position_embed_dim:int=128, map_embed_dim:int=256, cond_dim:int=128):
        super().__init__()
        self.sg_dim = sg_dim
        self.condition_encoder:ConditionEncoder = ConditionEncoder(sg_dim, position_embed_dim, map_embed_dim, cond_dim)

        self.flow_model:nn.ModuleList = nn.ModuleList([
            CouplingBlock(cond_dim=cond_dim) for _ in range(num_blocks)
        ])

    def forward(self, gt_trajs:torch.Tensor, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor):
        cond = self.condition_encoder(map_img, start, goal)
        cond = cond.unsqueeze(1).expand(-1, gt_trajs.shape[1], -1)

        x = gt_trajs
        log_det = torch.zeros(gt_trajs.shape[0], device=gt_trajs.device)
        for block in self.flow_model:
            x, log_det_block = block(x, cond)
            log_det += log_det_block

        return x, log_det

    def sample(self, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor, num_samples:int=1000):
        cond = self.condition_encoder(map_img, start, goal)
        cond = cond.unsqueeze(1).expand(-1, num_samples, -1)

        batch_size = map_img.shape[0]
        z = torch.randn(batch_size, num_samples, self.sg_dim, device=map_img.device)
        for block in reversed(self.flow_model):
            z = block.inverse(z, cond)

        return z
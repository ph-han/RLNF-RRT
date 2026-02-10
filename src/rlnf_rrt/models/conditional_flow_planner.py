import torch
import torch.nn as nn

from rlnf_rrt.models.condition_encoder import ConditionEncoderFiLMRawSG
from rlnf_rrt.models.coupling_block import CouplingBlock

class ConditionalFlowPlanner(nn.Module):
    def __init__(self, num_blocks:int=4, sg_dim:int=2, map_embed_dim:int=256, cond_dim:int=128, hidden_dim:int=128, s_max:float=2.0):
        super().__init__()
        self.sg_dim = sg_dim
        self.condition_encoder:ConditionEncoderFiLMRawSG = ConditionEncoderFiLMRawSG(map_embed_dim, sg_dim)
        
        # ConditionEncoderFiLMRawSG outputs: start(d) + goal(d) + diff(d) + dist(1) = 3*d + 1
        encoded_sg_dim = sg_dim * 3 + 1

        self.flow_model:nn.ModuleList = nn.ModuleList([
            CouplingBlock(sg_dim=encoded_sg_dim, map_dim=map_embed_dim, hidden_dim=hidden_dim, s_max=s_max) for _ in range(num_blocks)
        ])

    def forward(self, gt_trajs:torch.Tensor, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor):
        # cond = self.condition_encoder(map_img, start, goal)
        # cond = cond.unsqueeze(1).expand(-1, gt_trajs.shape[1], -1)

        map_feat, sg_feat = self.condition_encoder(map_img, start, goal)  # (B,map_dim), (B,7)
        T = gt_trajs.shape[1]
        sg_feat_T = sg_feat.unsqueeze(1).repeat(1, T, 1)                  # (B,T,7)

        x = gt_trajs
        log_det = torch.zeros(gt_trajs.shape[0], device=gt_trajs.device)
        for block in self.flow_model:
            x, log_det_block = block(x, sg_feat_T, map_feat)
            log_det += log_det_block

            x = x[..., [1, 0]] # permutation
        return x, log_det

    def sample(self, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor, num_samples:int=1000):
        map_feat, sg_feat = self.condition_encoder(map_img, start, goal)  # (B,map_dim), (B,7)
        T = gt_trajs.shape[1]
        sg_feat_T = sg_feat.unsqueeze(1).repeat(1, T, 1)                  # (B,T,7)

        batch_size = map_img.shape[0]
        z = torch.randn(batch_size, num_samples, self.sg_dim, device=map_img.device)
        for block in reversed(self.flow_model):
            z = z[..., [1, 0]] # permutation
            z = block.inverse(z, sg_feat_T, map_feat)

        return z
    
    def sample_with_intermediates(self, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor, num_samples:int=1000):
        """Sample from the flow model and return intermediate transformations.
        
        Returns:
            intermediates: List of (num_blocks + 1) tensors, each of shape (batch_size, num_samples, 2)
                          [z0, z1, z2, ..., x_final]
        """
        map_feat, sg_feat = self.condition_encoder(map_img, start, goal)  # (B,map_dim), (B,7)
        T = gt_trajs.shape[1]
        sg_feat_T = sg_feat.unsqueeze(1).repeat(1, T, 1)                  # (B,T,7)

        batch_size = map_img.shape[0]
        z = torch.randn(batch_size, num_samples, self.sg_dim, device=map_img.device)
        
        # Store intermediates
        intermediates = [z.clone()]
        
        # Apply inverse transforms
        for block in reversed(self.flow_model):
            z = z[..., [1, 0]] # permutation
            z = block.inverse(z, sg_feat_T, map_feat)
            intermediates.append(z.clone())
        
        return intermediates

    def forward_with_intermediates(self, gt_trajs:torch.Tensor, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor):
        """Pass input through the flow model and return intermediate transformations.
        
        Args:
            gt_trajs: Input trajectories (batch_size, seq_len, 2)
            map_img: Map image (batch_size, 1, 64, 64)
            start: Start position (batch_size, 2)
            goal: Goal position (batch_size, 2)
            
        Returns:
            intermediates: List of (num_blocks + 1) tensors, each of shape (batch_size, seq_len, 2)
                          [x_data, x1, x2, ..., x_final(z)]
        """
        map_feat, sg_feat = self.condition_encoder(map_img, start, goal)  # (B,map_dim), (B,7)
        T = gt_trajs.shape[1]
        sg_feat_T = sg_feat.unsqueeze(1).repeat(1, T, 1)                  # (B,T,7)
        cond = torch.cat([map_feat, sg_feat_T], dim=-1)  # (B, T, cond_dim)

        x = gt_trajs
        
        # Store intermediates
        intermediates = [x.clone()]
        
        # Apply forward transforms
        for block in self.flow_model:
            x, _ = block(x, sg_feat_T, map_feat)  # We don't need log_det for visualization
            intermediates.append(x.clone())

            x = x[..., [1, 0]] # permutation
            
        return intermediates
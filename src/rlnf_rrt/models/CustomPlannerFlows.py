import math
import torch
import torch.nn as nn

from rlnf_rrt.models.CNF import ConditionalNF
from rlnf_rrt.models.MapEncoder import MapEncoder

class CustomPlannerFlows(nn.Module):
    def __init__(self, masks, hidden_dim, env_latent_dim, state_dim=2):
        super().__init__()
        self.map_encoder = MapEncoder(latent_dim=env_latent_dim)
        
        self.condition_dim = env_latent_dim + state_dim * 2
        
        self.flow = ConditionalNF(masks, hidden_dim, self.condition_dim)

    def _get_condition(self, map_img, start, goal):
        return torch.cat([self.map_encoder(map_img), start, goal], dim=-1)
        

    def forward(self, map_img, start, goal, num_samples=1):
        condition = self._get_condition(map_img, start, goal)
        
        batch_size = map_img.size(0)
        z = torch.randn(batch_size, num_samples, self.flow.layers[0].input_dim,
                        device=map_img.device, dtype=torch.float32) # gaussian distribution
    
        condition_rep = condition.unsqueeze(1).repeat(1, num_samples, 1)
        q_samples, log_det = self.flow.forward(z.view(-1, z.size(-1)), condition_rep.view(-1, condition_rep.size(-1)))
        
        return q_samples.view(batch_size, num_samples, -1), log_det.view(batch_size, num_samples)
    

    def inverse(self, gt_points, map_img, start, goal):
        condition = self._get_condition(map_img, start, goal)
        B, K, D = gt_points.shape # B: 배치, K: 포인트 수, D: 차원(2)
        
        gt_points = gt_points.view(B * K, D)
        condition = condition.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        
        z, log_det = self.flow.inverse(gt_points, condition)
        
        log_2pi = math.log(2.0 * math.pi)
        prior_log_prob = -0.5 * (torch.sum(z**2, dim=-1) + D * log_2pi)
        
        log_likelihood = prior_log_prob + log_det
        
        return -log_likelihood.mean()
import torch
import torch.nn as nn

from rlnf_rrt.models.CNF import ConditionalNF
from rlnf_rrt.models.CondiEncoder import CondiEncoder

class CustomPlannerFlows(nn.Module):
    def __init__(self, masks, hidden_dim, env_latent_dim, state_dim=2):
        super().__init__()
        self.encoder = CondiEncoder(latent_dim=env_latent_dim)
        
        self.condition_dim = env_latent_dim + state_dim * 2
        
        self.flow = ConditionalNF(masks, hidden_dim, self.condition_dim)

    def _get_condition(self, map_img, start, goal):
        return self.encoder(map_img, start, goal)
        

    def forward(self, map_img, start, goal, num_samples=1):
        condition = self._get_condition(map_img, start, goal)
        
        batch_size = map_img.size(0)
        z = torch.randn(batch_size, num_samples, self.flow.layers[0].input_dim).to(map_img.device)
    
        condition_rep = condition.unsqueeze(1).repeat(1, num_samples, 1)
        q_samples, ll = self.flow.forward(z.view(-1, z.size(-1)), condition_rep.view(-1, condition_rep.size(-1)))
        
        return q_samples.view(batch_size, num_samples, -1), ll

    def get_nll(self, gt_points, map_img, start, goal):
        condition = self._get_condition(map_img, start, goal)
        B, K, D = gt_points.shape
        gt_points = gt_points.view(B * K, D)
        condition = condition.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        z, log_det = self.flow.inverse(gt_points, condition)
        
        prior_log_prob = -0.5 * torch.sum(z**2, dim=-1)
        log_likelihood = prior_log_prob + log_det
        
        return -log_likelihood.mean()

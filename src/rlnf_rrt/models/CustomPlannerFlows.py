# rlnf_rrt/models/CustomPlannerFlows.py
import torch
import torch.nn as nn

from rlnf_rrt.models.CNF import ConditionalNF
from rlnf_rrt.models.MapEncoder import MapEncoder

class CustomPlannerFlows(nn.Module):
    """
    CKPT(v7 등)과 '호환'되는 버전:
    - base z ~ N(0, I)
    - flow coupling에는 condition만 들어감
    """
    def __init__(self, masks, hidden_dim, env_latent_dim, state_dim=2):
        super().__init__()
        self.map_encoder = MapEncoder(latent_dim=env_latent_dim)
        self.condition_dim = env_latent_dim + state_dim * 2
        self.flow = ConditionalNF(masks, hidden_dim, self.condition_dim)
        self.data_dim = self.flow.layers[0].input_dim  # usually 2

    def _get_condition(self, map_img, start, goal):
        return torch.cat([self.map_encoder(map_img), start, goal], dim=-1)

    @torch.no_grad()
    def sample(self, map_img, start, goal, num_samples=1):
        condition = self._get_condition(map_img, start, goal)  # (B,C)
        B = condition.size(0)
        cond_rep = condition.unsqueeze(1).expand(B, num_samples, -1).reshape(B*num_samples, -1)

        z = torch.randn(B*num_samples, self.data_dim, device=map_img.device)
        x, log_det = self.flow.forward(z, cond_rep)  # z -> x
        return x.view(B, num_samples, -1), log_det.view(B, num_samples)

    def nll(self, gt_points, map_img, start, goal):
        """
        gt_points: (B,K,D) in same x-space that the model was trained on (e.g., [-1,1])
        return: scalar NLL
        """
        condition = self._get_condition(map_img, start, goal)  # (B,C)
        B, K, D = gt_points.shape
        x = gt_points.reshape(B*K, D)
        cond_rep = condition.unsqueeze(1).expand(B, K, -1).reshape(B*K, -1)

        z, log_det = self.flow.inverse(x, cond_rep)  # x -> z
        # standard normal prior log prob
        log_2pi = 1.8378770664093453  # log(2*pi)
        prior_logprob = -0.5 * (z.pow(2).sum(dim=-1) + D*log_2pi)
        ll = prior_logprob + log_det
        return -ll.mean()

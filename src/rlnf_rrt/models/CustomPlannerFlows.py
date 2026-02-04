# CustomPlannerFlows.py (수정본 핵심)
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
        self.data_dim = self.flow.layers[0].input_dim  # 보통 2

        # ✅ Conditional prior nets
        self.prior_mu = nn.Sequential(
            nn.Linear(self.condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.data_dim),
        )
        self.prior_logsigma = nn.Sequential(
            nn.Linear(self.condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.data_dim),
        )

        # 안정화: 초기엔 mu≈0, logsigma≈0 (즉 sigma≈1) 되게
        nn.init.zeros_(self.prior_mu[-1].weight)
        nn.init.zeros_(self.prior_mu[-1].bias)
        nn.init.zeros_(self.prior_logsigma[-1].weight)
        nn.init.zeros_(self.prior_logsigma[-1].bias)

        self.logsigma_clip = 5.0  # sigma 폭주 방지

    def _get_condition(self, map_img, start, goal):
        return torch.cat([self.map_encoder(map_img), start, goal], dim=-1)

    def _prior_params(self, condition):
        mu = self.prior_mu(condition)
        log_sigma = self.prior_logsigma(condition).clamp(-self.logsigma_clip, self.logsigma_clip)
        return mu, log_sigma

    def forward(self, map_img, start, goal, num_samples=1):
        condition = self._get_condition(map_img, start, goal)          # (B, C)
        B = condition.size(0)

        cond_rep = condition.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)

        # ✅ conditional prior로 z 샘플링
        mu, log_sigma = self._prior_params(cond_rep)                   # (B*num_samples, D)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(log_sigma) * eps

        q_samples, log_det = self.flow.forward(z, cond_rep)            # z -> x
        q_samples = q_samples.view(B, num_samples, -1)
        log_det = log_det.view(B, num_samples)
        return q_samples, log_det

    def inverse(self, gt_points, map_img, start, goal):
        condition = self._get_condition(map_img, start, goal)          # (B, C)
        B, K, D = gt_points.shape

        x = gt_points.reshape(B * K, D)
        cond_rep = condition.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

        z, log_det = self.flow.inverse(x, cond_rep)                    # x -> z

        # ✅ conditional prior log prob
        mu, log_sigma = self._prior_params(cond_rep)
        inv_sigma = torch.exp(-log_sigma)
        log_2pi = math.log(2.0 * math.pi)
        # log N(z; mu, sigma)
        prior_log_prob = -0.5 * (
            torch.sum(((z - mu) * inv_sigma) ** 2, dim=-1)
            + 2.0 * torch.sum(log_sigma, dim=-1)
            + D * log_2pi
        )

        log_likelihood = prior_log_prob + log_det
        return -log_likelihood.mean()

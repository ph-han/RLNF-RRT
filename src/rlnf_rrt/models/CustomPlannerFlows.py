# CustomPlannerFlows.py
import math
import torch
import torch.nn as nn

from rlnf_rrt.models.CNF import ConditionalNF
from rlnf_rrt.models.MapEncoder import MapEncoder


class CustomPlannerFlows(nn.Module):
    def __init__(self, masks, hidden_dim, env_latent_dim, state_dim=2):
        super().__init__()

        # -----------------------------
        # Condition encoder
        # -----------------------------
        self.map_encoder = MapEncoder(latent_dim=env_latent_dim)
        self.condition_dim = env_latent_dim + state_dim * 2

        # -----------------------------
        # Flow
        # -----------------------------
        self.flow = ConditionalNF(masks, hidden_dim, self.condition_dim)
        self.data_dim = self.flow.layers[0].input_dim  # usually 2

        # -----------------------------
        # Conditional prior (PlannerFlows-style)
        # -----------------------------
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

        # Identity-like init (important)
        nn.init.zeros_(self.prior_mu[-1].weight)
        nn.init.zeros_(self.prior_mu[-1].bias)
        nn.init.zeros_(self.prior_logsigma[-1].weight)
        nn.init.zeros_(self.prior_logsigma[-1].bias)

        # -----------------------------
        # 🔥 핵심 안정화 파라미터
        # -----------------------------
        self.min_log_sigma = -1.5   # sigma >= exp(-1.5) ≈ 0.22
        self.max_log_sigma =  2.0   # sigma <= exp(2.0)  ≈ 7.4
        self.logsigma_reg_weight = 1e-3

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------
    def _get_condition(self, map_img, start, goal):
        return torch.cat(
            [self.map_encoder(map_img), start, goal],
            dim=-1
        )

    def _prior_params(self, condition):
        mu = self.prior_mu(condition)
        log_sigma = self.prior_logsigma(condition).clamp(
            self.min_log_sigma,
            self.max_log_sigma
        )
        return mu, log_sigma

    # -------------------------------------------------
    # Forward: sample (Gaussian -> path space)
    # -------------------------------------------------
    def forward(self, map_img, start, goal, num_samples=1):
        condition = self._get_condition(map_img, start, goal)   # (B, C)
        B = condition.size(0)

        cond_rep = (
            condition.unsqueeze(1)
            .expand(B, num_samples, -1)
            .reshape(B * num_samples, -1)
        )

        # Conditional prior sampling
        mu, log_sigma = self._prior_params(cond_rep)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(log_sigma) * eps

        x, log_det = self.flow.forward(z, cond_rep)

        x = x.view(B, num_samples, -1)
        log_det = log_det.view(B, num_samples)

        return x, log_det

    # -------------------------------------------------
    # Inverse: training (path space -> Gaussian)
    # -------------------------------------------------
    def inverse(self, gt_points, map_img, start, goal):
        """
        gt_points: (B, K, 2)
        """
        condition = self._get_condition(map_img, start, goal)
        B, K, D = gt_points.shape

        x = gt_points.reshape(B * K, D)
        cond_rep = (
            condition.unsqueeze(1)
            .expand(B, K, -1)
            .reshape(B * K, -1)
        )

        # Inverse flow
        z, log_det = self.flow.inverse(x, cond_rep)

        # Conditional prior
        mu, log_sigma = self._prior_params(cond_rep)
        inv_sigma = torch.exp(-log_sigma)

        log_2pi = math.log(2.0 * math.pi)

        prior_log_prob = -0.5 * (
            torch.sum(((z - mu) * inv_sigma) ** 2, dim=-1)
            + 2.0 * torch.sum(log_sigma, dim=-1)
            + D * log_2pi
        )

        log_likelihood = prior_log_prob + log_det

        # 🔥 sigma collapse 방지 정규화
        logsigma_reg = self.logsigma_reg_weight * (log_sigma ** 2).mean()

        loss = -log_likelihood.mean() + logsigma_reg
        return loss

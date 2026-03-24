from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlnf_rrt.models.cond_encoder import MapEncoder

class SubGoalPolicy(nn.Module):
    """
    Sub-goal policy for REINFORCE training.

    (cond_image, start, goal) → (alpha, beta)  for Beta distribution
    sub_goal ~ Beta(alpha, beta) ∈ [0, 1]²

    mean은 start-goal midpoint로 초기화되어 있어, 학습 전에도 합리적인 예측.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        backbone: str = "resnet34",
        num_subgoals: int = 1,
        pretrained_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.num_subgoals = num_subgoals
        out_dim = num_subgoals * 2

        if pretrained_encoder is not None:
            self.map_encoder = pretrained_encoder
        else:
            self.map_encoder = MapEncoder(latent_dim=latent_dim, backbone=backbone)
        feat_dim = latent_dim + 4  # map_feat + start(2) + goal(2)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head   = nn.Linear(hidden_dim, out_dim)
        self.conc_head = nn.Linear(hidden_dim, out_dim)

        # zeros init: 기본 출력 0 → mean=midpoint, conc=softplus(0)+2≈2.69
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.conc_head.weight)
        nn.init.zeros_(self.conc_head.bias)

    def forward(
        self,
        cond_image: torch.Tensor,  # (3, H, W) or (B, 3, H, W)
        start: torch.Tensor,       # (2,) or (B, 2)
        goal: torch.Tensor,        # (2,) or (B, 2)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unbatched = cond_image.dim() == 3
        if unbatched:
            cond_image = cond_image.unsqueeze(0)
            start = start.unsqueeze(0)
            goal = goal.unsqueeze(0)

        feat = self.map_encoder(cond_image)             # (B, latent_dim)
        feat = torch.cat([feat, start, goal], dim=-1)   # (B, latent_dim+4)
        h = self.mlp(feat)

        mean = torch.sigmoid(self.mu_head(h))
        conc = F.softplus(self.conc_head(h)) + 2

        alpha = mean * conc
        beta  = (1 - mean) * conc

        if unbatched:
            alpha = alpha.squeeze(0)
            beta  = beta.squeeze(0)

        return alpha, beta


class AutoregressiveSubGoalPolicyCount(nn.Module):
    """
    Autoregressive sub-goal policy with variable count prediction.

    1단계: count head로 K ~ Categorical(0..max_subgoals) 샘플링
    2단계: AR로 K개 sub-goal 순서대로 예측
           SG_i ~ Beta(f(map, SG_{i-1}, goal))  where SG_0 = start

    Weight sharing: 모든 AR step이 같은 MLP/mu/conc head 파라미터 사용.
    """

    def __init__(
        self,
        max_subgoals: int = 4,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        backbone: str = "resnet34",
        pretrained_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.max_subgoals = max_subgoals

        if pretrained_encoder is not None:
            self.map_encoder = pretrained_encoder
        else:
            self.map_encoder = MapEncoder(latent_dim=latent_dim, backbone=backbone)
        feat_dim = latent_dim + 4  # map_feat + prev_point(2) + goal(2)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # AR heads: 한 번에 SG 1개(x,y)만 출력 — weight shared across AR steps
        self.mu_head    = nn.Linear(hidden_dim, 2)
        self.conc_head  = nn.Linear(hidden_dim, 2)
        # Count head: initial (map, start, goal) 기반으로 K 예측
        self.count_head = nn.Linear(hidden_dim, max_subgoals + 1)

        for head in (self.mu_head, self.conc_head, self.count_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def encode_map(self, cond_image: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, latent_dim). 에피소드당 한 번만 호출."""
        return self.map_encoder(cond_image)

    def forward_step(
        self,
        map_feat: torch.Tensor,    # (B, latent_dim)
        prev_point: torch.Tensor,  # (B, 2)
        goal: torch.Tensor,        # (B, 2)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """단일 AR step. returns alpha, beta: (B, 2)."""
        feat = torch.cat([map_feat, prev_point, goal], dim=-1)
        h = self.mlp(feat)
        mean = torch.sigmoid(self.mu_head(h))
        conc = F.softplus(self.conc_head(h)) + 2
        return mean * conc, (1 - mean) * conc

    # ------------------------------------------------------------------
    # 롤아웃용 forward (sampling 포함)
    # ------------------------------------------------------------------

    def forward(
        self,
        cond_image: torch.Tensor,  # (3, H, W) or (B, 3, H, W)
        start: torch.Tensor,       # (2,) or (B, 2)
        goal: torch.Tensor,        # (2,) or (B, 2)
        force_k: int | None = None,  # 커리큘럼 워밍업: K를 이 값으로 고정
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            count_logits : (B, max_subgoals+1)
            K            : (B,) int — sampled sub-goal counts (or force_k if set)
            sampled_sgs  : (B, max_subgoals, 2) — zero-padded for positions >= K
        """
        unbatched = cond_image.dim() == 3
        if unbatched:
            cond_image = cond_image.unsqueeze(0)
            start = start.unsqueeze(0)
            goal  = goal.unsqueeze(0)

        B = cond_image.shape[0]
        map_feat = self.encode_map(cond_image)  # (B, latent)

        # 1. Count 예측
        feat0 = torch.cat([map_feat, start, goal], dim=-1)
        h0 = self.mlp(feat0)
        count_logits = self.count_head(h0)  # (B, max_K+1)
        if force_k is not None:
            K = torch.full((B,), force_k, dtype=torch.long, device=cond_image.device)
        else:
            K = torch.distributions.Categorical(logits=count_logits).sample()  # (B,) int

        # 2. AR sampling — K_max번 루프, 비활성 위치는 0으로 padding
        K_max = int(K.max().item())
        prev = start.clone()  # (B, 2)
        sampled_sgs = torch.zeros(B, self.max_subgoals, 2, device=cond_image.device, dtype=start.dtype)

        for i in range(K_max):
            alpha_i, beta_i = self.forward_step(map_feat, prev, goal)
            sg_i = torch.distributions.Beta(
                alpha_i.clamp(min=0.1), beta_i.clamp(min=0.1)
            ).sample().clamp(0.001, 0.999)  # (B, 2)

            active = (i < K).float().unsqueeze(-1)  # (B, 1)
            sampled_sgs[:, i, :] = sg_i * active
            prev = sg_i * active + prev * (1.0 - active)

        if unbatched:
            count_logits = count_logits.squeeze(0)
            K            = K.squeeze(0)
            sampled_sgs  = sampled_sgs.squeeze(0)

        return count_logits, K, sampled_sgs

    # ------------------------------------------------------------------
    # 학습용 forward (teacher forcing, 벡터화)
    # ------------------------------------------------------------------

    def forward_ar_teacher(
        self,
        cond_image: torch.Tensor,       # (B, 3, H, W)
        start: torch.Tensor,            # (B, 2)
        goal: torch.Tensor,             # (B, 2)
        b_action: torch.Tensor,         # (B, max_subgoals, 2) — rollout 버퍼, detached
        b_K: torch.Tensor,              # (B,) int — sampled counts
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher forcing으로 count_logits, alpha, beta 재계산.

        Returns:
            count_logits : (B, max_subgoals+1)
            alpha        : (B, max_subgoals, 2)
            beta         : (B, max_subgoals, 2)
        """
        B = cond_image.shape[0]
        N = self.max_subgoals
        map_feat = self.encode_map(cond_image)  # (B, latent)

        # Count logits 재계산
        feat0 = torch.cat([map_feat, start, goal], dim=-1)
        h0 = self.mlp(feat0)
        count_logits = self.count_head(h0)  # (B, max_K+1)

        # Teacher forcing prev_points 구성:
        #   prev_points[b, 0]   = start[b]
        #   prev_points[b, i>0] = b_action[b, i-1]
        prev_points = torch.cat(
            [start.unsqueeze(1),       # (B, 1, 2)
             b_action[:, :-1, :]],     # (B, N-1, 2)
            dim=1,
        )  # (B, N, 2)

        # (B*N, ...) 로 flatten → MLP 한 번에 통과
        map_feat_rep = map_feat.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        prev_flat    = prev_points.reshape(B * N, 2)
        goal_rep     = goal.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 2)

        feat = torch.cat([map_feat_rep, prev_flat, goal_rep], dim=-1)
        h    = self.mlp(feat)
        mean = torch.sigmoid(self.mu_head(h)).view(B, N, 2)
        conc = (F.softplus(self.conc_head(h)) + 2).view(B, N, 2)

        alpha = mean * conc          # (B, N, 2)
        beta  = (1 - mean) * conc    # (B, N, 2)
        return count_logits, alpha, beta

# class AutoregressiveSubGoalPolicyCount(nn.Module):
#     """
#     Autoregressive sub-goal policy with variable count prediction.

#     1단계: count head로 K ~ Categorical(0..max_subgoals) 샘플링
#     2단계: AR로 K개 sub-goal 순서대로 예측
#            SG_i ~ Beta(f(map, SG_{i-1}, goal))  where SG_0 = start
#     """

#     def __init__(
#         self,
#         max_subgoals: int = 4,
#         latent_dim: int = 128,
#         hidden_dim: int = 128,
#         backbone: str = "resnet34",
#         pretrained_encoder: nn.Module | None = None,
#     ):
#         super().__init__()
#         self.max_subgoals = max_subgoals

#         if pretrained_encoder is not None:
#             self.map_encoder = pretrained_encoder
#         else:
#             # MapEncoder 클래스가 정의되어 있다고 가정합니다.
#             self.map_encoder = MapEncoder(latent_dim=latent_dim, backbone=backbone)
#         feat_dim = latent_dim + 4  # map_feat + prev_point(2) + goal(2)

#         # =====================================================================
#         # [수정 1] Count와 AR(위치 예측)의 MLP 분리 (Representation Bottleneck 해결)
#         # =====================================================================
        
#         # 1. 개수(Count) 예측 전용 네트워크
#         self.count_mlp = nn.Sequential(
#             nn.Linear(feat_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#         )
#         self.count_head = nn.Linear(hidden_dim, max_subgoals + 1)

#         # 2. 위치(AR Sub-goal) 예측 전용 네트워크
#         self.ar_mlp = nn.Sequential(
#             nn.Linear(feat_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#         )
#         self.mu_head    = nn.Linear(hidden_dim, 2)
#         self.conc_head  = nn.Linear(hidden_dim, 2)

#         for head in (self.mu_head, self.conc_head, self.count_head):
#             nn.init.zeros_(head.weight)
#             nn.init.zeros_(head.bias)

#     # ------------------------------------------------------------------
#     # 내부 헬퍼
#     # ------------------------------------------------------------------

#     def encode_map(self, cond_image: torch.Tensor) -> torch.Tensor:
#         """(B, 3, H, W) → (B, latent_dim). 에피소드당 한 번만 호출."""
#         return self.map_encoder(cond_image)

#     def forward_step(
#         self,
#         map_feat: torch.Tensor,    # (B, latent_dim)
#         prev_point: torch.Tensor,  # (B, 2)
#         goal: torch.Tensor,        # (B, 2)
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """단일 AR step. returns alpha, beta: (B, 2)."""
#         feat = torch.cat([map_feat, prev_point, goal], dim=-1)
#         h = self.ar_mlp(feat)  # [수정 1 적용] ar_mlp 사용
        
#         mean = torch.sigmoid(self.mu_head(h))
#         # =====================================================================
#         # [수정 2] 수치적 안정성 확보 (0이나 1로 수렴하여 NaN 발생 방지)
#         # =====================================================================
#         mean = mean.clamp(min=1e-4, max=1.0 - 1e-4) 
#         conc = F.softplus(self.conc_head(h)) + 2
        
#         return mean * conc, (1 - mean) * conc

#     # ------------------------------------------------------------------
#     # 롤아웃용 forward (sampling 포함)
#     # ------------------------------------------------------------------

#     def forward(
#         self,
#         cond_image: torch.Tensor,  # (3, H, W) or (B, 3, H, W)
#         start: torch.Tensor,       # (2,) or (B, 2)
#         goal: torch.Tensor,        # (2,) or (B, 2)
#         force_k: int | None = None,  # 커리큘럼 워밍업: K를 이 값으로 고정
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Returns:
#             count_logits : (B, max_subgoals+1)
#             K            : (B,) int — sampled sub-goal counts (or force_k if set)
#             sampled_sgs  : (B, max_subgoals, 2) — zero-padded for positions >= K
#         """
#         unbatched = cond_image.dim() == 3
#         if unbatched:
#             cond_image = cond_image.unsqueeze(0)
#             start = start.unsqueeze(0)
#             goal  = goal.unsqueeze(0)

#         B = cond_image.shape[0]
#         map_feat = self.encode_map(cond_image)  # (B, latent)

#         # 1. Count 예측
#         feat0 = torch.cat([map_feat, start, goal], dim=-1)
#         h0 = self.count_mlp(feat0)  # [수정 1 적용] count_mlp 사용
#         count_logits = self.count_head(h0)  # (B, max_K+1)
        
#         if force_k is not None:
#             K = torch.full((B,), force_k, dtype=torch.long, device=cond_image.device)
#         else:
#             K = torch.distributions.Categorical(logits=count_logits).sample()  # (B,) int

#         # 2. AR sampling — K_max번 루프, 비활성 위치는 0으로 padding
#         K_max = int(K.max().item())
#         prev = start.clone()  # (B, 2)
#         sampled_sgs = torch.zeros(B, self.max_subgoals, 2, device=cond_image.device, dtype=start.dtype)

#         for i in range(K_max):
#             alpha_i, beta_i = self.forward_step(map_feat, prev, goal)
#             sg_i = torch.distributions.Beta(
#                 alpha_i.clamp(min=0.1), beta_i.clamp(min=0.1)
#             ).sample().clamp(0.001, 0.999)  # (B, 2)

#             active = (i < K).float().unsqueeze(-1)  # (B, 1)
#             sampled_sgs[:, i, :] = sg_i * active
#             prev = sg_i * active + prev * (1.0 - active)

#         if unbatched:
#             count_logits = count_logits.squeeze(0)
#             K            = K.squeeze(0)
#             sampled_sgs  = sampled_sgs.squeeze(0)

#         return count_logits, K, sampled_sgs

#     # ------------------------------------------------------------------
#     # 학습용 forward (teacher forcing, 벡터화)
#     # ------------------------------------------------------------------

#     def forward_ar_teacher(
#         self,
#         cond_image: torch.Tensor,       # (B, 3, H, W)
#         start: torch.Tensor,            # (B, 2)
#         goal: torch.Tensor,             # (B, 2)
#         b_action: torch.Tensor,         # (B, max_subgoals, 2) — rollout 버퍼, detached
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Teacher forcing으로 count_logits, alpha, beta 재계산.

#         Returns:
#             count_logits : (B, max_subgoals+1)
#             alpha        : (B, max_subgoals, 2)
#             beta         : (B, max_subgoals, 2)
#         """
#         # [최적화] b_K는 여기서 사용되지 않으므로 파라미터에서 제거했습니다.
#         B = cond_image.shape[0]
#         N = self.max_subgoals
#         map_feat = self.encode_map(cond_image)  # (B, latent)

#         # 1. Count logits 재계산
#         feat0 = torch.cat([map_feat, start, goal], dim=-1)
#         h0 = self.count_mlp(feat0) # [수정 1 적용] count_mlp 사용
#         count_logits = self.count_head(h0)  # (B, max_K+1)

#         # 2. Teacher forcing prev_points 구성
#         prev_points = torch.cat(
#             [start.unsqueeze(1),       # (B, 1, 2)
#              b_action[:, :-1, :]],     # (B, N-1, 2)
#             dim=1,
#         )  # (B, N, 2)

#         # =====================================================================
#         # [최적화] 차원을 납작하게(flatten) 만들 필요 없이, PyTorch nn.Linear 기능을 활용해
#         # (B, N, ...) 다차원 텐서 형태로 한 번에 처리하도록 최적화했습니다.
#         # =====================================================================
#         map_feat_rep = map_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, latent)
#         goal_rep     = goal.unsqueeze(1).expand(-1, N, -1)      # (B, N, 2)

#         feat = torch.cat([map_feat_rep, prev_points, goal_rep], dim=-1) # (B, N, feat_dim)
#         h    = self.ar_mlp(feat)                                        # (B, N, hidden_dim)

#         mean = torch.sigmoid(self.mu_head(h))                           # (B, N, 2)
#         mean = mean.clamp(min=1e-4, max=1.0 - 1e-4)                     # [수정 2 적용] NaN 방지
#         conc = F.softplus(self.conc_head(h)) + 2                        # (B, N, 2)

#         alpha = mean * conc          # (B, N, 2)
#         beta  = (1 - mean) * conc    # (B, N, 2)
        
#         return count_logits, alpha, beta


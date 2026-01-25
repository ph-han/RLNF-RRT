import torch
import torch.nn as nn

from rlnf_rrt.models.CNF import ConditionalNF
from rlnf_rrt.models.MapEncoder import MapEncoder

class PlannerFlows(nn.Module):
    def __init__(self, masks, hidden_dim, env_latent_dim, state_dim=2):
        super().__init__()
        # 1. 환경 인코더 (w 추출)
        self.encoder = MapEncoder(latent_dim=env_latent_dim)
        
        # 2. 전체 조건 차원 계산: w + q_init + q_target
        self.condition_dim = env_latent_dim + (state_dim * 2)
        
        # 3. 조건부 노멀라이징 플로우
        self.flow = ConditionalNF(masks, hidden_dim, self.condition_dim)

    def _get_condition(self, map_img, start, goal):
        # 환경 정보 인코딩 
        w = self.encoder(map_img)
        # 모든 조건을 하나로 결합 [cite: 132]
        condition = torch.cat([w, start, goal], dim=-1)
        return condition

    def forward(self, map_img, start, goal, num_samples=1):
        """추론 단계: 잠재 공간 z에서 실제 샘플 q를 생성 [cite: 10, 36]"""
        condition = self._get_condition(map_img, start, goal)
        
        # 표준 가우시안 분포에서 z 샘플링 [cite: 108]
        # batch_size와 num_samples를 고려하여 확장
        batch_size = map_img.size(0)
        z = torch.randn(batch_size, num_samples, self.flow.layers[0].input_dim).to(map_img.device)
        
        # 생성 (z -> q) [cite: 104]
        # condition을 num_samples만큼 반복하여 맞춰줌
        condition_rep = condition.unsqueeze(1).repeat(1, num_samples, 1)
        q_samples, _ = self.flow.forward(z.view(-1, z.size(-1)), condition_rep.view(-1, condition_rep.size(-1)))
        
        return q_samples.view(batch_size, num_samples, -1)

    def get_nll(self, gt_point, map_img, start, goal):
        """학습 단계: 데이터 q의 음의 로그 가능도 계산 [cite: 164]"""
        condition = self._get_condition(map_img, start, goal)
        
        # 역변환 (q -> z) [cite: 115]
        z, log_det = self.flow.inverse(gt_point, condition)
        
        # 가우시안 로그 확률 + 야코비안 보정 [cite: 168]
        prior_log_prob = -0.5 * torch.sum(z**2, dim=-1)
        log_likelihood = prior_log_prob + log_det
        
        return -log_likelihood.mean() # NLL Loss [cite: 164]
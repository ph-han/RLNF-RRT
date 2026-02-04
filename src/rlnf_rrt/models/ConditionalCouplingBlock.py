# ConditionalCouplingBlock.py
import torch
import torch.nn as nn

class ConditionalAffineCouplingBlock2Step(nn.Module):
    """
    PlannerFlows/cINN 스타일:
      Step1: (1-mask) 파트 업데이트 (mask 파트로 조건)
      Step2: mask 파트 업데이트 (업데이트된 (1-mask) 파트로 조건)
    """
    def __init__(self, mask, hidden_dim, condition_dim, s_max=0.5, log_sigma_clip=5.0):
        super().__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.s_max = s_max

        m = torch.tensor(mask, dtype=torch.float32)
        self.register_buffer("mask", m)                 # 1인 곳이 A 파트
        self.register_buffer("inv_mask", 1.0 - m)       # 1인 곳이 B 파트

        # 네트워크 입력은 "conditioning에 쓰는 파트 + cond"
        # Step1은 A(mask=1) + cond -> s1,t1로 B 업데이트
        # Step2는 B + cond -> s2,t2로 A 업데이트
        in_dim = self.input_dim + self.condition_dim

        self.s1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.input_dim),
        )
        self.t1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.input_dim),
        )

        self.s2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.input_dim),
        )
        self.t2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.input_dim),
        )

        # 안정화: 마지막 레이어 0 초기화 → 초기엔 거의 identity
        for net in (self.s1, self.t1, self.s2, self.t2):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

        # (선택) gain 쓰고 싶으면 여기에 두되, 우선은 빼는 걸 추천
        # self.s_gain = nn.Parameter(torch.ones(self.input_dim))

    def _scale(self, s_raw):
        # exp 폭주 방지
        return torch.tanh(s_raw) * self.s_max

    def forward(self, x, condition):
        # A,B split
        a = x * self.mask
        b = x * self.inv_mask

        # Step1: update B conditioned on A
        inp1 = torch.cat([a, condition], dim=-1)
        s1 = self._scale(self.s1(inp1)) * self.inv_mask
        t1 = self.t1(inp1) * self.inv_mask
        b1 = b * torch.exp(s1) + t1
        logdet1 = torch.sum(s1, dim=-1)

        # Step2: update A conditioned on updated B
        inp2 = torch.cat([b1, condition], dim=-1)
        s2 = self._scale(self.s2(inp2)) * self.mask
        t2 = self.t2(inp2) * self.mask
        a2 = a * torch.exp(s2) + t2
        logdet2 = torch.sum(s2, dim=-1)

        y = a2 + b1
        return y, (logdet1 + logdet2)

    def inverse(self, y, condition):
        # A,B split
        a2 = y * self.mask
        b1 = y * self.inv_mask

        # Inverse Step2: recover A from (A2, B1)
        inp2 = torch.cat([b1, condition], dim=-1)
        s2 = self._scale(self.s2(inp2)) * self.mask
        t2 = self.t2(inp2) * self.mask
        a = (a2 - t2) * torch.exp(-s2)
        logdet2 = -torch.sum(s2, dim=-1)

        # Inverse Step1: recover B from (A, B1)
        inp1 = torch.cat([a, condition], dim=-1)
        s1 = self._scale(self.s1(inp1)) * self.inv_mask
        t1 = self.t1(inp1) * self.inv_mask
        b = (b1 - t1) * torch.exp(-s1)
        logdet1 = -torch.sum(s1, dim=-1)

        x = a + b
        return x, (logdet1 + logdet2)

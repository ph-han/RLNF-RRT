# import torch
# import torch.nn as nn

# class STNet(nn.Module):
#     def __init__(self, cond_dim:int=128, hidden_dim:int=128):
#         super().__init__()
#         # Input: z_component (1) + condition (cond_dim)
#         # Output: 1 (scale or translation for that component)
#         self.net = nn.Sequential(
#             nn.Linear(1 + cond_dim, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         return self.net(x)

# class CouplingBlock(nn.Module):
#     def __init__(self, cond_dim:int=128, hidden_dim:int=128, s_max:float=2.0):
#         super().__init__()
#         self.cond_dim = cond_dim
#         self.s_max = s_max  # scale factor for s_raw

#         # Networks for component a (conditioned on b)
#         self.s_a = STNet(cond_dim, hidden_dim)
#         self.t_a = STNet(cond_dim, hidden_dim)

#         # Networks for component b (conditioned on a)
#         self.s_b = STNet(cond_dim, hidden_dim)
#         self.t_b = STNet(cond_dim, hidden_dim)

#     def _scale(self, s_raw):
#         return torch.tanh(s_raw) * self.s_max

#     def forward(self, x:torch.Tensor, cond:torch.Tensor):
#         # x is complexity distribution that represents gt_trajs (B, T, 2)
#         # cond is condition (map_img, start, goal) for motion planning (B, T, cond_dim)
        
#         z_a:torch.Tensor = x[:, :, 0:1]  # (B, T, 1) - keep dim for concat
#         z_b:torch.Tensor = x[:, :, 1:2]  # (B, T, 1)

#         # component 1: z_a (conditioned on z_b)
#         s_a:torch.Tensor = self._scale(self.s_a(torch.cat([z_b, cond], dim=-1)))  # (B, T, 1)
#         t_a:torch.Tensor = self.t_a(torch.cat([z_b, cond], dim=-1))  # (B, T, 1)
#         z_a:torch.Tensor = z_a * torch.exp(s_a) + t_a

#         # component 2: z_b (conditioned on updated z_a)
#         s_b:torch.Tensor = self._scale(self.s_b(torch.cat([z_a, cond], dim=-1)))  # (B, T, 1)
#         t_b:torch.Tensor = self.t_b(torch.cat([z_a, cond], dim=-1))  # (B, T, 1)
#         z_b:torch.Tensor = z_b * torch.exp(s_b) + t_b

#         # calc log det (sum over the 1-dim)
#         log_det:torch.Tensor = s_a.sum(dim=(1, 2)) + s_b.sum(dim=(1, 2))  # (B,)

#         # out
#         out:torch.Tensor = torch.cat([z_a, z_b], dim=-1)  # (B, T, 2)

#         return out, log_det
        

#     def inverse(self, z:torch.Tensor, cond:torch.Tensor):
#         # z is simple distribution like gaussian distribution (B, T, 2)
#         z_a:torch.Tensor = z[:, :, 0:1]  # (B, T, 1)
#         z_b:torch.Tensor = z[:, :, 1:2]  # (B, T, 1)
    
#         # Reverse order: first undo z_b transform (using z_a which wasn't changed yet in inverse)
#         s_b:torch.Tensor = self._scale(self.s_b(torch.cat([z_a, cond], dim=-1)))
#         t_b:torch.Tensor = self.t_b(torch.cat([z_a, cond], dim=-1))
#         x_b:torch.Tensor = (z_b - t_b) * torch.exp(-s_b)
        
#         # Then undo z_a transform (using recovered x_b)
#         s_a:torch.Tensor = self._scale(self.s_a(torch.cat([x_b, cond], dim=-1)))
#         t_a:torch.Tensor = self.t_a(torch.cat([x_b, cond], dim=-1))
#         x_a:torch.Tensor = (z_a - t_a) * torch.exp(-s_a)
        
#         out:torch.Tensor = torch.cat([x_a, x_b], dim=-1)  # (B, T, 2)
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, map_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma = nn.Linear(map_dim, hidden_dim)
        self.beta  = nn.Linear(map_dim, hidden_dim)

        # 초기엔 identity에 가깝게 시작시키는 게 안정적
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h: torch.Tensor, map_feat: torch.Tensor):
        """
        h: (B,T,H)
        map_feat: (B,map_dim)
        """
        B, T, H = h.shape
        mf = map_feat.unsqueeze(1).expand(B, T, map_feat.shape[-1])  # (B,T,map_dim)

        gamma = self.gamma(mf)  # (B,T,H)
        beta  = self.beta(mf)   # (B,T,H)

        # (1+gamma) so identity is easy
        return h * (1.0 + gamma) + beta


class STNetFiLM(nn.Module):
    """
    Input:
      z_component: (B,T,1)
      sg_feat:     (B,T,sg_dim)   here sg_dim=7
      map_feat:    (B,map_dim)
    Output:
      (B,T,1) for s or t
    """
    def __init__(self, sg_dim: int, map_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(1 + sg_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film1 = FiLM(map_dim, hidden_dim)
        self.film2 = FiLM(map_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, z1: torch.Tensor, sg_feat: torch.Tensor, map_feat: torch.Tensor):
        x = torch.cat([z1, sg_feat], dim=-1)          # (B,T,1+sg_dim)
        h = F.leaky_relu(self.fc1(x), 0.2)            # (B,T,H)
        h = self.film1(h, map_feat)                   # FiLM injection
        h = F.leaky_relu(self.fc2(h), 0.2)
        h = self.film2(h, map_feat)                   # optional 2nd FiLM
        return self.out(h)                            # (B,T,1)


class CouplingBlock(nn.Module):
    def __init__(self, sg_dim:int=7, map_dim:int=256, hidden_dim:int=128, s_max:float=2.0):
        super().__init__()
        self.sg_dim = sg_dim
        self.map_dim = map_dim
        self.s_max = s_max  # scale factor for s_raw

        # Networks for component a (conditioned on b)
        self.s_a = STNetFiLM(sg_dim, map_dim, hidden_dim)
        self.t_a = STNetFiLM(sg_dim, map_dim, hidden_dim)

        # Networks for component b (conditioned on a)
        self.s_b = STNetFiLM(sg_dim, map_dim, hidden_dim)
        self.t_b = STNetFiLM(sg_dim, map_dim, hidden_dim)

    def _scale(self, s_raw):
        return torch.tanh(s_raw) * self.s_max

    def forward(self, x:torch.Tensor, sg_feat:torch.Tensor, map_feat:torch.Tensor):
        # x is complexity distribution that represents gt_trajs (B, T, 2)
        # cond is condition (map_img, start, goal) for motion planning (B, T, cond_dim)
        
        z_a:torch.Tensor = x[:, :, 0:1]  # (B, T, 1) - keep dim for concat
        z_b:torch.Tensor = x[:, :, 1:2]  # (B, T, 1)

        # component 1: z_a (conditioned on z_b)
        s_a:torch.Tensor = self._scale(self.s_a(z_b, sg_feat, map_feat))  # (B, T, 1)
        t_a:torch.Tensor = self.t_a(z_b, sg_feat, map_feat)  # (B, T, 1)
        z_a:torch.Tensor = z_a * torch.exp(s_a) + t_a

        # component 2: z_b (conditioned on updated z_a)
        s_b:torch.Tensor = self._scale(self.s_b(z_a, sg_feat, map_feat))  # (B, T, 1)
        t_b:torch.Tensor = self.t_b(z_a, sg_feat, map_feat)  # (B, T, 1)
        z_b:torch.Tensor = z_b * torch.exp(s_b) + t_b

        # calc log det (sum over the 1-dim)
        log_det:torch.Tensor = s_a.sum(dim=(1, 2)) + s_b.sum(dim=(1, 2))  # (B,)

        # out
        out:torch.Tensor = torch.cat([z_a, z_b], dim=-1)  # (B, T, 2)

        return out, log_det
        

    def inverse(self, z:torch.Tensor, sg_feat:torch.Tensor, map_feat:torch.Tensor):
        # z is simple distribution like gaussian distribution (B, T, 2)
        z_a:torch.Tensor = z[:, :, 0:1]  # (B, T, 1)
        z_b:torch.Tensor = z[:, :, 1:2]  # (B, T, 1)
    
        # Reverse order: first undo z_b transform (using z_a which wasn't changed yet in inverse)
        s_b:torch.Tensor = self._scale(self.s_b(z_a, sg_feat, map_feat))
        t_b:torch.Tensor = self.t_b(z_a, sg_feat, map_feat)
        x_b:torch.Tensor = (z_b - t_b) * torch.exp(-s_b)
        
        # Then undo z_a transform (using recovered x_b)
        s_a:torch.Tensor = self._scale(self.s_a(x_b, sg_feat, map_feat))
        t_a:torch.Tensor = self.t_a(x_b, sg_feat, map_feat)
        x_a:torch.Tensor = (z_a - t_a) * torch.exp(-s_a)
        
        out:torch.Tensor = torch.cat([x_a, x_b], dim=-1)  # (B, T, 2)
        return out
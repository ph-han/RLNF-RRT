import torch
import torch.nn as nn

import torch
import torch.nn as nn


class STNet(nn.Module):
    """
    STNet: (z_keep, cond) -> (s, t)
    - 입력:  (B,T, zdim + cond_dim)
    - 출력:  (B,T, 2*zdim)  -> chunk -> s_raw, t
    """
    def __init__(self, z_keep_dim: int, cond_dim: int, hidden_dim: int = 128, s_max: float = 1.5):
        super().__init__()
        self.s_max = s_max
        in_dim = z_keep_dim + cond_dim
        out_dim = 2 * z_keep_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z_keep: torch.Tensor, cond_bt: torch.Tensor):
        """
        z_keep : (B,T,z_keep_dim)
        cond_bt: (B,T,cond_dim)
        returns:
          s: (B,T,z_keep_dim)  (clamped)
          t: (B,T,z_keep_dim)
        """
        h = torch.cat([z_keep, cond_bt], dim=-1)
        st = self.net(h)
        s_raw, t = st.chunk(2, dim=-1)
        s = self.s_max * torch.tanh(s_raw)
        return s, t


class AffineCouplingBlock(nn.Module):
    """
    PlannerFlows-style 2-step coupling for 2D points.
    x: (B,T,2)  where T is number of sampled points (e.g., 1024)
    cond: (B,C) where C=132 (map_latent + start + goal)

    Step 1: update a using b
      a <- a * exp(s_a(b,cond)) + t_a(b,cond)
    Step 2: update b using updated a
      b <- b * exp(s_b(a,cond)) + t_b(a,cond)

    log_det = sum(s_a) + sum(s_b) over (T and dims)
    """
    def __init__(self, cond_dim: int, hidden_dim: int = 128, s_max: float = 1.5):
        super().__init__()
        self.st_a = STNet(z_keep_dim=1, cond_dim=cond_dim, hidden_dim=hidden_dim, s_max=s_max)
        self.st_b = STNet(z_keep_dim=1, cond_dim=cond_dim, hidden_dim=hidden_dim, s_max=s_max)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        Forward: x -> z
        returns: z, log_det  where log_det: (B,)
        """
        B, T, D = x.shape

        # broadcast cond to (B,T,C)
        cond_bt = cond[:, None, :].expand(B, T, cond.size(-1))

        z_a = x[:, :, 0:1]  # (B,T,1) -> x
        z_b = x[:, :, 1:2]  # (B,T,1) -> y

        # component 1: z_a 
        s_a, t_a = self.st_a(z_keep=z_b, cond_bt=cond_bt)  # (B,T,1), (B,T,1)
        z_a = z_a * torch.exp(s_a) + t_a

        # component 2: z_b
        s_b, t_b = self.st_b(z_keep=z_a, cond_bt=cond_bt)
        z_b = z_b * torch.exp(s_b) + t_b

        z = torch.cat([z_a, z_b], dim=-1)  # (B,T,2)

        # logdet
        log_det = s_a.sum(dim=(1, 2)) + s_b.sum(dim=(1, 2))  # (B,)
        return z, log_det

    def inverse(self, z: torch.Tensor, cond: torch.Tensor):
        """
        Inverse: z -> x
        returns: x, log_det_inv  where log_det_inv: (B,)
        """
        B, T, D = z.shape

        cond_bt = cond[:, None, :].expand(B, T, cond.size(-1))

        x_a = z[:, :, 0:1]
        x_b = z[:, :, 1:2]

        # inverse order is reversed

        # component 2: z_b
        s_b, t_b = self.st_b(z_keep=x_a, cond_bt=cond_bt)
        x_b = (x_b - t_b) * torch.exp(-s_b)

        # component 1: z_a 
        s_a, t_a = self.st_a(z_keep=x_b, cond_bt=cond_bt)
        x_a = (x_a - t_a) * torch.exp(-s_a)

        x = torch.cat([x_a, x_b], dim=-1)

        # inverse logdet is negative of forward's
        log_det = -(s_a.sum(dim=(1, 2)) + s_b.sum(dim=(1, 2)))
        return x, log_det

import torch
import torch.nn as nn

class STNet(nn.Module):
    def __init__(self, z_keep_dim: int, cond_dim: int, hidden_dim: int = 128, s_max: float = 1.5):
        super().__init__()
        self.s_max = s_max
        in_dim = z_keep_dim + cond_dim
        out_dim = z_keep_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z_keep: torch.Tensor, cond_bt: torch.Tensor):
        return self.net(torch.cat([z_keep, cond_bt], dim=-1))
class AffineCouplingBlock(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int = 128, s_max: float = 1.5):
        super().__init__()
        self.s_a = STNet(z_keep_dim=1, cond_dim=cond_dim, hidden_dim=hidden_dim, s_max=s_max)
        self.s_b = STNet(z_keep_dim=1, cond_dim=cond_dim, hidden_dim=hidden_dim, s_max=s_max)
        self.t_a = STNet(z_keep_dim=1, cond_dim=cond_dim, hidden_dim=hidden_dim, s_max=s_max)
        self.t_b = STNet(z_keep_dim=1, cond_dim=cond_dim, hidden_dim=hidden_dim, s_max=s_max)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        B, T, D = x.shape

        # broadcast cond to (B,T,C)
        cond_bt = cond[:, None, :].expand(B, T, cond.size(-1))

        z_a = x[:, :, 0:1]  # (B,T,1) -> x
        z_b = x[:, :, 1:2]  # (B,T,1) -> y

        # component 1: z_a 
        s_a = self.s_a(z_keep=z_b, cond_bt=cond_bt)  # (B,T,1)
        s_a = torch.tanh(s_a) * self.s_a.s_max
        t_a = self.t_a(z_keep=z_b, cond_bt=cond_bt)  # (B,T,1)
        z_a = z_a * torch.exp(s_a) + t_a

        # component 2: z_b
        s_b = self.s_b(z_keep=z_a, cond_bt=cond_bt)
        s_b = torch.tanh(s_b) * self.s_b.s_max
        t_b = self.t_b(z_keep=z_a, cond_bt=cond_bt)
        z_b = z_b * torch.exp(s_b) + t_b

        z = torch.cat([z_a, z_b], dim=-1)  # (B,T,2)

        # logdet
        log_det = s_a.sum(dim=(1, 2)) + s_b.sum(dim=(1, 2))  # (B,)
        return z, log_det
    
    @torch.no_grad()
    def inverse(self, z: torch.Tensor, cond: torch.Tensor):
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

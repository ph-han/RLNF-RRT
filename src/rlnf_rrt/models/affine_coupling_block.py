import torch
import torch.nn as nn
import torch.nn.functional as F

class STNet(nn.Module):
    def __init__(self, z_keep_dim: int, feat_ch: int, cond_dim: int, hidden_dim: int = 128, s_max: float = 1.5):
        super().__init__()
        self.s_max = s_max
        in_dim = z_keep_dim + feat_ch + cond_dim
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
        
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            if last.bias is not None:
                nn.init.zeros_(last.bias)
            if last.weight is not None:
                nn.init.zeros_(last.weight)

    def forward(self, z_keep: torch.Tensor, feat_map: torch.Tensor, cond_vec: torch.Tensor, current_z: torch.Tensor):
        grid = current_z.unsqueeze(2) * 2 - 1  # (B,T,2) -> (B,T,2) in range [-1,1]
        local_feat = F.grid_sample(feat_map, grid, align_corners=True)  # (B,C,H',W') + (B,T,cond_dim) -> (B,T,C+cond_dim)
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)  # (B,C,T) -> (B,T,C)

        B, T, _ = z_keep.shape
        cond_bt = cond_vec[:, None, :].expand(B, T, -1)
        
        combined_cond = torch.cat([z_keep, local_feat, cond_bt], dim=-1)  # (B,T,cond_dim)
        return self.net(combined_cond)
    
class AffineCouplingBlock(nn.Module):
    def __init__(self, feat_ch:int, cond_dim: int, hidden_dim: int = 128, s_max: float = 1.5):
        super().__init__()
        self.s_a = STNet(z_keep_dim=1, feat_ch=feat_ch, cond_dim=cond_dim, hidden_dim=hidden_dim, s_max=s_max)
        self.s_b = STNet(z_keep_dim=1, feat_ch=feat_ch, cond_dim=cond_dim, hidden_dim=hidden_dim, s_max=s_max)
        self.t_a = STNet(z_keep_dim=1, feat_ch=feat_ch, cond_dim=cond_dim, hidden_dim=hidden_dim, s_max=s_max)
        self.t_b = STNet(z_keep_dim=1, feat_ch=feat_ch, cond_dim=cond_dim, hidden_dim=hidden_dim, s_max=s_max)

    def forward(self, x: torch.Tensor, w: torch.Tensor, cond: torch.Tensor):

        z_a = x[:, :, 0:1]  # (B,T,1) -> x
        z_b = x[:, :, 1:2]  # (B,T,1) -> y

        # component 1: z_a 
        s_a = self.s_a(z_keep=z_b, feat_map=w, cond_vec=cond, current_z=x)  # (B,T,1)
        s_a = torch.tanh(s_a) * self.s_a.s_max
        t_a = self.t_a(z_keep=z_b, feat_map=w, cond_vec=cond, current_z=x)  # (B,T,1)
        z_a = z_a * torch.exp(s_a) + t_a

        # component 2: z_b
        _x = torch.cat([z_a, z_b], dim=-1)
        s_b = self.s_b(z_keep=z_a, feat_map=w, cond_vec=cond, current_z=_x)
        s_b = torch.tanh(s_b) * self.s_b.s_max
        t_b = self.t_b(z_keep=z_a, feat_map=w, cond_vec=cond, current_z=_x)
        z_b = z_b * torch.exp(s_b) + t_b

        z = torch.cat([z_a, z_b], dim=-1)  # (B,T,2)

        # logdet
        log_det = s_a.sum(dim=(1, 2)) + s_b.sum(dim=(1, 2))  # (B,)
        return z, log_det
    
    @torch.no_grad()
    def inverse(self, z: torch.Tensor, w: torch.Tensor, cond: torch.Tensor):
        x_a = z[:, :, 0:1]
        x_b = z[:, :, 1:2]

        # inverse order is reversed

        # component 2: z_b
        s_b = self.s_b(z_keep=x_a, feat_map=w, cond_vec=cond, current_z=z)
        s_b = torch.tanh(s_b) * self.s_b.s_max
        t_b = self.t_b(z_keep=x_a, feat_map=w, cond_vec=cond, current_z=z)
        x_b = (x_b - t_b) * torch.exp(-s_b)

        # component 1: z_a 
        _z = torch.cat([x_a, x_b], dim=-1)
        s_a = self.s_a(z_keep=x_b, feat_map=w, cond_vec=cond, current_z=_z)
        s_a = torch.tanh(s_a) * self.s_a.s_max
        t_a = self.t_a(z_keep=x_b, feat_map=w, cond_vec=cond, current_z=_z)
        x_a = (x_a - t_a) * torch.exp(-s_a)

        x = torch.cat([x_a, x_b], dim=-1)

        # inverse logdet is negative of forward's
        log_det = -(s_a.sum(dim=(1, 2)) + s_b.sum(dim=(1, 2)))
        return x, log_det

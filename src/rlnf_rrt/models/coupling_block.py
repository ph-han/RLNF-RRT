import torch
import torch.nn as nn

class STNet(nn.Module):
    """Scale-Translation Network for affine coupling.
    
    Takes z_component (1-dim) concatenated with condition (cond_dim),
    outputs a 1-dimensional scale or translation value.
    """
    def __init__(self, cond_dim:int=128, hidden_dim:int=128):
        super().__init__()
        # Input: z_component (1) + condition (cond_dim)
        # Output: 1 (scale or translation for that component)
        self.net = nn.Sequential(
            nn.Linear(1 + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CouplingBlock(nn.Module):
    """Affine Coupling Block for Real-NVP style normalizing flow.
    
    Splits input into two components (z_a, z_b), applies affine transforms
    conditioned on the other component and external condition.
    """
    def __init__(self, cond_dim:int=128, hidden_dim:int=128, s_max:float=2.0):
        super().__init__()
        self.cond_dim = cond_dim
        self.s_max = s_max  # scale factor for s_raw

        # Networks for component a (conditioned on b)
        self.s_a = STNet(cond_dim, hidden_dim)
        self.t_a = STNet(cond_dim, hidden_dim)

        # Networks for component b (conditioned on a)
        self.s_b = STNet(cond_dim, hidden_dim)
        self.t_b = STNet(cond_dim, hidden_dim)

    def _scale(self, s_raw):
        return torch.tanh(s_raw) * self.s_max

    def forward(self, x:torch.Tensor, cond:torch.Tensor):
        # x is complexity distribution that represents gt_trajs (B, T, 2)
        # cond is condition (map_img, start, goal) for motion planning (B, T, cond_dim)
        
        z_a:torch.Tensor = x[:, :, 0:1]  # (B, T, 1) - keep dim for concat
        z_b:torch.Tensor = x[:, :, 1:2]  # (B, T, 1)

        # component 1: z_a (conditioned on z_b)
        s_a:torch.Tensor = self._scale(self.s_a(torch.cat([z_b, cond], dim=-1)))  # (B, T, 1)
        t_a:torch.Tensor = self.t_a(torch.cat([z_b, cond], dim=-1))  # (B, T, 1)
        z_a:torch.Tensor = z_a * torch.exp(s_a) + t_a

        # component 2: z_b (conditioned on updated z_a)
        s_b:torch.Tensor = self._scale(self.s_b(torch.cat([z_a, cond], dim=-1)))  # (B, T, 1)
        t_b:torch.Tensor = self.t_b(torch.cat([z_a, cond], dim=-1))  # (B, T, 1)
        z_b:torch.Tensor = z_b * torch.exp(s_b) + t_b

        # calc log det (sum over the 1-dim)
        log_det:torch.Tensor = s_a.sum(dim=(1, 2)) + s_b.sum(dim=(1, 2))  # (B,)

        # out
        out:torch.Tensor = torch.cat([z_a, z_b], dim=-1)  # (B, T, 2)

        return out, log_det
        

    def inverse(self, z:torch.Tensor, cond:torch.Tensor):
        # z is simple distribution like gaussian distribution (B, T, 2)
        z_a:torch.Tensor = z[:, :, 0:1]  # (B, T, 1)
        z_b:torch.Tensor = z[:, :, 1:2]  # (B, T, 1)
    
        # Reverse order: first undo z_b transform (using z_a which wasn't changed yet in inverse)
        s_b:torch.Tensor = self._scale(self.s_b(torch.cat([z_a, cond], dim=-1)))
        t_b:torch.Tensor = self.t_b(torch.cat([z_a, cond], dim=-1))
        x_b:torch.Tensor = (z_b - t_b) * torch.exp(-s_b)
        
        # Then undo z_a transform (using recovered x_b)
        s_a:torch.Tensor = self._scale(self.s_a(torch.cat([x_b, cond], dim=-1)))
        t_a:torch.Tensor = self.t_a(torch.cat([x_b, cond], dim=-1))
        x_a:torch.Tensor = (z_a - t_a) * torch.exp(-s_a)
        
        out:torch.Tensor = torch.cat([x_a, x_b], dim=-1)  # (B, T, 2)
        return out
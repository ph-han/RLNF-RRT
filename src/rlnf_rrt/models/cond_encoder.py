import torch
from torch import nn

def add_coords(x):
    """
    x: (B, 1, H, W) 형태의 바이너리 맵
    결과: (B, 3, H, W) 형태 (원래 맵 + x좌표 + y좌표)
    """
    batch, _, h, w = x.size()
    # x 좌표 맵 생성 (-1 ~ 1 사이 값)
    xx_range = torch.linspace(-1, 1, w, device=x.device)
    xx_channel = xx_range.view(1, 1, 1, w).expand(batch, 1, h, w)

    # y 좌표 맵 생성 (-1 ~ 1 사이 값)
    yy_range = torch.linspace(-1, 1, h, device=x.device)
    yy_channel = yy_range.view(1, 1, h, 1).expand(batch, 1, h, w)

    return torch.cat([x, xx_channel, yy_channel], dim=1)

class MapEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, channels=(32, 48, 64, 96, 128)):
        super().__init__()
        layers = []
        in_ch = 3  # input channels: binary map + x coord + y coord
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(4, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(4, out_ch),
                nn.SiLU(),
            ]
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, binary_map: torch.Tensor) -> torch.Tensor:
        bmap = add_coords(binary_map)  # (B, 3, H, W)
        w_spatial = self.backbone(bmap)

        w_global = self.global_pool(w_spatial).view(w_spatial.size(0), -1) # (B, C)
        w_global = self.proj(w_global) 
        return w_spatial, w_global


class CondEncoder(nn.Module):
    def __init__(self, sg_dim: int = 2, latent_dim: int = 128, channels=(32, 48, 64, 96, 128)):
        super().__init__()

        self.map_encoder = MapEncoder(latent_dim, channels)

        self.sg_dim = sg_dim

    def forward(self, map: torch.Tensor, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        w_spatial, w_global = self.map_encoder(map)
        return w_spatial, torch.cat([w_global, start, goal], dim=-1)

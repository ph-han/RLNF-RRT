import torch
from torch import nn

class MapEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, channels=(32, 48, 64, 96, 128), norm: str = "bn"):
        super().__init__()

        def Norm(c: int):
            if norm == "bn":
                return nn.BatchNorm2d(c)
            if norm == "gn":
                g = min(8, c)
                while c % g != 0 and g > 1:
                    g -= 1
                return nn.GroupNorm(g, c)
            if norm == "none":
                return nn.Identity()
            raise ValueError("norm must be 'bn', 'gn', or 'none'.")

        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                Norm(out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                Norm(out_ch),
                nn.SiLU(),
            ]
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], latent_dim)

    def forward(self, map: torch.Tensor) -> torch.Tensor:
        x = self.backbone(map)
        x = x.mean(dim=(2, 3)) # GAP 
        return self.proj(x)


class CondEncoder(nn.Module):
    def __init__(self, sg_dim: int = 2, latent_dim: int = 128, channels=(32, 48, 64, 96, 128), norm: str = "bn"):
        super().__init__()

        self.map_encoder = MapEncoder(latent_dim, channels, norm)

        self.sg_dim = sg_dim

    def forward(self, map: torch.Tensor, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        w = self.map_encoder(map)
        return torch.cat([w, start, goal], dim=-1)
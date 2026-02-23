import torch
from torch import nn
from torchvision.models import resnet18, resnet34


class MapEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, backbone="resnet34"):
        super().__init__()
        if backbone == "resnet18":
            base = resnet18(weights=None)
        elif backbone == "resnet34":
            base = resnet34(weights=None)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.backbone = nn.Sequential(
            base.conv1,   # 224 -> 112
            base.bn1,
            base.relu,
            # base.maxpool,
            base.layer1,  # 112x112
            base.layer2,  # 56x56
            base.layer3,  # 28x28
            base.layer4   # 14x14 
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_spatial = self.backbone(x)
        
        w_global = self.global_pool(w_spatial).view(w_spatial.size(0), -1)
        w_global = self.proj(w_global) # (B, latent_dim)
        
        return w_global


class CondEncoder(nn.Module):
    def __init__(self, sg_dim: int = 2, latent_dim: int = 128, backbone="resnet34"):
        super().__init__()
        self.map_encoder = MapEncoder(latent_dim=latent_dim, backbone=backbone)
        self.sg_dim = sg_dim

    def forward(self, cond_image: torch.Tensor, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        w_global = self.map_encoder(cond_image)
        return torch.cat([w_global, start, goal], dim=-1)

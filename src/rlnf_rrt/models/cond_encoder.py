import torch
from torch import nn
from torchvision.models import resnet18
import numpy as np
from scipy import ndimage as ndi


def _make_input(bmap: torch.Tensor, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Build 3-channel map encoder input: binary map + start/goal + normalized SDF."""
    bmap = bmap.float().clamp(0.0, 1.0)
    bsz, _, height, width = bmap.shape
    device = bmap.device
    dtype = bmap.dtype

    sg_channel = torch.zeros((bsz, 1, height, width), device=device, dtype=dtype)
    sx = (start[:, 0] * (width - 1)).round().long().clamp(0, width - 1)
    sy = (start[:, 1] * (height - 1)).round().long().clamp(0, height - 1)
    gx = (goal[:, 0] * (width - 1)).round().long().clamp(0, width - 1)
    gy = (goal[:, 1] * (height - 1)).round().long().clamp(0, height - 1)
    batch_idx = torch.arange(bsz, device=device)
    sg_channel[batch_idx, 0, sy, sx] = 1.0
    sg_channel[batch_idx, 0, gy, gx] = -1.0

    bmap_np = (bmap[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    sdf_list: list[np.ndarray] = []
    norm = float(max(height, width))
    for i in range(bsz):
        free = bmap_np[i].astype(bool)
        obstacle = ~free
        dist_to_obstacle = ndi.distance_transform_edt(free)
        dist_to_free = ndi.distance_transform_edt(obstacle)
        sdf = (dist_to_obstacle - dist_to_free) / max(norm, 1.0) # pyright: ignore[reportOperatorIssue]
        sdf = np.clip(sdf, -1.0, 1.0).astype(np.float32)
        sdf_list.append(sdf)

    sdf_channel = torch.from_numpy(np.stack(sdf_list, axis=0)).to(device=device, dtype=dtype).unsqueeze(1)
    return torch.cat([bmap, sg_channel, sdf_channel], dim=1)


class MapEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        base = resnet18(weights=None)
        
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w_spatial = self.backbone(x)
        
        w_global = self.global_pool(w_spatial).view(w_spatial.size(0), -1)
        w_global = self.proj(w_global) # (B, latent_dim)
        
        return w_spatial, w_global


class CondEncoder(nn.Module):
    def __init__(self, sg_dim: int = 2, latent_dim: int = 128, channels=(32, 48, 64, 96, 128)):
        super().__init__()
        self.map_encoder = MapEncoder(latent_dim=latent_dim)
        self.sg_dim = sg_dim

    def forward(self, map: torch.Tensor, start: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model_input = _make_input(map, start, goal)
        w_spatial, w_global = self.map_encoder(model_input)
        return w_spatial, torch.cat([w_global, start, goal], dim=-1)

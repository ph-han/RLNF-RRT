import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class ConditionEncoder(nn.Module):
    def __init__(self, sg_dim:int=2, position_embed_dim:int=128, map_embed_dim:int=256, cond_dim:int=128):
        super().__init__()
        weights:MobileNet_V3_Small_Weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        m:nn.Module = mobilenet_v3_small(weights=weights)

        # --- change first conv 3->1 but KEEP pretrained info ---
        old_conv:nn.Conv2d = m.features[0][0]
        new_conv:nn.Conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            padding_mode=old_conv.padding_mode,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        m.features[0][0] = new_conv

        # use backbone only (no classifier)
        self.map_backbone:nn.Module = m.features
        self.map_pool:nn.Module = nn.AdaptiveAvgPool2d(1)
        self.map_proj:nn.Linear = nn.Linear(576, map_embed_dim)  # MNv3-small last ch = 576

        self.start_goal_encoder:nn.Sequential = nn.Sequential(
            nn.Linear(sg_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, position_embed_dim),
            nn.ReLU(),
        )

        # fuse everything into final condition vector for NF
        self.fuse:nn.Sequential = nn.Sequential(
            nn.Linear(map_embed_dim + position_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, cond_dim),
        )

    def forward(self, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor):
        # map_img: (B,1,H,W) in {0,1} or [0,1]
        x:torch.Tensor = self.map_backbone(map_img)
        x = self.map_pool(x).flatten(1)       # (B,576)
        map_emb:torch.Tensor = self.map_proj(x)            # (B,map_embed_dim)

        sg:torch.Tensor = torch.cat([start, goal], dim=-1) # (B,sg_dim * 2)
        sg_emb:torch.Tensor = self.start_goal_encoder(sg)  # (B,position_embed_dim)

        cond:torch.Tensor = self.fuse(torch.cat([map_emb, sg_emb], dim=-1))  # (B,cond_dim)
        return cond

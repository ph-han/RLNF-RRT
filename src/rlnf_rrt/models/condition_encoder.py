import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetMapEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 1-channel conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # ✅ GAP instead of flatten(7*7)
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B,512,1,1)
        self.proj = nn.Linear(512, out_dim)  # -> (B,out_dim)

    def forward(self, x):
        # x: (B,1,224,224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)            # (B,512,7,7)

        x = self.pool(x).flatten(1)   # (B,512)
        x = self.proj(x)              # (B,out_dim)
        return x


class ConditionEncoder(nn.Module):
    def __init__(self, sg_dim:int=2, position_embed_dim:int=128, map_embed_dim:int=256, cond_dim:int=128):
        super().__init__()
        
        # UPGRADED: Use ResNet-18 (Pretrained, preserving spatial features)
        self.map_encoder = ResNetMapEncoder(out_dim=map_embed_dim)
        
        self.start_encoder = nn.Sequential(
            nn.Linear(sg_dim, 64),
            nn.ReLU(),
            nn.Linear(64, position_embed_dim)
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(sg_dim, 64),
            nn.ReLU(),
            nn.Linear(64, position_embed_dim)
        )

        self.fusion = nn.Sequential(
            # map_embed + start_embed + goal_embed
            nn.Linear(map_embed_dim + 2 * position_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, cond_dim)
        )
    def forward(self, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor):
        # map_img: (B, 1, 224, 224)
        map_feat = self.map_encoder(map_img) # (B, map_embed_dim)
        
        start_feat = self.start_encoder(start)
        goal_feat = self.goal_encoder(goal)

        # fuse (B, map_embed_dim + 2 * position_embed_dim)
        combined = torch.cat([map_feat, start_feat, goal_feat], dim=1)
        return self.fusion(combined)

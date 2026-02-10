import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetMapEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        # Use ResNet18 (pretrained)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify first layer for 1-channel input (Grayscale)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Remove average pooling and fc layer
        # Output of layer4 is (B, 512, 7, 7) for 224x224 input
        self.flatten_dim = 512 * 7 * 7 
        
        self.proj = nn.Linear(self.flatten_dim, out_dim)

    def forward(self, x):
        # x: (B, 1, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (B, 512, 7, 7)
        
        x = x.flatten(1)   # (B, 25088)
        return self.proj(x)

class ConditionEncoder(nn.Module):
    def __init__(self, sg_dim:int=2, position_embed_dim:int=128, map_embed_dim:int=256, cond_dim:int=128):
        super().__init__()
        
        # UPGRADED: Use ResNet-18 (Pretrained, preserving spatial features)
        self.map_encoder = ResNetMapEncoder(out_dim=map_embed_dim)
        
        self.start_encoder = nn.Sequential(
            nn.Linear(sg_dim, position_embed_dim),
            nn.ReLU(),
            nn.Linear(position_embed_dim, position_embed_dim)
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(sg_dim, position_embed_dim),
            nn.ReLU(),
            nn.Linear(position_embed_dim, position_embed_dim)
        )

        self.fusion = nn.Sequential(
            # map_embed + start_embed + goal_embed
            nn.Linear(map_embed_dim + 2 * position_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, cond_dim)
        )
        # Dropout to prevent over-reliance on start/goal
        self.dropout = nn.Dropout(p=0.4)
    
    def forward(self, map_img:torch.Tensor, start:torch.Tensor, goal:torch.Tensor):
        # map_img: (B, 1, 64, 64)
        map_feat = self.map_encoder(map_img) # (B, map_embed_dim)
        
        start_feat = self.start_encoder(start)
        goal_feat = self.goal_encoder(goal)

        # Apply Dropout to force map usage
        start_feat = self.dropout(start_feat)
        goal_feat = self.dropout(goal_feat)

        # fuse
        combined = torch.cat([map_feat, start_feat, goal_feat], dim=1)
        return self.fusion(combined)

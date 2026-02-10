import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMapEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (B, 1, 224, 224)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 32, 112, 112)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 64, 56, 56)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 128, 28, 28)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # -> (B, 256, 1, 1)
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.proj(x)

class ConditionEncoder(nn.Module):
    def __init__(self, sg_dim:int=2, position_embed_dim:int=128, map_embed_dim:int=256, cond_dim:int=128):
        super().__init__()
        
        # UPGRADED: Use Simple Custom CNN instead of MobileNet
        self.map_encoder = SimpleMapEncoder(out_dim=map_embed_dim)
        
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

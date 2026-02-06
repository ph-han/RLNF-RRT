import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class MapEncoder(nn.Module):
    """
    Map encoder using pretrained ResNet18 with 3-channel input:
      - Channel 0: occupancy grid (0/1)
      - Channel 1: start position Gaussian heatmap
      - Channel 2: goal position Gaussian heatmap
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        # Load pretrained ResNet18 with ImageNet weights
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # conv1 already accepts 3 channels (RGB), no modification needed
        # Replace final fc layer to output latent_dim
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, latent_dim)

    def forward(self, x):
        return self.resnet(x)

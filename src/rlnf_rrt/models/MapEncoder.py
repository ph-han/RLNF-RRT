import torch
import torch.nn as nn
from torchvision import models

# class MapEncoder(nn.Module):
#     def __init__(self, latent_dim=128):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(128 * 14 * 14, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, latent_dim)
#         )

#     def forward(self, x):
#         return self.conv_block(x)

class MapEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, latent_dim)

    def forward(self, x):
        return self.resnet(x)

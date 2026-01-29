import torch
import torch.nn as nn

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


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.i_downsample = i_downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        out += identity
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, latent_dim=128, num_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 맵 특징 추출을 위한 4단계 레이어
        self.layer1 = self._make_layer(ResBlock, 64,  layer_list[0], stride=1)
        self.layer2 = self._make_layer(ResBlock, 128, layer_list[1], stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, layer_list[2], stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, layer_list[3], stride=2)

        # Global Average Pooling을 추가하면 입력 이미지 크기에 상관없이 작동하며 속도가 빨라집니다.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, latent_dim)
        
    def forward(self, x):
        x = self.max_pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def _make_layer(self, ResBlock, planes, blocks, stride=1):
        ii_downsample = None
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )
        layers = [ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride)]
        self.in_channels = planes * ResBlock.expansion
        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, planes))
        return nn.Sequential(*layers)

def MapEncoder(latent_dim=128, channels=1):
    # ResNet18 구조가 PlannerFlows의 실시간 샘플링에 가장 적합합니다.
    return ResNet(BasicBlock, [2, 2, 2, 2], latent_dim, channels)
# model_crnn_deep.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.leaky_relu(out)


class CRNN(nn.Module):
    def __init__(self, num_classes, dropout_fc=0.2):
        super().__init__()
        # Encoder
        self.layer1 = ResidualBlock(1, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 64, stride=1)
        self.layer4 = ResidualBlock(64, 128, stride=2)
        self.layer5 = ResidualBlock(128, 128, stride=1)
        self.layer6 = ResidualBlock(128, 256, stride=2)

        # Глобальное усреднение по высоте (оставляем только ширину как "время")
        self.global_pool = nn.AdaptiveAvgPool2d((1, None))  # H → 1

        self.drop_fc = nn.Dropout(dropout_fc)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # [B,1,H,W]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)  # [B,C,H’,W’]

        x = self.global_pool(x)  # [B,C,1,W]
        x = x.squeeze(2).permute(0, 2, 1)  # [B,W,C]

        x = self.drop_fc(x)
        x = self.fc(x)  # [B,W,num_classes]

        x = F.log_softmax(x, dim=-1)
        return x.permute(1, 0, 2)  # [W,B,num_classes]


# --------- Тестовый запуск ---------
if __name__ == "__main__":
    model = CRNNDeep(num_classes=11)
    data = torch.randn(8, 1, 64, 256)  # [B,1,H,W]
    out = model(data)
    print(f"Input: {data.shape}, Output: {out.shape}")  # [W,B,num_classes]

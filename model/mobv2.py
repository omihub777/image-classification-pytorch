import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import sys,os
sys.path.append(os.path.abspath("model"))
from layers import InvBottleneck

class MobileNetV2(nn.Module):
    def __init__(self, in_c, num_classes):
        super(MobileNetV2, self).__init__()
        self.blc1 = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(True)
        )
        self.blc2 = nn.Sequential(
            InvBottleneck(32, 16, expansion=1),
            InvBottleneck(16, 24),
            InvBottleneck(24, 24)
        )#32x32
        self.blc3 = nn.Sequential(
            InvBottleneck(24, 32, s=2),
            InvBottleneck(32, 32),
            InvBottleneck(32, 32)
        )#16x16
        self.blc4 = nn.Sequential(
            InvBottleneck(32, 64, s=2),
            InvBottleneck(64, 64),
            InvBottleneck(64, 64),
            InvBottleneck(64, 64),
            InvBottleneck(64, 96),
            InvBottleneck(96, 96),
            InvBottleneck(96, 96)
        )#8x8
        self.blc5 = nn.Sequential(
            InvBottleneck(96, 160, s=2),
            InvBottleneck(160, 160),
            InvBottleneck(160, 160),
            InvBottleneck(160, 320),
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )#4x4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(1280, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.blc1(x)
        out = self.blc2(out)
        out = self.blc3(out)
        out = self.blc4(out)
        out = self.blc5(out)
        out = self.gap(out)
        out = self.fc(out)
        return out.flatten(1)

if __name__ == "__main__":
    b, c, h, w = 4, 3, 224, 224
    x = torch.randn(b, c, h, w)
    n = MobileNetV2(c, 10)
    out = n(x)
    torchsummary.summary(n, (c,h,w))
    print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import sys, os
sys.path.append(os.path.abspath("model"))
from layers import SEInvBottleneck

class MobileNetV3(nn.Module):
    def __init__(self, in_c, num_classes):
        super(MobileNetV3, self).__init__()

        self.blc1 = nn.Sequential(
            nn.Conv2d(in_c, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(True)
        )
        self.blc2 = nn.Sequential(
            SEInvBottleneck(16, 16, 16, act='relu'),
            SEInvBottleneck(16, 64, 24, act='relu'),
            SEInvBottleneck(24, 72, 24, act='relu')
        )
        self.blc3 = nn.Sequential(
            SEInvBottleneck(24, 72, 40, k=5, s=2, p=2, act='relu', se=True),
            SEInvBottleneck(40, 120, 40, k=5, s=1, p=2, act='relu', se=True),
            SEInvBottleneck(40, 120, 40, k=5, s=1, p=2, act='relu', se=True)
        )
        self.blc4 = nn.Sequential(
            SEInvBottleneck(40, 240, 80, s=2, act='hswish'),
            SEInvBottleneck(80, 200, 80, act='hswish'),
            SEInvBottleneck(80, 184, 80, act='hswish'),
            SEInvBottleneck(80, 184, 80, act='hswish'),
            SEInvBottleneck(80,480, 112, act='hswish', se=True),
            SEInvBottleneck(112, 672, 112, act='hswish', se=True)
        )
        self.blc5 = nn.Sequential(
            SEInvBottleneck(112, 672, 160, k=5, s=2, p=2,act='hswish', se=True),
            SEInvBottleneck(160, 960, 160, k=5, s=1, p=2, act='hswish', se=True),
            SEInvBottleneck(160, 960, 160, k=5, s=1, p=2, act='hswish', se=True),
            nn.Conv2d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960),
            nn.Hardswish(True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(960, 1280, kernel_size=1),
            nn.Hardswish(True)
        )
        self.fc2 = nn.Conv2d(1280, num_classes, kernel_size=1)
    def forward(self, x):
        out = self.blc1(x)
        out = self.blc2(out)
        out = self.blc3(out)
        out = self.blc4(out)
        out = self.blc5(out)
        out = self.gap(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out.flatten(1)

if __name__ == "__main__":
    b,c,h,w = 4, 3, 32, 32
    x = torch.randn(b,c,h,w)
    n = MobileNetV3(c, 10)
    out = n(x)
    print(out.shape)
    torchsummary.summary(n, (c,h,w))
    
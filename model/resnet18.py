import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import sys, os
sys.path.append(os.path.abspath("model"))
from layers import ResBlock        

class ResNet18(nn.Module):
    def __init__(self, in_c, num_classes):
        super(ResNet18, self).__init__()
        self.blc1 = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.blc2 = nn.Sequential(
            ResBlock(64, 64, s=1),
            ResBlock(64, 64)
        )

        self.blc3 = nn.Sequential(
            ResBlock(64, 128, s=2),
            ResBlock(128, 128)
        )

        self.blc4 = nn.Sequential(
            ResBlock(128, 256, s=2),
            ResBlock(256, 256)
        )

        self.blc5 = nn.Sequential(
            ResBlock(256, 512, s=2),
            ResBlock(512, 512)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.blc1(x)
        out = self.blc2(out)
        out = self.blc3(out)
        out = self.blc4(out)
        out = self.blc5(out)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    b,c,h,w = 4, 3, 32, 32
    # x = torch.randn(b, c, h, w)
    net = ResNet18(c, 10)
    torchsummary.summary(net, input_size=(c,h,w))
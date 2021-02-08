import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import sys, os
sys.path.append(os.path.abspath("model"))
from layers import DepSepConvBlock, ConvBlock

class MobileNetV1(nn.Module):
    def __init__(self, in_c, num_classes):
        super(MobileNetV1, self).__init__()
        self.blc1 = nn.Sequential(
            ConvBlock(in_c, 32),
            DepSepConvBlock(32, 64)
        )
        self.blc2 = nn.Sequential(
            DepSepConvBlock(64, 128),
            DepSepConvBlock(128, 128)
        )
        self.blc3 = nn.Sequential(
            DepSepConvBlock(128, 256, s=2),
            DepSepConvBlock(256, 256)
        )
        self.blc4 = nn.Sequential(
            DepSepConvBlock(256, 512, s=2),
            DepSepConvBlock(512, 512),
            DepSepConvBlock(512, 512),
            DepSepConvBlock(512, 512),
            DepSepConvBlock(512, 512),
            DepSepConvBlock(512, 512)
        )
        self.blc5 = nn.Sequential(
            DepSepConvBlock(512, 1024, s=2),
            DepSepConvBlock(1024, 1024)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

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
    net = MobileNetV1(c, 10)
    x = torch.randn(b, c, h, w)
    torchsummary.summary(net, input_size=(c,h,w))
    out = net(x)
    print(out.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import os, sys
sys.path.append(os.path.abspath("model"))
from layers import PreActBottleneck, ConvBlock


class PreAct50(nn.Module):
    def __init__(self, in_c, num_classes):
        super(PreAct50, self).__init__()
        self.blc1 = ConvBlock(in_c, 64, 3, 1, 1)
        self.blc2 = nn.Sequential(
            PreActBottleneck(64, 256),
            PreActBottleneck(256, 256),
            PreActBottleneck(256, 256)
        )
        self.blc3 = nn.Sequential(
            PreActBottleneck(256, 512,s=2),
            PreActBottleneck(512, 512),
            PreActBottleneck(512, 512),
            PreActBottleneck(512, 512),
        )
        self.blc4 = nn.Sequential(
            PreActBottleneck(512, 1024,s=2),
            PreActBottleneck(1024, 1024),
            PreActBottleneck(1024, 1024),
            PreActBottleneck(1024, 1024),
            PreActBottleneck(1024, 1024),
            PreActBottleneck(1024, 1024)
        )
        self.blc5 = nn.Sequential(
            PreActBottleneck(1024, 2048,s=2),
            PreActBottleneck(2048, 2048),
            PreActBottleneck(2048, 2048)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    
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
    b, c, h, w = 4 ,3, 32, 32
    n = PreAct50(c, 10)
    torchsummary.summary(n , (c, h, w))
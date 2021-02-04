import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import os, sys
sys.path.append(os.path.abspath("model"))
from layers import PreActBlock, ConvBlock

class PreAct34(nn.Module):
    def __init__(self, in_c, num_classes):
        super(PreAct34, self).__init__()
        self.blc1 = ConvBlock(in_c, 64, 3, 1, 1)
        self.blc2 = nn.Sequential(
            PreActBlock(64, 64),
            PreActBlock(64, 64),
            PreActBlock(64, 64)
        )
        self.blc3 = nn.Sequential(
            PreActBlock(64, 128, s=2),
            PreActBlock(128, 128),
            PreActBlock(128, 128),
            PreActBlock(128, 128)
        )
        self.blc4 = nn.Sequential(
            PreActBlock(128, 256, s=2),
            PreActBlock(256, 256),
            PreActBlock(256, 256),
            PreActBlock(256, 256),
            PreActBlock(256, 256),
            PreActBlock(256, 256)
        )
        self.blc5 = nn.Sequential(
            PreActBlock(256, 512, s=2),
            PreActBlock(512, 512),
            PreActBlock(512, 512)
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
    x = torch.randn(b,c,h,w)
    net = PreAct34(c, 10)
    out = net(x)
    torchsummary.summary(net, (c,h,w))
    print(out.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import os, sys
sys.path.append(os.path.abspath("model"))
from layers import SEPreActBlock, ConvBlock

class SEPreAct34(nn.Module):
    def __init__(self, in_c, num_classes, r=16):
        super(SEPreAct34, self).__init__()
        self.blc1 = ConvBlock(in_c, 64, 3, 1, 1)
        self.blc2 = nn.Sequential(
            SEPreActBlock(64, 64, r=r),
            SEPreActBlock(64, 64, r=r),
            SEPreActBlock(64, 64, r=r)
        )
        self.blc3 = nn.Sequential(
            SEPreActBlock(64, 128, s=2, r=r),
            SEPreActBlock(128, 128, r=r),
            SEPreActBlock(128, 128, r=r),
            SEPreActBlock(128, 128, r=r)
        )
        self.blc4 = nn.Sequential(
            SEPreActBlock(128, 256, s=2, r=r),
            SEPreActBlock(256, 256, r=r),
            SEPreActBlock(256, 256, r=r),
            SEPreActBlock(256, 256, r=r),
            SEPreActBlock(256, 256, r=r),
            SEPreActBlock(256, 256, r=r)
        )
        self.blc5 = nn.Sequential(
            SEPreActBlock(256, 512, s=2, r=r),
            SEPreActBlock(512, 512, r=r),
            SEPreActBlock(512, 512, r=r)
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
    net = SEPreAct34(c, 10)
    out = net(x)
    torchsummary.summary(net, (c,h,w))
    print(out.shape)

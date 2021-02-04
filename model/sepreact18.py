import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import sys, os
sys.path.append(os.path.abspath("model"))
from layers import SEPreActBlock

class SEPreAct18(nn.Module):
    def __init__(self, in_c, num_classes, bias=False, r=16):
        super(SEPreAct18, self).__init__()
        self.blc1 = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.blc2 = nn.Sequential(
            SEPreActBlock(64, 64, bias=bias, r=r),
            SEPreActBlock(64, 64, bias=bias, r=r)
        )

        self.blc3 = nn.Sequential(
            SEPreActBlock(64, 128, s=2, bias=bias, r=r),
            SEPreActBlock(128, 128, bias=bias, r=r)
        )


        self.blc4 = nn.Sequential(
            SEPreActBlock(128, 256, s=2, bias=bias, r=r),
            SEPreActBlock(256, 256, bias=bias, r=r)
        )

        self.blc5 = nn.Sequential(
            SEPreActBlock(256, 512, s=2, bias=bias, r=r),
            SEPreActBlock(512, 512, bias=bias, r=r)
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
    b, c, h, w = 4, 128, 32, 32
    x = torch.randn(b, c, h, w)
    net = SEPreAct18(in_c=c, num_classes=10)
    torchsummary.summary(net, input_size=(c,h,w))
    print(net(x).shape)
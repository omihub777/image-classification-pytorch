import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import sys, os
sys.path.append(os.path.abspath("model"))
from layers import WideResBlock, ConvBlock

class WideResNet(nn.Module):
    def __init__(self, in_c, num_classes, l=22, widen=8, se=False, r=16):
        super(WideResNet, self).__init__()
        num_block = int((l-4)/6)
        self.blc1 = ConvBlock(in_c, 16)
        self.blc2 = nn.Sequential(*self._make_layer(16, 16*widen, 1, num_block,se,r))
        self.blc3 = nn.Sequential(*self._make_layer(16*widen, 32*widen, 2, num_block,se,r))
        self.blc4 = nn.Sequential(*self._make_layer(32*widen, 64*widen, 2, num_block,se,r))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64*widen, num_classes)

    @staticmethod
    def _make_layer(in_c, out_c, s, num_block,se,r):
        blc_list = []
        for i in range(num_block):
            blc_list.append(WideResBlock(in_c, out_c, s=s,se=se,r=r))
            if i==0:
                in_c = out_c
                s = 1
        return blc_list


    def forward(self, x):
        out = self.blc1(x)
        out = self.blc2(out)
        out = self.blc3(out)
        out = self.blc4(out)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    b,c,h,w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    n = WideResNet(c, 10, l=16, widen=10,se=True, r=16)
    out = n(x)
    torchsummary.summary(n, (c,h,w))
    print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c,k=3,s=1,p=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out

class AllCNNC(nn.Module):
    def __init__(self, in_c, num_classes):
        super(AllCNNC, self).__init__()
        self.blc1 = nn.Sequential(
            ConvBlock(in_c, 96),
            ConvBlock(96, 96),
            ConvBlock(96,96, s=2)
        )
        self.blc2 = nn.Sequential(
            ConvBlock(96, 192),
            ConvBlock(192, 192),
            ConvBlock(192, 192, s=2)
        )
        self.blc3 = ConvBlock(192, 192)
        self.blc4 = ConvBlock(192, 192, k=1, p=0)
        self.blc5 = ConvBlock(192, num_classes, k=1, p=0)
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        out = self.blc1(x)
        out = self.blc2(out)
        out = self.blc3(out)
        out = self.blc4(out)
        out = self.blc5(out)
        out = self.gap(out)
        return out


if __name__ == "__main__":
    b,c,h,w = 4, 3, 32, 32
    net = AllCNNC(c, 10)
    torchsummary.summary(net, (c,h,w))


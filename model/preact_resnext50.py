import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import sys,os
sys.path.append(os.path.abspath("model"))
from layers import PreActResNeXtBottleneck


class PreActResNeXt50(nn.Module):
    """PreActResNext50
    Args:
        se: If True, Bottleneck uses Squeeze-and-Excitation Module    
    """
    def __init__(self, in_c, num_classes,cardinality=32, se=False, r=16):
        super(PreActResNeXt50, self).__init__()
        self.blc1 = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.blc2 = nn.Sequential(
            PreActResNeXtBottleneck(64, 256, cardinality=cardinality, se=se, r=r),
            PreActResNeXtBottleneck(256, 256, cardinality=cardinality, se=se, r=r),
            PreActResNeXtBottleneck(256,256, cardinality=cardinality, se=se, r=r)
        )
        self.blc3 = nn.Sequential(
            PreActResNeXtBottleneck(256, 512, cardinality=cardinality, se=se, r=r, s=2),
            PreActResNeXtBottleneck(512, 512, cardinality=cardinality, se=se, r=r),
            PreActResNeXtBottleneck(512, 512, cardinality=cardinality, se=se, r=r),
            PreActResNeXtBottleneck(512, 512, cardinality=cardinality, se=se, r=r)
        )
        self.blc4 = nn.Sequential(
            PreActResNeXtBottleneck(512, 1024, cardinality=cardinality, se=se, r=r, s=2),
            PreActResNeXtBottleneck(1024, 1024, cardinality=cardinality, se=se, r=r),
            PreActResNeXtBottleneck(1024, 1024, cardinality=cardinality, se=se, r=r),
            PreActResNeXtBottleneck(1024, 1024, cardinality=cardinality, se=se, r=r),
            PreActResNeXtBottleneck(1024, 1024, cardinality=cardinality, se=se, r=r),
            PreActResNeXtBottleneck(1024, 1024, cardinality=cardinality, se=se, r=r)
        )
        self.blc5 = nn.Sequential(
            PreActResNeXtBottleneck(1024, 2048, cardinality=cardinality, se=se, r=r, s=2),
            PreActResNeXtBottleneck(2048, 2048, cardinality=cardinality, se=se, r=r),
            PreActResNeXtBottleneck(2048, 2048, cardinality=cardinality, se=se, r=r)
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
    b, c, h, w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    n = PreActResNeXt50(c, 10, se=True)
    # out = n(x)
    torchsummary.summary(n, (c,h,w))
    # print(out.shape)
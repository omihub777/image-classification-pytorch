import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class PreActBlock(nn.Module):
    """Modified PreAct: relu(bn(x)) is used for skip."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=k, stride=1, padding=p, bias=bias)

        if s!=1 or in_c!=out_c:
            self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=s, padding=0,bias=bias)
        else:
            self.skip = nn.Sequential()

    def forward(self, x):
        x = F.relu(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(F.relu(self.bn2(out)))

        return out + self.skip(x)

class PreAct18(nn.Module):
    def __init__(self, in_c, num_classes, bias=False):
        super(PreAct18, self).__init__()
        self.blc1 = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.blc2 = nn.Sequential(
            PreActBlock(64, 64, s=2, bias=bias),
            PreActBlock(64, 64, bias=bias)
        )

        self.blc3 = nn.Sequential(
            PreActBlock(64, 128, s=2, bias=bias),
            PreActBlock(128, 128, bias=bias)
        )


        self.blc4 = nn.Sequential(
            PreActBlock(128, 256, s=2, bias=bias),
            PreActBlock(256, 256, bias=bias)
        )

        self.blc5 = nn.Sequential(
            PreActBlock(256, 512, s=2, bias=bias),
            PreActBlock(512, 512, bias=bias)
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
    b, c, h, w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    # blc = PreActBlock(c, 16, k=3, s=1, p=1)
    net = PreAct18(in_c=c, num_classes=10)
    torchsummary.summary(net, input_size=(c,h,w))
    print(net(x).shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class SEModule(nn.Module):
    def __init__(self, in_c, r=16):
        super(SEModule, self).__init__()
        hidden = in_c//r
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, in_c, kernel_size=1)

    def forward(self, x):
        out = self.gap(x)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return x*out.expand_as(x)


class PreActSEBlock(nn.Module):
    """Modified PreAct: relu(bn(x)) is used for skip."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False, r=16):
        super(PreActSEBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=k, stride=1, padding=p, bias=bias)
        self.se = SEModule(out_c, r=r)

        if s!=1 or in_c!=out_c:
            self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=s, padding=0,bias=bias)
        else:
            self.skip = nn.Sequential()

    def forward(self, x):
        x = F.relu(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.se(out)

        return out + self.skip(x)

class PreActSE18(nn.Module):
    def __init__(self, in_c, num_classes, bias=False, r=16):
        super(PreActSE18, self).__init__()
        self.blc1 = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.blc2 = nn.Sequential(
            PreActSEBlock(64, 64, bias=bias, r=r),
            PreActSEBlock(64, 64, bias=bias, r=r)
        )

        self.blc3 = nn.Sequential(
            PreActSEBlock(64, 128, s=2, bias=bias, r=r),
            PreActSEBlock(128, 128, bias=bias, r=r)
        )


        self.blc4 = nn.Sequential(
            PreActSEBlock(128, 256, s=2, bias=bias, r=r),
            PreActSEBlock(256, 256, bias=bias, r=r)
        )

        self.blc5 = nn.Sequential(
            PreActSEBlock(256, 512, s=2, bias=bias, r=r),
            PreActSEBlock(512, 512, bias=bias, r=r)
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
    net = PreActSE18(in_c=c, num_classes=10)
    # net = SEModule(c, r=1)
    torchsummary.summary(net, input_size=(c,h,w))
    print(net(x).shape)
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

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, k, s, p, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, k, 1, p, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_c)

        if s!=1 or in_c!=out_c:
            self.skip = nn.Conv2d(in_c, out_c, 1, s, bias=bias)
        else:
            self.skip = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out+self.skip(x))

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

class PreActBottleneck(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c//4, 1, 1, 0, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_c//4)
        self.conv2 = nn.Conv2d(out_c//4, out_c//4, 3, s, 1, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_c//4)
        self.conv3 = nn.Conv2d(out_c//4, out_c, 1, 1, 0, bias=bias)
        
        if s!=1 or in_c!=out_c:
            self.skip = nn.Conv2d(in_c, out_c, 1, s, 0, bias=bias)
        else:
            self.skip = nn.Sequential()


    def forward(self, x):
        x = F.relu(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))

        return out + self.skip(x)

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


class SEPreActBlock(nn.Module):
    """Modified PreAct: relu(bn(x)) is used for skip."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False, r=16):
        super(SEPreActBlock, self).__init__()
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

class SEPreActBottleneck(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False, r=16):
        super(SEPreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c//4, 1, 1, 0, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_c//4)
        self.conv2 = nn.Conv2d(out_c//4, out_c//4, 3, s, 1, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_c//4)
        self.conv3 = nn.Conv2d(out_c//4, out_c, 1, 1, 0, bias=bias)
        
        self.se = SEModule(out_c, r=r)

        if s!=1 or in_c!=out_c:
            self.skip = nn.Conv2d(in_c, out_c, 1, s, 0, bias=bias)
        else:
            self.skip = nn.Sequential()


    def forward(self, x):
        x = F.relu(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = self.se(out)
        return out + self.skip(x)

class DepSepConvBlock(nn.Module):
    def __init__(self, in_c):
        ...

    def forward(self, x):
        ...

if __name__ == "__main__":
    b, c, h, w = 4, 512, 32, 32
    x = torch.randn(b, c, h, w)
    n = PreActBottleneck(c, 1024, s=2)
    torchsummary.summary(n, (c, h, w))
    print(n(x).shape)
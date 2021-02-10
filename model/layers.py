import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import sys, os
sys.path.append(os.path.abspath("model"))

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
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False):
        super(DepSepConvBlock, self).__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=k, stride=s, padding=p, groups=in_c,bias=bias)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        out = F.relu(self.bn1(self.dw(x)))
        out = F.relu(self.bn2(self.pw(out)))
        return out

class InvBottleneck(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False, expansion=6):
        super(InvBottleneck, self).__init__()
        self.skip = s==1 and in_c==out_c
        h_c = in_c*expansion
        self.pw1 = nn.Conv2d(in_c, h_c, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(h_c)
        self.dw = nn.Conv2d(h_c, h_c, kernel_size=k, stride=s, padding=p, bias=bias, groups=in_c)
        self.bn2 = nn.BatchNorm2d(h_c)
        self.pw2 = nn.Conv2d(h_c, out_c, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        out = F.relu6(self.bn1(self.pw1(x)))
        out = F.relu6(self.bn2(self.dw(out)))
        out = self.bn3(self.pw2(out))
        if self.skip:
            out = out+x
        return out

class HSEModule(nn.Module):
    def __init__(self, in_c, r=4):
        """using Hard Sigmoid for the activation in the last layer of Excitation."""
        super(HSEModule, self).__init__()
        hidden = in_c//r
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, in_c, kernel_size=1)

    def forward(self, x):
        out = self.gap(x)
        out = F.relu(self.fc1(out))
        out = F.hardsigmoid(self.fc2(out))
        return x*out.expand_as(x)


class SEInvBottleneck(nn.Module):
    def __init__(self, in_c,h_c, out_c,k=3, s=1,p=1, bias=False, act='relu', se=False, r=4):
        """
        Args:
            act: 'relu' or 'hswish'
            se: whether use SE block or not
        """
        super(SEInvBottleneck, self).__init__()
        self.pw1 = nn.Conv2d(in_c, h_c, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(h_c)
        self.dw = nn.Conv2d(h_c, h_c, kernel_size=k, stride=s, padding=p, groups=h_c,bias=bias)
        self.bn2 = nn.BatchNorm2d(h_c)

        if se:
            self.se = HSEModule(h_c, r=r)
        else:
            self.se = nn.Sequential()
        self.pw2 = nn.Conv2d(h_c, out_c, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.skip = in_c==out_c and s==1

        if act=='relu':
            self.act = nn.ReLU(True)
        elif act=='hswish':
            self.act = nn.Hardswish(True) 
        else:
            raise ValueError(f"{act}")

    def forward(self, x):
        out = self.act(self.bn1(self.pw1(x)))
        out = self.act(self.bn2(self.dw(out)))
        out = self.se(out)
        out = self.bn3(self.pw2(out))
        if self.skip:
            out = out+x
        return out

class WideResBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False, dropout_rate=0.3):
        super(WideResBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=k, stride=1, padding=p, bias=bias)

        if in_c!=out_c or s!=1:
            self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=s, bias=False)
        else:
            self.skip = nn.Sequential()

    def forward(self, x):
        x = F.relu(self.bn1(x))
        out = F.dropout(self.conv1(x), self.dropout_rate, training=self.training)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.skip(x)




if __name__ == "__main__":
    b, c, h, w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    # n = PreActBottleneck(c, 1024, s=2)
    # n = SEInvBottleneck(in_c=160, h_c=960, out_c=160, k=5, s=1,p=2, se=True, act='hswish')
    widen=8
    n = WideResBlock(c, 16*widen)
    torchsummary.summary(n, (c, h, w))
    print(n(x).shape)
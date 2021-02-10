import torch
import torch.nn as nn
import torch.nn.functional as F

def hswish(x):
    out = x*F.relu6(x+3.)/6.
    return out


if __name__ == "__main__":
    x = torch.randn(4,16)
    out = hswish(x)
    print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel,
                          kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel,
                               kernel_size=3, stride=1, relu=True)
        self.cafe = CAHE(in_channel) if filter else nn.Identity()
        self.conv2 = BasicConv(out_channel, out_channel,
                               kernel_size=5, stride=1, relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.cafe(out)
        out = self.conv2(out)
        return out + x

# Physical Guided-based Refiner (PGR)

class PGR(nn.Module):
    def __init__(self, in_dim):
        super(PGR, self).__init__()
        self.sa = SpatialAttention()
        self.merge = nn.Conv2d(in_dim*2, in_dim, kernel_size=1, padding=0)
        self.ca = ChannelAttention(in_dim)

    def forward(self, x):
        a = self.ca(x)
        t = self.sa(x)
        x_1 = a*(1-t)
        x_2 = x * t
        out = x_1+x_2
        out = torch.cat([out, x], dim=1)
        out = self.merge(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa1 = nn.Conv2d(2, 1, 7, padding=3, bias=True, groups=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        T1 = self.sa1(x2)
        return self.act(T1)


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.conv0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 2, 1),
            nn.ReLU(),
            nn.Conv2d(2, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.conv0(x)
        return a


# Channel Attention-based High-frequency Enhancement (CAHE)

class CAHE(nn.Module):
    def __init__(self, features, M=4, r=2, L=32) -> None:
        super().__init__()
        d = max(int(features/r), L)
        self.features = features
        # 3×3卷积，dilated=1
        self.conv1 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3,
                      padding=1, dilation=1, groups=1),
            nn.GELU()
        )
        # 3×3卷积，dilated=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3,
                      padding=3, dilation=3, groups=1),
            nn.GELU()
        )

        # 3×3卷积，dilated=5
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3,
                      padding=5, dilation=5, groups=1),
            nn.GELU()
        )
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(features, features, 1),
            nn.Softmax(dim=1)
        )
        self.out2 = nn.Conv2d(features*2, features, kernel_size=1, padding=0)

    def forward(self, x):
        # 通过四个3×3卷积
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        emerge = x_1+x_2+x_3
        out = self.out(emerge) * emerge
        out = torch.cat([out, x], dim=1)
        out = self.out2(out)
        return out
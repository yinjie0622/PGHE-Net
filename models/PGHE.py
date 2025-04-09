
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Encoder


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel)
                  for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Dncoder


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# InputLayer


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            # Conv3×3
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            # Conv1×1
            BasicConv(out_plane // 4, out_plane // 2,
                      kernel_size=1, stride=1, relu=True),
            # Conv3×3
            BasicConv(out_plane // 2, out_plane // 2,
                      kernel_size=3, stride=1, relu=True),
            # Conv1×1
            BasicConv(out_plane // 2, out_plane,
                      kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(
            channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.merge(x)


class PGHE(nn.Module):
    def __init__(self, num_res=8):
        super(PGHE, self).__init__()

        base_channel = 24

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])
        # 用于上下采样
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2,
                      kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4,
                      kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4,
                      relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4,
                      relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])
        self.Refine = nn.ModuleList([
            PGR(base_channel * 4),
            PGR(base_channel * 2),
            PGR(base_channel * 1),
        ])
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2,
                      kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel,
                      kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3,
                          relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3,
                          relu=False, stride=1)
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2, _ = self.Encoder[1](z)
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z = self.Decoder[0](z)
        z = self.Refine[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z = self.Refine[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.Refine[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


def build_net():
    return PGHE()

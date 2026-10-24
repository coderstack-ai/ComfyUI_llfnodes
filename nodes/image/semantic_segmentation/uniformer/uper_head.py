"""UPerNet decode / FCN auxiliary heads matching CondadosAI UniFormer ADE20K keys."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.batch_norm(self.conv(x)))


class UPerHead(nn.Module):
    def __init__(
        self,
        in_channels=(64, 128, 320, 512),
        channels=512,
        num_classes=150,
        pool_scales=(1, 2, 3, 6),
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.psp_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvBNReLU(in_channels[-1], channels, kernel_size=1, padding=0),
                )
                for scale in pool_scales
            ]
        )
        self.bottleneck = ConvBNReLU(in_channels[-1] + len(pool_scales) * channels, channels, 3, 1)
        self.lateral_convs = nn.ModuleList(
            [ConvBNReLU(c, channels, kernel_size=1, padding=0) for c in in_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList(
            [ConvBNReLU(channels, channels, 3, 1) for _ in in_channels[:-1]]
        )
        self.fpn_bottleneck = ConvBNReLU(len(in_channels) * channels, channels, 3, 1)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def _psp_forward(self, x):
        psp_outs = [x]
        for ppm in self.psp_modules:
            y = ppm(x)
            y = F.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)
            psp_outs.append(y)
        return self.bottleneck(torch.cat(psp_outs, dim=1))

    def forward(self, inputs):
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, inputs[:-1])]
        laterals.append(self._psp_forward(inputs[-1]))
        for i in range(len(laterals) - 1, 0, -1):
            prev = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + prev
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(len(fpn_outs) - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=False)
        feats = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        return self.classifier(feats)


class FCNAuxHead(nn.Module):
    def __init__(self, in_channels=320, channels=256, num_classes=150):
        super().__init__()
        self.convs = nn.ModuleList([ConvBNReLU(in_channels, channels, 3, 1)])
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return self.classifier(x)

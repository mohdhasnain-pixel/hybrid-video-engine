"""RIFE v4.26 IFNet (inference only).

Adapted from Practical-RIFE (MIT licence, Zhewei Huang / hzwer):
https://github.com/hzwer/Practical-RIFE — train_log/IFNet_HDv3.py (v4.26).
Training-only branches were removed; the network is otherwise unchanged so the
official ``flownet.pkl`` weights load as-is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_grid_cache: dict = {}


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    key = (str(flow.device), flow.dtype, tuple(flow.shape))
    if key not in _grid_cache:
        n, _, h, w = flow.shape
        gx = torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype).view(1, 1, 1, w).expand(n, -1, h, -1)
        gy = torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype).view(1, 1, h, 1).expand(n, -1, -1, w)
        _grid_cache.clear()
        _grid_cache[key] = torch.cat([gx, gy], 1)
    flow = torch.cat([flow[:, 0:1] / ((img.shape[3] - 1.0) / 2.0), flow[:, 1:2] / ((img.shape[2] - 1.0) / 2.0)], 1)
    grid = (_grid_cache[key] + flow).permute(0, 2, 3, 1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def conv(cin, cout, k=3, s=1, p=1, d=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=True), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(self, img0, img1, timestep: float, scale_list, f0=None, f1=None):
        flow, mask, w0, w1 = self.flow_mask(img0, img1, timestep, scale_list, f0, f1)
        m = torch.sigmoid(mask)
        return w0 * m + w1 * (1 - m)

    def flow_mask(self, img0, img1, timestep: float, scale_list, f0=None, f1=None):
        """Bidirectional flow (4 ch), blend-mask logits and the two warped frames."""
        t = img0[:, :1] * 0 + timestep
        f0 = self.encode(img0) if f0 is None else f0
        f1 = self.encode(img1) if f1 is None else f1
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        flow = mask = feat = None
        w0, w1 = img0, img1
        for i, block in enumerate(blocks):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, t), 1), None, scale=scale_list[i])
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, mask, feat = block(torch.cat((w0, w1, wf0, wf1, t, mask, feat), 1), flow, scale=scale_list[i])
                flow = flow + fd
            w0 = warp(img0, flow[:, :2])
            w1 = warp(img1, flow[:, 2:4])
        return flow, mask, w0, w1


def load_state(path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    # official checkpoints are saved from DataParallel ("module." prefix)
    return {k.replace("module.", ""): v for k, v in sd.items() if "module." in k} or sd

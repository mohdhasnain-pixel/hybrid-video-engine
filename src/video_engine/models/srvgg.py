"""SRVGGNetCompact — the compact Real-ESRGAN architecture (realesr-general-x4v3).

Re-implemented from Real-ESRGAN (BSD-3-Clause, Xintao Wang et al.):
https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def load_state(path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def dni(strong: dict, weak: dict, strength: float) -> dict:
    """Deep network interpolation: blend strong-denoise and weak-denoise weights."""
    return {k: strength * strong[k] + (1 - strength) * weak[k] for k in strong}

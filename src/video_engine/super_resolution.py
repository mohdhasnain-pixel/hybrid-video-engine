"""Phase 4 — AI super-resolution adapter (Real-ESRGAN compact, PyTorch MPS).

Loads the model once, processes one frame at a time, tiles automatically for
large inputs, and halves the tile size when the GPU runs out of memory.
When the model's native 4× output is larger than the requested size, the
result is resized on the GPU so oversized frames never cross to the CPU.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from . import models_registry
from .ai import GPU_LOCK, empty_cache, is_oom, select_device, to_frame, to_tensor
from .models.srvgg import SRVGGNetCompact, dni, load_state

log = logging.getLogger(__name__)

# full-frame inference only for small inputs: 512-px tiles are steadier on MPS for
# large frames (measured 4.2 s/frame tiled vs 1.5-14 s erratic full-frame at 1080x1920)
AUTO_TILE_PIXELS = 1_000_000
RELEASE_EVERY = 100  # frames between MPS cache releases (keeps long jobs from creeping into swap)
TILE_PAD = 16


class SuperResolver:
    def __init__(self, model: str, out_size: tuple[int, int], denoise_strength: float = 0.5,
                 tile: int = 0, half: bool = True, device: str | None = None):
        if model != "realesr-general-x4v3":
            raise ValueError(f"Unsupported SR model {model}")
        self.name = model
        self.device = select_device(device)
        self.dtype = torch.float16 if half and self.device.type != "cpu" else torch.float32
        self.out_w, self.out_h = out_size
        self.denoise_strength = denoise_strength
        self.tile = tile
        self.scale = 4

        strong = load_state(models_registry.path_for(model, "realesr-general-x4v3.pth"))
        if denoise_strength < 1.0:
            weak = load_state(models_registry.path_for(model, "realesr-general-wdn-x4v3.pth"))
            state = dni(strong, weak, denoise_strength)
        else:
            state = strong
        net = SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4)
        net.load_state_dict(state, strict=True)
        self.net = net.eval().to(self.device, self.dtype)
        self.calls = 0

    def info(self) -> dict:
        return {"model": self.name, "device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""),
                "denoise_strength": self.denoise_strength, "tile": self.tile,
                **models_registry.model_info(self.name)}

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        with GPU_LOCK:
            out = to_frame(self.upscale_tensor(to_tensor(frame, self.device, self.dtype)))
            self.calls += 1
            if self.calls % RELEASE_EVERY == 0:
                empty_cache(self.device)
        return out

    @torch.inference_mode()
    def upscale_tensor(self, x: torch.Tensor) -> torch.Tensor:
        while True:
            try:
                tile = self.tile or (0 if x.shape[2] * x.shape[3] <= AUTO_TILE_PIXELS else 512)
                y = self.net(x) if tile == 0 else self._tiled(x, tile)
                if (y.shape[3], y.shape[2]) != (self.out_w, self.out_h):
                    # antialiased bicubic works for any ratio on MPS ("area" needs integer ratios there)
                    y = F.interpolate(y, size=(self.out_h, self.out_w), mode="bicubic",
                                      align_corners=False, antialias=y.shape[3] > self.out_w)
                return y
            except RuntimeError as exc:
                if not is_oom(exc) or (self.tile and self.tile <= 64):
                    raise
                empty_cache(self.device)
                self.tile = max(64, (self.tile or 512) // 2)
                log.warning("SR out of memory; retrying with tile=%d", self.tile)

    def _tiled(self, x: torch.Tensor, tile: int) -> torch.Tensor:
        _, c, h, w = x.shape
        s = self.scale
        out = torch.empty((1, c, h * s, w * s), device=x.device, dtype=x.dtype)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                py0, px0 = max(y0 - TILE_PAD, 0), max(x0 - TILE_PAD, 0)
                py1, px1 = min(y1 + TILE_PAD, h), min(x1 + TILE_PAD, w)
                o = self.net(x[:, :, py0:py1, px0:px1])
                oy, ox = (y0 - py0) * s, (x0 - px0) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = o[:, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
        return out

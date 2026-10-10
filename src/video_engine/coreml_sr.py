"""Core ML / Apple Neural Engine runtime for the compact Real-ESRGAN model.

The Neural Engine needs fixed tensor shapes, so each job uses one tile shape
chosen to divide the frame evenly (minimal wasted pixels), and the whole frame
goes through Core ML as a single batched prediction. Converted models are
cached in ``models/coreml/`` keyed by (denoise strength, tile shape, batch);
the first job with a new shape spends ~15 s converting.

Measured on an M4 (16 GB): ~0.32 µs per input pixel on the Neural Engine vs
~1 µs on the GPU via MPS, with mean absolute error ~6e-4 vs. the fp32 model.
"""

from __future__ import annotations

import logging
import math
import platform
import threading

import numpy as np
import torch
import torch.nn.functional as F

from . import models_registry
from .ai import GPU_LOCK, empty_cache, select_device, to_frame
from .config import get_settings
from .models.srvgg import SRVGGNetCompact, dni, load_state

log = logging.getLogger(__name__)

PAD = 16  # context around each tile (input pixels)
MAX_CORE = 736  # tile core size target; 736 + 2*16 = 768
_convert_lock = threading.Lock()


def available() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def tile_grid(w: int, h: int) -> tuple[int, int, int, int]:
    """(nx, ny, core_w, core_h) covering a w×h frame with near-equal tiles."""
    nx, ny = math.ceil(w / MAX_CORE), math.ceil(h / MAX_CORE)
    return nx, ny, math.ceil(w / nx), math.ceil(h / ny)


class CoreMLSuperResolver:
    def __init__(self, model: str, src_size: tuple[int, int], out_size: tuple[int, int],
                 denoise_strength: float = 0.5):
        if model != "realesr-general-x4v3":
            raise ValueError(f"Unsupported SR model {model}")
        import coremltools as ct

        self.name = model
        self.scale = 4
        self.w, self.h = src_size
        self.out_w, self.out_h = out_size
        self.denoise_strength = round(float(denoise_strength), 2)
        self.nx, self.ny, self.cw, self.ch = tile_grid(self.w, self.h)
        self.tiles = self.nx * self.ny
        th, tw = self.ch + 2 * PAD, self.cw + 2 * PAD
        self.tile_shape = (th, tw)
        cache = get_settings().models_dir / "coreml"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"srvgg-x4v3_dn{self.denoise_strength:.2f}_{th}x{tw}.mlpackage"
        with _convert_lock:
            if not path.exists():
                log.info("Converting SR model to Core ML for tile %dx%d (one-off)", tw, th)
                # one tile per prediction: measured faster on the ANE than a batched call
                self._convert(ct, path, (1, 3, th, tw))
        self.mlmodel = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.path = path
        # assembly and the final resize run on the GPU, which is otherwise idle
        self.device = select_device()

    def _convert(self, ct, path, shape) -> None:
        strong = load_state(models_registry.path_for(self.name, "realesr-general-x4v3.pth"))
        if self.denoise_strength < 1.0:
            weak = load_state(models_registry.path_for(self.name, "realesr-general-wdn-x4v3.pth"))
            strong = dni(strong, weak, self.denoise_strength)
        net = SRVGGNetCompact()
        net.load_state_dict(strong)
        net.eval()
        # trace on a tiny example (tracing at full batch size needs GBs of RAM); the
        # graph is shape-agnostic and Core ML specialises it to `shape` below
        with torch.no_grad():
            traced = torch.jit.trace(net, torch.rand(1, 3, 64, 64))
        mlm = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS14,
        )
        tmp = path.with_suffix(".tmp.mlpackage")
        mlm.save(str(tmp))
        tmp.rename(path)

    def info(self) -> dict:
        return {"model": self.name, "device": "coreml:neural-engine", "dtype": "float16",
                "denoise_strength": self.denoise_strength,
                "tile": f"{self.tiles}×{self.tile_shape[1]}×{self.tile_shape[0]}",
                **models_registry.model_info(self.name)}

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) != (self.w, self.h):
            raise ValueError(f"CoreML SR built for {self.w}x{self.h}, got {w}x{h}")
        s, p = self.scale, PAD * self.scale
        ph, pw = self.ny * self.ch - h, self.nx * self.cw - w
        x = np.pad(frame, ((PAD, PAD + ph), (PAD, PAD + pw), (0, 0)), mode="edge")
        x = x.transpose(2, 0, 1).astype(np.float16) * np.float16(1 / 255)
        th, tw = self.tile_shape
        cores = []
        for ty in range(self.ny):  # Neural Engine work: no GPU lock held
            for tx in range(self.nx):
                y0, x0 = ty * self.ch, tx * self.cw
                tile = np.ascontiguousarray(x[None, :, y0:y0 + th, x0:x0 + tw])
                out = self.mlmodel.predict({"x": tile})["y"]
                cores.append((y0, x0, out[:, :, p:p + self.ch * s, p:p + self.cw * s]))
        with GPU_LOCK:  # assembly + resize on the GPU
            full = torch.empty((1, 3, self.ny * self.ch * s, self.nx * self.cw * s), dtype=torch.float16,
                               device=self.device)
            for y0, x0, core in cores:
                full[:, :, y0 * s:(y0 + self.ch) * s, x0 * s:(x0 + self.cw) * s] = \
                    torch.from_numpy(core).to(self.device, torch.float16)
            y = full[:, :, :h * s, :w * s]
            if (w * s, h * s) != (self.out_w, self.out_h):
                y = F.interpolate(y, size=(self.out_h, self.out_w), mode="bicubic", align_corners=False,
                                  antialias=w * s > self.out_w)
            frame = to_frame(y)
            self.calls = getattr(self, "calls", 0) + 1
            if self.calls % 100 == 0:
                empty_cache(self.device)
        return frame

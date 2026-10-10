"""Phase 5 — frame interpolation (RIFE v4.26) and timeline resampling.

:class:`TimelineResampler` turns a constant-rate source stream into a
constant-rate target stream. Output frame ``k`` sits at source position
``p = k * base / target``; when ``p`` is an integer the source frame is passed
through untouched, otherwise an intermediate frame is synthesised at timestep
``p - floor(p)``. This handles 24/25/30/50 → 60 with one code path, and is
exact (rational arithmetic), so duration never drifts.

Scene cuts are detected so the model never morphs between two different shots;
the nearest real frame is used instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

from . import models_registry
from .ai import GPU_LOCK, empty_cache, select_device, to_frame, to_tensor
from .models.rife import IFNet, load_state, warp


def _thumb(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    sy, sx = max(1, h // 72), max(1, w // 128)
    return frame[::sy, ::sx].mean(axis=2, dtype=np.float32)


def is_scene_cut(a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
    """Normalised mean abs difference of tiny luma thumbnails."""
    return float(np.abs(_thumb(a) - _thumb(b)).mean() / 255.0) > threshold


def _gpu_locked(fn):
    def wrapper(*a, **kw):
        with GPU_LOCK:
            return fn(*a, **kw)
    wrapper.__name__ = fn.__name__
    return wrapper


class RifeInterpolator:
    name = "rife-v4.26"

    def __init__(self, scale: float = 1.0, half: bool = True, device: str | None = None):
        self.device = select_device(device)
        self.dtype = torch.float16 if half and self.device.type != "cpu" else torch.float32
        self.scale = scale
        net = IFNet()
        missing, _unexpected = net.load_state_dict(load_state(models_registry.path_for(self.name)), strict=False)
        if missing:
            raise RuntimeError(f"RIFE weights incomplete: missing {missing[:5]}")
        self.net = net.eval().to(self.device, self.dtype)
        # id(frame) -> (frame, padded tensor, features); holding the frame keeps its id unique
        self._cache: dict[int, tuple[np.ndarray, torch.Tensor, torch.Tensor]] = {}

    def info(self) -> dict:
        return {"model": self.name, "device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""),
                "scale": self.scale, **models_registry.model_info(self.name)}

    def _prep(self, frame: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded tensor + encoder features, cached per frame object."""
        key = id(frame)
        if key not in self._cache:
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            x = to_tensor(frame, self.device, self.dtype)
            h, w = x.shape[2:]
            m = max(64, int(64 / self.scale))
            ph, pw = math.ceil(h / m) * m, math.ceil(w / m) * m
            x = F.pad(x, (0, pw - w, 0, ph - h))
            self._cache[key] = (frame, x, self.net.encode(x))
        return self._cache[key][1:]

    @_gpu_locked
    @torch.inference_mode()
    def __call__(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        x0, f0 = self._prep(a)
        x1, f1 = self._prep(b)
        s = self.scale
        out = self.net(x0, x1, t, [16 / s, 8 / s, 4 / s, 2 / s, 1 / s], f0, f1)
        h, w = a.shape[:2]
        frame = to_frame(out[:, :, :h, :w])
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls % 200 == 0:
            empty_cache(self.device)  # release cached blocks so long jobs don't creep into swap
        return frame

    @_gpu_locked
    @torch.inference_mode()
    def interpolate_full(self, a: np.ndarray, b: np.ndarray, big_a: np.ndarray, big_b: np.ndarray,
                         t: float) -> np.ndarray:
        """Estimate motion on small frames ``a``/``b``, apply it to large ``big_a``/``big_b``.

        Used when interpolating *after* super-resolution: flow is computed at source
        resolution (cheap, ~1 GB) and upsampled to warp the enhanced frames, instead
        of running RIFE on 5.3K frames (~6-7 GB).
        """
        x0, f0 = self._prep(a)
        x1, f1 = self._prep(b)
        s = self.scale
        flow, mask, _, _ = self.net.flow_mask(x0, x1, t, [16 / s, 8 / s, 4 / s, 2 / s, 1 / s], f0, f1)
        h, w = a.shape[:2]
        H, W = big_a.shape[:2]
        flow, mask = flow[:, :, :h, :w], mask[:, :, :h, :w]
        flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
        flow = flow * torch.tensor([W / w, H / h, W / w, H / h], device=flow.device, dtype=flow.dtype).view(1, 4, 1, 1)
        m = torch.sigmoid(F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False))
        A = to_tensor(big_a, self.device, self.dtype)
        B = to_tensor(big_b, self.device, self.dtype)
        frame = to_frame(warp(A, flow[:, :2]) * m + warp(B, flow[:, 2:4]) * (1 - m))
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls % 200 == 0:
            empty_cache(self.device)
        return frame


class TimelineResampler:
    def __init__(self, base_fps: Fraction, target_fps: Fraction,
                 interpolate: Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None,
                 scene_change_threshold: float = 0.12,
                 view: Callable = lambda f: f, out: Callable = lambda f: f):
        """``view`` picks the frame used for scene-cut checks and ``out`` the frame emitted;
        both default to the item itself (items may be (source, enhanced) pairs)."""
        self.ratio = Fraction(base_fps) / Fraction(target_fps)  # source frames per output frame
        self.interpolate = interpolate
        self.threshold = scene_change_threshold
        self.view, self.out = view, out
        self.generated = 0
        self.passthrough = 0
        self.scene_cuts = 0
        self.tail_holds = 0

    def __call__(self, frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        it = iter(frames)
        prev = next(it, None)
        if prev is None:
            return
        cur = next(it, None)
        idx = 0  # index of `prev`
        k = 0
        cut = cur is not None and is_scene_cut(self.view(prev), self.view(cur), self.threshold)
        while True:
            p = k * self.ratio
            i, frac = math.floor(p), p - math.floor(p)
            while i > idx:  # advance window
                if cur is None:
                    return
                prev, cur, idx = cur, next(it, None), idx + 1
                cut = cur is not None and is_scene_cut(self.view(prev), self.view(cur), self.threshold)
            if frac == 0:
                self.passthrough += 1
                yield self.out(prev)
            elif cur is None:
                self.tail_holds += 1
                yield self.out(prev)  # last source frame keeps its full display interval
            elif cut or self.interpolate is None:
                self.scene_cuts += int(cut)
                yield self.out(prev if frac < 0.5 else cur)
            else:
                self.generated += 1
                yield self.interpolate(prev, cur, float(frac))
            k += 1

    def stats(self) -> dict:
        return {"generated": self.generated, "passthrough": self.passthrough, "scene_cut_holds": self.scene_cuts,
                "tail_holds": self.tail_holds}

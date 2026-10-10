"""Shared AI runtime helpers: device selection, tensor conversion, OOM detection.

Model adapters (:mod:`super_resolution`, :mod:`interpolation`) operate on
``HxWx3 uint8`` numpy frames so the rest of the pipeline never sees tensors,
and the encoder never needs to know which model produced a frame.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np
import torch

from .config import get_settings


class FrameModel(Protocol):
    name: str

    def info(self) -> dict: ...


# MPS is not safe for concurrent use from several threads (hard abort), so every
# GPU section takes this lock. Neural Engine (Core ML) predictions run outside it,
# which is what lets SR on the ANE overlap with RIFE on the GPU.
GPU_LOCK = threading.RLock()


def select_device(preference: str | None = None) -> torch.device:
    pref = preference or get_settings().device
    if pref in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor(frame: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # move uint8 (3x smaller than float) to the device, convert there
    t = torch.from_numpy(np.ascontiguousarray(frame)).to(device, non_blocking=True)
    return t.permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)


def to_frame(t: torch.Tensor) -> np.ndarray:
    t = t.squeeze(0).clamp_(0, 1).mul_(255.0).round_().to(torch.uint8)
    return t.permute(1, 2, 0).contiguous().cpu().numpy()


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def device_memory(device: torch.device) -> int | None:
    try:
        if device.type == "mps":
            return int(torch.mps.driver_allocated_memory())
        if device.type == "cuda":
            return int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001 - diagnostic only
        return None
    return None

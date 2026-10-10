"""Degradation estimate from sampled frames (rule-based, no ML yet).

All spatial statistics are computed on a *native-resolution* centre crop so
that rescaling does not hide compression blocks or noise:

* noise sigma  — Immerkær's fast noise estimator (8-bit scale)
* blockiness   — gradient energy on the 8-px block grid vs. off-grid
* blur         — Crété-Roffet re-blur metric (0 sharp .. 1 very blurry),
                 measured per axis so directional (motion) blur shows up
* motion / duplicates — from a short low-resolution scan of consecutive frames

Thresholds live in :data:`THRESHOLDS` so they can be tuned against the
benchmark set; later this module can be replaced by a trained classifier.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field

import numpy as np

from .analyzer import SourceProfile
from .config import get_settings

THRESHOLDS = {
    "noise_light": 3.0,
    "noise_medium": 6.0,
    "noise_strong": 10.0,
    "blockiness": 1.12,
    "low_bpp": 0.04,
    "soft": 0.40,
    "strong_blur": 0.50,
    "directional_blur": 0.07,
    "high_motion": 6.0,
    "duplicate_diff": 0.35,
}

CROP_W, CROP_H = 960, 540


@dataclass
class Degradation:
    noise_sigma: float
    blockiness: float
    blur: float
    blur_h: float
    blur_v: float
    motion: float  # mean abs diff between consecutive low-res frames (0-255)
    duplicate_ratio: float
    samples: int
    labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def noise_level(self) -> str:
        t = THRESHOLDS
        if self.noise_sigma >= t["noise_strong"]:
            return "strong"
        if self.noise_sigma >= t["noise_medium"]:
            return "medium"
        if self.noise_sigma >= t["noise_light"]:
            return "light"
        return "none"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["noise_level"] = self.noise_level
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Degradation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --- frame access ------------------------------------------------------------


def _grab_gray(path: str, t: float, vf: str, w: int, h: int, frames: int = 1) -> np.ndarray | None:
    s = get_settings()
    cmd = [
        s.ffmpeg, "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path,
        "-frames:v", str(frames), "-vf", f"{vf},format=gray", "-f", "rawvideo", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    n = len(out) // (w * h)
    if n == 0:
        return None
    return np.frombuffer(out[: n * w * h], np.uint8).reshape(n, h, w)


# --- metrics -----------------------------------------------------------------


def noise_sigma(img: np.ndarray) -> float:
    """Immerkær (1996): sigma = sqrt(pi/2) / (6 (W-2)(H-2)) * sum |I * M|."""
    i = img.astype(np.float32)
    conv = (
        i[:-2, :-2] - 2 * i[:-2, 1:-1] + i[:-2, 2:]
        - 2 * i[1:-1, :-2] + 4 * i[1:-1, 1:-1] - 2 * i[1:-1, 2:]
        + i[2:, :-2] - 2 * i[2:, 1:-1] + i[2:, 2:]
    )
    h, w = img.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def blockiness(img: np.ndarray) -> float:
    i = img.astype(np.float32)
    dx = np.abs(np.diff(i, axis=1)).mean(axis=0)
    dy = np.abs(np.diff(i, axis=0)).mean(axis=1)

    def ratio(d: np.ndarray) -> float:
        idx = np.arange(d.size)
        on = d[idx % 8 == 7]
        off = d[idx % 8 != 7]
        return float(on.mean() / max(off.mean(), 1e-6)) if on.size and off.size else 1.0

    return (ratio(dx) + ratio(dy)) / 2


def _box_blur_1d(i: np.ndarray, axis: int, k: int = 9) -> np.ndarray:
    pad = [(0, 0), (0, 0)]
    pad[axis] = (k // 2, k // 2)
    p = np.pad(i, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    c = np.insert(c, 0, 0, axis=axis)
    n = i.shape[axis]
    hi = np.take(c, np.arange(k, k + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / k).astype(np.float32)


def blur_metric(img: np.ndarray) -> tuple[float, float]:
    """Crété-Roffet et al. (2007) no-reference blur, per axis (h, v)."""
    i = img.astype(np.float32)
    res = []
    for axis in (1, 0):
        b = _box_blur_1d(i, axis)
        d_f = np.abs(np.diff(i, axis=axis))
        d_b = np.abs(np.diff(b, axis=axis))
        v = np.maximum(0, d_f - d_b)
        s_f = d_f.sum()
        res.append(float((s_f - v.sum()) / s_f) if s_f > 0 else 1.0)
    return res[0], res[1]


# --- main entry --------------------------------------------------------------


def estimate(profile: SourceProfile, samples: int = 6) -> Degradation:
    path = profile.path
    dur = max(profile.duration_seconds, 0.1)
    # centre crop at native (post-rotation) resolution
    cw, ch = min(CROP_W, profile.display_width), min(CROP_H, profile.display_height)
    cw, ch = cw - cw % 2, ch - ch % 2
    crop = f"crop={cw}:{ch}"

    noise, block, bh, bv = [], [], [], []
    times = [dur * (k + 0.5) / samples for k in range(samples)]
    for t in times:
        frame = _grab_gray(path, min(t, max(dur - 0.05, 0)), crop, cw, ch)
        if frame is None:
            continue
        f = frame[0]
        noise.append(noise_sigma(f))
        block.append(blockiness(f))
        h, v = blur_metric(f)
        bh.append(h)
        bv.append(v)

    # short low-res scan of consecutive frames: motion level + duplicated frames
    sw, sh = 256, max(2, round(256 * profile.display_height / profile.display_width / 2) * 2)
    start = min(dur * 0.1, max(dur - 3, 0))
    seq = _grab_gray(path, start, f"scale={sw}:{sh}:flags=area", sw, sh, frames=90)
    motion, dup = 0.0, 0.0
    if seq is not None and len(seq) > 2:
        diffs = np.abs(np.diff(seq.astype(np.float32), axis=0)).mean(axis=(1, 2))
        motion = float(np.median(diffs))
        dup = float((diffs < THRESHOLDS["duplicate_diff"]).mean())
        # a fully static shot is not "duplicated frames"
        if diffs.max() < THRESHOLDS["duplicate_diff"]:
            dup = 0.0

    def med(xs, default):
        return float(np.median(xs)) if xs else default

    d = Degradation(
        noise_sigma=round(med(noise, 0.0), 3),
        blockiness=round(med(block, 1.0), 4),
        blur=round(max(med(bh, 0.0), med(bv, 0.0)), 4),
        blur_h=round(med(bh, 0.0), 4),
        blur_v=round(med(bv, 0.0), 4),
        motion=round(motion, 3),
        duplicate_ratio=round(dup, 3),
        samples=len(noise),
    )
    classify(d, profile)
    return d


def classify(d: Degradation, profile: SourceProfile) -> None:
    t = THRESHOLDS
    labels, notes = [], []
    if d.noise_level != "none":
        labels.append("noisy")
        notes.append(f"Estimated noise sigma {d.noise_sigma:.1f} ({d.noise_level}).")
    low_bpp = profile.bits_per_pixel is not None and profile.bits_per_pixel < t["low_bpp"]
    if d.blockiness >= t["blockiness"] or (low_bpp and d.blockiness >= 1.05):
        labels.append("compressed")
        notes.append(f"Block-grid energy ratio {d.blockiness:.2f}"
                     + (f", only {profile.bits_per_pixel:.3f} bits/pixel." if low_bpp else "."))
    elif low_bpp:
        notes.append(f"Low bitrate ({profile.bits_per_pixel:.3f} bits/pixel) but no visible blocking measured.")
    if d.blur >= t["soft"]:
        directional = abs(d.blur_h - d.blur_v) >= t["directional_blur"]
        if d.blur >= t["strong_blur"] and directional and d.motion >= t["high_motion"] / 2:
            labels.append("motion_blur")
            notes.append(f"Directional blur (h={d.blur_h:.2f}, v={d.blur_v:.2f}) with motion — likely motion blur.")
        elif d.blur >= t["strong_blur"]:
            labels.append("out_of_focus")
            notes.append(f"Strong isotropic blur ({d.blur:.2f}) — detail cannot be truly recovered.")
        else:
            labels.append("soft")
            notes.append(f"Mildly soft image (blur metric {d.blur:.2f}).")
    if profile.display_width * profile.display_height >= 5312 * 2988 * 0.95:
        labels.append("high_resolution")
    if d.duplicate_ratio >= 0.3:
        notes.append(f"{d.duplicate_ratio:.0%} of sampled consecutive frames are duplicates — "
                     "unique motion is lower than the tagged frame rate.")
    if not labels or labels == ["high_resolution"]:
        labels.insert(0, "clean")
    d.labels, d.notes = labels, notes

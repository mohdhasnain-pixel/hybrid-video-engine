"""Comparison assets for the web UI.

* ``source_preview.mp4`` — browser-playable H.264 copy of the *same time window*
  of the source, framed exactly like the output (same aspect/padding) but at no
  more than the source's own resolution, so a wipe slider lines up.
* Still crops at 1:1 output pixels from the same timestamp:
  ``crop_<region>_output.png``, ``crop_<region>_lanczos.png`` (conventional
  upscale of the source) and ``crop_<region>_source.png`` (nearest-neighbour —
  the original pixels).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .config import get_settings
from .decoder import window_args
from .media import has_encoder, run
from .planner import JobPlan

CROP_W, CROP_H = 960, 540


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def source_preview(source: Path, plan: JobPlan, out: Path) -> Path:
    s = get_settings()
    # never larger than the source picture; capped for smooth browser playback
    k = min(plan.source_width / plan.content_width, 1920 / plan.target_width, 1.0)
    cw, ch = _even(plan.content_width * k), _even(plan.content_height * k)
    tw, th = _even(plan.target_width * k), _even(plan.target_height * k)
    geo = [f"scale={cw}:{ch}:flags=bicubic"]
    if plan.aspect_mode == "crop":
        geo.append(f"crop={tw}:{th}")
    elif (cw, ch) != (tw, th):
        geo.append(f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black")
    vf = ",".join([*[f for f in plan.decode_filters if f.startswith(("bwdif", "scale="))], *geo, "format=yuv420p"])
    enc = (["-c:v", "h264_videotoolbox", "-q:v", "60", "-allow_sw", "1"] if has_encoder("h264_videotoolbox")
           else ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast"])
    run([s.ffmpeg, "-nostdin", "-v", "error", "-y", *window_args(plan), "-i", str(source),
         "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf, *enc, "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", str(out)], timeout=3600)
    return out


def poster(video: Path, out: Path) -> Path:
    """First frame as JPEG, shown before the video loads."""
    run([get_settings().ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(video), "-frames:v", "1", "-q:v", "3",
         str(out)], timeout=300)
    return out


def output_preview(output: Path, plan: JobPlan, out: Path, max_side: int = 3840) -> Path:
    """H.264 copy of the output for the browser player (Chrome often refuses 5.3K HEVC).

    Long side capped at 3840 px (still well above any source we upscale from).
    """
    s = get_settings()
    k = min(1.0, max_side / max(plan.target_width, plan.target_height))
    w, h = _even(plan.target_width * k), _even(plan.target_height * k)
    vf = f"scale={w}:{h}:flags=lanczos,format=yuv420p" if k < 1 else "format=yuv420p"
    enc = (["-c:v", "h264_videotoolbox", "-q:v", "75", "-allow_sw", "1"]
           if has_encoder("h264_videotoolbox") and max(w, h) <= 4096
           else ["-c:v", "libx264", "-crf", "16", "-preset", "veryfast", "-level:v", "6.0"])
    run([s.ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(output), "-map", "0:v:0", "-map", "0:a:0?",
         "-vf", vf, *enc, "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)], timeout=3600)
    return out


def _frame_at(path: Path, t: float, vf: str | None, w: int, h: int) -> np.ndarray:
    s = get_settings()
    cmd = [s.ffmpeg, "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=600).stdout
    if len(raw) < w * h * 3:
        raise RuntimeError(f"Could not grab frame at {t:.2f}s from {path.name}")
    return np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3)


def _save_png(frame: np.ndarray, out: Path) -> None:
    s = get_settings()
    h, w = frame.shape[:2]
    subprocess.run([s.ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-i", "-", "-frames:v", "1", str(out)], input=frame.tobytes(), check=True, timeout=120)


def _detail_region(frame: np.ndarray, cw: int, ch: int) -> tuple[int, int]:
    """Top-left of the crop window with the most gradient energy (most detail)."""
    g = frame[::8, ::8].mean(axis=2, dtype=np.float32)
    e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]
    bh, bw = max(1, ch // 8), max(1, cw // 8)
    best, pos = -1.0, (0, 0)
    for y in range(0, max(1, e.shape[0] - bh), max(1, bh // 3)):
        for x in range(0, max(1, e.shape[1] - bw), max(1, bw // 3)):
            v = float(e[y:y + bh, x:x + bw].mean())
            if v > best:
                best, pos = v, (y * 8, x * 8)
    return pos


def still_crops(source: Path, output: Path, plan: JobPlan, out_dir: Path) -> list[dict]:
    tw, th = plan.target_width, plan.target_height
    t = plan.duration_seconds / 2
    out_frame = _frame_at(output, t, None, tw, th)
    geo = (f"crop={tw}:{th}" if plan.aspect_mode == "crop"
           else f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black")
    pre = ",".join(f for f in plan.decode_filters if f.startswith(("bwdif", "scale=")))
    pre = pre + "," if pre else ""
    src_t = plan.start_seconds + t
    lanczos = _frame_at(source, src_t, f"{pre}scale={plan.content_width}:{plan.content_height}:flags=lanczos,{geo}",
                        tw, th)
    nearest = _frame_at(source, src_t, f"{pre}scale={plan.content_width}:{plan.content_height}:flags=neighbor,{geo}",
                        tw, th)

    cw, ch = min(CROP_W, tw), min(CROP_H, th)
    regions = {"center": ((th - ch) // 2, (tw - cw) // 2), "detail": _detail_region(out_frame, cw, ch)}
    crops = []
    for name, (y, x) in regions.items():
        y, x = min(max(0, y), th - ch), min(max(0, x), tw - cw)
        item = {"region": name, "x": x, "y": y, "width": cw, "height": ch, "time": round(t, 3)}
        for label, frame in (("output", out_frame), ("lanczos", lanczos), ("source", nearest)):
            path = out_dir / f"crop_{name}_{label}.png"
            _save_png(frame[y:y + ch, x:x + cw], path)
            item[label] = path.name
        crops.append(item)
    _save_jpeg_preview(out_frame, out_dir / "frame_output.jpg")
    _save_jpeg_preview(lanczos, out_dir / "frame_lanczos.jpg")
    return crops


def _save_jpeg_preview(frame: np.ndarray, out: Path) -> None:
    s = get_settings()
    h, w = frame.shape[:2]
    subprocess.run([s.ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-i", "-", "-frames:v", "1", "-q:v", "2", str(out)], input=frame.tobytes(), check=True,
                   timeout=120)

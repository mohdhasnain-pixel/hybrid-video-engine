"""Thin helpers around the ffmpeg / ffprobe executables."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from .config import get_settings


class MediaError(RuntimeError):
    """An ffmpeg/ffprobe invocation failed."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        raise MediaError(f"{Path(cmd[0]).name} failed (exit {proc.returncode}): {tail(stderr)}", stderr)
    return proc


def ffprobe_json(path: Path | str, *args: str) -> dict:
    s = get_settings()
    proc = run([s.ffprobe, "-v", "error", "-of", "json", *args, str(path)], timeout=300)
    return json.loads(proc.stdout or b"{}")


def tail(text: str, lines: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def parse_rate(value: str | None) -> Fraction | None:
    """Parse ffprobe rates such as ``30000/1001``; returns None for 0/0."""
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


@lru_cache
def ffmpeg_capabilities() -> dict:
    """Which encoders and filters this ffmpeg build provides."""
    s = get_settings()
    enc = subprocess.run([s.ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    flt = subprocess.run([s.ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True).stdout

    def has(listing: str, name: str) -> bool:
        return any(len(p := line.split()) > 1 and p[1] == name for line in listing.splitlines())

    encoders = ["hevc_videotoolbox", "h264_videotoolbox", "libx265", "libx264", "aac"]
    filters = ["hqdn3d", "nlmeans", "deblock", "bwdif", "minterpolate", "cas", "fps", "mpdecimate", "tonemap"]
    return {
        "encoders": {e: has(enc, e) for e in encoders},
        "filters": {f: has(flt, f) for f in filters},
    }


def has_encoder(name: str) -> bool:
    return ffmpeg_capabilities()["encoders"].get(name, False)


def has_filter(name: str) -> bool:
    return ffmpeg_capabilities()["filters"].get(name, False)

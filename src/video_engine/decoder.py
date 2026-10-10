"""FFmpeg decoder: source file -> constant-rate RGB24 frames over a pipe."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .config import get_settings
from .media import MediaError, tail
from .planner import JobPlan


def window_args(plan: JobPlan) -> list[str]:
    """Input-side seek/limit options shared by decoder, audio input and proxies."""
    args = []
    if plan.start_seconds > 0:
        args += ["-ss", f"{plan.start_seconds:.3f}"]
    args += ["-t", f"{plan.duration_seconds:.3f}"]
    return args


class FrameDecoder:
    def __init__(self, source: Path, plan: JobPlan, log_path: Path):
        self.width, self.height = plan.source_width, plan.source_height
        self.frame_bytes = self.width * self.height * 3
        self.log_path = log_path
        vf = ",".join([*plan.decode_filters, "format=rgb24"])
        self.cmd = [
            get_settings().ffmpeg, "-nostdin", "-v", "error", *window_args(plan), "-i", str(source),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        self.proc: subprocess.Popen | None = None

    def frames(self) -> Iterator[np.ndarray]:
        with open(self.log_path, "wb") as log:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=log, bufsize=0)
            stdout = self.proc.stdout
            assert stdout is not None
            while True:
                buf = bytearray(self.frame_bytes)
                view, got = memoryview(buf), 0
                while got < self.frame_bytes:
                    n = stdout.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got < self.frame_bytes:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
            rc = self.proc.wait()
        if rc not in (0, -9, -15):
            err = self.log_path.read_text(errors="replace")
            raise MediaError(f"Decoder failed (exit {rc}): {tail(err)}", err)

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

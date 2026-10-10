"""FFmpeg encoder: final scaling/padding, colour tagging, video encode, audio mux.

Used two ways:
* :class:`FrameEncoder` — receives processed RGB frames over stdin (AI path).
* :func:`ffmpeg_only_command` — one ffmpeg process does everything when no AI
  stage is needed (fast path, no Python frame copies at all).
"""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from .config import get_settings
from .decoder import window_args
from .media import MediaError, tail
from .planner import JobPlan

COLOR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def _color_filter(plan: JobPlan) -> str:
    return (f"scale=out_color_matrix=bt709:out_range=tv,format={plan.encoder['pix_fmt']},"
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv")


def _audio_args(plan: JobPlan, input_index: int) -> list[str]:
    action = plan.audio["action"]
    if action == "none":
        return ["-an"]
    args = ["-map", f"{input_index}:a?"]
    if action == "copy":
        return args + ["-c:a", "copy"]
    return args + ["-c:a", "aac", "-b:a", plan.audio.get("bitrate", "256k")]


def _rate(r: Fraction) -> str:
    return f"{r.numerator}/{r.denominator}"


def ffmpeg_only_command(source: Path, plan: JobPlan, output: Path) -> list[str]:
    vf = ",".join([*plan.decode_filters, *plan.encode_filters, _color_filter(plan)])
    return [
        get_settings().ffmpeg, "-nostdin", "-v", "error", "-y", *window_args(plan), "-i", str(source),
        "-map", "0:v:0", "-vf", vf, "-r", _rate(plan.target_fps_fraction), "-fps_mode", "cfr",
        *plan.encoder["args"], *COLOR_TAGS, *_audio_args(plan, 0),
        "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(output),
    ]


class FrameEncoder:
    def __init__(self, source: Path, plan: JobPlan, width: int, height: int, output: Path, log_path: Path):
        self.width, self.height = width, height
        self.log_path = log_path
        s = get_settings()
        vf = ",".join([*plan.encode_filters, _color_filter(plan)])
        cmd = [s.ffmpeg, "-nostdin", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-framerate", plan.stream_fps or plan.target_fps, "-i", "-"]
        if plan.audio["action"] != "none":
            cmd += [*window_args(plan), "-i", str(source)]
        cmd += ["-map", "0:v", "-vf", vf, *plan.encoder["args"], *COLOR_TAGS]
        cmd += _audio_args(plan, 1) if plan.audio["action"] != "none" else ["-an"]
        cmd += ["-movflags", "+faststart", str(output)]
        self.cmd = cmd
        self._log = open(log_path, "wb")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Encoder expects {self.width}x{self.height}, got {frame.shape[1]}x{frame.shape[0]}")
        try:
            self.proc.stdin.write(memoryview(np.ascontiguousarray(frame)))
        except BrokenPipeError:
            self.proc.wait()
            raise MediaError(f"Encoder exited early: {tail(self.log_path.read_text(errors='replace'))}") from None
        self.frames += 1

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = self.proc.wait()
        self._log.close()
        if rc != 0:
            err = self.log_path.read_text(errors="replace")
            raise MediaError(f"Encoder failed (exit {rc}): {tail(err)}", err)

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self._log.close()


def run_ffmpeg_only(cmd: list[str], log_path: Path, on_frame=None, should_stop=None) -> None:
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            if should_stop and should_stop():
                proc.kill()
                break
            if line.startswith("frame=") and on_frame:
                try:
                    on_frame(int(line.split("=", 1)[1]))
                except ValueError:
                    pass
        rc = proc.wait()
    if should_stop and should_stop():
        return
    if rc != 0:
        err = log_path.read_text(errors="replace")
        raise MediaError(f"FFmpeg failed (exit {rc}): {tail(err)}", err)

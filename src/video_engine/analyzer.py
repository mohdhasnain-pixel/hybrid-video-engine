"""Phase 1 — video analyzer.

Builds a machine-readable :class:`SourceProfile` from real ffprobe output (and
a short timestamp scan) instead of assuming CFR / 16:9 / H.264.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from .media import MediaError, ffprobe_json, parse_rate

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


@dataclass
class AudioStream:
    index: int
    codec: str
    channels: int | None
    sample_rate: int | None
    bitrate: int | None
    duration_seconds: float | None
    start_time: float | None


@dataclass
class SourceProfile:
    path: str
    container: str
    size_bytes: int
    width: int  # coded width
    height: int
    display_width: int  # after SAR and rotation — what a player shows
    display_height: int
    rotation: int  # degrees, normalised to 0/90/180/270
    sample_aspect_ratio: str
    display_aspect_ratio: float
    codec: str
    profile: str | None
    pixel_format: str
    bit_depth: int
    nominal_fps: float  # r_frame_rate
    nominal_fps_fraction: str
    avg_fps: float
    is_vfr: bool
    timestamp_jitter: float  # stdev / mean of frame intervals in the scanned window
    duration_seconds: float
    frame_count: int | None
    video_bitrate: int | None
    bits_per_pixel: float | None
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    color_range: str | None
    is_hdr: bool
    field_order: str | None
    is_interlaced: bool
    start_time: float
    audio: list[AudioStream] = field(default_factory=list)

    @property
    def has_audio(self) -> bool:
        return bool(self.audio)

    @property
    def fps_fraction(self) -> Fraction:
        return Fraction(self.nominal_fps_fraction)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["has_audio"] = self.has_audio
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SourceProfile":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d["audio"] = [AudioStream(**a) for a in d.get("audio", [])]
        return cls(**d)


def _rotation(stream: dict) -> int:
    rot = 0
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if not rot and "rotate" in (stream.get("tags") or {}):
        rot = int(stream["tags"]["rotate"])
    return rot % 360


def _bit_depth(stream: dict) -> int:
    if stream.get("bits_per_raw_sample", "").isdigit():
        return int(stream["bits_per_raw_sample"])
    pix = stream.get("pix_fmt", "")
    for depth in (16, 12, 10):
        if str(depth) in pix:
            return depth
    return 8


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scan_timestamps(path: Path, seconds: float = 20.0) -> list[float]:
    """Presentation timestamps of the first ``seconds`` of video packets."""
    data = ffprobe_json(
        path,
        "-select_streams", "v:0",
        "-read_intervals", f"%+{seconds}",
        "-show_entries", "packet=pts_time,flags",
    )
    pts = sorted(t for p in data.get("packets", []) if (t := _float(p.get("pts_time"))) is not None)
    return pts


def analyze(path: Path | str) -> SourceProfile:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = ffprobe_json(path, "-show_streams", "-show_format")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    if video is None:
        raise MediaError(f"{path.name} has no video stream")

    width, height = int(video["width"]), int(video["height"])
    rotation = _rotation(video)
    sar = parse_rate((video.get("sample_aspect_ratio") or "1:1").replace(":", "/")) or Fraction(1)
    disp_w, disp_h = round(width * sar), height
    if rotation in (90, 270):
        disp_w, disp_h = disp_h, disp_w

    nominal = parse_rate(video.get("r_frame_rate")) or Fraction(30)
    avg = parse_rate(video.get("avg_frame_rate")) or nominal
    duration = _float(video.get("duration")) or _float(fmt.get("duration")) or 0.0
    frame_count = _int(video.get("nb_frames"))

    # Timestamp scan: VFR footage (phones, screen recordings) has irregular
    # intervals even when r_frame_rate looks like a clean number.
    jitter = 0.0
    try:
        pts = scan_timestamps(path)
        intervals = [b - a for a, b in zip(pts, pts[1:]) if b > a]
        if len(intervals) > 5:
            mean = statistics.fmean(intervals)
            jitter = statistics.pstdev(intervals) / mean if mean else 0.0
            measured = 1 / statistics.median(intervals)
            if abs(measured - float(nominal)) / float(nominal) > 0.05:
                nominal = Fraction(measured).limit_denominator(1001)
    except MediaError:
        pass
    is_vfr = jitter > 0.05 or abs(float(avg) - float(nominal)) / float(nominal) > 0.02

    bitrate = _int(video.get("bit_rate"))
    if bitrate is None:
        total = _int(fmt.get("bit_rate"))
        bitrate = total
    fps_for_bpp = float(avg) or float(nominal)
    bpp = bitrate / (width * height * fps_for_bpp) if bitrate and fps_for_bpp else None

    transfer = video.get("color_transfer")
    field_order = video.get("field_order")
    audio = [
        AudioStream(
            index=int(s["index"]),
            codec=s.get("codec_name", "unknown"),
            channels=_int(s.get("channels")),
            sample_rate=_int(s.get("sample_rate")),
            bitrate=_int(s.get("bit_rate")),
            duration_seconds=_float(s.get("duration")),
            start_time=_float(s.get("start_time")),
        )
        for s in streams
        if s.get("codec_type") == "audio"
    ]

    return SourceProfile(
        path=str(path),
        container=fmt.get("format_name", "unknown"),
        size_bytes=path.stat().st_size,
        width=width,
        height=height,
        display_width=disp_w,
        display_height=disp_h,
        rotation=rotation,
        sample_aspect_ratio=f"{sar.numerator}:{sar.denominator}",
        display_aspect_ratio=round(disp_w / disp_h, 4),
        codec=video.get("codec_name", "unknown"),
        profile=video.get("profile"),
        pixel_format=video.get("pix_fmt", "unknown"),
        bit_depth=_bit_depth(video),
        nominal_fps=round(float(nominal), 3),
        nominal_fps_fraction=f"{nominal.numerator}/{nominal.denominator}",
        avg_fps=round(float(avg), 3),
        is_vfr=is_vfr,
        timestamp_jitter=round(jitter, 4),
        duration_seconds=round(duration, 3),
        frame_count=frame_count,
        video_bitrate=bitrate,
        bits_per_pixel=round(bpp, 4) if bpp else None,
        color_primaries=video.get("color_primaries"),
        color_transfer=transfer,
        color_space=video.get("color_space"),
        color_range=video.get("color_range"),
        is_hdr=transfer in HDR_TRANSFERS,
        field_order=field_order,
        is_interlaced=field_order not in (None, "progressive", "unknown"),
        start_time=_float(video.get("start_time")) or 0.0,
        audio=audio,
    )

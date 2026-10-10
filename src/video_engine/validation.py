"""Output integrity checks — a file is only marked complete after these pass.

Critical checks (dimensions, frame rate, duration, playable video) fail the
job; advisory checks (A/V duration difference, colour tags) are reported.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from .analyzer import SourceProfile
from .media import MediaError, ffprobe_json, parse_rate
from .planner import JobPlan


def _check(name: str, ok: bool, expected, actual, critical: bool = True, note: str = "") -> dict:
    return {"name": name, "passed": bool(ok), "expected": expected, "actual": actual,
            "critical": critical, "note": note}


def validate(output: Path, plan: JobPlan, source: SourceProfile) -> dict:
    checks: list[dict] = []
    try:
        data = ffprobe_json(output, "-count_packets", "-show_streams", "-show_format")
    except MediaError as exc:
        return {"passed": False, "checks": [_check("readable", False, "ffprobe can read output", str(exc))]}

    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        return {"passed": False, "checks": [_check("video_stream", False, "1 video stream", "none")]}

    checks.append(_check("resolution", (v["width"], v["height"]) == (plan.target_width, plan.target_height),
                         f"{plan.target_width}x{plan.target_height}", f"{v['width']}x{v['height']}"))
    rate = parse_rate(v.get("r_frame_rate")) or Fraction(0)
    target = Fraction(plan.target_fps)
    checks.append(_check("frame_rate", abs(float(rate) - float(target)) < 0.01,
                         f"{float(target):.3f}", f"{float(rate):.3f}"))

    frame_dur = 1 / float(target)
    duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    tol = max(3 * frame_dur, 0.1)
    checks.append(_check("duration", abs(duration - plan.duration_seconds) <= tol,
                         f"{plan.duration_seconds:.3f}s ±{tol:.3f}", f"{duration:.3f}s"))

    packets = int(v.get("nb_read_packets") or 0)
    checks.append(_check("frame_count", abs(packets - plan.expected_frames) <= 3,
                         plan.expected_frames, packets,
                         note="frames actually present in the stream"))
    checks.append(_check("codec", v.get("codec_name") == plan.encoder["codec"],
                         plan.encoder["codec"], v.get("codec_name"), critical=False))
    checks.append(_check("color_tags", v.get("color_primaries") == "bt709" and v.get("color_transfer") == "bt709",
                         "bt709", f"{v.get('color_primaries')}/{v.get('color_transfer')}", critical=False))

    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if plan.audio["action"] == "none":
        checks.append(_check("audio_streams", not audio, 0, len(audio), critical=False))
    else:
        want = plan.audio.get("streams", 1)
        checks.append(_check("audio_streams", len(audio) == want, want, len(audio)))
        for i, a in enumerate(audio):
            ad = float(a.get("duration") or 0)
            # sync: audio and video must cover the same span (allow one audio frame + one video frame)
            drift_ok = abs(ad - duration) <= max(0.1, frame_dur + 0.05)
            checks.append(_check(f"audio_{i}_duration", drift_ok, f"{duration:.3f}s", f"{ad:.3f}s",
                                 critical=False, note="audio vs video span"))
            if plan.audio["action"] == "copy":
                src_codec = source.audio[i].codec if i < len(source.audio) else None
                checks.append(_check(f"audio_{i}_codec", a.get("codec_name") == src_codec, src_codec,
                                     a.get("codec_name"), critical=False, note="copied bit-exact"))

    passed = all(c["passed"] for c in checks if c["critical"])
    return {"passed": passed, "checks": checks,
            "output": {"width": v["width"], "height": v["height"], "fps": round(float(rate), 3),
                       "duration_seconds": round(duration, 3), "frames": packets, "codec": v.get("codec_name"),
                       "pix_fmt": v.get("pix_fmt"), "size_bytes": output.stat().st_size,
                       "bitrate": int(data.get("format", {}).get("bit_rate") or 0)}}

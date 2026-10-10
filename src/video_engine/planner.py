"""Phase 2 — the intelligent (but deterministic) processing planner.

Given a :class:`SourceProfile`, a :class:`Degradation` estimate, the user's
:class:`JobOptions` and a :class:`Preset`, produce a :class:`JobPlan`: which
stages run, with which backend and parameters, and *why*. The plan is pure
data; the pipeline executes it without making further decisions.

Key policies
* 60 FPS is a target policy, not a reason to interpolate everything.
* Restoration runs at source resolution (cheap, and before SR amplifies it).
* AI super-resolution only when the needed scale factor makes it worthwhile.
* Interpolation defaults to the source resolution (before SR).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, Field

from . import models_registry, perf
from .analyzer import SourceProfile
from .config import Preset
from .degradation import Degradation
from .media import has_encoder, has_filter

TARGET_5K3 = (5312, 2988)
STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
                  Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60), Fraction(100),
                  Fraction(120000, 1001), Fraction(120), Fraction(240)]
MP4_COPYABLE_AUDIO = {"aac", "mp3", "alac", "ac3", "eac3"}


class JobOptions(BaseModel):
    """User-facing settings for one job."""

    preset: str = "balanced"
    output_policy: Literal["5k3", "native"] = "5k3"
    target_width: int = TARGET_5K3[0]
    target_height: int = TARGET_5K3[1]
    target_fps: float = 60.0
    interpolation: Literal["auto", "on", "off"] = "auto"
    aspect_mode: Literal["fit", "crop"] = "fit"
    codec: Literal["preset", "hevc", "h264"] = "preset"
    max_seconds: float | None = Field(default=None, gt=0)
    start_seconds: float = Field(default=0.0, ge=0)
    preserve_audio: bool = True


@dataclass
class Operation:
    name: str  # decode / deinterlace / denoise / deblock / super_resolution / ...
    backend: str  # ffmpeg / ai:<model> / rife / none
    reason: str
    params: dict = field(default_factory=dict)


@dataclass
class JobPlan:
    source_width: int
    source_height: int  # display orientation
    target_width: int
    target_height: int
    content_width: int  # picture size inside the target frame (rest is padding)
    content_height: int
    aspect_mode: str
    base_fps: str  # CFR rate frames are decoded at (fraction string)
    target_fps: str
    fps_action: str  # preserve / interpolate / repeat / decimate / resample
    start_seconds: float
    duration_seconds: float
    expected_frames: int
    decode_filters: list[str]
    encode_filters: list[str]
    sr: dict
    interpolation: dict
    encoder: dict
    audio: dict
    streaming: bool  # True when frames go through the Python AI worker
    operations: list[Operation]
    warnings: list[str]
    preset: str
    queue_size: int
    temporal_checks: bool
    estimate: dict = field(default_factory=dict)
    stream_fps: str = ""  # rate frames leave the AI stage (== target_fps unless repeated in the encoder)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JobPlan":
        d = dict(d)
        d["operations"] = [Operation(**o) for o in d["operations"]]
        return cls(**d)

    @property
    def base_fps_fraction(self) -> Fraction:
        return Fraction(self.base_fps)

    @property
    def target_fps_fraction(self) -> Fraction:
        return Fraction(self.target_fps)


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def snap_rate(rate: float) -> Fraction:
    """Nearest standard rate within 0.05% (so 30 stays 30 and 29.97 stays 29.97)."""
    best = min(STANDARD_RATES, key=lambda r: abs(float(r) - rate))
    if abs(float(best) - rate) / float(best) < 0.0005:
        return best
    return Fraction(rate).limit_denominator(1001)


def _fmt_rate(r: Fraction) -> str:
    return f"{float(r):.3f}".rstrip("0").rstrip(".")


def plan_job(profile: SourceProfile, deg: Degradation, opts: JobOptions, preset: Preset) -> JobPlan:
    ops: list[Operation] = [Operation("analyze", "ffprobe", "Probe streams, timestamps and degradation.")]
    warnings: list[str] = []
    decode_filters: list[str] = []
    encode_filters: list[str] = []
    labels = set(deg.labels)
    src_w, src_h = profile.display_width, profile.display_height

    # ---------------- timeline window ----------------------------------------
    start = min(opts.start_seconds, max(profile.duration_seconds - 0.1, 0))
    duration = max(profile.duration_seconds - start, 0.0)
    if opts.max_seconds:
        duration = min(duration, opts.max_seconds)
    if duration <= 0:
        raise ValueError("Nothing to process: empty time window")

    # ---------------- square pixels -------------------------------------------
    if profile.sample_aspect_ratio not in ("1:1", "0:1"):
        decode_filters.append(f"scale={_even(src_w)}:{_even(src_h)},setsar=1")
        src_w, src_h = _even(src_w), _even(src_h)
        ops.append(Operation("square_pixels", "ffmpeg:scale",
                             f"Anamorphic source (SAR {profile.sample_aspect_ratio}) resampled to square pixels."))

    # ---------------- deinterlace --------------------------------------------
    if profile.is_interlaced:
        decode_filters.append("bwdif=mode=send_frame")
        ops.append(Operation("deinterlace", "ffmpeg:bwdif", f"Source is interlaced ({profile.field_order})."))

    if profile.is_hdr:
        warnings.append("HDR source: this build processes frames as SDR BT.709 (no zscale tone-mapping "
                        "available) — highlights may clip and colours will be compressed.")

    # ---------------- frame-rate policy --------------------------------------
    src_rate = snap_rate(profile.avg_fps if profile.is_vfr else profile.nominal_fps)
    if profile.is_vfr:
        ops.append(Operation("timeline", "ffmpeg:fps", f"Variable frame rate (jitter {profile.timestamp_jitter:.1%}); "
                             f"resampled to a constant {_fmt_rate(src_rate)} FPS timeline using source timestamps, "
                             "so duration and A/V sync are preserved."))
    if (deg.duplicate_ratio >= 0.4 and preset.temporal_checks and src_rate >= 48
            and (half := snap_rate(float(src_rate) / 2)) in STANDARD_RATES):
        warnings.append(f"About {deg.duplicate_ratio:.0%} duplicated frames: treating source as "
                        f"{_fmt_rate(half)} FPS of unique motion before interpolation.")
        src_rate = half

    if opts.output_policy == "native" and opts.interpolation != "on":
        target_rate = src_rate
    else:
        target_rate = snap_rate(opts.target_fps)

    interp_mode = opts.interpolation if opts.interpolation != "auto" else preset.interpolation.mode
    rife_ok = models_registry.is_installed("rife-v4.26")
    interp = {"enabled": False, "backend": None, "order": preset.interpolation.order,
              "from_fps": str(src_rate), "to_fps": str(target_rate),
              "rife_scale": preset.interpolation.rife_scale,
              "scene_change_threshold": preset.interpolation.scene_change_threshold}
    base_rate = src_rate
    if src_rate == target_rate:
        fps_action = "preserve"
        ops.append(Operation("frame_rate", "none", f"Source is already {_fmt_rate(src_rate)} FPS — preserved, "
                             "no interpolation."))
    elif src_rate > target_rate:
        fps_action = "decimate"
        base_rate = target_rate
        exact = (src_rate / target_rate).denominator == 1
        ops.append(Operation("frame_rate", "ffmpeg:fps", f"{_fmt_rate(src_rate)} → {_fmt_rate(target_rate)} FPS by "
                             + ("dropping every other frame (exact)." if exact else "timestamp-based frame selection.")))
    elif abs(float(src_rate) - float(target_rate)) / float(target_rate) < 0.01:
        fps_action = "resample"
        base_rate = target_rate
        ops.append(Operation("frame_rate", "ffmpeg:fps", f"{_fmt_rate(src_rate)} → {_fmt_rate(target_rate)} FPS: "
                             "~1 frame per 1000 repeated so duration stays exact."))
    elif interp_mode == "off":
        fps_action = "repeat"
        base_rate = target_rate
        ops.append(Operation("frame_rate", "ffmpeg:fps", f"{_fmt_rate(src_rate)} → {_fmt_rate(target_rate)} FPS by "
                             "repeating frames (interpolation disabled) — no new motion is created."))
    else:
        fps_action = "interpolate"
        backend = preset.interpolation.backend
        if backend == "rife" and not rife_ok:
            warnings.append("RIFE weights not installed (run `video-engine download-models`); "
                            "falling back to FFmpeg motion-compensated interpolation.")
            backend = "ffmpeg"
        if backend == "ffmpeg" and not has_filter("minterpolate"):
            raise RuntimeError("No interpolation backend available")
        interp.update(enabled=True, backend=backend)
        # reason is completed once the stage order is known (after the SR decision)
        interp_op = Operation("frame_interpolation", "ai:rife-v4.26" if backend == "rife" else "ffmpeg:minterpolate",
                              f"{_fmt_rate(src_rate)} → {_fmt_rate(target_rate)} FPS: synthesise intermediate frames "
                              f"({'timestamp-aware arbitrary timesteps' if backend == 'rife' else 'motion-compensated'})")
        ops.append(interp_op)

    if fps_action != "preserve" or profile.is_vfr:
        # a clean CFR timeline at base_rate, derived from source timestamps
        decode_filters.append(f"fps={base_rate.numerator}/{base_rate.denominator}")

    # ---------------- geometry ------------------------------------------------
    if opts.output_policy == "native":
        tw, th = _even(src_w), _even(src_h)
    else:
        tw, th = opts.target_width, opts.target_height
        if (src_h > src_w) != (th > tw):  # orientation-aware: portrait in, portrait out
            tw, th = th, tw
    if opts.aspect_mode == "crop":
        scale = max(tw / src_w, th / src_h)
    else:
        scale = min(tw / src_w, th / src_h)
    cw, chh = (min(_even(src_w * scale), tw), min(_even(src_h * scale), th)) if opts.aspect_mode == "fit" \
        else (_even(src_w * scale), _even(src_h * scale))
    if opts.aspect_mode == "fit" and abs(cw / chh - tw / th) < 0.01:
        cw, chh = tw, th  # rounding noise: fill the frame exactly

    # ---------------- restoration (source resolution) -------------------------
    sp = preset.super_resolution
    use_ai_sr = (sp.backend == "ai" and scale >= sp.min_factor_for_ai
                 and models_registry.is_installed(sp.model))
    if sp.backend == "ai" and scale >= sp.min_factor_for_ai and not use_ai_sr:
        warnings.append(f"AI model {sp.model} not installed (run `video-engine download-models`); "
                        "using FFmpeg Lanczos scaling instead.")

    strength_order = ["none", "light", "medium", "strong"]
    rp = preset.restoration
    noise = deg.noise_level
    if rp.enabled == "off":
        noise = "none"
    elif rp.enabled == "on" and noise == "none":
        noise = "light"
    noise = strength_order[min(strength_order.index(noise), strength_order.index(rp.max_strength))]
    # the AI model's own denoiser (DNI) handles light noise; avoid double denoising
    ff_noise = noise if not (use_ai_sr and noise == "light") else "none"
    if ff_noise == "light" and has_filter("hqdn3d"):
        decode_filters.append("hqdn3d=1.5:1.5:3:3")
        ops.append(Operation("denoise", "ffmpeg:hqdn3d", f"Light noise (sigma {deg.noise_sigma:.1f}); conservative "
                             "spatio-temporal denoise at source resolution."))
    elif ff_noise == "medium" and has_filter("hqdn3d"):
        decode_filters.append("hqdn3d=3:2.5:5:4")
        ops.append(Operation("denoise", "ffmpeg:hqdn3d", f"Medium noise (sigma {deg.noise_sigma:.1f})."))
    elif ff_noise == "strong":
        if has_filter("nlmeans"):
            decode_filters.append("nlmeans=s=3.5:p=7:r=9")
            ops.append(Operation("denoise", "ffmpeg:nlmeans", f"Strong noise (sigma {deg.noise_sigma:.1f}); "
                                 "non-local means at source resolution."))
        elif has_filter("hqdn3d"):
            decode_filters.append("hqdn3d=5:4:7:6")
            ops.append(Operation("denoise", "ffmpeg:hqdn3d", f"Strong noise (sigma {deg.noise_sigma:.1f})."))
    if "compressed" in labels and rp.deblock and rp.enabled != "off" and has_filter("deblock"):
        decode_filters.append("deblock=filter=strong:block=8")
        ops.append(Operation("deblock", "ffmpeg:deblock", "Compression blocking detected; deblock before upscaling "
                             "so SR does not sharpen block edges."))
    if labels & {"motion_blur", "out_of_focus"}:
        kind = "motion blur" if "motion_blur" in labels else "defocus blur"
        ops.append(Operation("deblur", "none", f"Strong {kind} detected. No dedicated deblur model is installed; "
                             "the SR model (trained with blur degradations) is used and no extra sharpening is "
                             "applied, to avoid ringing."))
        warnings.append(f"Strong {kind}: lost detail cannot be recovered, only plausibly re-synthesised.")

    # ---------------- super-resolution ----------------------------------------
    sr: dict = {"enabled": False, "backend": "none", "model": None, "scale_factor": round(scale, 3)}
    if use_ai_sr:
        spec = models_registry.get(sp.model)
        if sp.denoise_strength == "auto":
            dn = {"none": 0.3, "light": 0.5, "medium": 0.7, "strong": 0.85}[deg.noise_level]
            if "compressed" in labels:
                dn = max(dn, 0.7)
        else:
            dn = float(sp.denoise_strength)
        native_w, native_h = src_w * spec.scale, src_h * spec.scale
        # never push frames larger than needed through the pipe: when the model's
        # native output exceeds the content size, it is resized on the GPU.
        out_w, out_h = (native_w, native_h) if native_w <= cw and native_h <= chh else (cw, chh)
        runtime = _sr_runtime(sp.runtime)
        sr.update(enabled=True, backend="ai", model=sp.model, model_scale=spec.scale, runtime=runtime,
                  denoise_strength=round(dn, 2), out_width=out_w, out_height=out_h,
                  tile=sp.tile, half_precision=sp.half_precision)
        ops.append(Operation(
            "super_resolution", f"ai:{sp.model} ({RUNTIME_LABEL[runtime]})",
            f"{src_w}×{src_h} needs {scale:.2f}× → {spec.scale}× AI model "
            f"({out_w}×{out_h})" + (f", then Lanczos to {cw}×{chh} (adds pixels, not detail)."
                                    if (out_w, out_h) != (cw, chh) else "."),
            {"denoise_strength": round(dn, 2)}))
        if preset.hallucination_warning or "out_of_focus" in labels or scale > 3:
            warnings.append("AI super-resolution synthesises plausible detail; faces, text, logos and number "
                            "plates may not match the real scene.")
    elif (cw, chh) != (src_w, src_h):
        sr.update(backend="ffmpeg", algorithm="lanczos")
        why = "conventional resize" if scale < sp.min_factor_for_ai else "AI disabled by preset"
        ops.append(Operation("resize", "ffmpeg:lanczos", f"{src_w}×{src_h} → {cw}×{chh} ({scale:.2f}×, {why})."))
    else:
        ops.append(Operation("resize", "none", "Source already matches the target size — no scaling."))

    # ---------------- stage order (interpolation vs SR) -----------------------
    n_src = max(1, math.floor(duration * base_rate + 1e-6))
    n_out = max(1, math.floor(duration * target_rate + 1e-6))
    if interp["enabled"]:
        order, why = interp["order"], "set by preset"
        if not sr["enabled"]:
            order, why = "interpolate_first", "no super-resolution in this plan"
        elif order == "auto":
            order, why = _choose_order(interp["backend"], sr, src_w, src_h, n_src, n_out)
        interp["order"] = order
        # RIFE always runs on source-sized frames; coarser flow for >4 MP sources
        interp["rife_scale"] = min(interp["rife_scale"], 0.5) if src_w * src_h > 4e6 else 1.0
        interp_op.reason += (f", {'before' if order == 'interpolate_first' else 'after'} super-resolution "
                             f"({why}).")
        interp_op.params = {"order": order}

    # ---------------- assemble filter chains ---------------------------------
    if interp["enabled"] and interp["backend"] == "ffmpeg" and interp["order"] == "interpolate_first":
        decode_filters.append(_minterpolate(target_rate))
    if (cw, chh) != (sr.get("out_width", src_w), sr.get("out_height", src_h)):
        encode_filters.append(f"scale={cw}:{chh}:flags=lanczos")
    if interp["enabled"] and interp["backend"] == "ffmpeg" and interp["order"] == "upscale_first":
        encode_filters.append(_minterpolate(target_rate))
    if sp.final_sharpen == "cas" and not use_ai_sr and has_filter("cas") and scale > 1.2:
        encode_filters.append("cas=strength=0.4")
        ops.append(Operation("sharpen", "ffmpeg:cas", "Mild contrast-adaptive sharpening after Lanczos upscale."))
    if opts.aspect_mode == "crop" and (cw, chh) != (tw, th):
        encode_filters.append(f"crop={tw}:{th}")
        ops.append(Operation("crop", "ffmpeg", f"Centre-crop to fill {tw}×{th}."))
    elif (cw, chh) != (tw, th):
        encode_filters.append(f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black")
        ops.append(Operation("pad", "ffmpeg", f"Aspect ratio {src_w / src_h:.3f} preserved; padded to {tw}×{th}."))

    streaming = sr["enabled"] or (interp["enabled"] and interp["backend"] == "rife")

    # Frame repetition (interpolation off) happens AFTER the AI stage, in the encoder:
    # otherwise every repeated frame would be upscaled again for nothing.
    stream_fps = target_rate
    if streaming and fps_action == "repeat":
        decode_filters = [f for f in decode_filters if not f.startswith("fps=")]
        if profile.is_vfr:
            decode_filters.insert(1 if profile.is_interlaced else 0, f"fps={src_rate.numerator}/{src_rate.denominator}")
        encode_filters.insert(0, f"fps={target_rate.numerator}/{target_rate.denominator}")
        base_rate = stream_fps = src_rate
        n_src = max(1, math.floor(duration * src_rate + 1e-6))

    # ---------------- encoder --------------------------------------------------
    ep = preset.encoder
    codec = ep.codec if opts.codec == "preset" else opts.codec
    encoder = _choose_encoder(codec, ep, tw, th)
    ops.append(Operation("encode", encoder["name"], encoder["why"], {k: v for k, v in encoder.items()
                                                                    if k not in ("why", "args")}))

    # ---------------- audio ----------------------------------------------------
    if not profile.has_audio or not opts.preserve_audio:
        audio = {"action": "none"}
        if profile.has_audio:
            ops.append(Operation("audio", "none", "Audio removed by user setting."))
    elif all(a.codec in MP4_COPYABLE_AUDIO for a in profile.audio):
        audio = {"action": "copy", "streams": len(profile.audio)}
        ops.append(Operation("audio", "copy", f"{len(profile.audio)} audio stream(s) copied bit-exact "
                             f"({', '.join(a.codec for a in profile.audio)})."))
    else:
        audio = {"action": "aac", "bitrate": "256k", "streams": len(profile.audio)}
        codecs = ", ".join(a.codec for a in profile.audio)
        ops.append(Operation("audio", "ffmpeg:aac", f"{codecs} cannot be stored in MP4 — transcoded to AAC 256k."))
        warnings.append(f"Audio re-encoded from {codecs} to AAC (MP4 container compatibility).")

    expected = n_out
    estimate = _estimate(streaming, sr, interp, src_w, src_h, tw, th, n_src, n_out,
                         n_stream=max(1, math.floor(duration * stream_fps + 1e-6)))
    return JobPlan(
        source_width=src_w, source_height=src_h, target_width=tw, target_height=th,
        content_width=cw, content_height=chh, aspect_mode=opts.aspect_mode,
        base_fps=str(base_rate), target_fps=str(target_rate), fps_action=fps_action,
        start_seconds=round(start, 3), duration_seconds=round(duration, 3), expected_frames=expected,
        decode_filters=decode_filters, encode_filters=encode_filters, sr=sr, interpolation=interp,
        encoder=encoder, audio=audio, streaming=streaming, operations=ops, warnings=warnings,
        preset=preset.name, queue_size=preset.queue_size, temporal_checks=preset.temporal_checks,
        estimate=estimate, stream_fps=str(stream_fps),
    )


RUNTIME_LABEL = {"coreml": "Neural Engine", "mps": "GPU", "cpu": "CPU"}


def _sr_runtime(pref: str) -> str:
    from . import coreml_sr

    if pref in ("auto", "coreml") and coreml_sr.available():
        return "coreml"
    import torch

    return "mps" if pref in ("auto", "mps", "coreml") and torch.backends.mps.is_available() else "cpu"


def _stage_seconds(backend: str, sr: dict, src_w: int, src_h: int, n_src: int, n_out: int, order: str) -> tuple:
    """(sr_seconds, interpolation_seconds, interpolation_gb) for one stage order."""
    r = perf.rates()
    src_mp = src_w * src_h / 1e6
    sr_cost = perf.sr_rate(sr["runtime"]) * src_mp if sr.get("enabled") else 0.0
    n_new = max(0, n_out - n_src)
    if order == "interpolate_first":
        rife = n_new * r["rife_s_per_mp"] * src_mp if backend == "rife" else 0.0
        return n_out * sr_cost, rife, 0.0
    # upscale_first: flow on source frames, then warp the enhanced frames at full size
    out_mp = sr.get("out_width", src_w) * sr.get("out_height", src_h) / 1e6
    per_new = r["rife_s_per_mp"] * src_mp + r["warp_s_per_output_mp"] * out_mp
    return n_src * sr_cost, n_new * per_new, r["rife_gb_per_mp"] * src_mp + r["warp_gb_per_output_mp"] * out_mp


def _choose_order(backend, sr, src_w, src_h, n_src, n_out) -> tuple[str, str]:
    """Pick the faster order that fits in memory, using the measured cost model."""
    if backend != "rife":
        return "interpolate_first", "FFmpeg interpolation is cheapest at source resolution"
    parallel = sr["runtime"] == "coreml"  # SR on the Neural Engine overlaps RIFE on the GPU

    def wall(a, b):
        return max(a, b) if parallel else a + b

    sa, ia, ga = _stage_seconds(backend, sr, src_w, src_h, n_src, n_out, "interpolate_first")
    sb, ib, gb = _stage_seconds(backend, sr, src_w, src_h, n_src, n_out, "upscale_first")
    ta, tb = wall(sa, ia), wall(sb, ib)
    budget = 0.35 * perf.ram_gb()
    if tb < ta * 0.9 and gb <= budget:
        return "upscale_first", f"estimated {tb / 60:.0f} min vs {ta / 60:.0f} min, RIFE needs ~{gb:.1f} GB"
    if tb < ta * 0.9:
        return "interpolate_first", (f"upscaling first would be faster ({tb / 60:.0f} vs {ta / 60:.0f} min) but "
                                     f"RIFE at full size needs ~{gb:.1f} GB, over the {budget:.1f} GB memory budget")
    return "interpolate_first", f"estimated {ta / 60:.0f} min vs {tb / 60:.0f} min upscaling first"


def _estimate(streaming, sr, interp, src_w, src_h, tw, th, n_src, n_out, n_stream) -> dict:
    r = perf.rates()
    out_mp = tw * th / 1e6
    stages: dict[str, float] = {}
    if not streaming:
        stages["ffmpeg"] = n_out * out_mp * r["ffmpeg_s_per_output_mp"]
        wall = stages["ffmpeg"]
    else:
        backend = interp["backend"] if interp["enabled"] else None
        if interp["enabled"]:
            s_sr, s_int, _ = _stage_seconds(backend or "", sr, src_w, src_h, n_src, n_out, interp["order"])
        else:  # SR runs once per frame leaving the AI stage
            s_sr, s_int, _ = _stage_seconds("", sr, src_w, src_h, n_stream, n_stream, "interpolate_first")
        stages["super_resolution"] = s_sr
        stages["interpolation"] = s_int
        stages["encode"] = n_out * out_mp * r["encode_s_per_output_mp"]
        parallel = sr.get("runtime") == "coreml"
        wall = (max(s_sr, s_int) if parallel else s_sr + s_int) + stages["encode"]
    total = wall + r["fixed_overhead_s"]
    return {"seconds": round(total), "stages": {k: round(v) for k, v in stages.items()},
            "sr_runtime": sr.get("runtime"), "frames_out": n_out,
            "basis": perf.basis()}


def _minterpolate(rate: Fraction) -> str:
    return (f"minterpolate=fps={rate.numerator}/{rate.denominator}:mi_mode=mci:mc_mode=aobmc"
            ":me_mode=bidir:vsbmc=1:scd=fdiff")


def _choose_encoder(codec: str, ep, w: int, h: int) -> dict:
    ten = ep.ten_bit and codec == "hevc"
    if codec == "hevc":
        if ep.hardware and has_encoder("hevc_videotoolbox"):
            return {"name": "hevc_videotoolbox", "codec": "hevc", "ten_bit": ten, "hardware": True,
                    "args": ["-c:v", "hevc_videotoolbox", "-q:v", str(ep.quality), "-allow_sw", "1",
                             "-profile:v", "main10" if ten else "main", "-tag:v", "hvc1"],
                    "pix_fmt": "p010le" if ten else "yuv420p",
                    "why": f"Hardware HEVC (VideoToolbox, constant quality {ep.quality}"
                           f"{', 10-bit' if ten else ''})."}
        return {"name": "libx265", "codec": "hevc", "ten_bit": ten, "hardware": False,
                "args": ["-c:v", "libx265", "-crf", str(ep.crf), "-preset", "medium", "-tag:v", "hvc1"],
                "pix_fmt": "yuv420p10le" if ten else "yuv420p",
                "why": f"Software HEVC (libx265, CRF {ep.crf})."}
    # VideoToolbox H.264 cannot open sessions above 4096 px wide on Apple Silicon
    if ep.hardware and has_encoder("h264_videotoolbox") and max(w, h) <= 4096:
        return {"name": "h264_videotoolbox", "codec": "h264", "ten_bit": False, "hardware": True,
                "args": ["-c:v", "h264_videotoolbox", "-q:v", str(ep.quality), "-allow_sw", "1"],
                "pix_fmt": "yuv420p", "why": f"Hardware H.264 (VideoToolbox, quality {ep.quality})."}
    return {"name": "libx264", "codec": "h264", "ten_bit": False, "hardware": False,
            "args": ["-c:v", "libx264", "-crf", str(ep.crf), "-preset", "medium", "-level:v", "6.0"],
            "pix_fmt": "yuv420p",
            "why": f"Software H.264 (libx264, CRF {ep.crf})"
                   + ("; VideoToolbox H.264 is limited to 4096 px." if max(w, h) > 4096 else ".")}

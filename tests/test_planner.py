from dataclasses import replace

import pytest

from video_engine.analyzer import AudioStream, SourceProfile
from video_engine.config import get_preset
from video_engine.degradation import Degradation
from video_engine.planner import JobOptions, plan_job, snap_rate


def profile(w=1280, h=720, fps="30/1", **kw) -> SourceProfile:
    from fractions import Fraction
    r = Fraction(fps)
    base = dict(path="x.mp4", container="mp4", size_bytes=1, width=w, height=h, display_width=w, display_height=h,
                rotation=0, sample_aspect_ratio="1:1", display_aspect_ratio=w / h, codec="h264", profile=None,
                pixel_format="yuv420p", bit_depth=8, nominal_fps=float(r), nominal_fps_fraction=fps,
                avg_fps=float(r), is_vfr=False, timestamp_jitter=0.0, duration_seconds=10.0, frame_count=None,
                video_bitrate=8_000_000, bits_per_pixel=0.2, color_primaries="bt709", color_transfer="bt709",
                color_space="bt709", color_range="tv", is_hdr=False, field_order="progressive",
                is_interlaced=False, start_time=0.0,
                audio=[AudioStream(1, "aac", 2, 48000, 128000, 10.0, 0.0)])
    base.update(kw)
    return SourceProfile(**base)


def deg(**kw) -> Degradation:
    d = Degradation(noise_sigma=1.0, blockiness=1.0, blur=0.3, blur_h=0.3, blur_v=0.3, motion=3.0,
                    duplicate_ratio=0.0, samples=6, labels=["clean"])
    return replace(d, **kw)


def plan(p=None, d=None, preset="balanced", **opts):
    return plan_job(p or profile(), d or deg(), JobOptions(preset=preset, **opts), get_preset(preset))


def names(pl):
    return [o.name for o in pl.operations]


def test_snap_rate_keeps_ntsc_distinct():
    assert float(snap_rate(30.0)) == 30
    assert abs(float(snap_rate(29.97)) - 30000 / 1001) < 1e-9
    assert float(snap_rate(59.94)) == pytest.approx(60000 / 1001)


def test_720p30_interpolates_and_upscales():
    pl = plan()
    assert (pl.target_width, pl.target_height) == (5312, 2988)
    assert pl.fps_action == "interpolate" and pl.interpolation["backend"] in ("rife", "ffmpeg")
    assert pl.expected_frames == 600
    assert "frame_interpolation" in names(pl)
    if pl.sr["enabled"]:  # models installed
        assert (pl.sr["out_width"], pl.sr["out_height"]) == (5120, 2880)
        assert "scale=5312:2988:flags=lanczos" in pl.encode_filters


def test_60fps_source_is_preserved():
    pl = plan(profile(fps="60/1"))
    assert pl.fps_action == "preserve"
    assert not pl.interpolation["enabled"]
    assert not any(f.startswith("fps=") for f in pl.decode_filters)


def test_120fps_decimated_in_5k3_mode_but_kept_native():
    assert plan(profile(fps="120/1")).fps_action == "decimate"
    native = plan(profile(fps="120/1"), output_policy="native")
    assert native.fps_action == "preserve" and native.target_fps == "120"


def test_fast_preset_repeats_frames_without_ai():
    pl = plan(preset="fast")
    assert pl.fps_action == "repeat"
    assert not pl.streaming and not pl.sr["enabled"]
    assert "fps=60/1" in pl.decode_filters or "fps=60" in ",".join(pl.decode_filters)


def test_interpolation_off_override():
    assert plan(interpolation="off").fps_action == "repeat"


def test_clean_5k3_source_is_not_upscaled():
    pl = plan(profile(5312, 2988, fps="60/1"))
    assert not pl.sr["enabled"]
    assert pl.encode_filters == []
    assert next(o for o in pl.operations if o.name == "resize").backend == "none"


def test_4k_uses_lanczos_below_ai_threshold():
    pl = plan(profile(3840, 2160, fps="60/1"))
    assert not pl.sr["enabled"] and pl.sr["backend"] == "ffmpeg"


def test_portrait_output_is_portrait():
    pl = plan(profile(720, 1280))
    assert (pl.target_width, pl.target_height) == (2988, 5312)


def test_4_3_source_is_padded_not_stretched():
    pl = plan(profile(640, 480))
    assert (pl.content_width, pl.content_height) == (3984, 2988)
    assert any(f.startswith("pad=5312:2988") for f in pl.encode_filters)


def test_crop_mode_fills_frame():
    pl = plan(profile(640, 480), aspect_mode="crop")
    assert any(f.startswith("crop=5312:2988") for f in pl.encode_filters)


def test_noise_and_compression_trigger_restoration():
    d = deg(noise_sigma=8.0, blockiness=1.4, labels=["noisy", "compressed"])
    pl = plan(d=d)
    assert any(f.startswith("hqdn3d") for f in pl.decode_filters)
    assert any(f.startswith("deblock") for f in pl.decode_filters)


def test_restoration_off_never_denoises():
    d = deg(noise_sigma=12.0, labels=["noisy"])
    from video_engine.config import get_preset as gp
    preset = gp("balanced").model_copy(deep=True)
    preset.restoration.enabled = "off"
    pl = plan_job(profile(), d, JobOptions(), preset)
    assert not any(f.startswith(("hqdn3d", "nlmeans")) for f in pl.decode_filters)


def test_motion_blur_is_reported_honestly():
    pl = plan(d=deg(blur=0.6, blur_h=0.6, blur_v=0.45, motion=8, labels=["motion_blur"]))
    deblur = next(o for o in pl.operations if o.name == "deblur")
    assert deblur.backend == "none"
    assert any("cannot be recovered" in w for w in pl.warnings)


def test_vfr_gets_cfr_timeline():
    pl = plan(profile(fps="30/1", is_vfr=True, avg_fps=29.5, timestamp_jitter=0.2))
    assert pl.decode_filters[0].startswith("fps=")


def test_interlaced_is_deinterlaced_first():
    pl = plan(profile(field_order="tt", is_interlaced=True))
    assert pl.decode_filters[0].startswith("bwdif")


def test_pcm_audio_is_transcoded_and_warned():
    p = profile(audio=[AudioStream(1, "pcm_s16le", 2, 48000, None, 10.0, 0.0)])
    pl = plan(p)
    assert pl.audio["action"] == "aac"
    assert any("AAC" in w for w in pl.warnings)


def test_trim_window():
    pl = plan(max_seconds=2.5, start_seconds=1)
    assert pl.duration_seconds == 2.5 and pl.start_seconds == 1
    assert pl.expected_frames == 150


def test_h264_above_4k_uses_software():
    pl = plan(codec="h264")
    assert pl.encoder["name"] == "libx264"


def test_plan_roundtrip():
    pl = plan()
    assert type(pl).from_dict(pl.to_dict()) == pl


def test_repeat_mode_upscales_each_source_frame_once():
    pl = plan(interpolation="off")
    if not pl.streaming:
        pytest.skip("AI models not installed")
    assert not any(f.startswith("fps=") for f in pl.decode_filters)
    assert pl.encode_filters[0] == "fps=60/1"
    assert pl.stream_fps == "30" and pl.expected_frames == 600


def test_auto_order_prefers_upscale_first_with_low_memory_warp():
    pl = plan(profile(1080, 1920))
    if not pl.sr["enabled"]:
        pytest.skip("AI models not installed")
    assert pl.interpolation["order"] == "upscale_first"
    reason = next(o.reason for o in pl.operations if o.name == "frame_interpolation")
    assert "after super-resolution" in reason and "GB" in reason


def test_estimate_present_and_scales_with_duration():
    a = plan(max_seconds=5).estimate
    b = plan(max_seconds=10).estimate
    assert a["seconds"] > 0 and b["seconds"] > a["seconds"]
    assert a["frames_out"] == 300 and "basis" in a

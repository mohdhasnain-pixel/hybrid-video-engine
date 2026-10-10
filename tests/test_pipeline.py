"""End-to-end runs on tiny clips (small targets keep them fast)."""

import pytest

from video_engine import models_registry
from video_engine.analyzer import analyze
from video_engine.engine import enhance_video
from video_engine.planner import JobOptions

ai = pytest.mark.skipif(not (models_registry.is_installed("realesr-general-x4v3")
                             and models_registry.is_installed("rife-v4.26")),
                        reason="run `video-engine download-models` first")


def run(src, tmp_path, **opts):
    out = tmp_path / "out.mp4"
    result = enhance_video(src, out, JobOptions(**opts), work_dir=tmp_path / "work")
    return out, result


def test_ffmpeg_only_5k3(clips, tmp_path):
    out, r = run(clips["small30"], tmp_path, preset="fast")
    assert r["validation"]["passed"]
    p = analyze(out)
    assert (p.width, p.height, p.nominal_fps) == (5312, 2988, 60)
    assert r["metrics"]["mode"] == "ffmpeg_only"
    assert not out.with_name("out.partial.mp4").exists()


@ai
def test_ai_sr_and_rife(clips, tmp_path):
    out, r = run(clips["small30"], tmp_path, preset="balanced", target_width=640, target_height=360)
    assert r["validation"]["passed"], r["validation"]
    m = r["metrics"]
    assert m["mode"] == "streaming"
    assert m["frames_decoded"] == 30 and m["frames_written"] == 60
    assert m["interpolation"]["generated"] == 29
    assert set(m["models"]) == {"super_resolution", "interpolation"}
    p = analyze(out)
    assert (p.width, p.height, p.nominal_fps) == (640, 360, 60)


@ai
def test_60fps_source_is_not_interpolated(clips, tmp_path):
    out, r = run(clips["small60"], tmp_path, preset="balanced", target_width=640, target_height=360)
    assert r["plan"]["fps_action"] == "preserve"
    assert "interpolation" not in r["metrics"]["models"]
    assert r["metrics"]["frames_written"] == 60


@ai
def test_upscale_first_order(clips, tmp_path):
    out, r = run(clips["small30"], tmp_path, preset="restoration", target_width=640, target_height=360)
    assert r["validation"]["passed"]
    assert r["plan"]["interpolation"]["order"] == "upscale_first"


def test_vfr_keeps_duration(clips, tmp_path):
    src = analyze(clips["vfr"])
    out, r = run(clips["vfr"], tmp_path, preset="fast", output_policy="native")
    assert r["validation"]["passed"]
    assert abs(analyze(out).duration_seconds - src.duration_seconds) < 0.1


def test_portrait_rotation(clips, tmp_path):
    out, r = run(clips["portrait"], tmp_path, preset="fast", target_width=640, target_height=360)
    p = analyze(out)
    assert (p.display_width, p.display_height) == (360, 640)


def test_anamorphic_square_pixels(clips, tmp_path):
    out, r = run(clips["anamorphic"], tmp_path, preset="fast", output_policy="native")
    p = analyze(out)
    assert p.sample_aspect_ratio == "1:1" and p.height == 480 and abs(p.width - 853) <= 1


def test_cancel(clips, tmp_path):
    import threading

    from video_engine.pipeline import Cancelled

    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        enhance_video(clips["720p30"], tmp_path / "x.mp4", JobOptions(preset="fast"), tmp_path / "w", cancel=ev)
    assert not (tmp_path / "x.mp4").exists()


@ai
def test_sr_downscale_non_integer_ratio():
    """1080x1920 portrait -> 4x = 4320x7680 -> 2988x5312 (regression: MPS area-pool crash)."""
    import numpy as np

    from video_engine.super_resolution import SuperResolver

    sr = SuperResolver("realesr-general-x4v3", (299, 531), 0.5)
    out = sr(np.random.randint(0, 255, (192, 108, 3), np.uint8))  # 4x = 432x768, ratio 1.446
    assert out.shape == (531, 299, 3)


@ai
def test_flow_transfer_matches_shape_and_is_a_blend():
    import numpy as np

    from video_engine.interpolation import RifeInterpolator

    ri = RifeInterpolator()
    a = np.zeros((90, 160, 3), np.uint8)
    b = np.full((90, 160, 3), 100, np.uint8)
    big_a, big_b = np.zeros((360, 640, 3), np.uint8), np.full((360, 640, 3), 100, np.uint8)
    out = ri.interpolate_full(a, b, big_a, big_b, 0.5)
    assert out.shape == (360, 640, 3)
    assert 0 < out.mean() < 100  # a blend of the two enhanced frames, nothing outside their range


@pytest.mark.skipif(not __import__("video_engine.coreml_sr", fromlist=["x"]).available(), reason="Core ML unavailable")
@ai
def test_coreml_matches_mps(tmp_path, monkeypatch):
    import numpy as np

    from video_engine import config
    from video_engine.coreml_sr import CoreMLSuperResolver
    from video_engine.super_resolution import SuperResolver

    f = np.clip(np.cumsum(np.random.default_rng(0).normal(size=(90, 160, 3)), axis=1) * 4 + 128, 0, 255).astype(np.uint8)
    a = CoreMLSuperResolver("realesr-general-x4v3", (160, 90), (640, 360), 0.5)(f)
    b = SuperResolver("realesr-general-x4v3", (640, 360), 0.5, half=False)(f)
    assert a.shape == b.shape == (360, 640, 3)
    # compare the interior: the runtimes pad frame borders differently (edge vs zero)
    mse = np.mean((a[64:-64, 64:-64].astype(float) - b[64:-64, 64:-64].astype(float)) ** 2)
    assert 10 * np.log10(255 ** 2 / max(mse, 1e-9)) > 40  # visually identical


@ai
def test_upscale_first_with_flow_transfer_end_to_end(clips, tmp_path):
    out, r = run(clips["small30"], tmp_path, preset="restoration", target_width=640, target_height=360)
    assert r["validation"]["passed"]
    m = r["metrics"]
    assert m["frames_decoded"] == 30 and m["frames_written"] == 60
    assert "warp of enhanced" in m["models"]["interpolation"]["mode"]


@ai
def test_repeat_mode_end_to_end(clips, tmp_path):
    out, r = run(clips["small30"], tmp_path, preset="balanced", interpolation="off", target_width=640, target_height=360)
    assert r["validation"]["passed"]
    assert r["metrics"]["frames_written"] == 30  # SR ran once per source frame; encoder repeated to 60
    assert analyze(out).nominal_fps == 60

from video_engine import degradation
from video_engine.analyzer import analyze


def test_basic_profile(clips):
    p = analyze(clips["720p30"])
    assert (p.width, p.height) == (1280, 720)
    assert p.nominal_fps == 30 and not p.is_vfr
    assert p.codec == "h264" and p.has_audio and p.audio[0].codec == "aac"
    assert abs(p.duration_seconds - 2.0) < 0.1
    assert p.bit_depth == 8 and not p.is_hdr and not p.is_interlaced


def test_vfr_detected(clips):
    p = analyze(clips["vfr"])
    assert p.is_vfr
    assert p.timestamp_jitter > 0.05


def test_rotation_gives_portrait_display(clips):
    p = analyze(clips["portrait"])
    assert p.rotation in (90, 270)
    assert (p.display_width, p.display_height) == (180, 320)


def test_anamorphic_display_size(clips):
    p = analyze(clips["anamorphic"])
    assert p.sample_aspect_ratio == "32:27"
    assert p.display_width == 853 and p.display_height == 480


def test_noise_is_detected(clips):
    noisy = degradation.estimate(analyze(clips["noisy"]))
    clean = degradation.estimate(analyze(clips["720p30"]))
    assert noisy.noise_sigma > clean.noise_sigma + 3
    assert "noisy" in noisy.labels


def test_profile_roundtrip(clips):
    p = analyze(clips["720p30"])
    assert type(p).from_dict(p.to_dict()) == p

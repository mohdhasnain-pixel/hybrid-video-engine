from video_engine.analyzer import analyze
from video_engine.engine import enhance_video
from video_engine.planner import JobOptions


def test_aac_copied_and_in_sync(clips, tmp_path):
    out = tmp_path / "o.mp4"
    r = enhance_video(clips["720p30"], out, JobOptions(preset="fast"), tmp_path / "w")
    p = analyze(out)
    assert p.audio and p.audio[0].codec == "aac"
    assert abs(p.audio[0].duration_seconds - p.duration_seconds) < 0.1
    checks = {c["name"]: c for c in r["validation"]["checks"]}
    assert checks["audio_0_codec"]["passed"] and checks["audio_0_duration"]["passed"]


def test_pcm_transcoded_to_aac(clips, tmp_path):
    out = tmp_path / "o.mp4"
    r = enhance_video(clips["pcm"], out, JobOptions(preset="fast"), tmp_path / "w")
    assert r["plan"]["audio"]["action"] == "aac"
    assert analyze(out).audio[0].codec == "aac"


def test_trimmed_window_audio_matches(clips, tmp_path):
    out = tmp_path / "o.mp4"
    enhance_video(clips["720p30"], out, JobOptions(preset="fast", start_seconds=0.5, max_seconds=1), tmp_path / "w")
    p = analyze(out)
    assert abs(p.duration_seconds - 1.0) < 0.06
    assert abs(p.audio[0].duration_seconds - 1.0) < 0.1


def test_audio_can_be_dropped(clips, tmp_path):
    out = tmp_path / "o.mp4"
    enhance_video(clips["720p30"], out, JobOptions(preset="fast", preserve_audio=False), tmp_path / "w")
    assert not analyze(out).has_audio

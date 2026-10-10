import subprocess
from pathlib import Path

import pytest

from video_engine import config, storage


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Every test gets its own storage/work/jobs dirs; models are shared."""
    monkeypatch.setenv("VE_LOCAL_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("VE_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("VE_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("VE_STORAGE_BACKEND", "local")
    monkeypatch.setenv("VE_LEARN_PERF", "false")  # test clips must not skew real time estimates
    config.get_settings.cache_clear()
    storage.get_storage.cache_clear()
    yield
    config.get_settings.cache_clear()
    storage.get_storage.cache_clear()


@pytest.fixture(scope="session")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    c = {
        "720p30": d / "720p30.mp4",
        "small30": d / "small30.mp4",
        "small60": d / "small60.mp4",
        "vfr": d / "vfr.mkv",
        "portrait": d / "portrait.mp4",
        "noisy": d / "noisy.mp4",
        "pcm": d / "pcm.mov",
        "anamorphic": d / "anamorphic.mp4",
    }
    sine = ["-f", "lavfi", "-i", "sine=f=440:r=48000"]
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30", *sine, "-t", "2", "-c:v", "libx264", "-crf", "23",
           "-c:a", "aac", "-shortest", str(c["720p30"]))
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x90:r=30", *sine, "-t", "1", "-c:v", "libx264", "-c:a", "aac",
           "-shortest", str(c["small30"]))
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x90:r=60", "-t", "1", "-c:v", "libx264", str(c["small60"]))
    # irregular timestamps: random per-frame jitter, stored in MKV (keeps pts as-is)
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-t", "3",
           "-vf", "setpts='N/30/TB+if(mod(N,3),0.012,-0.008)/TB'", "-fps_mode", "passthrough",
           "-c:v", "libx264", str(c["vfr"]))
    # portrait via rotation metadata (display matrix)
    tmp = d / "landscape.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-t", "1", "-c:v", "libx264", str(tmp))
    ffmpeg("-display_rotation", "90", "-i", str(tmp), "-c", "copy", str(c["portrait"]))
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=640x360:r=30", "-t", "1",
           "-vf", "noise=alls=40:allf=t", "-c:v", "libx264", "-crf", "12", str(c["noisy"]))
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x90:r=25", *sine, "-t", "1", "-c:v", "libx264",
           "-c:a", "pcm_s16le", "-shortest", str(c["pcm"]))
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=720x480:r=30", "-t", "1", "-vf", "setsar=32/27",
           "-c:v", "libx264", str(c["anamorphic"]))
    return c

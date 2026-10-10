"""FastAPI backend + single-page HTML frontend.

    uv run video-engine serve            # http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import ValidationError

from . import models_registry
from .ai import select_device
from .config import PROJECT_ROOT, get_settings, load_presets
from .jobs import JobManager, job_view
from .media import ffmpeg_capabilities
from .planner import JobOptions
from .storage import LocalStorage, get_storage

log = logging.getLogger(__name__)
WEB_DIR = PROJECT_ROOT / "web"
manager: JobManager | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global manager
    s = get_settings()
    s.work_dir.mkdir(parents=True, exist_ok=True)
    get_storage()  # fail fast on bad storage configuration (e.g. s3 without a bucket)
    manager = JobManager()
    log.info("Storage backend: %s", get_storage().describe())
    yield


app = FastAPI(title="Hybrid Video Enhancement Engine", lifespan=lifespan)


def _manager() -> JobManager:
    assert manager is not None
    return manager


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text())


@app.get("/api/config")
def config() -> dict:
    s = get_settings()
    storage = get_storage()
    return {
        "storage": storage.describe(),
        "storage_flag": "VE_STORAGE_BACKEND",
        "presets": [{"name": p.name, "label": p.label, "description": p.description}
                    for p in load_presets().values()],
        "models": models_registry.status(),
        "device": str(select_device()),
        "ffmpeg": ffmpeg_capabilities(),
        "max_upload_mb": s.max_upload_mb,
        "defaults": JobOptions().model_dump(),
    }


@app.post("/api/jobs", status_code=201)
async def create_job(
    file: Annotated[UploadFile, File()],
    preset: Annotated[str, Form()] = "balanced",
    output_policy: Annotated[str, Form()] = "5k3",
    interpolation: Annotated[str, Form()] = "auto",
    aspect_mode: Annotated[str, Form()] = "fit",
    codec: Annotated[str, Form()] = "preset",
    max_seconds: Annotated[float | None, Form()] = None,
    start_seconds: Annotated[float, Form()] = 0.0,
    preserve_audio: Annotated[bool, Form()] = True,
) -> dict:
    s = get_settings()
    if preset not in load_presets():
        raise HTTPException(422, f"Unknown preset {preset!r}")
    try:
        opts = JobOptions(preset=preset, output_policy=output_policy, interpolation=interpolation,
                          aspect_mode=aspect_mode, codec=codec, max_seconds=max_seconds or None,
                          start_seconds=start_seconds, preserve_audio=preserve_audio)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False)) from exc

    limit = s.max_upload_mb * 1024 * 1024
    s.work_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=s.work_dir, suffix=Path(file.filename or "").suffix)
    tmp = Path(tmp_name)
    written = 0
    try:
        with open(fd, "wb") as out:
            while chunk := await file.read(8 << 20):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"Upload exceeds {s.max_upload_mb} MB")
                out.write(chunk)
        if written == 0:
            raise HTTPException(422, "Empty upload")
        job = _manager().submit(tmp, file.filename or "input.mp4", opts)
    finally:
        tmp.unlink(missing_ok=True)
    return job_view(job)


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [job_view(j) for j in _manager().store.list()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _manager().store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job_view(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if not _manager().cancel(job_id):
        raise HTTPException(409, "Job is not running")
    return {"cancelled": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    if not _manager().delete(job_id):
        raise HTTPException(404, "Job not found")
    return {"deleted": True}


@app.get("/files/{key:path}")
def serve_file(key: str, download: Annotated[str | None, Query()] = None):
    """Local-storage files (range requests supported, so videos can seek)."""
    storage = get_storage("local")
    assert isinstance(storage, LocalStorage)
    try:
        path = storage.path(key)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=download, content_disposition_type="attachment" if download else "inline")


@app.get("/api/estimate")
def estimate(
    width: int, height: int, duration: float, fps: float = 30.0, preset: str = "balanced",
    output_policy: str = "5k3", interpolation: str = "auto", max_seconds: float | None = None,
    start_seconds: float = 0.0,
) -> dict:
    """Processing-time estimate before upload (browser sends the file's size and duration).

    Browsers cannot read a file's frame rate, so ``fps`` defaults to 30. With
    interpolation the number of upscaled frames barely depends on it.
    """
    from fractions import Fraction

    from .analyzer import SourceProfile
    from .config import get_preset
    from .degradation import Degradation
    from .planner import plan_job

    if width <= 0 or height <= 0 or duration <= 0:
        raise HTTPException(422, "width, height and duration must be positive")
    r = Fraction(fps).limit_denominator(1001)
    prof = SourceProfile(
        path="", container="", size_bytes=0, width=width, height=height, display_width=width,
        display_height=height, rotation=0, sample_aspect_ratio="1:1", display_aspect_ratio=width / height,
        codec="h264", profile=None, pixel_format="yuv420p", bit_depth=8, nominal_fps=float(r),
        nominal_fps_fraction=f"{r.numerator}/{r.denominator}", avg_fps=float(r), is_vfr=False,
        timestamp_jitter=0.0, duration_seconds=duration, frame_count=None, video_bitrate=None,
        bits_per_pixel=None, color_primaries=None, color_transfer=None, color_space=None, color_range=None,
        is_hdr=False, field_order=None, is_interlaced=False, start_time=0.0, audio=[])
    deg = Degradation(noise_sigma=0, blockiness=1, blur=0.3, blur_h=0.3, blur_v=0.3, motion=0,
                      duplicate_ratio=0, samples=0, labels=["clean"])
    try:
        opts = JobOptions(preset=preset, output_policy=output_policy, interpolation=interpolation,
                          max_seconds=max_seconds or None, start_seconds=start_seconds)
        pl = plan_job(prof, deg, opts, get_preset(preset))
    except (ValueError, KeyError, ValidationError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {**pl.estimate, "order": pl.interpolation.get("order") if pl.interpolation.get("enabled") else None,
            "fps_action": pl.fps_action, "target": f"{pl.target_width}x{pl.target_height}",
            "duration_seconds": pl.duration_seconds, "assumed_fps": fps}


@app.get("/api/storage")
def storage_info() -> dict:
    return get_storage().describe() | {"work_dir_free_bytes": shutil.disk_usage(get_settings().work_dir).free}

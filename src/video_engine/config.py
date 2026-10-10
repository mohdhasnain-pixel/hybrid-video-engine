"""Runtime settings and quality presets.

Settings come from environment variables (prefix ``VE_``) or a ``.env`` file.
The most important one is the storage feature flag::

    VE_STORAGE_BACKEND=local   # videos live on this computer's disk (default)
    VE_STORAGE_BACKEND=s3      # videos live in an S3 bucket

Presets are YAML files in ``configs/``; they describe *policies* (which
capabilities may run and how aggressively), not hard-coded models.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VE_", env_file=".env", extra="ignore")

    # --- Storage feature flag -------------------------------------------------
    storage_backend: Literal["local", "s3"] = "local"
    local_storage_dir: Path = PROJECT_ROOT / "data" / "storage"

    s3_bucket: str = ""
    s3_prefix: str = "video-engine/"
    s3_region: str | None = None
    s3_endpoint_url: str | None = None  # e.g. MinIO / LocalStack
    s3_presign_expiry_seconds: int = 3600

    # --- Processing -----------------------------------------------------------
    # Local scratch space used while a job runs. Always on this machine, even in
    # S3 mode, because FFmpeg and the models need real files; cleaned afterwards.
    work_dir: Path = PROJECT_ROOT / "data" / "work"
    jobs_dir: Path = PROJECT_ROOT / "data" / "jobs"
    models_dir: Path = PROJECT_ROOT / "models"
    configs_dir: Path = PROJECT_ROOT / "configs"
    device: Literal["auto", "mps", "cpu", "cuda"] = "auto"
    max_upload_mb: int = 4096
    keep_work_files: bool = False
    learn_perf: bool = True  # refine time estimates from finished jobs

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class RestorationPreset(BaseModel):
    enabled: Literal["auto", "on", "off"] = "auto"
    # ffmpeg: hqdn3d / nlmeans / deblock at source resolution (cheap, faithful)
    denoise_backend: Literal["ffmpeg"] = "ffmpeg"
    max_strength: Literal["light", "medium", "strong"] = "medium"
    deblock: bool = True


class SuperResolutionPreset(BaseModel):
    backend: Literal["ai", "ffmpeg"] = "ai"
    model: str = "realesr-general-x4v3"
    # Real-ESRGAN DNI: 0 keeps grain/noise, 1 removes noise strongly.
    # "auto" lets the planner pick from the degradation estimate.
    denoise_strength: float | Literal["auto"] = "auto"
    min_factor_for_ai: float = 1.5  # below this, conventional Lanczos is used
    tile: int = 0  # 0 = automatic
    half_precision: bool = True
    final_sharpen: Literal["none", "cas"] = "none"
    # coreml = Apple Neural Engine (fastest on Apple Silicon), mps = GPU
    runtime: Literal["auto", "coreml", "mps"] = "auto"


class InterpolationPreset(BaseModel):
    mode: Literal["auto", "on", "off"] = "auto"
    backend: Literal["rife", "ffmpeg"] = "rife"
    # auto = pick the faster order that fits in memory (perf.py cost model)
    order: Literal["auto", "interpolate_first", "upscale_first"] = "auto"
    rife_scale: float = 1.0  # RIFE flow scale; 0.5 for >=4K inputs
    scene_change_threshold: float = 0.12


class EncoderPreset(BaseModel):
    codec: Literal["hevc", "h264"] = "hevc"
    hardware: bool = True
    quality: int = 60  # VideoToolbox constant quality (1-100)
    crf: int = 20  # software fallback
    ten_bit: bool = False


class Preset(BaseModel):
    name: str
    label: str
    order: int = 100  # position in the UI
    description: str = ""
    restoration: RestorationPreset = Field(default_factory=RestorationPreset)
    super_resolution: SuperResolutionPreset = Field(default_factory=SuperResolutionPreset)
    interpolation: InterpolationPreset = Field(default_factory=InterpolationPreset)
    encoder: EncoderPreset = Field(default_factory=EncoderPreset)
    temporal_checks: bool = False
    hallucination_warning: bool = False
    queue_size: int = 6


def load_presets(configs_dir: Path | None = None) -> dict[str, Preset]:
    configs_dir = configs_dir or get_settings().configs_dir
    presets: dict[str, Preset] = {}
    for path in sorted(configs_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        data.setdefault("name", path.stem)
        preset = Preset.model_validate(data)
        presets[preset.name] = preset
    if not presets:
        raise RuntimeError(f"No presets found in {configs_dir}")
    return dict(sorted(presets.items(), key=lambda kv: (kv[1].order, kv[0])))


def get_preset(name: str, configs_dir: Path | None = None) -> Preset:
    presets = load_presets(configs_dir)
    if name not in presets:
        raise KeyError(f"Unknown preset {name!r}; available: {', '.join(presets)}")
    return presets[name]

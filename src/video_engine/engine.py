"""Phase 10 — the Python processing API.

``enhance_video`` orchestrates the independent components; it holds no
processing logic of its own. The job manager (web) and the CLI both use it.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import degradation, models_registry, perf
from .analyzer import analyze
from .config import Preset, get_preset
from .pipeline import Pipeline
from .planner import JobOptions, plan_job
from .validation import validate

StateFn = Callable[[str, dict | None], None]


class EngineError(RuntimeError):
    def __init__(self, stage: str, message: str, detail: str = ""):
        super().__init__(message)
        self.stage, self.detail = stage, detail


def estimate_output_bytes(plan) -> int:
    """Rough upper bound used for the disk-space preflight (≈ 0.12 bits/pixel)."""
    return int(plan.target_width * plan.target_height * float(plan.target_fps_fraction)
               * plan.duration_seconds * 0.12 / 8 * 1.5)


def enhance_video(
    input_path: Path,
    output_path: Path,
    options: JobOptions | None = None,
    work_dir: Path | None = None,
    on_state: StateFn | None = None,
    on_progress: Callable[[dict], None] | None = None,
    cancel: threading.Event | None = None,
    preset: Preset | None = None,
) -> dict:
    """Analyze, restore, upscale, optionally interpolate, encode and validate.

    Writes to ``<output>.partial.mp4`` and only renames to ``output_path`` once
    validation passes. Returns a structured job result.
    """
    options = options or JobOptions()
    on_state = on_state or (lambda _s, _d=None: None)
    input_path, output_path = Path(input_path), Path(output_path)
    work_dir = Path(work_dir or output_path.parent / f".{output_path.stem}.work")
    work_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"options": options.model_dump(), "timings": {}}
    t = time.perf_counter()

    on_state("ANALYZING", None)
    profile = analyze(input_path)
    deg = degradation.estimate(profile)
    result["profile"], result["degradation"] = profile.to_dict(), deg.to_dict()
    result["timings"]["analyze"] = round(time.perf_counter() - t, 2)
    on_state("ANALYZING", {"profile": result["profile"], "degradation": result["degradation"]})

    on_state("PLANNING", None)
    preset = preset or get_preset(options.preset)
    plan = plan_job(profile, deg, options, preset)
    result["plan"] = plan.to_dict()
    result["preset"] = preset.model_dump()
    result["models_available"] = models_registry.status()
    on_state("PLANNING", {"plan": result["plan"]})

    need = estimate_output_bytes(plan)
    free = shutil.disk_usage(work_dir).free
    if free < need + 1_000_000_000:
        raise EngineError("PLANNING", f"Not enough disk space: ~{need / 1e9:.1f} GB needed for output, "
                                      f"{free / 1e9:.1f} GB free in {work_dir}")

    partial = output_path.with_name(output_path.stem + ".partial.mp4")
    on_state("PROCESSING", None)
    t = time.perf_counter()
    pipe = Pipeline(input_path, plan, partial, work_dir, on_progress, cancel)
    result["metrics"] = pipe.run()
    result["timings"]["process"] = round(time.perf_counter() - t, 2)

    on_state("VALIDATING", None)
    report = validate(partial, plan, profile)
    result["validation"] = report
    if not report["passed"]:
        failed = [c for c in report["checks"] if c["critical"] and not c["passed"]]
        raise EngineError("VALIDATING", "Output failed validation: " + "; ".join(
            f"{c['name']} expected {c['expected']}, got {c['actual']}" for c in failed))
    partial.replace(output_path)
    try:
        perf.learn(result["plan"], result["metrics"])  # sharpen future time estimates
    except Exception:  # noqa: BLE001 - estimates are advisory
        pass
    return result

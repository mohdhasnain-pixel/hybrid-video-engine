"""Processing-time cost model (used for the time estimate and to pick the stage order).

Rates are seconds per megapixel of *model input*. Defaults were measured on an
Apple M4 (16 GB, on AC power); ``video-engine calibrate`` re-measures them on the
current machine and stores them in ``models/perf.json``.
"""

from __future__ import annotations

import json
import os
import time

from .config import get_settings

DEFAULTS = {
    # Neural Engine incl. tiling + GPU resize. Peak is ~0.45; sustained inside the full
    # pipeline it was 0.9-1.2 on an M4 at low battery, so start in between and learn.
    "sr_coreml_s_per_mp": 0.75,
    "sr_mps_s_per_mp": 1.0,  # GPU; varies 0.6-2.5 under sustained load
    "sr_cpu_s_per_mp": 20.0,
    "rife_s_per_mp": 0.066,  # flow scale 1.0 (source-resolution frames)
    "rife_gb_per_mp": 0.55,  # GPU memory for RIFE per source MP
    "warp_s_per_output_mp": 0.006,  # flow upsample + warp of enhanced frames, incl. transfers
    "warp_gb_per_output_mp": 0.10,
    "ffmpeg_s_per_output_mp": 0.012,  # FFmpeg-only path (scale + encode)
    "encode_s_per_output_mp": 0.004,  # VideoToolbox encode in the AI path
    "fixed_overhead_s": 25.0,  # analysis, comparison assets, validation
}


def rates() -> dict:
    p = get_settings().models_dir / "perf.json"
    data = dict(DEFAULTS)
    if p.is_file():
        try:
            data.update(json.loads(p.read_text()).get("rates", {}))
        except (ValueError, OSError):
            pass
    return data


def _load() -> dict:
    p = get_settings().models_dir / "perf.json"
    try:
        return json.loads(p.read_text()) if p.is_file() else {}
    except (ValueError, OSError):
        return {}


def calibrated() -> bool:
    return bool(_load())


def basis() -> str:
    d = _load()
    if d.get("jobs_learned"):
        return f"learned from {d['jobs_learned']} job(s) on this Mac"
    if d.get("measured_at"):
        return "calibrated on this Mac"
    return "default Apple M4 measurements (improves after your first job)"


def learn(plan: dict, metrics: dict, alpha: float = 0.5) -> None:
    """Fold the real per-stage rates of a finished job into perf.json (moving average).

    Stage timings are measured inside each thread, so they already include the
    slow-down from stages competing for power, memory bandwidth and the GPU.
    """
    if not get_settings().learn_perf:
        return
    st = metrics.get("stage_seconds") or {}
    sr, interp = plan.get("sr") or {}, plan.get("interpolation") or {}
    src_mp = plan["source_width"] * plan["source_height"] / 1e6
    if src_mp < 0.3:  # tiny clips are dominated by per-frame overhead, not representative
        return
    observed: dict[str, float] = {}
    if sr.get("enabled") and st.get("super_resolution"):
        pairs = interp.get("enabled") and interp.get("order") == "upscale_first"
        sr_frames = metrics["frames_decoded"] if pairs or not interp.get("enabled") else metrics["frames_written"]
        if sr_frames >= 20:
            runtime = (metrics.get("models", {}).get("super_resolution", {}).get("device") or "")
            key = "sr_coreml_s_per_mp" if runtime.startswith("coreml") else "sr_mps_s_per_mp"
            observed[key] = st["super_resolution"] / (sr_frames * src_mp)
    gen = (metrics.get("interpolation") or {}).get("generated", 0)
    if st.get("interpolation") and gen >= 20:
        per = st["interpolation"] / gen
        if interp.get("order") == "upscale_first" and sr.get("enabled"):
            out_mp = sr["out_width"] * sr["out_height"] / 1e6
            r = rates()
            observed["warp_s_per_output_mp"] = max(per - r["rife_s_per_mp"] * src_mp, 0.0) / out_mp
        else:
            observed["rife_s_per_mp"] = per / src_mp
    if not observed:
        return
    d = _load()
    cur = {**DEFAULTS, **d.get("rates", {})}
    for k, v in observed.items():
        cur[k] = round(alpha * v + (1 - alpha) * cur[k], 4)
    d["rates"] = cur
    d["jobs_learned"] = d.get("jobs_learned", 0) + 1
    d["last_learned_at"] = time.strftime("%Y-%m-%d %H:%M")
    (get_settings().models_dir / "perf.json").write_text(json.dumps(d, indent=2))


def ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        return 16.0


def sr_rate(runtime: str) -> float:
    r = rates()
    return {"coreml": r["sr_coreml_s_per_mp"], "mps": r["sr_mps_s_per_mp"]}.get(runtime, r["sr_cpu_s_per_mp"])


def calibrate(log=print) -> dict:
    """Measure the real rates on this machine (≈1-2 minutes)."""
    import numpy as np

    from . import coreml_sr
    from .interpolation import RifeInterpolator
    from .super_resolution import SuperResolver

    def bench(fn, n=5):
        fn()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n

    w, h = 1280, 720
    mp = w * h / 1e6
    frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    out = dict(rates())
    sr = SuperResolver("realesr-general-x4v3", (w * 4, h * 4), 0.5)
    out["sr_mps_s_per_mp"] = round(bench(lambda: sr(frame)) / mp, 4)
    log(f"  SR on GPU (MPS): {out['sr_mps_s_per_mp']} s/MP")
    del sr
    if coreml_sr.available():
        cm = coreml_sr.CoreMLSuperResolver("realesr-general-x4v3", (w, h), (w * 4, h * 4), 0.5)
        out["sr_coreml_s_per_mp"] = round(bench(lambda: cm(frame)) / mp, 4)
        log(f"  SR on Neural Engine (Core ML): {out['sr_coreml_s_per_mp']} s/MP")
        del cm
    ri = RifeInterpolator(scale=1.0)
    b = np.roll(frame, 4, axis=1)
    out["rife_s_per_mp"] = round(bench(lambda: ri(frame.copy(), b.copy(), 0.5)) / mp, 4)
    log(f"  RIFE (scale 1.0): {out['rife_s_per_mp']} s/MP")
    big = (np.random.rand(h * 4, w * 4, 3) * 255).astype(np.uint8)
    big_b = np.roll(big, 16, axis=1)
    full = bench(lambda: ri.interpolate_full(frame.copy(), b.copy(), big, big_b, 0.5), 3)
    out["warp_s_per_output_mp"] = round(max(full - out["rife_s_per_mp"] * mp, 0.0) / (16 * mp), 4)
    log(f"  Full-size warp: {out['warp_s_per_output_mp']} s/MP")
    d = _load()
    d.update({"measured_at": time.strftime("%Y-%m-%d %H:%M"), "rates": out})
    (get_settings().models_dir / "perf.json").write_text(json.dumps(d, indent=2))
    return out

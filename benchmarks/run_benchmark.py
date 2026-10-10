"""Quality benchmark: Baseline vs AI-only vs Hybrid A vs Hybrid B.

    uv run python benchmarks/run_benchmark.py [--dataset benchmarks/dataset.yaml]

For each clean reference clip: degrade (downscale, 30 fps, noise, compression),
run every pipeline to the reference size/rate, then measure
  * PSNR / SSIM against the reference (ffmpeg psnr/ssim filters)
  * temporal error: mean |(out_t - out_t-1) - (ref_t - ref_t-1)| (flicker / bad motion)
  * wall time and throughput
Same output size, frame rate and encoder for every pipeline, so the comparison is fair.
Results: benchmarks/results/<timestamp>.json and .md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_engine.config import get_preset  # noqa: E402
from video_engine.engine import enhance_video  # noqa: E402
from video_engine.planner import JobOptions  # noqa: E402


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def pipelines():
    base = get_preset("balanced")

    def variant(name, **changes):
        p = base.model_copy(deep=True)
        p.name = name
        for path, value in changes.items():
            section, field = path.split("__")
            setattr(getattr(p, section), field, value)
        p.encoder.quality = 75  # identical, high encode quality for all pipelines
        return p

    return {
        "baseline_lanczos": variant("baseline", super_resolution__backend="ffmpeg", restoration__enabled="off",
                                    interpolation__mode="off", super_resolution__final_sharpen="none"),
        "ai_only": variant("ai_only", restoration__enabled="off", interpolation__mode="off"),
        "hybrid_a": variant("hybrid_a", interpolation__order="interpolate_first"),
        "hybrid_b": variant("hybrid_b", interpolation__order="upscale_first"),
    }


def frames(path: Path, w: int, h: int, step: int = 4) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={w // step}:{h // step}:flags=area,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    n = len(raw) // ((w // step) * (h // step))
    return np.frombuffer(raw[: n * (w // step) * (h // step)], np.uint8).reshape(n, h // step, w // step).astype(np.float32)


def quality(out: Path, ref: Path, w: int, h: int) -> dict:
    log = subprocess.run(["ffmpeg", "-i", str(out), "-i", str(ref), "-lavfi",
                          "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]psnr;[0:v]setpts=PTS-STARTPTS[c];"
                          "[1:v]setpts=PTS-STARTPTS[d];[c][d]ssim", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    psnr = re.search(r"PSNR .* average:([\d.inf]+)", log)
    ssim = re.search(r"SSIM .* All:([\d.]+)", log)
    a, b = frames(out, w, h), frames(ref, w, h)
    n = min(len(a), len(b))
    terr = float(np.abs(np.diff(a[:n], axis=0) - np.diff(b[:n], axis=0)).mean()) if n > 1 else None
    return {"psnr": float(psnr.group(1)) if psnr else None, "ssim": float(ssim.group(1)) if ssim else None,
            "temporal_error": round(terr, 3) if terr is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "benchmarks" / "dataset.yaml")
    ap.add_argument("--only", nargs="*", help="pipeline names to run")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.dataset.read_text())
    dg = cfg["degradation"]
    work = ROOT / "data" / "benchmark"
    work.mkdir(parents=True, exist_ok=True)
    results = []
    for clip in cfg["clips"]:
        name, secs = clip["name"], clip.get("seconds", 3)
        ref = work / f"{name}_ref.mp4"
        if clip.get("path"):
            ff("-i", clip["path"], "-t", str(secs), "-an", "-c:v", "libx264", "-crf", "10", str(ref))
        else:
            ff("-f", "lavfi", "-i", clip["generate"], "-t", str(secs), "-c:v", "libx264", "-crf", "10", str(ref))
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json",
                                          str(ref)], capture_output=True, check=True).stdout)["streams"][0]
        w, h = info["width"], info["height"]
        fps = eval(info["r_frame_rate"])  # noqa: S307 - ffprobe fraction from our own file
        dw, dh = w // dg["scale_divisor"] // 2 * 2, h // dg["scale_divisor"] // 2 * 2
        degraded = work / f"{name}_degraded.mp4"
        ff("-i", str(ref), "-vf", f"fps={dg['fps']},scale={dw}:{dh}:flags=area,noise=alls={dg['noise']}:allf=t",
           "-c:v", "libx264", "-crf", str(dg["crf"]), str(degraded))
        print(f"\n== {name}: reference {w}x{h}@{fps:g}, degraded {dw}x{dh}@{dg['fps']}")
        for pname, preset in pipelines().items():
            if args.only and pname not in args.only:
                continue
            out = work / f"{name}_{pname}.mp4"
            t = time.time()
            r = enhance_video(degraded, out, JobOptions(target_width=w, target_height=h, target_fps=fps),
                              work / "tmp", preset=preset)
            q = quality(out, ref, w, h)
            row = {"clip": name, "pipeline": pname, **q, "wall_seconds": round(time.time() - t, 1),
                   "throughput_fps": r["metrics"]["throughput_fps"],
                   "operations": [f"{o['name']}:{o['backend']}" for o in r["plan"]["operations"]]}
            results.append(row)
            print(f"  {pname:<18} PSNR {q['psnr']:.2f}  SSIM {q['ssim']:.4f}  temporal err {q['temporal_error']}"
                  f"  {row['wall_seconds']}s")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / "benchmarks" / "results"
    (outdir / f"{stamp}.json").write_text(json.dumps({"degradation": dg, "results": results}, indent=2))
    md = ["| clip | pipeline | PSNR ↑ | SSIM ↑ | temporal err ↓ | time |", "|---|---|---|---|---|---|"]
    md += [f"| {r['clip']} | {r['pipeline']} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['temporal_error']} | {r['wall_seconds']}s |"
           for r in results]
    (outdir / f"{stamp}.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()

"""Command-line interface.

    video-engine download-models
    video-engine analyze input.mp4
    video-engine plan input.mp4 --preset quality
    video-engine enhance input.mp4 output.mp4 --preset balanced --max-seconds 10
    video-engine serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _opts(a) -> "JobOptions":  # noqa: F821
    from .planner import JobOptions

    return JobOptions(preset=a.preset, output_policy=a.policy, interpolation=a.interpolation,
                      aspect_mode=a.aspect, codec=a.codec, max_seconds=a.max_seconds, start_seconds=a.start)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="video-engine", description="Hybrid AI + FFmpeg video enhancement engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download-models", help="Download model weights into models/")
    sub.add_parser("status", help="Show models, device and ffmpeg capabilities")
    sub.add_parser("calibrate", help="Measure model speeds on this Mac for accurate time estimates")

    a = sub.add_parser("analyze", help="Probe a video and estimate its degradation")
    a.add_argument("input", type=Path)

    def job_args(sp):
        sp.add_argument("--preset", default="balanced")
        sp.add_argument("--policy", choices=["5k3", "native"], default="5k3")
        sp.add_argument("--interpolation", choices=["auto", "on", "off"], default="auto")
        sp.add_argument("--aspect", choices=["fit", "crop"], default="fit")
        sp.add_argument("--codec", choices=["preset", "hevc", "h264"], default="preset")
        sp.add_argument("--max-seconds", type=float, default=None)
        sp.add_argument("--start", type=float, default=0.0)

    pl = sub.add_parser("plan", help="Show the processing plan without running it")
    pl.add_argument("input", type=Path)
    job_args(pl)

    e = sub.add_parser("enhance", help="Enhance a video")
    e.add_argument("input", type=Path)
    e.add_argument("output", type=Path)
    e.add_argument("--report", type=Path, help="Write the full JSON job result here")
    job_args(e)

    s = sub.add_parser("serve", help="Run the web UI + API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true", help="Restart automatically when the code changes")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "download-models":
        from . import models_registry

        for name in models_registry.REGISTRY:
            print(name)
            models_registry.download(name)
        return 0

    if args.cmd == "calibrate":
        from . import perf

        print("Measuring (about a minute)...")
        perf.calibrate()
        print(f"Saved to {perf.get_settings().models_dir / 'perf.json'}")
        return 0

    if args.cmd == "status":
        from . import models_registry
        from .ai import select_device
        from .media import ffmpeg_capabilities

        print(json.dumps({"device": str(select_device()), "models": models_registry.status(),
                          "ffmpeg": ffmpeg_capabilities()}, indent=2))
        return 0

    if args.cmd == "analyze":
        from . import degradation
        from .analyzer import analyze

        prof = analyze(args.input)
        print(json.dumps({"source": prof.to_dict(), "degradation": degradation.estimate(prof).to_dict()}, indent=2))
        return 0

    if args.cmd == "plan":
        from . import degradation
        from .analyzer import analyze
        from .config import get_preset
        from .planner import plan_job

        prof = analyze(args.input)
        deg = degradation.estimate(prof)
        plan = plan_job(prof, deg, _opts(args), get_preset(args.preset))
        print(json.dumps(plan.to_dict(), indent=2))
        return 0

    if args.cmd == "enhance":
        from .engine import enhance_video

        def progress(pr: dict) -> None:
            eta = pr.get("eta_seconds")
            sys.stderr.write(f"\r{pr.get('stage', '')}: {pr.get('frames_done', 0)}/{pr.get('frames_total', '?')} "
                             f"frames  {pr.get('fps', 0)} fps  ETA {eta if eta is not None else '?'}s   ")
            sys.stderr.flush()

        result = enhance_video(args.input, args.output, _opts(args),
                               on_state=lambda st, _d=None: print(f"\n[{st}]", file=sys.stderr),
                               on_progress=progress)
        print()
        for op in result["plan"]["operations"]:
            print(f"  {op['name']:<20} {op['backend']:<28} {op['reason']}")
        for w in result["plan"]["warnings"]:
            print(f"  WARNING: {w}")
        out = result["validation"]["output"]
        est = result["plan"].get("estimate", {}).get("seconds")
        if est:
            print(f"  estimated {est / 60:.1f} min, actual {result['metrics']['wall_seconds'] / 60:.1f} min")
        print(f"\n{args.output}: {out['width']}x{out['height']} @ {out['fps']} fps, {out['duration_seconds']}s, "
              f"{out['size_bytes'] / 1e6:.1f} MB (source {result['profile']['size_bytes'] / 1e6:.1f} MB), "
              f"processed in {result['metrics']['wall_seconds']}s")
        if args.report:
            args.report.write_text(json.dumps(result, indent=2, default=str))
        return 0

    if args.cmd == "serve":
        import uvicorn

        from .config import PROJECT_ROOT

        uvicorn.run("video_engine.server:app", host=args.host, port=args.port, log_level="info",
                    reload=args.reload,
                    reload_dirs=[str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "configs")] if args.reload else None,
                    reload_includes=["*.py", "*.yaml"] if args.reload else None)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 6 — streaming execution of a :class:`JobPlan`.

    ffmpeg decode ─▶ [bounded queue] ─▶ AI worker ─▶ [bounded queue] ─▶ ffmpeg encode
     (reader thread)                (interpolate / SR)              (writer thread)

The queues are bounded, so memory stays constant regardless of video length:
when the encoder is slow, the AI worker blocks; when the AI worker is slow,
the decoder blocks. Errors in any thread stop the others; cancellation kills
both ffmpeg processes. Plans that need no AI run as a single ffmpeg process.
"""

from __future__ import annotations

import logging
import queue
import resource
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np

from .decoder import FrameDecoder
from .encoder import FrameEncoder, ffmpeg_only_command, run_ffmpeg_only
from .planner import JobPlan

log = logging.getLogger(__name__)
_END = object()


class Cancelled(Exception):
    pass


ProgressFn = Callable[[dict], None]


class StageTimer:
    def __init__(self):
        self.seconds: dict[str, float] = {}

    def wrap(self, name: str, fn):
        def timed(*a, **kw):
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t
        return timed


class TemporalProbe:
    """Cheap flicker check: does SR change frame-to-frame differences?

    Compares mean absolute temporal difference of the SR input and output on a
    coarse luma grid. A ratio near 1.0 means SR adds no flicker of its own.
    """

    GRID = (54, 96)

    def __init__(self):
        self.prev_in = self.prev_out = None
        self.d_in: list[float] = []
        self.d_out: list[float] = []

    def _grid(self, f: np.ndarray) -> np.ndarray:
        h, w = f.shape[:2]
        ys = np.linspace(0, h - 1, self.GRID[0]).astype(int)
        xs = np.linspace(0, w - 1, self.GRID[1]).astype(int)
        return f[np.ix_(ys, xs)].mean(axis=2, dtype=np.float32)

    def add(self, src: np.ndarray, out: np.ndarray) -> None:
        gi, go = self._grid(src), self._grid(out)
        if self.prev_in is not None:
            self.d_in.append(float(np.abs(gi - self.prev_in).mean()))
            self.d_out.append(float(np.abs(go - self.prev_out).mean()))
        self.prev_in, self.prev_out = gi, go

    def report(self) -> dict:
        if not self.d_in:
            return {}
        din, dout = np.array(self.d_in), np.array(self.d_out)
        return {
            "mean_temporal_diff_in": round(float(din.mean()), 3),
            "mean_temporal_diff_out": round(float(dout.mean()), 3),
            "flicker_ratio": round(float(dout.mean() / max(din.mean(), 1e-6)), 3),
            "max_excess_flicker": round(float((dout - din).max()), 3),
        }


class Pipeline:
    def __init__(self, source: Path, plan: JobPlan, output: Path, work_dir: Path,
                 progress: ProgressFn | None = None, cancel: threading.Event | None = None):
        self.source, self.plan, self.output, self.work_dir = source, plan, output, work_dir
        self.progress = progress or (lambda _p: None)
        self.cancel = cancel or threading.Event()
        self.timer = StageTimer()
        self.frames_in = 0
        self.frames_out = 0
        self.models: dict[str, dict] = {}
        self.extra: dict = {}

    # ------------------------------------------------------------------ public
    def run(self) -> dict:
        t0 = time.perf_counter()
        if self.plan.streaming:
            self._run_streaming()
        else:
            self._run_ffmpeg_only()
        if self.cancel.is_set():
            raise Cancelled()
        wall = time.perf_counter() - t0
        return {
            "mode": "streaming" if self.plan.streaming else "ffmpeg_only",
            "wall_seconds": round(wall, 2),
            "frames_decoded": self.frames_in,
            "frames_written": self.frames_out,
            "throughput_fps": round(self.frames_out / wall, 3) if wall else None,
            "stage_seconds": {k: round(v, 2) for k, v in self.timer.seconds.items()},
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
            "models": self.models,
            **self.extra,
        }

    def _emit(self, stage: str, done: int, t0: float) -> None:
        total = self.plan.expected_frames
        if self.plan.streaming and self.plan.stream_fps:  # frames repeated later by the encoder
            done = round(done * float(Fraction(self.plan.target_fps) / Fraction(self.plan.stream_fps)))
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed > 0 and done else 0.0
        eta = (total - done) / rate if rate else None
        self.progress({"stage": stage, "frames_done": done, "frames_total": total,
                       "fraction": min(done / total, 1.0) if total else 0.0,
                       "fps": round(rate, 2), "eta_seconds": round(eta, 1) if eta is not None else None})

    # ------------------------------------------------------------- fast path
    def _run_ffmpeg_only(self) -> None:
        cmd = ffmpeg_only_command(self.source, self.plan, self.output)
        t0 = time.perf_counter()

        def on_frame(n: int) -> None:
            self.frames_out = n
            self._emit("ffmpeg", n, t0)

        run_ffmpeg_only(cmd, self.work_dir / "ffmpeg.log", on_frame, self.cancel.is_set)
        self.frames_in = self.frames_out
        self.timer.seconds["ffmpeg"] = time.perf_counter() - t0

    # ------------------------------------------------------- streaming path
    def _make_sr(self):
        plan = self.plan
        args = (plan.sr["model"],)
        out = (plan.sr["out_width"], plan.sr["out_height"])
        if plan.sr.get("runtime") == "coreml":
            try:
                from .coreml_sr import CoreMLSuperResolver

                return CoreMLSuperResolver(*args, (plan.source_width, plan.source_height), out,
                                           plan.sr["denoise_strength"])
            except Exception as exc:  # noqa: BLE001 - fall back to the GPU path, but say so
                log.warning("Core ML SR unavailable (%s); falling back to MPS", exc)
                self.extra.setdefault("warnings", []).append(f"Neural Engine SR failed ({exc}); used the GPU instead.")
        from .super_resolution import SuperResolver

        return SuperResolver(*args, out, plan.sr["denoise_strength"], plan.sr["tile"], plan.sr["half_precision"])

    def _build_stages(self) -> tuple[list[tuple[str, Callable[[Iterable[np.ndarray]], Iterator[np.ndarray]]]],
                                     tuple[int, int]]:
        """Ordered (name, generator-transform) stages; each runs in its own thread."""
        plan = self.plan
        out_size = (plan.source_width, plan.source_height)
        stages: list[tuple[str, Callable[[Iterable[np.ndarray]], Iterator[np.ndarray]]]] = []
        interp = plan.interpolation
        rife = interp["enabled"] and interp["backend"] == "rife"
        resampler = None
        if rife:
            from .interpolation import RifeInterpolator, TimelineResampler

            model = RifeInterpolator(scale=interp["rife_scale"])
            self.models["interpolation"] = model.info()
            if interp["order"] == "upscale_first" and plan.sr["enabled"]:
                # items are (source, enhanced) pairs: motion from source, warp the enhanced frames
                self.models["interpolation"]["mode"] = "flow from source frames, warp of enhanced frames"
                fn = self.timer.wrap("interpolation",
                                     lambda pa, pb, t: model.interpolate_full(pa[0], pb[0], pa[1], pb[1], t))
                resampler = TimelineResampler(Fraction(plan.base_fps), Fraction(plan.target_fps), fn,
                                              interp["scene_change_threshold"],
                                              view=lambda p: p[0], out=lambda p: p[1])
            else:
                resampler = TimelineResampler(Fraction(plan.base_fps), Fraction(plan.target_fps),
                                              self.timer.wrap("interpolation", model),
                                              interp["scene_change_threshold"])
            self.extra["interpolation_resampler"] = resampler

        if rife and interp["order"] == "interpolate_first":
            stages.append(("interpolation", resampler))

        if plan.sr["enabled"]:
            sr = self._make_sr()
            self.models["super_resolution"] = sr.info()
            sr_call = self.timer.wrap("super_resolution", sr)
            probe = TemporalProbe()
            self.extra["temporal_probe"] = probe

            pairs = rife and interp["order"] == "upscale_first"

            def sr_stage(frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
                for f in frames:
                    out = sr_call(f)
                    probe.add(f, out)
                    yield (f, out) if pairs else out

            stages.append(("super_resolution", sr_stage))
            out_size = (plan.sr["out_width"], plan.sr["out_height"])

        if rife and interp["order"] == "upscale_first":
            stages.append(("interpolation", resampler))
        return stages, out_size

    def _run_streaming(self) -> None:
        plan = self.plan
        stages, (ow, oh) = self._build_stages()
        decoder = FrameDecoder(self.source, plan, self.work_dir / "decoder.log")
        encoder = FrameEncoder(self.source, plan, ow, oh, self.output, self.work_dir / "encoder.log")
        # queues[0]: decoder -> first stage ... queues[-1]: last stage -> encoder
        queues = [queue.Queue(maxsize=plan.queue_size) for _ in range(len(stages) + 1)]
        q_in, q_out = queues[0], queues[-1]
        errors: list[BaseException] = []
        stop = threading.Event()

        def put(q: queue.Queue, item) -> bool:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def drain(q: queue.Queue) -> Iterator[np.ndarray]:
            while True:
                try:
                    item = q.get(timeout=0.2)
                except queue.Empty:
                    if stop.is_set():
                        return
                    continue
                if item is _END:
                    return
                yield item

        def guarded(fn):
            def run():
                try:
                    fn()
                except BaseException as exc:  # noqa: BLE001 - surfaced to caller
                    errors.append(exc)
                    stop.set()
            return run

        def reader():
            for frame in decoder.frames():
                if not put(q_in, frame):
                    decoder.kill()
                    return
                self.frames_in += 1
            put(q_in, _END)

        def stage_worker(fn, src: queue.Queue, dst: queue.Queue):
            def run():
                for frame in fn(drain(src)):
                    if not put(dst, frame):
                        return
                put(dst, _END)
            return run

        def writer():
            w = self.timer.wrap("encode_write", encoder.write)
            for frame in drain(q_out):
                w(frame)
                self.frames_out += 1

        workers = [(f"ve-{name}", stage_worker(fn, queues[i], queues[i + 1]))
                   for i, (name, fn) in enumerate(stages)]
        threads = [threading.Thread(target=guarded(fn), name=name, daemon=True)
                   for name, fn in [("ve-reader", reader), *workers, ("ve-writer", writer)]]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                if self.cancel.is_set():
                    stop.set()
                    decoder.kill()
                    encoder.kill()
                    break
                if errors:
                    break
                self._emit("ai", self.frames_out, t0)
                time.sleep(0.5)
            for t in threads:
                t.join(timeout=10)
        finally:
            if errors or self.cancel.is_set():
                stop.set()
                decoder.kill()
                encoder.kill()
        if self.cancel.is_set():
            raise Cancelled()
        if errors:
            raise errors[0]
        self.progress({"stage": "finalizing", "frames_done": self.frames_out,
                       "frames_total": plan.expected_frames, "fraction": 1.0, "eta_seconds": 0})
        t = time.perf_counter()
        encoder.close()
        self.timer.seconds["encode_flush"] = time.perf_counter() - t
        if "interpolation_resampler" in self.extra:
            self.extra["interpolation"] = self.extra.pop("interpolation_resampler").stats()
        if "temporal_probe" in self.extra:
            self.extra["temporal"] = self.extra.pop("temporal_probe").report()

# Architecture

<p align="center"><img src="images/pipeline.svg" alt="Pipeline" width="100%"></p>

The engine has six independent components that talk through plain data. The AI stages never know where a frame came
from, and the encoder never knows which model produced it.

```
upload ─▶ ANALYZE ─▶ PLAN ─▶ DECODE ─▶ SUPER-RESOLUTION ─▶ INTERPOLATION ─▶ ENCODE ─▶ VALIDATE ─▶ STORE
          ffprobe    rules +   FFmpeg    Real-ESRGAN          RIFE v4.26        FFmpeg     ffprobe     local | S3
          + stats    cost      (restore  Neural Engine        GPU (MPS)         VideoTool-
                     model     at source)                                       box HEVC
```

## Modules

| Module | Role |
|---|---|
| `analyzer.py` | ffprobe → `SourceProfile`: resolution, SAR, rotation, nominal/avg FPS, **VFR detection from a timestamp scan**, codec, bit depth, bitrate and bits/pixel, colour/HDR, interlacing, audio streams |
| `degradation.py` | Native-resolution centre crops: Immerkær noise σ, 8-px block-grid energy (compression), Crété-Roffet blur per axis (soft, defocus or motion blur), duplicate-frame and motion scan. Rule-based, with thresholds in one dict |
| `planner.py` | Deterministic `JobPlan`: FPS policy, geometry, restoration filters, SR runtime, stage order, encoder, audio, warnings and a time estimate. Every operation carries a human-readable reason |
| `perf.py` | Cost model (seconds per megapixel per stage). Chooses the stage order, estimates time, learns from finished jobs, and backs `video-engine calibrate` |
| `decoder.py` / `encoder.py` | FFmpeg subprocesses over pipes. CFR timeline from source timestamps, BT.709 tagging, audio muxed from the source for the same time window |
| `pipeline.py` | One thread per stage with bounded queues (constant memory), cancellation, error propagation, per-stage timing and a flicker probe. Plans with no AI run as a single FFmpeg process |
| `coreml_sr.py` | Real-ESRGAN on the **Apple Neural Engine**. One fixed tile shape per job, chosen to divide the frame evenly; converted models cached in `models/coreml/` |
| `super_resolution.py` | Same model on the GPU (MPS) or CPU. 512-px tiles, OOM back-off, GPU-side resize |
| `interpolation.py` | RIFE v4.26 with arbitrary timesteps, `TimelineResampler` (exact rational maths), scene-cut protection, **motion transfer** (`interpolate_full`) |
| `validation.py` | Resolution, FPS, duration, real frame count, audio streams and sync span, codec, colour tags. Output is written as `.partial.mp4` and only renamed after critical checks pass |
| `compare.py` | Browser-safe H.264 previews of source and output (same framing), posters, 1:1 crops (original pixels, Lanczos, enhanced) at the centre and at the most detailed region |
| `jobs.py` / `server.py` | Job state machine, JSON persistence, single GPU worker, FastAPI REST API, single-file web UI |
| `storage.py` | `LocalStorage` / `S3Storage` behind `VE_STORAGE_BACKEND` |

## Job states

```
QUEUED → ANALYZING → PLANNING → PROCESSING → VALIDATING → STORING → COMPLETED
                                     ↘ FAILED (stage + error recorded) / CANCELLED
```

Jobs run one at a time: the GPU and Neural Engine are the bottleneck, so running two jobs at once would only make
them compete for unified memory. If the server restarts mid-job, the job is marked `FAILED` with a reason
instead of hanging in a running state.

## Decision policies

### Frame rate: 60 FPS is a target, not an excuse to interpolate

| Source | 5.3K mode | Native mode |
|---|---|---|
| 60 FPS | preserved, no interpolation | preserved |
| 24 / 25 / 30 / 50 FPS | RIFE interpolation (preset allowing), else frames repeated *after* SR | preserved unless interpolation is forced on |
| 59.94 FPS | resampled to 60 (~1 frame per 1000 repeated), duration exact | preserved |
| 120 FPS | every other frame dropped (exact) | preserved |
| VFR | first placed on a constant-rate timeline built from its timestamps | same |
| Tagged 60 with ~50% duplicate frames | (quality presets) treated as 30 FPS of unique motion | — |

`TimelineResampler` places output frame *k* at source position *k × base / target* using rational arithmetic.
Integer positions pass real frames through untouched; fractional positions are synthesised. The duration
cannot drift, and the last frame is held for its full display interval.

### Resolution

- **720p → 5.3K:** 4× AI to 5120×2880, then a small Lanczos step to 5312×2988. The plan says this last step
  *adds pixels, not detail*.
- **1080p and up:** the 4× output is resized on the GPU, so oversized frames never cross the pipe.
- **Below the preset's minimum factor** (1.5× by default, e.g. 4K → 5.3K): conventional Lanczos.
- **Clean 5.3K source:** no scaling.
- **Portrait input → portrait output** (2988×5312). The aspect ratio is preserved by padding, or by cropping with `aspect_mode=crop`.

### Restoration (always at source resolution, before SR)

- Noise → `hqdn3d` (light/medium) or `nlmeans` (strong). Light noise is left to the SR model's own denoiser (DNI blend
  of the strong- and weak-denoise weights), so the frame isn't denoised twice.
- Compression blocking → `deblock`, so SR doesn't sharpen block edges.
- Interlaced → `bwdif`.
- Strong motion or defocus blur → reported honestly. No sharpening is added (it rings), and the plan says the detail can
  only be re-synthesised.

### Stage order (picked per job by the cost model)

<p align="center"><img src="images/motion-transfer.svg" alt="Motion transfer" width="100%"></p>

| Order | Upscaled frames | Interpolation runs on | Memory |
|---|---|---|---|
| interpolate → upscale | all output frames (3,600 for 60 s) | source frames | low |
| **upscale → interpolate, with motion transfer** | real frames only (1,800) | flow on source frames, warp on 5.3K frames | ~2.7 GB |

The planner computes both with `perf.py` (taking into account that the Neural Engine and GPU run in parallel) and
picks the faster one that fits in 35% of RAM. Its reasoning appears in the plan, e.g. *"after super-resolution
(estimated 27 min vs 54 min, RIFE needs ~2.7 GB)"*.

## Presets

| Preset | Restoration | Super-resolution | Interpolation | Encode |
|---|---|---|---|---|
| `fast` | light, auto | FFmpeg Lanczos + CAS | off (frame repeat) | HEVC VT q55 |
| `balanced` | auto, ≤ medium | Real-ESRGAN compact | RIFE when needed, auto order | HEVC VT q62 |
| `quality` | auto, ≤ strong | AI from 1.3× | RIFE, stricter scene cuts, auto order | 10-bit HEVC q70, temporal report |
| `restoration` | always on | AI from 1.1× | RIFE after SR | 10-bit HEVC q72, hallucination warning |

## Reproducibility

Every job records the preset, the full plan, model names, licences and SHA-256 checksums of the weights, the
device and precision used, per-stage timings, and the validation report. Model weights are downloaded by
`video-engine download-models` with pinned checksums and are never committed.

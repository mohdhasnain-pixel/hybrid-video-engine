# Performance

<p align="center"><img src="images/performance.svg" alt="Processing time for a 60 s clip" width="100%"></p>

## Where the time goes

5.3K at 60 FPS is about **950 megapixels per second of video**. For a 60 s, 1080p, 30 FPS clip:

- **3,600 output frames.** Only 1,800 are real; the rest are interpolated.
- **AI upscaling** costs about 5 TFLOP per 1080×1920 frame and dominates everything else.
- **Encoding alone** (5.3K HEVC) is about 10 minutes for 60 s on an M4, even with no AI at all.

## What the engine does about it

| Optimisation | Effect (1080×1920 source, M4 16 GB) |
|---|---|
| **SR on the Neural Engine** via Core ML (`coreml_sr.py`), one fixed tile shape per job | ~0.9 s/frame at peak vs 2–5 s/frame on the GPU (MPS); output matches the GPU path at 58 dB PSNR |
| **Upscale first, then interpolate with motion transfer** | SR on 1,800 frames instead of 3,600; RIFE ~2.7 GB instead of 6.8 GB at full size |
| **Parallel stages**: decoder, RIFE (GPU), SR (Neural Engine) and encoder in separate threads | Interpolation hides behind SR. One GPU lock keeps MPS thread-safe |
| **Frame repetition after SR** when interpolation is off | Repeated frames aren't upscaled twice |
| **Periodic GPU cache release, 512-px MPS tiles** | Memory stays flat on long jobs; steady per-frame times |

## Measurements (M4 MacBook Pro, 16 GB)

| Run | Time | Notes |
|---|---|---|
| 5 s of 1080×1920 30 FPS → 5.3K60, first version (GPU) | 10.7 min | measured |
| same clip, current Balanced | **5.2 min** (estimate 4.6 min) | measured at 8% battery |
| 60 s clip, first version | ≈ 7 h | observed pace, 0.14 output fps |
| 60 s clip, current Balanced | **≈ 30 min** | observed pace, 1.98 output fps |
| 60 s clip, `fast` preset (no AI) | ≈ 10.5 min | estimated from 5.7 fps measured |

> **Power matters.** Under sustained load, the Neural Engine and GPU slow down when the Mac is power-limited.
> At 8% battery, Neural Engine tile time rose from 0.127 s to 0.196 s within seconds. Charge first, and keep the Mac on AC
> power for long jobs.

## Time estimates

`perf.py` models each stage in seconds per megapixel:

```
estimate = max(SR time, interpolation time)    # Neural Engine and GPU run in parallel
         + encode time + fixed overhead
```

- The UI calls `/api/estimate` with the file's size and duration (read locally in the browser) as soon as you pick it.
- Each finished job updates the rates in `models/perf.json` (moving average), so estimates converge on your Mac's
  real sustained speed, including stages competing for power and memory.
- `uv run video-engine calibrate` re-measures every rate from scratch in about a minute.

## Going faster

These figures are estimates: none of the cloud options have been benchmarked with this engine yet.

| Option | 60 s clip at 5.3K60 | Quality |
|---|---|---|
| Current Balanced on M4 | ~30 min | full AI |
| AI on half-resolution input (540p → 4× → 5.3K) | ~10–12 min | ~4K-level real detail |
| `fast` preset | ~10 min | conventional upscale |
| 1× NVIDIA T4 + TensorRT FP16 + NVENC | ~8–12 min | full AI |
| 1× L4 / A10G / L40S | ~3–5 min | full AI |
| 4× T4, video split into segments | ~3–4 min | full AI |

The engine is split so only the model runtime needs to change: a CUDA/TensorRT SR runtime, GPU colour conversion
and NVENC. Storage already supports S3, so a cloud worker can take jobs from the same bucket.

## Quality benchmark

`benchmarks/run_benchmark.py` degrades clean reference clips (÷4 resolution, 30 FPS, noise, CRF 32), runs four
pipelines back to the reference size and frame rate with identical encoding, and reports PSNR, SSIM and a
temporal error (frame-to-frame change compared with the reference).

```bash
uv run python benchmarks/run_benchmark.py      # clips listed in benchmarks/dataset.yaml
```

Result on the built-in synthetic clip (a Mandelbrot zoom, **not natural footage**):

| Pipeline | PSNR ↑ | SSIM ↑ | Temporal error ↓ |
|---|---|---|---|
| Baseline Lanczos | 27.72 | 0.8809 | 3.54 |
| AI-only (SR, frames repeated) | 27.28 | 0.8821 | 3.63 |
| Hybrid A (restore → interpolate → SR) | 27.91 | 0.8878 | 2.84 |
| Hybrid B (restore → SR → interpolate) | 27.99 | 0.8888 | 2.80 |

On synthetic content, SR alone barely moves reference metrics: it creates *plausible* texture, not the true texture.
Most of the measurable gain comes from restoration plus real interpolation. Treat this as a harness check, not a
verdict. Add real footage (faces, foliage, sports, low light) and a perceptual metric such as LPIPS before tuning
defaults.

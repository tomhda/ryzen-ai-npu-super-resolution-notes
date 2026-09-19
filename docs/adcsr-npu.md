# Running AdcSR (a one-step diffusion super-resolution model) on an AMD Ryzen AI NPU (XDNA2)

*日本語の詳細は [ja/npu-research.md](ja/npu-research.md) の 5.6「AdcSR」を参照。*

This document describes how we run **AdcSR** — a one-step generative
super-resolution model distilled from Stable Diffusion 2.1-base (a UNet plus
a VAE decoder, CVPR 2025) — on the **NPU** of an AMD Ryzen AI 300-series
laptop with ONNX Runtime's VitisAI Execution Provider, and the runtime bug we
had to work around to make repeated inference return finite values.
As of September 2026, we are not aware of any prior public implementation of
AdcSR running on a client NPU. (Diffusion-based SR on mobile accelerators has
been demonstrated before, e.g. Edge-SD-SR on Samsung devices; the claim here
is limited to AdcSR and to a reproducible ONNX/VitisAI deployment.) This page
documents the conversion pipeline, the NPU-specific graph rewrites, measured
performance and the runtime workarounds, so that the next person searching
"AdcSR NPU", "VitisAI NaN second run" or "Ryzen AI diffusion NPU" finds the
recipe.

## Result

- Model: AdcSR `net_params_200` (code Apache-2.0; weights derive from
  SD2.1-base, OpenRAIL-M). Fixed-shape export 128x128 -> 512x512 (x4),
  converted to BF16 with Quark (`convert_fp32_to_bf16 --format with_cast`).
- Hardware/stack: Ryzen AI 7 PRO 350 (XDNA2 NPU, 8 columns), Windows 11,
  Ryzen AI Software 1.8.0, onnxruntime 1.27.0 VitisAI EP, XRT 2.19.0,
  NPU driver 32.0.203.329.
- VAIML accepts the whole graph after one graph rewrite (below): 6317 ops /
  1158 GOPs per tile as a **single 100% NPU subgraph**. The graph is then split
  in two (also below): front 4786/4797 ops (~93 min compile), back 1520/1520
  ops (~30 min). Later sessions load from cache in ~8 s and ~1 s.
- Fidelity: 45.4 dB PSNR against the same model on DirectML fp32 for a full
  1280x534 photo (crossfade tiling and seam correction identical on both).
- Speed: **~2.05 s per 128x128 tile** (front 0.71 s + back 1.33 s + 2.5 ms
  handoff), i.e. ~6.5 min for a 1280x534 photo (180 tiles at margin 32).
  DirectML on the iGPU does 1.3-1.6 s/tile.
- Why bother: the NPU lane leaves the GPU untouched (3D engine idle, CPU
  2-7%), so the highest-quality model runs in the background while the
  machine stays usable.

## Blocker 1: GroupNorm-style normalization on long 3-D tensors is not offloaded

The exported graph normalizes over `[1, C, L]` (channels x flattened spatial).
VAIML offloads the broadcast `Sub`/`Div` of that decomposition only while
`L <= 32768`; the 36 instances with `L >= 40960` fall back to the CPU, which
splits the model into 73 subgraphs and makes it 6x slower (11-15 s/tile).
Rewriting each normalization to operate on the 4-D `[1, C, H, W]` layout
(reshape, reduce, normalize, reshape back, then the per-channel affine) is
mathematically identical (PSNR 144 dB vs the original on CPU) and is fully
offloaded: one subgraph, 100%. Script:
[`scripts/adcsr/rewrite_in_to_n5.py`](../scripts/adcsr/rewrite_in_to_n5.py).

## Blocker 2: the second inference of the same session returns all-NaN

With the whole model resident on the NPU, the first `session.run()` is
correct and **every later run returns all-NaN**. None of 11 ONNX Runtime
session-option combinations change this (memory pattern, arena, graph
optimization level, threads, IO binding, fresh input copies, in-memory
cache...). The contamination is process-local and one-way:

- Cutting the graph right after the UNet output into a front half F
  (image -> latent + two 1x3x1x1 statistics) and a back half G
  (latent -> image): F alone and G alone run correctly indefinitely.
- In the same process, **running G once makes every later run of an F
  session created before it return NaN**. F never harms G. Deleting G does
  not heal F; recreating F does.
- Sessions in **separate processes never contaminate each other**
  (100/100 runs).

Full write-up, minimal reproducer (`repro.py` + fixed inputs + raw NaN
outputs) and the ONNX halves:
**[amd/RyzenAI-SW#402](https://github.com/amd/RyzenAI-SW/issues/402)** and the
release
[`adcsr-npu-repro-v1`](https://github.com/tomhda/ultraeasy-upscaler/releases/tag/adcsr-npu-repro-v1).
The failing layer (EP / VAIP / XRT / driver) is not identified; the report
lists what was ruled out.

## The workaround

Run the two halves in **two worker processes** and hand the 4 MiB latent
between them through the parent. Implementation in this repository:

- [`tools/npu-serve/npu_worker.py`](https://github.com/tomhda/ultraeasy-upscaler/blob/main/tools/npu-serve/npu_worker.py):
  a role-agnostic tensor executor (one VitisAI session per process). During
  session creation it redirects fd 1 to fd 2, because the AIE compiler
  writes `Old buffers:` spill reports to stdout mid-compile and would corrupt
  a stdout protocol.
- [`tools/npu-serve/npu_twostage.py`](https://github.com/tomhda/ultraeasy-upscaler/blob/main/tools/npu-serve/npu_twostage.py):
  the relay (self-test of two consecutive runs at start-up, finiteness
  checks on all boundary tensors and on the merged image, one in-image
  recovery by restarting both workers, one response per image), Windows Job
  Objects so that no worker outlives the app, crossfade tile merging and the
  seam-template correction applied in float before quantization.
- [`scripts/adcsr/split_adcsr_npu.py`](../scripts/adcsr/split_adcsr_npu.py):
  cuts the BF16 graph at the UNet output (exactly three crossing tensors, no
  extra Cast) and writes a manifest with SHA-256s and cache keys.

## Reproduce

```text
1. scripts/adcsr/export_adcsr.py       # AdcSR weights + SD2.1-base -> fixed-shape fp32 ONNX (128 -> 512)
2. scripts/adcsr/rewrite_in_to_n5.py   # normalization -> 4-D layout (VAIML-offloadable)
3. python -m quark.onnx.tools.convert_fp32_to_bf16 --input <n5 fp32> --output <n5 bf16> --format with_cast
4. scripts/adcsr/split_adcsr_npu.py    # front/back halves + manifest
5. Place the halves and the manifest in the models directory; the app starts
   two worker processes. First start compiles both halves (~2 h); later
   starts load the caches in seconds.
```

The app in this repository exposes the result as the `AdcSR` model choice on
both the DirectML and NPU backends. Set `UEU_ADCSR_NPU2=0` to force the
DirectML path.

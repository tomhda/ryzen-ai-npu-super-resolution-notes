# Super-resolution on the AMD Ryzen AI NPU: measurement notes

> **Unofficial.** This is an independent, personal project. It is not affiliated with, sponsored by, or
> endorsed by AMD. AMD, Ryzen, Ryzen AI and Radeon are trademarks of Advanced Micro Devices, Inc.

Notes, workarounds and scripts from running image super-resolution models (Real-ESRGAN, SPAN, SwinIR,
AdcSR) on the NPU of an AMD Ryzen AI laptop with the ONNX Runtime VitisAI execution provider
(VAIML bf16 flow), and comparing it with the iGPU of the same chip. Everything here was measured on
one machine; guesses are marked as guesses.

The models run in production in a Windows upscaler app:
[tomhda/ultraeasy-upscaler](https://github.com/tomhda/ultraeasy-upscaler). This repository holds the
technical background.

## Test machine

- AMD Ryzen AI 7 PRO 350 (XDNA2 NPU, 8 columns) / Radeon 860M iGPU / 32 GB LPDDR5-8000, Windows 11
- Ryzen AI Software 1.8.0 (onnxruntime 1.27 VitisAI EP, XRT 2.19.0), NPU driver 32.0.203.329
- bf16 models: `quark.onnx.tools.convert_fp32_to_bf16 --format with_cast`, compiled by the EP on first session creation
- NPU power mode Default unless noted. iGPU numbers are DirectML with fp32 ONNX.

## Results at a glance

One 853x480 frame → 4x (3412x1920), including tiling, merge and colour conversion, resident session:

| Model | iGPU (DirectML, fp32) | NPU (Default) | NPU (Turbo) | NPU vs fp32 (PSNR) |
|---|---|---|---|---|
| 4xNomosUni SPAN | 0.51 s | **0.25 s** | 0.18 s | 46.9 dB |
| realesr-animevideov3 (SRVGGNetCompact) | 0.46 s | 0.47 s | 0.35 s | 49.4 dB |
| Real-ESRGAN, reduced RRDB | 2.78 s | 2.07 s | not measured | 37.9 dB |
| SwinIR-M (real-world SR x4) | ~53 s | ~79 s | 256 tile: 6.73 → 3.39 s | 38.5 dB |
| AdcSR (one-step diffusion SR) | ~1.3-1.6 s / 128 tile | ~2.05 s / 128 tile | ~1.05 s / 128 tile | 45.4 dB (vs the iGPU output) |

While the NPU runs, the iGPU 3D engine stays at idle level and CPU use is 2-7%.

## Pages

| Page | What it covers |
|---|---|
| [DepthToSpace tail-cut](docs/tail-cut.md) | The pixel-shuffle tail took ~70% of NPU time for SPAN / SRVGGNetCompact. Cutting it off and doing it in numpy halves NPU time; SPAN becomes ~2x faster than the iGPU. Includes how to read AI Analyzer layer timers, the host-side changes, NPU power mode (Turbo ≈ 2x), and why NPU + iGPU together add no throughput. |
| [SwinIR on the Ryzen AI NPU](docs/swinir-npu.md) | VAIML compiler assertion on the `torch.roll` export pattern (negative `Slice` bounds + `Concat`), a bit-exact rewrite that avoids it, resource-usage comparison, TDR live dumps on long inferences. |
| [AdcSR on the Ryzen AI NPU](docs/adcsr-npu.md) | All-NaN output from the second run on, isolating it to cross-session contamination inside one process, and the two-process workaround. GroupNorm-style 3D normalisation rewritten to 4D to get a single subgraph. |
| [研究ノート（日本語、全記録）](docs/ja/npu-research.md) | The full record in Japanese: int8 (XIR) vs bf16 (VAIML), the PReLU miscompile and its bisection, compiler-option sweep, per-model profiles (RRDB, SwinIR attention rewrite experiment), SwinIR-S and SinSR (rejected), Windows ML route, history. |

## What this NPU is good and bad at (short version)

- `xrt-smi validate --run gemm` reports 51.3 TOPS, but SR models run at roughly 1/50-1/100 of that.
  The compute units are not the limit.
- Time goes into moving and rearranging large activations. AI Analyzer layer timers: SPAN spends
  ~70% in the `DepthToSpace` tail; the AdcSR VAE-decoder half spends 92% in `L3_OFM_Buffer_spill`
  layers; SwinIR-M spends 50% in spill layers; RRDB costs about 1 ms per layer across all 156 convolutions.
- NPU time is close to linear in pixels. Whole-frame inference (854x480 in one dispatch) compiles but
  fails at run time with `ERT_CMD_STATE_TIMEOUT`; tiling is required.
- bf16 through VAIML needs no calibration and was both faster and closer to fp32 than the older int8
  (XIR) flow, which split Real-ESRGAN into 11 subgraphs.
- Recompiling the same graph changes run time by several percent (0.375 s vs 0.406 s on a
  two-block SwinIR model), so small speedups need repeated compiles on both sides.

## Issues reported to AMD

- [amd/RyzenAI-SW#397](https://github.com/amd/RyzenAI-SW/issues/397): native assertion on a `Slice` pair with negative bounds + `Concat` (the `torch.roll` export pattern)
- [amd/RyzenAI-SW#398](https://github.com/amd/RyzenAI-SW/issues/398): TDR live dumps during long NPU kernels
- [amd/RyzenAI-SW#402](https://github.com/amd/RyzenAI-SW/issues/402): running one session makes another session in the same process return all-NaN

## Scripts

Copies of the conversion and reproduction scripts used in the app repository (which holds the
maintained versions and their tests).

| Script | Purpose |
|---|---|
| `scripts/npu/export_spandrel.py` | spandrel-supported models (SPAN, SwinIR, ...) → fixed-size fp32 ONNX; rewrites negative `Slice` bounds |
| `scripts/npu/export_animevideov3.py` | SRVGGNetCompact → fixed-size fp32 ONNX, optional PReLU decomposition |
| `scripts/npu/export_x4plus_anime.py` | RRDBNet (6B) → fixed-size fp32 ONNX |
| `scripts/npu/split_tail.py`, `tools/npu-serve/npu_tail.py` | tail-cut: split before the final `DepthToSpace`, verify against the original on CPU, numpy post-processing |
| `scripts/npu/bisect_vaiml_bf16*.py` | bisection of the PReLU bf16 miscompile (Ryzen AI 1.7.1) |
| `scripts/npu/verify_animevideov3_npu.py` | compile, check partitioning, PSNR and speed in one go |
| `scripts/adcsr/export_adcsr.py` | AdcSR → fixed 128x128 fp32 ONNX |
| `scripts/adcsr/rewrite_in_to_n5.py` | rewrite 3D normalisation to 4D so VAIML offloads it |
| `scripts/adcsr/split_adcsr_npu.py` | split AdcSR into the UNet half and the VAE-decoder half |

No model weights are included. Each model keeps its own licence (Real-ESRGAN BSD-3-Clause,
4xNomosUni SPAN CC-BY-4.0, SwinIR Apache-2.0, AdcSR Apache-2.0 with a Stable Diffusion 2.1 base under
CreativeML Open RAIL-M, AMD's reduced RRDB model research-only RAIL-MS).

## Licence and attribution

- Text and scripts in this repository: [MIT](LICENSE)
- Comparison images use frames from [Big Buck Bunny](https://peach.blender.org) and
  [Tears of Steel](https://mango.blender.org) (© Blender Foundation, CC-BY 3.0) and Superman (1941, public domain).

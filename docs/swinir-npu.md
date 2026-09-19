# Running SwinIR on an AMD Ryzen AI NPU (XDNA2)

*日本語の詳細は [ja/npu-research.md](ja/npu-research.md) の 5.5「SwinIR-M」を参照。*

This document describes how we run **SwinIR** — a Swin-Transformer
(window-attention) super-resolution model — entirely on the **NPU** of an AMD
Ryzen AI 300-series laptop, using ONNX Runtime's VitisAI Execution Provider.
As of 2026-08 we are not aware of any prior public example of SwinIR (or any
Swin-family attention SR model) running on a client NPU, so this page exists
mainly so that the next person searching "SwinIR NPU" finds the working recipe
and the compiler workaround it depends on.

## Result

- Model: `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN` (real-world SR, Apache-2.0),
  exported as a fixed-shape 256x256 ONNX graph, converted to BF16 with
  Quark (`convert_fp32_to_bf16 --format with_cast`).
- Hardware/stack: Ryzen AI 7 PRO 350 (XDNA2 NPU), Windows 11,
  Ryzen AI Software 1.8.0, onnxruntime 1.27.0 VitisAI EP.
- The VAIML compiler accepts the whole graph — **4971 ops / 2969 GOPs as a
  single 100% NPU subgraph** (no CPU fallback). First compile takes ~51 min;
  later sessions load from cache in seconds.
- Fidelity: PSNR 38.5 dB vs the fp32 CPU reference on real image tiles.
- Speed: ~6.6 s per 256x256 tile (x4). A 1080p still takes ~5 min. This is
  slower than the same model on the iGPU via DirectML (~4.5 s/tile).
- Why bother: **resource occupancy**. While the NPU runs SwinIR, iGPU 3D
  utilization stays at idle level (2.4%) and the inference process uses ~0.01%
  CPU — the machine stays fully usable. The DirectML path pegs the 3D engine
  at ~99%. So the NPU lane is a "slow but invisible" background
  highest-quality upscaler.

## The blocker: a VAIML compiler crash on `torch.roll` exports

Compiling the BF16 SwinIR initially kills the host process ~3 s into session
creation with a native assertion in `vaiml.dll`:

```
Assertion failed: all_equal({range_size(t), range_size(u), range_size(args)...})
&& "Iteratees do not have equal length", llvm-aie .../STLExtras.h:867
Exception Code: 0x80000003
```

Bisection down to a **3-node reproducer** showed the trigger is the export
form of `torch.roll(x, shifts=+s)` used by SwinIR's shifted windows:
two `Slice` nodes with **negative `starts`/`ends` literals** feeding one
`Concat`. Each piece compiles fine in isolation; the combination crashes the
ONNX→ONNX-MLIR lowering. Every attention building block (LayerNormalization,
Softmax, 4-D MatMul, masked window attention, window partition
transpose/reshape chains) compiles and runs on the NPU individually — only
this literal-handling bug stood between SwinIR and the NPU.

Full analysis and the minimal reproducer:
**[amd/RyzenAI-SW#397](https://github.com/amd/RyzenAI-SW/issues/397)**

## The workaround

Rewrite negative `Slice` bound literals to their positive equivalents
(`value + dim_size`) before BF16 conversion. For static-shape exports this is
fully determined, bit-exact, and changes no node/op. Our exporter applies it
automatically: see `_rewrite_slice_nonneg` in
[`scripts/npu/export_spandrel.py`](../scripts/npu/export_spandrel.py)
(72 rewrite sites in the full SwinIR-M graph).

## Reproduce

```text
1. python scripts/get_ai_models.py --download swinir        # weights + SHA-256
2. python scripts/npu/export_spandrel.py --weights <pth> --name swinir --tile 256
   # fixed-shape fp32 ONNX; the VAIML-safe Slice rewrite is applied here
3. python -m quark.onnx.tools.convert_fp32_to_bf16 \
       --input swinir_nchw_256x256_fp32.onnx \
       --output swinir_nchw_256x256_bf16cast.onnx --format with_cast
4. Create a VitisAI EP session (cache_dir/cache_key provider options).
   First session creation runs the ~51 min VAIML compile; afterwards the
   cache loads in seconds.
```

The app in this repository exposes the result as the `SwinIR-M` model choice
on both the DirectML and NPU backends.

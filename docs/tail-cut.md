# DepthToSpace tail-cut: halving NPU time for pixel-shuffle SR models on AMD Ryzen AI (XDNA2)

Most lightweight super-resolution networks (SRVGGNetCompact / Real-ESRGAN "compact", SPAN, and many
others) end with a `DepthToSpace` (PixelShuffle) node. On the Ryzen AI NPU this single
data-rearrangement step, together with the memory spill around it, took about 70% of the NPU time of
the whole model. Cutting the graph just before `DepthToSpace` and doing the pixel shuffle on the CPU
with numpy roughly halves the per-tile NPU time. The output is numerically the same operation, and
the CPU work overlaps with the next tile's NPU run.

Measured on one machine: AMD Ryzen AI 7 PRO 350 (XDNA2 NPU, Radeon 860M iGPU), 32 GB LPDDR5-8000,
Ryzen AI Software 1.8.0 (onnxruntime 1.27 VitisAI EP, VAIML bf16 flow), NPU driver 32.0.203.329,
NPU power mode Default unless noted.

## Result

One 853x480 frame → 4x (3412x1920), including tiling, merge and uint8 conversion, resident session:

| Model | iGPU (DirectML, fp32) | NPU, whole model | NPU, tail-cut | NPU, tail-cut + host-side changes |
|---|---|---|---|---|
| 4xNomosUni SPAN (48nf) | 0.51 s | 0.61 s | 0.36 s | **0.25 s** |
| realesr-animevideov3 (SRVGGNetCompact) | 0.46 s | 1.21 s | 0.62 s | **0.47 s** |

Per 512x512 tile, NPU only: SPAN 0.261 → 0.115 s, animevideov3 0.553 → 0.239 s.
Video (3 s, 72 frames, end to end): SPAN 1.57 → 3.61 fps, animevideov3 0.79 → 1.95 fps.

With the NPU power mode set to Turbo (`xrt-smi configure --pmode turbo`, AC power required, no
recompilation), the same frame takes 0.18 s (SPAN) and 0.35 s (animevideov3); see
[Power mode](#npu-power-mode).

## How the bottleneck was found

Compile with the AI Analyzer provider options into a separate cache key and run a few inferences:

```python
provider_options = [{
    "cache_dir": cache_dir,
    "cache_key": "modelcachekey_prof_span",
    "ai_analyzer_profiling": True,
    "ai_analyzer_visualization": True,
}]
```

This writes `record_timer_ts.json` (and a few other files) into the current directory, and keeps
`vaiml_par_0/0/aie_record_timer.json` and `layer_name_map.json` inside the cache directory.
`record_timer_ts.json` is a list of `{id, cycle}` timestamps at 1 GHz; the difference between
adjacent `cycle` values is the duration of a segment. Attribute each segment to the `layer_id` of the
later timestamp, then map `local_layer_id` → `user_friendly_name` with `layer_name_map.json` to get
per-operator times. Do not compare timings taken with profiling on against timings with it off.

For SPAN (512x512 tile, about 240 ms): about twenty mid-network layers cost a uniform 2.0-2.3 ms each
(42 ms in total), while the four layers of the upsampler tail (`DepthToSpace` and `L3` spill
layers) cost 17 / 42 / 20 / 18 ms (97 ms), followed by another 71 ms segment. The tail is about 70%
of the run. `xrt-smi validate --run gemm` reports 51.3 TOPS on the same machine, so the compute
units are fine; the time goes into moving and rearranging a large activation
(`[1,48,512,512]` → `[1,3,2048,2048]`).

## The cut

`DepthToSpace` has no weights. For `blocksize = r`, mode CRD:

```python
def pixel_shuffle_crd(x, r):            # x: [N, C*r*r, H, W]
    n, c, h, w = x.shape
    t = x.reshape(n, c // (r * r), r, r, h, w).transpose(0, 1, 4, 2, 5, 3)
    return t.reshape(n, c // (r * r), h * r, w * r)
# DCR: x.reshape(n, r, r, c // (r * r), h, w).transpose(0, 3, 4, 1, 5, 2)
```

- SPAN ends with `... → Conv → DepthToSpace`. The body outputs `[1,48,512,512]`; the CPU does the pixel shuffle.
- SRVGGNetCompact ends with `... → Conv → DepthToSpace → Add(nearest-4x of the input)`.
  The CPU does the pixel shuffle and adds the nearest-neighbour upsampled input.

The number of output elements is unchanged (48×512² = 3×2048²), so the amount of data copied out of the
NPU is the same.

[`scripts/npu/split_tail.py`](../scripts/npu/split_tail.py) detects these two tail shapes, extracts
the body with `onnx.utils.extract_model`, writes a small JSON manifest
(`blocksize`, `mode`, `add_nearest_input`, `scale`, `body_output`), and refuses to write anything
unless body + numpy post-processing matches the original model on the ONNX Runtime CPU EP
(max abs diff < 1e-5). Tails it does not recognise (for example a trailing `Clip`) are rejected.

```text
python scripts/npu/split_tail.py --input model_fp32.onnx
```

The shared numpy code is [`tools/npu-serve/npu_tail.py`](../tools/npu-serve/npu_tail.py).

## Host-side changes that mattered once the NPU part got short

After the cut (and even more with Turbo), 40-50% of the per-frame time was on the host:
merging float tiles (~45 ms), `clip(x*255).astype(uint8)` on the whole frame (~50 ms), CHW→HWC
(~20 ms), all executed serially after the last tile. For SRVGGNetCompact, building the 4x
nearest-neighbour input with two `np.repeat` calls cost about 120 ms per tile.

What was changed (the output stayed bit-identical, verified by SHA-256 of the frames):

- A second thread post-processes tile *n* while the NPU runs tile *n+1* (queue depth 2).
- Crop the valid core of each tile (without overlap and padding) on the low-resolution side first,
  so the full-size float tile and the float merge buffer are never created.
- Do the pixel-shuffle permutation and the uint8 cast as a single write into the final HWC uint8
  frame (`np.copyto(dst.reshape(h, r, w, r, c), shuffled_view, casting="unsafe")`).
- Add the nearest-neighbour input by broadcasting (`[h,1,w,1,c]`) instead of materialising it.
- Write the response without `tobytes()`, and `readinto` the destination array on the client.

| 853x480 frame | Default, before | Default, after | Turbo, before | Turbo, after |
|---|---|---|---|---|
| SPAN | 0.362 s | 0.250-0.274 s | 0.275-0.282 s | 0.173-0.181 s |
| animevideov3 | 0.615 s | 0.466-0.494 s | 0.447-0.455 s | 0.346-0.351 s |

## Things to check when you try this

- **FP32 direct input vs bf16 cast.** Ryzen AI 1.8 accepts FP32 ONNX and converts to BF16 at compile
  time. For the SPAN body this ran at the same speed as the Quark `with_cast` model and was closer to
  the FP32 reference (PSNR 44.51 → 46.94 dB). It is model dependent: the AdcSR VAE-decoder half
  compiled from FP32 put only 76 nodes (6 subgraphs) on the NPU and ran the rest on the CPU.
  After every compile, open `context.json` in the cache and confirm there is one `metaDef` and that
  it contains all nodes.
- **PReLU.** realesr-animevideov3 with its trained PReLU slopes was silently miscompiled by the VAIML
  bf16 flow in Ryzen AI 1.7.1 (finite but wrong output). Decompose
  `PReLU(x) = ReLU(x) − w⊙ReLU(−x)` before conversion
  ([`export_animevideov3.py --decompose-prelu`](../scripts/npu/export_animevideov3.py));
  the decomposed model is also about 2x faster.
- **Compiler options.** The bundled default configuration already uses optimize level 2. Level 1
  was 5.6x slower at run time; level 3 and `preferred_data_storage` (vectorized / unvectorized) made
  no difference for SPAN.
- **It only helps when the tail is a pure rearrangement.** The reduced RRDB Real-ESRGAN model ends
  with `Resize → Conv → Resize → Conv → Conv → Conv` at up to 1024x1024. Measured: whole model
  ~170 ms per 256 tile, body alone 158.7 ms, so the tail is only 10-15 ms on the NPU, while the same
  tail takes ~120 ms on the CPU (all cores). Not worth cutting. In SwinIR-M the high-resolution tail
  is about 1% of the NPU time.

## NPU power mode

`xrt-smi configure --pmode turbo` roughly doubles NPU speed for every model tested, with the same
compiled cache and bit-identical output:

| 1 run, median | Default | Turbo |
|---|---|---|
| SPAN, 512 tile (whole model) | 0.246 s | 0.130 s |
| animevideov3, 512 tile (whole model) | 0.524 s | 0.257 s |
| Real-ESRGAN reduced RRDB, 853x480 frame (12 tiles of 256) | 2.07 s | 1.23 s |
| SwinIR-M, 256 tile | 6.73 s | 3.39 s |
| AdcSR UNet half / VAE-decoder half, 128 tile | 0.738 / 1.359 s | 0.361 / 0.628 s |

NPU power as reported by `xrt-smi examine` while running SwinIR-M continuously (not an external
measurement): Default 0.6-1.2 W (about 0.85 W on average), Turbo a constant 2.4 W. With no process using the NPU the
reading is `N/A` (or 0.001 W) in both modes; while a session is loading it is 0.04-0.16 W (Default) and 0.1-0.4 W (Turbo). Per 256 tile that is about 5.6 J (Default, 6.55 s) vs 8.1 J (Turbo, 3.39 s) for
the NPU alone: Turbo is about 2x faster for about 1.45x the NPU energy. Whole-system energy was not
measured. After setting Turbo on AC power and then unplugging, the mode stayed Turbo and SwinIR-M ran
at 3.49 s per tile on battery.

Default follows the Windows power mode. Thermals were not measured. All other numbers
in this repository are in Default mode.

## Running the NPU and the iGPU at the same time does not add throughput

Processing different images concurrently on the NPU (VitisAI) and the iGPU (DirectML) with SwinIR-M:
about 35 s each when run alone, 60-74 s each when run together. Total throughput equals running
them one after the other. The cause was not identified; both devices share the same LPDDR5.

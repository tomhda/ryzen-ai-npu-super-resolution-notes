#!/usr/bin/env python
"""N5 書換え済み AdcSR bf16cast を UNet 出力直後で前半 F / 後半 G に切断する。

tmp/adcsr-npu/round5/b1/cut_b1.py の配布用移植。切断面は面 A (UNet 出力直後) に固定。
横断テンソルは入力依存の3本のみ (B1 scan_cut2 で確定):
  MAIN = /core/core.0/body/body.1/body/conv_out/Conv_output_0[_DequantizeLinear_Output]
  MEAN = /ReduceMean_3_output_0[_DequantizeLinear_Output]  ([1,3,1,1] FLOAT)
  STD  = /Sqrt_1_output_0[_DequantizeLinear_Output]       ([1,3,1,1] FLOAT)
bf16cast は DQ 付き名、fp32 は素の名。すべて既存 FLOAT のため Cast 追加なし。

front/back とマニフェスト (SHA-256、切断テンソル名、ツール版、元モデルの由来、
cache_key) を書き出す。2 段モードでは配布名を変えず、マニフェストの cache_key
で既存キャッシュを指定する (1 モデル経路の modelcachekey_<stem> 規約は不変)。

実行 (onnx・onnxruntime(CPU) が入った Python で):
  python scripts/adcsr/split_adcsr_npu.py --input <N5 bf16cast> --out-dir <dir> \
      --cache-key-front modelcachekey_xxx --cache-key-back modelcachekey_yyy \
      [--fp32 <N5 fp32> --ref-image <png> (fp32 連結一致の検証用)]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

TOOL_VERSION = "1"

# 切断面 A (bf16cast の DQ 付き名)。
MAIN_B = "/core/core.0/body/body.1/body/conv_out/Conv_output_0_DequantizeLinear_Output"
MEAN_B = "/ReduceMean_3_output_0_DequantizeLinear_Output"
STD_B = "/Sqrt_1_output_0_DequantizeLinear_Output"
BOUNDARY_B = [MAIN_B, MEAN_B, STD_B]
# fp32 側の素名。
MAIN_F = "/core/core.0/body/body.1/body/conv_out/Conv_output_0"
MEAN_F = "/ReduceMean_3_output_0"
STD_F = "/Sqrt_1_output_0"

DEFAULT_FRONT_NAME = "adcsr_front_nchw_128x128_bf16cast.onnx"
DEFAULT_BACK_NAME = "adcsr_back_nchw_128x128_bf16cast.onnx"
DEFAULT_MANIFEST_NAME = "adcsr_npu_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    front_name: str,
    front_sha256: str,
    front_cache_key: str,
    back_name: str,
    back_sha256: str,
    back_cache_key: str,
    boundary: list[str],
    tool_version: str,
    source_model: str,
) -> dict:
    """配布マニフェスト。本文は npu_twostage.load_manifest の必須キーを満たす。"""
    return {
        "model_family": "AdcSR",
        "tool": "scripts/adcsr/split_adcsr_npu.py",
        "tool_version": tool_version,
        "source_model": source_model,
        "cut": "UNet output (surface A): 3 input-dependent crossing tensors",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "front": {"file": front_name, "sha256": front_sha256, "cache_key": front_cache_key},
        "back": {"file": back_name, "sha256": back_sha256, "cache_key": back_cache_key},
        "boundary": list(boundary),
    }


def split_models(src: Path, front_path: Path, back_path: Path) -> None:
    import onnx
    import onnx.utils

    print("extracting F (bf16cast) ...", flush=True)
    onnx.utils.extract_model(str(src), str(front_path), ["input"], BOUNDARY_B)
    print("extracting G (bf16cast) ...", flush=True)
    onnx.utils.extract_model(str(src), str(back_path), BOUNDARY_B, ["output"])
    for path in (front_path, back_path):
        onnx.checker.check_model(onnx.load(str(path), load_external_data=True))
        print(f"checker OK: {path.name} ({path.stat().st_size}B)", flush=True)


def check_boundary_count(front_path: Path) -> None:
    import onnx

    model = onnx.load(str(front_path), load_external_data=True)
    names = [v.name for v in model.graph.output]
    if len(names) != 3 or set(names) != set(BOUNDARY_B):
        raise ValueError(f"front outputs must be the 3 boundary tensors, got {names}")
    print(f"boundary OK: {len(names)} tensors", flush=True)


def check_fp32_linkage(fp32_path: Path, ref_image: Path) -> float:
    """F32→G32 連結が fp32 本体と一致すること (max abs diff < 1e-4)。戻り値は max diff。"""
    import numpy as np
    import onnx.utils
    import onnxruntime as ort
    from PIL import Image

    import tempfile

    with tempfile.TemporaryDirectory() as work:
        f32_path = str(Path(work) / "front_fp32.onnx")
        g32_path = str(Path(work) / "back_fp32.onnx")
        onnx.utils.extract_model(str(fp32_path), f32_path, ["input"], [MAIN_F, MEAN_F, STD_F])
        onnx.utils.extract_model(str(fp32_path), g32_path, [MAIN_F, MEAN_F, STD_F], ["output"])
        img = Image.open(ref_image).convert("RGB")
        W, H = img.size
        x = (np.asarray(img.crop(((W - 128) // 2, (H - 128) // 2,
                                  (W + 128) // 2, (H + 128) // 2)), np.float32) / 255.0
             ).transpose(2, 0, 1)[None].copy()
        full = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]).run(
            ["output"], {"input": x})[0]
        f32 = ort.InferenceSession(f32_path, providers=["CPUExecutionProvider"])
        mid_main, mid_mean, mid_std = f32.run(None, {"input": x})
        pipe = ort.InferenceSession(g32_path, providers=["CPUExecutionProvider"]).run(
            ["output"], {MAIN_F: mid_main, MEAN_F: mid_mean, STD_F: mid_std})[0]
        diff = float(np.abs(pipe.astype(np.float64) - full.astype(np.float64)).max())
        print(f"pipe_vs_full: max_abs_diff={diff:.3e} "
              f"{'PASS' if diff < 1e-4 else 'FAIL'} (threshold 1e-4)", flush=True)
        if diff >= 1e-4:
            raise ValueError(f"fp32 linkage check failed: max_abs_diff={diff:.3e}")
        return diff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="N5 書換え済み AdcSR bf16cast ONNX")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--front-name", default=DEFAULT_FRONT_NAME)
    parser.add_argument("--back-name", default=DEFAULT_BACK_NAME)
    parser.add_argument("--cache-key-front", required=True)
    parser.add_argument("--cache-key-back", required=True)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--tool-version", default=TOOL_VERSION)
    parser.add_argument("--source-model", default="",
                        help="元モデルの由来 (例: adcsr_norm_bf16cast.onnx)")
    parser.add_argument("--fp32", default=None, help="N5 書換え済み fp32 (連結検証用)")
    parser.add_argument("--ref-image", default=None, help="連結検証用の参照画像")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    front_path = out_dir / args.front_name
    back_path = out_dir / args.back_name
    split_models(Path(args.input), front_path, back_path)
    check_boundary_count(front_path)
    if args.fp32 and args.ref_image:
        check_fp32_linkage(Path(args.fp32), Path(args.ref_image))
    elif bool(args.fp32) != bool(args.ref_image):
        raise SystemExit("--fp32 と --ref-image は両方指定すること")
    manifest = build_manifest(
        front_name=args.front_name,
        front_sha256=sha256_file(front_path),
        front_cache_key=args.cache_key_front,
        back_name=args.back_name,
        back_sha256=sha256_file(back_path),
        back_cache_key=args.cache_key_back,
        boundary=BOUNDARY_B,
        tool_version=args.tool_version,
        source_model=args.source_model or Path(args.input).name,
    )
    manifest_path = out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

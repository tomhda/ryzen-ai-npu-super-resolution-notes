"""VAIML bf16 が SRVGGNetCompact を誤コンパイルする犯人opを絞り込む。

数ノードの合成ONNXを op 別に作り、fp32→bf16(cast)→VitisAIコンパイル→
fp32(CPU)との出力差を測る。おまけに PReLU 分解版フルモデルも検証する。

Ryzen AI conda 環境でリポジトリルートから実行:
  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe scripts/npu/bisect_vaiml_bf16.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "tmp" / "npu-anime" / "bisect"
TILE = 256


class ConvOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.c2 = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        return self.c2(self.c1(x))


class ConvLRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.c2 = nn.Conv2d(32, 3, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.c2(self.act(self.c1(x)))


class ConvPRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.c2 = nn.Conv2d(32, 3, 3, 1, 1)
        self.act = nn.PReLU(num_parameters=32)

    def forward(self, x):
        return self.c2(self.act(self.c1(x)))


class ConvPixelShuffle(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 48, 3, 1, 1)
        self.ps = nn.PixelShuffle(4)

    def forward(self, x):
        return self.ps(self.c1(x))


class ResizeAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 48, 3, 1, 1)
        self.ps = nn.PixelShuffle(4)

    def forward(self, x):
        return self.ps(self.c1(x)) + F.interpolate(x, scale_factor=4, mode="nearest")


def build_cases() -> list[tuple[str, nn.Module]]:
    torch.manual_seed(20260814)
    return [
        ("conv_only", ConvOnly()),
        ("conv_lrelu", ConvLRelu()),
        ("conv_prelu", ConvPRelu()),
        ("conv_pixelshuffle", ConvPixelShuffle()),
        ("resize_add", ResizeAdd()),
    ]


def to_bf16(fp32: Path, bf16: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "quark.onnx.tools.convert_fp32_to_bf16",
         "--input", str(fp32), "--output", str(bf16), "--format", "with_cast"],
        check=True, capture_output=True,
    )


def evaluate(name: str, fp32: Path, bf16: Path) -> str:
    import onnxruntime as ort

    x = np.random.RandomState(7).rand(1, 3, TILE, TILE).astype(np.float32)
    ref_sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    ref = ref_sess.run(None, {ref_sess.get_inputs()[0].name: x})[0]

    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        str(bf16),
        providers=["VitisAIExecutionProvider"],
        provider_options=[{"cache_dir": str(WORK), "cache_key": f"cache_{name}",
                           "enable_cache_file_io_in_mem": 0}],
    )
    compile_s = time.perf_counter() - t0
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    mse = float(np.mean((ref.astype(np.float64) - got.astype(np.float64)) ** 2))
    rng = float(ref.max() - ref.min()) or 1.0
    psnr = 99.0 if mse == 0 else 10 * np.log10(rng ** 2 / mse)
    verdict = "OK" if psnr > 30 else "BROKEN"
    return (f"{name}: {verdict}  psnr={psnr:.1f}dB  max|diff|={diff.max():.4f}  "
            f"compile={compile_s:.0f}s")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    WORK.mkdir(parents=True, exist_ok=True)
    results = []
    for name, model in build_cases():
        model.eval()
        fp32 = WORK / f"{name}_fp32.onnx"
        bf16 = WORK / f"{name}_bf16.onnx"
        dummy = torch.rand(1, 3, TILE, TILE)
        with torch.no_grad():
            torch.onnx.export(model, dummy, str(fp32),
                              input_names=["input"], output_names=["output"],
                              opset_version=17, do_constant_folding=True)
        to_bf16(fp32, bf16)
        print(f"--- {name}: コンパイル開始", flush=True)
        try:
            line = evaluate(name, fp32, bf16)
        except Exception as exc:  # noqa: BLE001
            line = f"{name}: ERROR {type(exc).__name__}: {exc}"
        print(line, flush=True)
        results.append(line)

    print("===== 結果まとめ =====")
    for line in results:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

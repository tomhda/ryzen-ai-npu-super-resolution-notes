"""VAIML bf16 誤コンパイルの phase2: SRVGG構造を深さ・活性化違いで検証。

phase1 で単体opは全て正常だったため、フルグラフでの発現条件を探す。
  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe scripts/npu/bisect_vaiml_bf16_phase2.py
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


class DecomposedPReLU(nn.Module):
    """PReLU(x) = ReLU(x) - w * ReLU(-x)（数学的に等価な分解形）。"""

    def __init__(self, num_parameters: int):
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_parameters,), 0.25))

    def forward(self, x):
        w = self.weight.view(1, -1, 1, 1)
        return F.relu(x) - w * F.relu(-x)


class SrvggLike(nn.Module):
    """SRVGGNetCompact と同構造（深さ・活性化を可変に）。"""

    def __init__(self, num_conv: int, act: str):
        super().__init__()
        feat = 64
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(3, feat, 3, 1, 1))
        self.body.append(self._act(act, feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(feat, feat, 3, 1, 1))
            self.body.append(self._act(act, feat))
        self.body.append(nn.Conv2d(feat, 3 * 16, 3, 1, 1))
        self.ps = nn.PixelShuffle(4)

    @staticmethod
    def _act(act: str, feat: int) -> nn.Module:
        if act == "prelu":
            return nn.PReLU(num_parameters=feat)
        if act == "lrelu":
            return nn.LeakyReLU(0.1)
        if act == "dprelu":
            return DecomposedPReLU(feat)
        raise ValueError(act)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.ps(out) + F.interpolate(x, scale_factor=4, mode="nearest")


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

    mse = float(np.mean((ref.astype(np.float64) - got.astype(np.float64)) ** 2))
    rng = float(ref.max() - ref.min()) or 1.0
    psnr = 99.0 if mse == 0 else 10 * np.log10(rng ** 2 / mse)
    diff = float(np.abs(ref.astype(np.float64) - got.astype(np.float64)).max())
    verdict = "OK" if psnr > 30 else "BROKEN"
    return (f"{name}: {verdict}  psnr={psnr:.1f}dB  max|diff|={diff:.4f}  "
            f"compile={compile_s:.0f}s")


CASES = [
    ("srvgg2_prelu", 2, "prelu"),
    ("srvgg8_prelu", 8, "prelu"),
    ("srvgg16_prelu", 16, "prelu"),
    ("srvgg16_lrelu", 16, "lrelu"),
    ("srvgg16_dprelu", 16, "dprelu"),
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    WORK.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260814)
    results = []
    for name, depth, act in CASES:
        model = SrvggLike(depth, act).eval()
        fp32 = WORK / f"{name}_fp32.onnx"
        bf16 = WORK / f"{name}_bf16.onnx"
        with torch.no_grad():
            torch.onnx.export(model, torch.rand(1, 3, TILE, TILE), str(fp32),
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

    print("===== phase2 結果まとめ =====")
    for line in results:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

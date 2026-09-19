"""VAIML bf16 phase4: 負のPReLU slope 仮説の検証と、分解による修正候補の検証。

  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe scripts/npu/bisect_vaiml_bf16_phase4.py
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
BASE = REPO / "tmp" / "npu-anime"
WORK = BASE / "bisect"
WEIGHTS = BASE / "realesr-animevideov3.pth"
TILE = 256


class DecomposedPReLU(nn.Module):
    """PReLU(x) = ReLU(x) - w * ReLU(-x)（等価分解・PReLUカーネル回避）。"""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())

    def forward(self, x):
        w = self.weight.view(1, -1, 1, 1)
        return F.relu(x) - w * F.relu(-x)


class SrvggLike(nn.Module):
    def __init__(self, num_conv: int = 16, act: str = "prelu"):
        super().__init__()
        feat = 64
        self.act_kind = act
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(3, feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(feat, feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=feat))
        self.body.append(nn.Conv2d(feat, 3 * 16, 3, 1, 1))
        self.ps = nn.PixelShuffle(4)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.ps(out) + F.interpolate(x, scale_factor=4, mode="nearest")


def load_real_model(decompose: bool) -> nn.Module:
    sys.path.insert(0, str(REPO / "scripts" / "npu"))
    from export_animevideov3 import SRVGGNetCompact

    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    params = state.get("params", state)
    model = SRVGGNetCompact(upscale=4)
    model.load_state_dict(params, strict=True)
    if decompose:
        for i, layer in enumerate(model.body):
            if isinstance(layer, nn.PReLU):
                model.body[i] = DecomposedPReLU(layer.weight.data)
    return model.eval()


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


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    WORK.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260814)
    results = []

    # A: 合成 + 実物と同レンジの slope（負値・1超え込み）
    syn = SrvggLike(16)
    with torch.no_grad():
        for layer in syn.body:
            if isinstance(layer, nn.PReLU):
                layer.weight.uniform_(-1.4, 1.7)
    cases = [("syn16_negslope", syn)]

    # B: 実重み + PReLU分解（実slopeのまま）
    cases.append(("real_dprelu", load_real_model(decompose=True)))

    for name, model in cases:
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

    print("===== phase4 結果まとめ =====")
    for line in results:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

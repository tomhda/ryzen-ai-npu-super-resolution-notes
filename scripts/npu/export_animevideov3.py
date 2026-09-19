"""realesr-animevideov3 (SRVGGNetCompact) を NPU 用の固定形状 ONNX に出力する。

Ryzen AI conda 環境 (torch CPU + onnx + onnxsim) で実行する:
  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe scripts/npu/export_animevideov3.py

出力: tmp/npu-anime/animevideov3_nchw_256x256_fp32.onnx
入力仕様は既存の realesrgan NPU モデルに合わせる:
  float32 NCHW 1x3x256x256, RGB, [0,1] → 出力 1x3x1024x1024
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "tmp" / "npu-anime" / "realesr-animevideov3.pth"
OUT_DIR = REPO / "tmp" / "npu-anime"
TILE = int(sys.argv[sys.argv.index("--tile") + 1]) if "--tile" in sys.argv else 256
# --size WxH で非正方形の固定入力（例: 854x480 = フレームぴったり・タイル分割不要）
if "--size" in sys.argv:
    _w, _h = sys.argv[sys.argv.index("--size") + 1].lower().split("x")
    IN_W, IN_H = int(_w), int(_h)
else:
    IN_W = IN_H = TILE
SCALE = 4
# VAIML bf16 は実重みの PReLU で誤コンパイルするため、等価分解して回避する
# （PReLU(x) = ReLU(x) - w * ReLU(-x)。詳細は docs/npu-research.md）
DECOMPOSE_PRELU = "--decompose-prelu" in sys.argv


class SRVGGNetCompact(nn.Module):
    """BasicSR の SRVGGNetCompact 互換実装（realesr-animevideov3 用）。"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16,
                 upscale=4, act_type="prelu") -> None:
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._act(act_type, num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._act(act_type, num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale ** 2, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _act(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        if act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError(f"unsupported act_type: {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


class DecomposedPReLU(nn.Module):
    """PReLU(x) = ReLU(x) - w * ReLU(-x)（数学的等価・PReLUカーネル回避）。"""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.view(1, -1, 1, 1)
        return F.relu(x) - w * F.relu(-x)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not WEIGHTS.exists():
        print(f"重みがありません: {WEIGHTS}")
        return 1

    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    params = state.get("params", state)
    model = SRVGGNetCompact(upscale=SCALE)
    model.load_state_dict(params, strict=True)
    if DECOMPOSE_PRELU:
        for i, layer in enumerate(model.body):
            if isinstance(layer, nn.PReLU):
                model.body[i] = DecomposedPReLU(layer.weight.data)
        print("PReLU を分解形に置換")
    model.eval()
    print(f"重み読込OK: {len(params)} tensors")

    prefix = "animevideov3dp" if DECOMPOSE_PRELU else "animevideov3"
    dummy = torch.rand(1, 3, IN_H, IN_W, dtype=torch.float32)
    raw_path = OUT_DIR / f"{prefix}_nchw_{IN_W}x{IN_H}_fp32_raw.onnx"
    out_path = OUT_DIR / f"{prefix}_nchw_{IN_W}x{IN_H}_fp32.onnx"

    torch.onnx.export(
        model, dummy, str(raw_path),
        input_names=["input"], output_names=["output"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes=None,  # NPU 向けに完全固定形状
    )
    print(f"ONNX出力: {raw_path.name}")

    import onnx
    from onnxsim import simplify
    simplified, ok = simplify(onnx.load(str(raw_path)))
    if not ok:
        print("onnxsim簡略化に失敗（rawをそのまま使用）")
        out_path.write_bytes(raw_path.read_bytes())
    else:
        onnx.save(simplified, str(out_path))
        print(f"onnxsim簡略化OK: {out_path.name}")

    # 検証: torch と onnxruntime(CPU) の出力一致
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, IN_H, IN_W).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"input": x})[0]
    diff = float(np.abs(ref - got).max())
    print(f"出力形状: {got.shape}, torch/ORT 最大差: {diff:.3e}")
    if got.shape != (1, 3, IN_H * SCALE, IN_W * SCALE) or diff > 1e-4:
        print("検証NG")
        return 1
    print("検証OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

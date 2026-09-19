"""RealESRGAN_x4plus_anime_6B (RRDBNet) を NPU 用の固定形状 ONNX に出力する。

Ryzen AI conda 環境で実行:
  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe scripts/npu/export_x4plus_anime.py

出力: tmp/npu-anime/x4plus_anime_nchw_256x256_fp32.onnx
入力仕様は既存NPUモデルと同じ: float32 NCHW 1x3x256x256, RGB, [0,1]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "tmp" / "npu-anime" / "RealESRGAN_x4plus_anime_6B.pth"
OUT_DIR = REPO / "tmp" / "npu-anime"
TILE = 256
SCALE = 4


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """basicsr 互換 RRDBNet（x4plus_anime_6B: num_block=6）。"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6,
                 num_grow_ch=32) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    params = state.get("params_ema", state.get("params", state))
    model = RRDBNet()
    model.load_state_dict(params, strict=True)
    model.eval()
    print(f"重み読込OK: {len(params)} tensors")

    dummy = torch.rand(1, 3, TILE, TILE, dtype=torch.float32)
    raw_path = OUT_DIR / f"x4plus_anime_nchw_{TILE}x{TILE}_fp32_raw.onnx"
    out_path = OUT_DIR / f"x4plus_anime_nchw_{TILE}x{TILE}_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(raw_path),
            input_names=["input"], output_names=["output"],
            opset_version=17, do_constant_folding=True,
        )
    print(f"ONNX出力: {raw_path.name}")

    import onnx
    from onnxsim import simplify
    simplified, ok = simplify(onnx.load(str(raw_path)))
    if ok:
        onnx.save(simplified, str(out_path))
        print(f"onnxsim簡略化OK: {out_path.name}")
    else:
        out_path.write_bytes(raw_path.read_bytes())
        print("onnxsim失敗（rawを使用）")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, TILE, TILE).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"input": x})[0]
    diff = float(np.abs(ref - got).max())
    print(f"出力形状: {got.shape}, torch/ORT 最大差: {diff:.3e}")
    if got.shape != (1, 3, TILE * SCALE, TILE * SCALE) or diff > 1e-3:
        print("検証NG")
        return 1
    print("検証OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

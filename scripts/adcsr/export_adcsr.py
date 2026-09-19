"""AdcSRを固定形状ONNXへexportし、Torch/ORTの一致をPSNRで検証する。

環境例（ネットワーク接続または事前キャッシュが必要）::
    python -m venv .venv
    .venv\\Scripts\\python -m pip install torch==2.4.1+cpu diffusers==0.32.2 transformers==4.37.2 peft==0.13.2 onnx onnxsim onnxruntime

--repo は AdcSR clone、--sd はローカルの Stable Diffusion 2.1-base、
--weights は net_params_200.pkl、--half-decoder は halfDecoder.ckpt を指定する。
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.autoencoders.vae import Decoder


class AdcSRWrapper(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        lr = x * 2 - 1
        sr = self.core(lr)
        sr = (sr - sr.mean(dim=(2, 3), keepdim=True)) / sr.std(dim=(2, 3), keepdim=True)
        sr = sr * lr.std(dim=(2, 3), keepdim=True) + lr.mean(dim=(2, 3), keepdim=True)
        return (sr / 2 + 0.5).clamp(0, 1)


def build(repo: Path, sd: Path, weights: Path, half_decoder: Path):
    sys.path.insert(0, str(repo))
    from model import Net

    pipe = StableDiffusionPipeline.from_pretrained(str(sd), local_files_only=True).to("cpu")
    decoder = Decoder(in_channels=4, out_channels=3,
                      up_block_types=["UpDecoderBlock2D"] * 4,
                      block_out_channels=[64, 128, 256, 256], layers_per_block=2,
                      norm_num_groups=32, act_fn="silu", norm_type="group",
                      mid_block_add_attention=True)
    checkpoint = torch.load(half_decoder, map_location="cpu", weights_only=False)
    decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in checkpoint["state_dict"].items() if "decoder" in k})
    model = torch.nn.DataParallel(Net(pipe.unet, copy.deepcopy(decoder)))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False))
    return torch.nn.Sequential(model.module, *decoder.up_blocks,
                               decoder.conv_norm_out, decoder.conv_act,
                               decoder.conv_out).eval()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--sd", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--half-decoder", type=Path, required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.size != 128:
        p.error("この統合用exportは --size 128 のみ対応します")
    wrapped = AdcSRWrapper(build(args.repo, args.sd, args.weights, args.half_decoder))
    x = torch.zeros(1, 3, args.size, args.size, dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapped, x, str(args.out), opset_version=17,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes=None, do_constant_folding=True)
    import onnx
    model = onnx.load(str(args.out))
    onnx.checker.check_model(model)
    try:
        import onnxsim
        simplified, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(simplified, str(args.out))
    except Exception as exc:
        print(f"onnxsim skipped: {exc}")
    with torch.inference_mode():
        torch_out = wrapped(x).numpy()
    ort_out = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"]).run(["output"], {"input": x.numpy()})[0]
    value = psnr(torch_out, ort_out)
    print(f"exported={args.out} shape={tuple(ort_out.shape)} psnr={value:.6f} dB")
    if value < 50:
        raise RuntimeError(f"Torch/ORT PSNRが低すぎます: {value:.6f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

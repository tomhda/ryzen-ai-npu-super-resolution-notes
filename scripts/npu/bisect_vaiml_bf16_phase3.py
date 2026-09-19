"""VAIML bf16 誤コンパイル phase3: onnxsim か実重みかを切り分ける。

  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe scripts/npu/bisect_vaiml_bf16_phase3.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "tmp" / "npu-anime"
WORK = BASE / "bisect"
TILE = 256


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

    # 合成16層に onnxsim を掛けた版を用意
    import onnx
    from onnxsim import simplify
    syn_raw = WORK / "srvgg16_prelu_fp32.onnx"
    syn_sim = WORK / "srvgg16sim_fp32.onnx"
    model, ok = simplify(onnx.load(str(syn_raw)))
    if not ok:
        print("onnxsim失敗")
        return 1
    onnx.save(model, str(syn_sim))

    cases = [
        # (名前, fp32パス) — bf16はここで都度生成
        ("real_raw", BASE / "animevideov3_nchw_256x256_fp32_raw.onnx"),
        ("real_sim", BASE / "animevideov3_nchw_256x256_fp32.onnx"),  # 既知BROKENの対照
        ("syn16_sim", syn_sim),
    ]
    results = []
    for name, fp32 in cases:
        if not fp32.exists():
            results.append(f"{name}: SKIP (missing {fp32.name})")
            continue
        bf16 = WORK / f"{name}_bf16.onnx"
        to_bf16(fp32, bf16)
        print(f"--- {name}: コンパイル開始", flush=True)
        try:
            line = evaluate(name, fp32, bf16)
        except Exception as exc:  # noqa: BLE001
            line = f"{name}: ERROR {type(exc).__name__}: {exc}"
        print(line, flush=True)
        results.append(line)

    print("===== phase3 結果まとめ =====")
    for line in results:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

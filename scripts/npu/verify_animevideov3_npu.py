"""量子化済み animevideov3 を実機NPUでコンパイル・検証する。

Ryzen AI conda 環境でリポジトリルートから実行:
  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe scripts/npu/verify_animevideov3_npu.py

やること:
  1. vendor/amd-npu に置いた u8s8 ONNX で VitisAI セッション生成（初回はコンパイル数分）
  2. vitisai_ep_report.json で NPU/CPU ノード割当を確認（PReLUがCPU落ちしていないか）
  3. fp32 CPU出力との PSNR（量子化+NPU誤差）
  4. タイル毎の推論時間
  5. 目視用の比較画像を tmp/npu-anime/ab/ に保存
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[2]
FP32 = REPO / "tmp" / "npu-anime" / "animevideov3_nchw_256x256_fp32.onnx"
U8S8 = REPO / "vendor" / "amd-npu" / "onnx-models" / "animevideov3_nchw_256x256_u8s8.onnx"
CACHE_DIR = REPO / "vendor" / "amd-npu"
CACHE_KEY = "modelcachekey_animevideov3_nchw_256x256_u8s8"
CALIB_DIR = REPO / "tmp" / "npu-anime" / "calib"
AB_DIR = REPO / "tmp" / "npu-anime" / "ab"
N_QUALITY = 20
N_TIMING = 30


def _load(p: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    x = img[:, :, ::-1].astype("float32") / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


def _to_img(y: np.ndarray) -> np.ndarray:
    out = (y[0].transpose(1, 2, 0)[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for p in (FP32, U8S8):
        if not p.exists():
            print(f"モデルがありません: {p}")
            return 1

    print("[1/5] VitisAI セッション生成（初回はコンパイルで数分かかる）…")
    t0 = time.perf_counter()
    npu = ort.InferenceSession(
        str(U8S8),
        providers=["VitisAIExecutionProvider"],
        provider_options=[{
            "cache_dir": str(CACHE_DIR),
            "cache_key": CACHE_KEY,
            "enable_cache_file_io_in_mem": 0,
        }],
    )
    print(f"  セッション生成 {time.perf_counter() - t0:.1f}s")
    cpu = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
    in_npu = npu.get_inputs()[0].name
    in_cpu = cpu.get_inputs()[0].name

    print("[2/5] ノード割当レポート…")
    report_found = False
    for cand in (CACHE_DIR / CACHE_KEY / "vitisai_ep_report.json",
                 CACHE_DIR / "vitisai_ep_report.json"):
        if cand.exists():
            report = json.loads(cand.read_text(encoding="utf-8"))
            report_found = True
            devs = {}
            try:
                for stat in report["deviceStat"]:
                    devs[stat.get("name", "?")] = (
                        stat.get("nodeNum", 0), stat.get("supportedOpType", []))
            except (KeyError, TypeError):
                print(f"  レポート形式が想定外: {cand}")
                print(json.dumps(report, ensure_ascii=False)[:1500])
                break
            for name, (num, ops) in devs.items():
                print(f"  {name}: {num} nodes  {sorted(set(ops))[:12]}")
            cpu_nodes = devs.get("CPU", (0, []))[0]
            all_nodes = sum(n for n, _ in devs.values())
            print(f"  → NPUオフロード率: {(all_nodes - cpu_nodes)}/{all_nodes}")
            break
    if not report_found:
        print("  レポートJSONが見つからず（割当はタイル時間から間接判断）")

    print(f"[3/5] 画質: fp32(CPU) vs u8s8(NPU) PSNR ({N_QUALITY} patches)…")
    patches = sorted(CALIB_DIR.glob("*.png"))[-N_QUALITY:]  # 校正に使ってない側から
    psnrs = []
    for p in patches:
        x = _load(p)
        ref = _to_img(cpu.run(None, {in_cpu: x})[0])
        got = _to_img(npu.run(None, {in_npu: x})[0])
        psnrs.append(_psnr(ref, got))
    print(f"  PSNR: mean {np.mean(psnrs):.2f} dB / min {np.min(psnrs):.2f} dB")

    print(f"[4/5] 速度: NPUタイル推論 x{N_TIMING}…")
    x = _load(patches[0])
    for _ in range(3):
        npu.run(None, {in_npu: x})  # warmup
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        npu.run(None, {in_npu: x})
    dt = (time.perf_counter() - t0) / N_TIMING
    print(f"  {dt * 1000:.1f} ms/タイル ≒ {1 / dt:.1f} タイル/秒")

    print("[5/5] 目視用A/B画像を保存…")
    AB_DIR.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(patches[:3]):
        x = _load(p)
        ref = _to_img(cpu.run(None, {in_cpu: x})[0])
        got = _to_img(npu.run(None, {in_npu: x})[0])
        side = np.concatenate([ref, got], axis=1)
        ok, enc = cv2.imencode(".png", side)
        if ok:
            enc.tofile(str(AB_DIR / f"ab_{i}_left-fp32_right-npu.png"))
    print(f"  {AB_DIR} に保存（左=fp32基準 / 右=NPU int8）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

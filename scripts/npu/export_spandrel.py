"""spandrel の画像超解像モデルを NPU 用の固定形状 ONNX に出力する。

実行例:
  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe \
    scripts/npu/export_spandrel.py \
    --weights tmp/npu-anime/span/2x_ModernSpanimationV1.pth \
    --name modernspan --tile 256

入力は spandrel の ImageModelDescriptor が想定する RGB [0,1] NCHW
float32 とし、NPU 向けに空間次元を完全固定する。出力は
tmp/npu-anime/span/{name}_nchw_{W}x{H}_fp32.onnx。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from spandrel import ImageModelDescriptor, ModelLoader


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "tmp" / "npu-anime" / "span"


class ModelOutputAdapter(nn.Module):
    """ImageModelDescriptor.model を ONNX の単一 Tensor 出力へ適合させる。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"モデル出力がTensorではありません: {type(output)!r}")
        return output


# ⚠️ SiLU→x*sigmoid(x) のモジュール置換は行わないこと。
# spandrelのSPANは nn.SiLU(inplace=True) のインプレース副作用に依存しており、
# 数学的に等価な非インプレース置換でもデータフローが変わり出力が崩壊する
# （purephoto実測でPSNR22dB）。SiLUはONNXエクスポータが正しく展開する。


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--size は WxH 形式で指定してください") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--size の幅・高さは正数で指定してください")
    return width, height


def _rewrite_slice_nonneg(model):
    """負のSlice starts/endsリテラルを正の等価値に置換する（VAIML crash回避）。

    VitisAI(RyzenAI SW 1.8.0)のVAIMLコンパイラは「負のstarts/endsを持つ
    Slice2本＋Concat」（torch.rollのexport形）をONNX→MLIR loweringの
    assertionで落とす（2026-08-29切り分け、最小再現は3ノード）。
    固定形状exportなので負値は常に正の等価値へ確定でき、数値はbit-exact、
    ノード数・順序・op種も不変。他バックエンドにも無害なので常時適用する。
    戻り値は (書換済みmodel, 書換箇所数)。書換対象が無ければ None。
    """
    import onnx
    from onnx import numpy_helper, shape_inference

    try:
        model = shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:  # noqa: BLE001 - shape推論はベストエフォート
        pass
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    shapes: dict[str, list[int]] = {}
    for collection in (graph.value_info, graph.input, graph.output):
        for item in collection:
            proto_dims = item.type.tensor_type.shape.dim
            dims = [d.dim_value for d in proto_dims if d.HasField("dim_value")]
            if dims and len(dims) == len(proto_dims):
                shapes[item.name] = dims

    new_inits: dict[str, object] = {}
    count = 0
    for node in graph.node:
        if node.op_type != "Slice" or len(node.input) < 4:
            continue
        axes_init = inits.get(node.input[3])
        dims = shapes.get(node.input[0])
        if axes_init is None or dims is None:
            continue
        axes = numpy_helper.to_array(axes_init).reshape(-1)
        for position in (1, 2):  # starts, ends
            init = inits.get(node.input[position])
            if init is None:
                continue
            values = numpy_helper.to_array(init).reshape(-1).astype(np.int64)
            if not (values < 0).any():
                continue
            fixed = values.copy()
            ok = True
            for index, value in enumerate(values):
                if value >= 0:
                    continue
                axis = int(axes[index])
                if axis >= len(dims):
                    ok = False
                    break
                fixed[index] = max(0, min(int(dims[axis]), int(value) + int(dims[axis])))
            if not ok:
                continue
            name = f"{node.input[position]}__nonneg_{'_'.join(str(int(v)) for v in fixed)}"
            if name not in new_inits:
                new_inits[name] = numpy_helper.from_array(fixed, name=name)
            node.input[position] = name
            count += 1

    if count == 0:
        return None
    graph.initializer.extend(new_inits.values())
    used = {name for node in graph.node for name in node.input}
    keep = [t for t in graph.initializer if t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(keep)
    return model, count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--name", required=True, help="出力ONNXのプレフィックス")
    parser.add_argument("--tile", type=int, choices=(256, 512), default=256)
    parser.add_argument("--size", type=parse_size, help="固定入力サイズ WxH（指定時は--tileより優先）")
    return parser


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()

    weights = args.weights if args.weights.is_absolute() else REPO / args.weights
    weights = weights.resolve()
    if not weights.exists():
        print(f"重みがありません: {weights}")
        return 1

    in_w, in_h = args.size if args.size else (args.tile, args.tile)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"{args.name}_nchw_{in_w}x{in_h}_fp32_raw.onnx"
    out_path = OUT_DIR / f"{args.name}_nchw_{in_w}x{in_h}_fp32.onnx"

    print(f"重み: {weights}")
    print(f"入力サイズ: 1x3x{in_h}x{in_w}, RGB [0,1] NCHW, float32")

    try:
        descriptor = ModelLoader(device="cpu").load_from_file(weights)
    except Exception as exc:  # noqa: BLE001 - 個別モデルの失敗を呼出側で記録する
        print(f"spandrelロード失敗: {type(exc).__name__}: {exc}")
        return 1

    if not isinstance(descriptor, ImageModelDescriptor):
        print(f"ImageModelDescriptorではありません: {type(descriptor).__name__}")
        return 1
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        print(
            "RGB 3chモデルではありません: "
            f"input={descriptor.input_channels}, output={descriptor.output_channels}"
        )
        return 1

    # ImageModelDescriptor の SRモデルは RGB [0,1] の NCHW Tensor を受ける。
    # この2重量みでは正規化・NHWC変換は不要なので、modelをそのままアダプタへ渡す。
    # SiLUだけは、ネイティブTorch実装とORTのSigmoid/Mul展開で極端値の丸め差が
    # 最大差へ増幅されるため、意味を変えない明示演算へ先に置き換える。
    model = ModelOutputAdapter(descriptor.model).eval()
    scale = int(descriptor.scale)
    print(
        "モデル: "
        f"{descriptor.architecture.name if hasattr(descriptor.architecture, 'name') else descriptor.architecture}, "
        f"scale={scale}, tags={descriptor.tags}, "
        "入力アダプタ=RGB[0,1] NCHW identity"
    )

    # ⚠️ export前のforwardに inference_mode を使ってはいけない。
    # SPAN系(Conv3XC)はforward中に update_params() で .data 直接代入するため、
    # inference_mode製テンソルがモジュールへ混入し、直後のONNXトレースが
    # 「見た目は成功・出力は誤り(実画像でPSNR22dB)」の壊れたグラフになる。
    # ここでは no_grad のウォームアップのみ行い、参照出力は export後に
    # 「別途フレッシュにロードしたモデル」から取る（トレースで変異した
    # 同一オブジェクトと比較すると壊れを見逃すため）。
    # 検証入力は0.5中心の低振幅ノイズ。SPAN系は白色ノイズで再パラメータ化の
    # 数値差が増幅され、正しいグラフでも max_diff が1e-4を超えるため
    # （実画像ではPSNR 139dBで一致することを確認済み）。壊れたグラフは
    # フレッシュ参照との比較なら低振幅でも大差になるので検出できる。
    rng = np.random.default_rng(12345)
    x_np = np.clip(0.5 + 0.05 * rng.standard_normal((1, 3, in_h, in_w)), 0.0, 1.0).astype(np.float32)
    x = torch.from_numpy(x_np)
    with torch.no_grad():
        warm = model(x)
    expected_shape = (1, 3, in_h * scale, in_w * scale)
    if tuple(warm.shape) != expected_shape:
        print(f"Torch出力形状NG: got={tuple(warm.shape)}, expected={expected_shape}")
        return 1
    del warm
    print(f"入力仕様確認OK: 出力形状={expected_shape}")

    try:
        torch.onnx.export(
            model,
            x,
            str(raw_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    except Exception as exc:  # noqa: BLE001 - モデル単位で失敗扱いにする
        print(f"ONNX export失敗: {type(exc).__name__}: {exc}")
        return 1
    print(f"ONNX raw出力: {raw_path.name}")

    import onnx
    from onnxsim import simplify

    try:
        simplified, ok = simplify(onnx.load(str(raw_path)))
    except Exception as exc:  # noqa: BLE001 - rawをフォールバックにする
        print(f"onnxsim例外（rawを使用）: {type(exc).__name__}: {exc}")
        out_path.write_bytes(raw_path.read_bytes())
    else:
        if not ok:
            print("onnxsim簡略化NG（rawを使用）")
            out_path.write_bytes(raw_path.read_bytes())
        else:
            onnx.save(simplified, str(out_path))
            print(f"onnxsim簡略化OK: {out_path.name}")

    # VAIML対策の負Slice境界の正値化（適用理由は_rewrite_slice_nonnegのdocstring参照）。
    # この後のORT検証は書換済みグラフに対して行われる。
    rewritten = _rewrite_slice_nonneg(onnx.load(str(out_path)))
    if rewritten is not None:
        fixed_model, fixed_count = rewritten
        onnx.save(fixed_model, str(out_path))
        print(f"VAIML-safe化: 負のSlice境界 {fixed_count}箇所を正値へ書換")

    try:
        import onnxruntime as ort

        # 参照はフレッシュなモデルインスタンスから取得（上のコメント参照）
        fresh = ModelLoader(device="cpu").load_from_file(weights)
        fresh_model = ModelOutputAdapter(fresh.model).eval()
        with torch.no_grad():
            fresh_model(x)  # 再パラメータ化の状態安定用
            reference = fresh_model(x).cpu().numpy()

        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        got = session.run(None, {input_name: x_np})[0]
    except Exception as exc:  # noqa: BLE001 - モデル単位で失敗扱いにする
        print(f"ORT検証実行失敗: {type(exc).__name__}: {exc}")
        return 1

    diff = float(np.abs(reference - got).max())
    print(f"ORT出力形状: {tuple(got.shape)}, torch(fresh)/ORT最大差: {diff:.3e}")
    if tuple(got.shape) != expected_shape or diff > 1e-4:
        print(f"検証NG: expected_shape={expected_shape}, max_diff<=1e-4")
        return 1

    print(f"検証OK: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""ONNX 末尾の DepthToSpace 以降を切り離し、NPU 用 body と tail マニフェストを作る。

対応する末尾（どちらも DepthToSpace 1 個が条件）:
  (a) ``... → DepthToSpace(blocksize=r, mode) → output``（後段なし）
  (b) ``... → DepthToSpace ─┐
       input → Resize(nearest, 整数倍) ─┴→ Add → output``（Anime Video v3 形）

量子化ラッパー（Cast / QuantizeLinear / DequantizeLinear）の線形挿入は許す。
Clip を含む末尾や上記以外の構造は「非対応」で終了する（終了コード 2）。

切り出し後、CPU（ORT CPU EP）で ``body ＋ numpy 後処理 == 元モデル出力`` を
必ず検証する（max abs diff < 1e-5）。外れたらエラー（終了コード 1）。
bf16 roundtrip の Cast 挿入がある末尾は、bf16 丸めを再現した上で比較する。

使い方 (cwd=リポジトリ直下):
  .venv\\Scripts\\python.exe scripts\\npu\\split_tail.py
    --input models/ai/purephoto_nchw_512x512_fp32.onnx
    --body models/ai/purephoto_body_nchw_512x512_fp32.onnx
    --manifest models/ai/purephoto_body_nchw_512x512.tail.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import extract_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "npu-serve"))
from npu_tail import TAIL_MANIFEST_KIND, TAIL_MANIFEST_VERSION, tail_postprocess  # noqa: E402

#: tail 経路に挿入されていてよい量子化ラッパー。
WRAPPER_OPS = frozenset({"Cast", "QuantizeLinear", "DequantizeLinear"})

#: bf16 roundtrip として許す Cast の行き先（1=FLOAT、16=BFLOAT16）。
BF16_ROUNDTRIP_TARGETS = frozenset({1, 16})

VERIFY_TOL = 1e-5


class UnsupportedTail(Exception):
    """切断に非対応の末尾構造。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_attr(node: onnx.NodeProto, name: str) -> int | None:
    for attr in node.attribute:
        if attr.name == name and attr.type == onnx.AttributeProto.INT:
            return int(attr.i)
    return None


def _bytes_attr(node: onnx.NodeProto, name: str, default: bytes) -> bytes:
    for attr in node.attribute:
        if attr.name == name and attr.type == onnx.AttributeProto.STRING:
            return bytes(attr.s)
    return default


def _initializer_floats(model: onnx.ModelProto, name: str) -> np.ndarray | None:
    for init in model.graph.initializer:
        if init.name == name:
            try:
                return np.asarray(onnx.numpy_helper.to_array(init), dtype=np.float64)
            except Exception:
                return None
    return None


def bf16_round(a: np.ndarray) -> np.ndarray:
    """float32 配列を bf16 へ最近傍偶数丸めした値を float32 で返す。"""
    arr = np.ascontiguousarray(a, dtype=np.float32)
    u = arr.view(np.uint32)
    u = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    u = u & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def _consumer_counts(graph: onnx.GraphProto) -> Counter[str]:
    counts: Counter[str] = Counter()
    for node in graph.node:
        for name in node.input:
            if name:
                counts[name] += 1
    return counts


def _node_by_output(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    table: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for name in node.output:
            table[name] = node
    return table


def _follow_wrappers(
    graph: onnx.GraphProto,
    start: str,
    consumers: Counter[str],
) -> tuple[str, list[onnx.NodeProto]]:
    """start から出力方向へ量子化ラッパーの線形連鎖をたどる。

    分岐（consumer 数 != 1）やラッパー以外の op に当たったらそこで止め、
    末端テンソルと連鎖を返す。tail 専有の経路用（分岐は呼び出し側で拒否する）。
    """
    chain: list[onnx.NodeProto] = []
    current = start
    while True:
        if consumers.get(current, 0) != 1:
            return current, chain
        user = None
        for node in graph.node:
            if current in node.input:
                user = node
                break
        if user is None or user.op_type not in WRAPPER_OPS:
            return current, chain
        if len(user.output) != 1 or len(user.input) != 1:
            raise UnsupportedTail(f"wrapper {user.name} の入出力が 1 本でない")
        chain.append(user)
        current = user.output[0]
    # 到達しない


def _trace_backward(
    tensor: str,
    nodes: dict[str, onnx.NodeProto],
) -> tuple[str, list[onnx.NodeProto]]:
    """tensor から生産者方向へ量子化ラッパーをさかのぼり、始端と連鎖を返す。

    入力側の共有（body と同じテンソルを読む分岐）は許すため consumer 数は見ない。
    呼び出し側が連鎖の内容（Cast 行き先など）を検証する。
    """
    chain: list[onnx.NodeProto] = []
    current = tensor
    while current in nodes and nodes[current].op_type in WRAPPER_OPS:
        node = nodes[current]
        if len(node.output) != 1 or len(node.input) != 1:
            raise UnsupportedTail(f"wrapper {node.name} の入出力が 1 本でない")
        chain.append(node)
        current = node.input[0]
    return current, chain


def analyze_tail(model: onnx.ModelProto) -> dict:
    """末尾構造を調べて切断情報を返す。非対応なら UnsupportedTail。"""
    graph = model.graph
    if len(graph.output) != 1:
        raise UnsupportedTail(f"グラフ出力が {len(graph.output)} 個")
    if len(graph.input) != 1:
        raise UnsupportedTail(f"グラフ入力が {len(graph.input)} 個")
    model_input = graph.input[0].name
    graph_output = graph.output[0].name

    depth_nodes = [n for n in graph.node if n.op_type == "DepthToSpace"]
    if len(depth_nodes) != 1:
        raise UnsupportedTail(f"DepthToSpace が {len(depth_nodes)} 個")
    dts = depth_nodes[0]
    blocksize = _int_attr(dts, "blocksize")
    if blocksize is None or blocksize <= 0:
        raise UnsupportedTail("DepthToSpace に blocksize がない")
    mode = _bytes_attr(dts, "mode", b"DCR").decode("ascii")
    if mode not in ("CRD", "DCR"):
        raise UnsupportedTail(f"DepthToSpace の mode が {mode}")
    if len(dts.input) != 1 or len(dts.output) != 1:
        raise UnsupportedTail("DepthToSpace の入出力が 1 本でない")
    cut = dts.input[0]
    if cut == model_input:
        raise UnsupportedTail("DepthToSpace の入力がモデル入力そのもの")

    consumers = _consumer_counts(graph)
    nodes = _node_by_output(graph)
    if cut not in nodes:
        raise UnsupportedTail(f"切断テンソル {cut} を作るノードがない")
    dts_consumers = consumers.get(dts.output[0], 0)
    if dts_consumers not in (0, 1):
        raise UnsupportedTail("DepthToSpace の出力に分岐がある")
    if dts_consumers == 0 and dts.output[0] != graph_output:
        raise UnsupportedTail("DepthToSpace の出力がどこにもつながっていない")

    wrappers: list[onnx.NodeProto] = []
    # DepthToSpace の出力側の連鎖をたどる。
    endpoint, chain = _follow_wrappers(graph, dts.output[0], consumers)
    wrappers.extend(chain)

    add_nearest_input = False
    resize_input_quantized = False
    if endpoint == graph_output:
        # (a) 後段なし。endpoint までの連鎖がそのまま出力になる。
        pass
    else:
        # endpoint を消費するノードが後段である（生産者ではない）。
        endpoint_users = [node for node in graph.node if endpoint in node.input]
        if len(endpoint_users) != 1:
            raise UnsupportedTail(f"DepthToSpace 系の末端 {endpoint} の消費者が {len(endpoint_users)} 個")
        endpoint_node = endpoint_users[0]
        if endpoint_node.op_type != "Add":
            raise UnsupportedTail(
                f"DepthToSpace の後段が {endpoint_node.op_type} で非対応"
            )
    if endpoint != graph_output:
        # (b) Add 形。endpoint の唯一の消費者は上で Add と確認済み。
        endpoint_node = [node for node in graph.node if endpoint in node.input][0]
        if len(endpoint_node.input) != 2:
            raise UnsupportedTail("Add の入力が 2 本でない")
        other = endpoint_node.input[1] if endpoint_node.input[0] == endpoint else (
            endpoint_node.input[0] if endpoint_node.input[1] == endpoint else None
        )
        if other is None:
            raise UnsupportedTail("Add が DepthToSpace 系とつながっていない")
        # Add の相手側を生産者方向へさかのぼり、Resize を見つける。
        branch_src, branch_chain = _trace_backward(other, nodes)
        branch_node = nodes.get(branch_src)
        if branch_node is None or branch_node.op_type != "Resize":
            raise UnsupportedTail("Add の相手が Resize 分岐でない")
        # Resize 出力側のラッパーは tail 専有（分岐なし）であること。
        for wrapper in branch_chain:
            if consumers.get(wrapper.output[0], 0) != 1:
                raise UnsupportedTail("Resize 分岐に共有がある")
        wrappers.extend(branch_chain)
        resize_attrs = {a.name: a for a in branch_node.attribute}
        rmode = resize_attrs.get("mode")
        if rmode is None or rmode.type != onnx.AttributeProto.STRING or bytes(rmode.s) != b"nearest":
            raise UnsupportedTail("Resize が nearest でない")
        if len(branch_node.input) < 3 or not branch_node.input[2]:
            raise UnsupportedTail("Resize の scales 入力がない")
        scales = _initializer_floats(model, branch_node.input[2])
        if scales is None or scales.shape != (4,):
            raise UnsupportedTail("Resize の scales が定数 [1,1,s,s] でない")
        if not (scales[0] == 1 and scales[1] == 1):
            raise UnsupportedTail(f"Resize の scales が {scales}（N/C 方向が 1 でない）")
        if not (scales[2] == scales[3] == float(blocksize)):
            raise UnsupportedTail(
                f"Resize の倍率 {scales[2:]} が blocksize {blocksize} と不一致"
            )
        # Resize の入力は（ラッパー越しに）モデル入力であること。
        # 入力側の共有（body と同じテンソルを読む）は許す。
        resize_src, resize_chain = _trace_backward(branch_node.input[0], nodes)
        if resize_src != model_input:
            raise UnsupportedTail("Resize の入力がモデル入力でない")
        resize_targets = _cast_targets(resize_chain)
        if resize_targets is None or not resize_targets <= BF16_ROUNDTRIP_TARGETS:
            raise UnsupportedTail(
                f"Resize 入力側に再現できないラッパー（Cast 行き先 {resize_targets}）"
            )
        resize_input_quantized = bool(resize_chain)
        if consumers.get(branch_node.output[0], 0) != 1:
            raise UnsupportedTail("Resize の出力に分岐がある")
        # Add の出力側の連鎖をたどる（出力直結の consumer 数 0 を許す）。
        add_consumers = consumers.get(endpoint_node.output[0], 0)
        if add_consumers not in (0, 1):
            raise UnsupportedTail("Add の出力に分岐がある")
        if add_consumers == 0 and endpoint_node.output[0] != graph_output:
            raise UnsupportedTail("Add の出力がどこにもつながっていない")
        tail_end, tail_chain = _follow_wrappers(graph, endpoint_node.output[0], consumers)
        wrappers.extend(tail_chain)
        if tail_end != graph_output:
            raise UnsupportedTail("Add 以降が出力まで直結していない")
        add_nearest_input = True

    # tail 経路上の Clip を拒否する（後処理を numpy で再現できないため）。
    # _follow_wrappers / _trace_backward は WRAPPER_OPS しか集めないので、
    # ここは将来の収集漏れへの念押しである。
    for node in wrappers:
        if node.op_type == "Clip":
            raise UnsupportedTail("tail 経路に Clip がある")
    for node in wrappers:
        if node.op_type not in WRAPPER_OPS:
            raise UnsupportedTail(f"tail 経路に {node.op_type} がある")

    quantized = any(True for _ in wrappers)
    return {
        "blocksize": blocksize,
        "mode": mode,
        "cut": cut,
        "add_nearest_input": add_nearest_input,
        "scale": blocksize,
        "model_input": model_input,
        "graph_output": graph_output,
        "wrappers": wrappers,
        "quantized": quantized,
        "resize_input_quantized": resize_input_quantized,
    }


def _cast_targets(wrappers: list[onnx.NodeProto]) -> set[int] | None:
    """ラッパーがすべて Cast なら行き先の集合、違う op があれば None。"""
    targets: set[int] = set()
    for node in wrappers:
        if node.op_type != "Cast":
            return None
        to = _int_attr(node, "to")
        if to is None:
            return None
        targets.add(to)
    return targets


def _run_cpu(model_path: Path, feed: dict[str, np.ndarray]) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, feed)[0]


def verify(
    full_path: Path,
    body_path: Path,
    info: dict,
    manifest: dict,
    test_input: np.ndarray | None = None,
) -> float:
    """body＋numpy 後処理が元モデル出力と一致することを検証し、max abs diff を返す。"""
    from npu_tail import pixel_shuffle  # noqa: E402

    model_input = info["model_input"]
    full_model = onnx.load(str(full_path))
    dims = full_model.graph.input[0].type.tensor_type.shape.dim
    shape = [d.dim_value for d in dims]
    if any(v <= 0 for v in shape):
        raise UnsupportedTail(f"モデル入力が動的形状 {shape}")
    if test_input is None:
        rng = np.random.RandomState(0)
        test_input = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
    test_input = np.ascontiguousarray(test_input, dtype=np.float32)
    if list(test_input.shape) != shape:
        raise ValueError(f"テスト入力 {test_input.shape} がモデル入力 {shape} と不一致")

    ref = _run_cpu(full_path, {model_input: test_input})
    bout = _run_cpu(body_path, {model_input: test_input})

    if not info["quantized"]:
        rec = tail_postprocess(
            bout, test_input if manifest["add_nearest_input"] else None, manifest
        )
        if rec.shape != ref.shape:
            raise ValueError(f"再現出力 {rec.shape} が元出力 {ref.shape} と不一致")
        diff = float(np.abs(rec - ref).max())
        if diff >= VERIFY_TOL:
            raise ValueError(f"CPU 一致検証に失敗: max abs diff={diff:.3e}（許容 {VERIFY_TOL}）")
        return diff

    # 量子化あり: 切断以降のラッパーは bf16 roundtrip の Cast 挿入だけを許し、
    # 丸めを numpy で再現した上で比較する。
    targets = _cast_targets(info["wrappers"])
    if targets is None or not targets <= BF16_ROUNDTRIP_TARGETS:
        raise UnsupportedTail(f"再現できない量子化ラッパー（Cast 行き先 {targets}）")
    rounds = 16 in targets
    if rounds and not np.array_equal(bf16_round(bout), bout):
        raise UnsupportedTail("body 出力が bf16 表現可能でない（中間丸めを再現できない）")
    shuffled = pixel_shuffle(bout, manifest["blocksize"], manifest["mode"])
    if manifest["add_nearest_input"]:
        base = bf16_round(test_input) if info["resize_input_quantized"] else test_input
        scale = manifest["scale"]
        total = shuffled + np.repeat(np.repeat(base, scale, axis=2), scale, axis=3)
    else:
        total = shuffled
    rec = bf16_round(total) if rounds else total
    if rec.shape != ref.shape:
        raise ValueError(f"再現出力 {rec.shape} が元出力 {ref.shape} と不一致")
    diff = float(np.abs(rec - ref).max())
    if diff >= VERIFY_TOL:
        raise ValueError(
            f"CPU 一致検証に失敗（bf16 再現）: max abs diff={diff:.3e}（許容 {VERIFY_TOL}）"
        )
    return diff


def split_model(
    input_path: Path,
    body_path: Path,
    manifest_path: Path,
    test_input: np.ndarray | None = None,
) -> dict:
    """切断＋検証を実行し、マニフェスト内容を返す。"""
    model = onnx.load(str(input_path))
    info = analyze_tail(model)
    body_inputs = [i.name for i in model.graph.input]
    extract_model(str(input_path), str(body_path), body_inputs, [info["cut"]])
    body = onnx.load(str(body_path))
    body_outputs = [o.name for o in body.graph.output]
    if body_outputs != [info["cut"]]:
        raise ValueError(f"body 出力が {body_outputs}（期待 {info['cut']}）")

    manifest = {
        "version": TAIL_MANIFEST_VERSION,
        "kind": TAIL_MANIFEST_KIND,
        "blocksize": info["blocksize"],
        "mode": info["mode"],
        "add_nearest_input": info["add_nearest_input"],
        "scale": info["scale"],
        "body_output": info["cut"],
        "source_sha256": _sha256_file(input_path),
    }
    diff = verify(input_path, body_path, info, manifest, test_input)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, "max_abs_diff": diff}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="入力 ONNX（リポジトリ相対でも可）")
    parser.add_argument("--body", default=None, help="出力 body ONNX（既定: <stem>_body.onnx）")
    parser.add_argument("--manifest", default=None, help="出力 tail マニフェスト（既定: <stem>_body.tail.json）")
    parser.add_argument("--test-input", default=None, help="検証用入力（.npy）。省略時は決定論的乱数")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if args.body is None or args.manifest is None:
        default_body = input_path.parent / f"{input_path.stem}_body.onnx"
        default_manifest = input_path.parent / f"{input_path.stem}_body.tail.json"
    body_path = Path(args.body) if args.body else default_body
    manifest_path = Path(args.manifest) if args.manifest else default_manifest

    test_input = None
    if args.test_input is not None:
        test_input = np.load(args.test_input)

    try:
        result = split_model(input_path, body_path, manifest_path, test_input)
    except UnsupportedTail as exc:
        print(f"非対応: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"失敗: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1
    manifest = result["manifest"]
    print(f"body: {body_path}")
    print(f"manifest: {manifest_path}")
    print(
        "tail kind={kind} blocksize={blocksize} mode={mode} "
        "add_nearest_input={add} scale={scale} body_output={out}".format(
            kind=manifest["kind"], blocksize=manifest["blocksize"],
            mode=manifest["mode"], add=manifest["add_nearest_input"],
            scale=manifest["scale"], out=manifest["body_output"],
        )
    )
    print(f"verify max abs diff={result['max_abs_diff']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""AdcSR fp32 の InstanceNormalization 91個を N5 形に書き換える。

tmp/adcsr-npu/round4/rewrite_in_to_n5.py の配布用移植。算法は同一で、
入出力パスを引数化した (マシン固有パスを埋め込まない)。

N5 = 3D [1,C,L] -> Reshape [1,C,H',W'] -> 4D正規化 (ReduceMean[2,3]/Sub/Mul/
Sqrt/Reciprocal) -> Reshape [1,C,L] -> affine(Mul scale[1,C,1]+Add bias)。
小型 SA/SB で単一サブグラフ・精度正常を確認した形。
H',W' は各 IN の後段 Reshape 出力 (4D [Cd,Hd,Wd]) から H'=Hd, W'=C*L/C/Hd
(割り切れない場合は sqrt 近傍で因数分解)。数学的には任意の因数分解で等価。
scale/bias は共有初期化子を直接触らず、新規 [1,C,1,1] 初期化子を作る。
eps は継承。opset21/ir10。既存 ReduceMean 4個の axes 移行も行う。

実行 (onnx が入った Python で):
  python scripts/adcsr/rewrite_in_to_n5.py --input <fp32.onnx> --output <norm_fp32.onnx> [--map rewrite_n5.csv]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def factor_hw(length: int) -> tuple[int, int]:
    h = int(math.isqrt(length))
    while h > 0 and length % h != 0:
        h -= 1
    return (h, length // h) if h > 0 else (1, length)


def rewrite_file(src: Path, dst: Path) -> list[tuple]:
    """IN→N5 書換えを行い、(idx, orig_name, C, L, Hp, Wp, eps, n_consumers) を返す。"""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    print("loading fp32 model (takes a while)...", flush=True)
    m = onnx.load(str(src))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    init_arr = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    vi = {}
    for v in list(g.value_info) + list(g.input) + list(g.output):
        t = v.type.tensor_type
        if t.elem_type == 1 and len(t.shape.dim) > 0:
            vi[v.name] = [d.dim_value for d in t.shape.dim]

    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    new_nodes = []
    rows = []
    n_rw = 0
    for n in list(g.node):
        if n.op_type != "InstanceNormalization":
            new_nodes.append(n)
            continue
        x3, s1, b1 = n.input[0], n.input[1], n.input[2]
        assert s1 in inits and b1 in inits, f"non-init affine: {n.name}"
        sc = init_arr[s1].astype(np.float32).reshape(-1)
        bi = init_arr[b1].astype(np.float32).reshape(-1)
        C = int(sc.shape[0])
        assert bi.shape[0] == C
        shape3 = vi.get(x3)
        assert shape3 is not None and len(shape3) == 3, f"no 3D shape for {x3}: {shape3}"
        assert shape3[0] == 1 and shape3[1] == C, f"shape mismatch {n.name}: {shape3}"
        L = int(shape3[2])
        eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
        y3 = n.output[0]

        Hp, Wp = None, None
        for c in consumers.get(y3, []):
            if c.op_type == "Reshape" and c.input[0] == y3:
                dshape = vi.get(c.output[0])
                if dshape is not None and len(dshape) == 4 and int(np.prod(dshape)) == C * L:
                    Hp = int(dshape[2])
                    if L % Hp == 0:
                        Wp = L // Hp
                break
        if Hp is None:
            Hp, Wp = factor_hw(L)
        assert Hp * Wp == L, f"{n.name}: {Hp}x{Wp} != {L}"

        p = f"r4n5_{n_rw}"
        inits_new = [
            numpy_helper.from_array(sc.reshape(1, C, 1, 1), f"{p}_sc4"),
            numpy_helper.from_array(bi.reshape(1, C, 1, 1), f"{p}_bi4"),
            numpy_helper.from_array(np.array([1, C, Hp, Wp], np.int64), f"{p}_sh4"),
            numpy_helper.from_array(np.array([1, C, L], np.int64), f"{p}_sh3"),
            numpy_helper.from_array(np.array([2, 3], np.int64), f"{p}_ax23"),
            numpy_helper.from_array(np.array(eps, np.float32), f"{p}_eps"),
        ]
        for ii in inits_new:
            g.initializer.append(ii)

        chain = [
            helper.make_node("Reshape", [x3, f"{p}_sh4"], [f"{p}_x4"], name=f"{p}_Mid4D"),
            helper.make_node("ReduceMean", [f"{p}_x4", f"{p}_ax23"], [f"{p}_mean"],
                             name=f"{p}_Mean", keepdims=1),
            helper.make_node("Sub", [f"{p}_x4", f"{p}_mean"], [f"{p}_d"],
                             name=f"{p}_Centered"),
            helper.make_node("Mul", [f"{p}_d", f"{p}_d"], [f"{p}_sq"], name=f"{p}_Sq"),
            helper.make_node("ReduceMean", [f"{p}_sq", f"{p}_ax23"], [f"{p}_var"],
                             name=f"{p}_Var", keepdims=1),
            helper.make_node("Add", [f"{p}_var", f"{p}_eps"], [f"{p}_ve"],
                             name=f"{p}_VarEps"),
            helper.make_node("Sqrt", [f"{p}_ve"], [f"{p}_std"], name=f"{p}_Std"),
            helper.make_node("Reciprocal", [f"{p}_std"], [f"{p}_inv"], name=f"{p}_Inv"),
            helper.make_node("Mul", [f"{p}_d", f"{p}_inv"], [f"{p}_n4"],
                             name=f"{p}_Normalized"),
            helper.make_node("Mul", [f"{p}_n4", f"{p}_sc4"], [f"{p}_s"],
                             name=f"{p}_Scale"),
            helper.make_node("Add", [f"{p}_s", f"{p}_bi4"], [f"{p}_a"],
                             name=f"{p}_Shift"),
            helper.make_node("Reshape", [f"{p}_a", f"{p}_sh3"], [y3], name=f"{p}_Mid3D"),
        ]
        new_nodes.extend(chain)
        rows.append((n_rw, n.name, C, L, Hp, Wp, eps, len(consumers.get(y3, []))))
        n_rw += 1

    print(f"rewrote {n_rw} InstanceNormalization -> N5", flush=True)
    del g.node[:]
    g.node.extend(new_nodes)

    n_rm = 0
    for n in g.node:
        if n.op_type != "ReduceMean":
            continue
        axes, keep = None, None
        for a in n.attribute:
            if a.name == "axes":
                axes = list(a.ints)
            if a.name == "keepdims":
                keep = a.i
        if axes is None:
            continue
        init_name = f"round4_reduce_axes_{'_'.join(str(v) for v in axes)}"
        if init_name not in {i.name for i in g.initializer}:
            g.initializer.append(numpy_helper.from_array(
                np.array(axes, dtype=np.int64), init_name))
        if init_name not in n.input:
            n.input.append(init_name)
        del n.attribute[:]
        if keep is not None:
            n.attribute.extend([helper.make_attribute("keepdims", keep)])
        n_rm += 1
    print(f"migrated {n_rm} ReduceMean axes attr -> input", flush=True)

    for oi in m.opset_import:
        if oi.domain == "":
            oi.version = 21
    m.ir_version = 10
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    print(f"saved {dst}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="書換え前の AdcSR fp32 ONNX")
    parser.add_argument("--output", required=True, help="N5 書換え後の fp32 ONNX")
    parser.add_argument("--map", default=None, help="書換え対応表 CSV (既定は出力と同名 .csv)")
    args = parser.parse_args()
    dst = Path(args.output)
    rows = rewrite_file(Path(args.input), dst)
    map_path = Path(args.map) if args.map else dst.with_suffix(".csv")
    with open(map_path, "w", encoding="utf-8") as handle:
        handle.write("idx,orig_name,C,L,Hp,Wp,eps,n_consumers\n")
        for r in rows:
            handle.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]}\n")
    print(f"saved {map_path} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

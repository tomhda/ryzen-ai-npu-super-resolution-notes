#!/usr/bin/env python
"""DepthToSpace tail-cut の共有ロジック。

NPU 実行時間の約半分を占める末尾の DepthToSpace（と spill）を NPU から外し、
NPU は DepthToSpace の入力（``[N, C_out*r*r, H, W]``）までを出力する body を
実行、残りの pixel shuffle（と入力の最近傍加算）は CPU の numpy で行う。

``npu_serve.py``（実行時）と ``scripts/npu/split_tail.py``（body 切り出し時の
CPU 検証）が同じ関数を使う。onnxruntime に依存しない。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

TAIL_MANIFEST_VERSION = 1
TAIL_MANIFEST_KIND = "depth_to_space"

#: 後処理の並列スレッドへ渡すキューの深さ。NPU 実行と CPU 後処理を重ねる。
TAIL_QUEUE_DEPTH = 2


def load_tail_manifest(path: str | Path) -> dict[str, Any]:
    """tail マニフェスト JSON を読み、必須項目を検証して返す。"""
    raw = Path(path).read_bytes().decode("utf-8")
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"tail manifest is not JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"tail manifest must be an object: {path}")
    if manifest.get("version") != TAIL_MANIFEST_VERSION:
        raise ValueError(
            f"unsupported tail manifest version: {manifest.get('version')!r} ({path})"
        )
    if manifest.get("kind") != TAIL_MANIFEST_KIND:
        raise ValueError(f"unsupported tail manifest kind: {manifest.get('kind')!r} ({path})")
    blocksize = manifest.get("blocksize")
    if not isinstance(blocksize, int) or blocksize <= 0:
        raise ValueError(f"invalid blocksize: {blocksize!r} ({path})")
    if manifest.get("mode") not in ("CRD", "DCR"):
        raise ValueError(f"invalid mode: {manifest.get('mode')!r} ({path})")
    if not isinstance(manifest.get("add_nearest_input"), bool):
        raise ValueError(f"invalid add_nearest_input: {manifest.get('add_nearest_input')!r} ({path})")
    scale = manifest.get("scale")
    if not isinstance(scale, int) or scale <= 0:
        raise ValueError(f"invalid scale: {scale!r} ({path})")
    if not manifest.get("body_output") or not isinstance(manifest.get("body_output"), str):
        raise ValueError(f"invalid body_output: {manifest.get('body_output')!r} ({path})")
    return manifest


def pixel_shuffle(x: np.ndarray, blocksize: int, mode: str = "CRD") -> np.ndarray:
    """DepthToSpace と等価な numpy の pixel shuffle。x は ``[N, C, H, W]``。

    CRD は ``reshape [N, c_out, r, r, H, W] → transpose(0,1,4,2,5,3)``、
    DCR は ``reshape [N, r, r, c_out, H, W] → transpose(0,3,4,1,5,2)``。
    """
    if x.ndim != 4:
        raise ValueError(f"pixel_shuffle needs 4D input, got shape {x.shape}")
    if mode not in ("CRD", "DCR"):
        raise ValueError(f"unknown pixel shuffle mode: {mode!r}")
    batch, channels, height, width = (int(v) for v in x.shape)
    rank = blocksize * blocksize
    if channels % rank != 0:
        raise ValueError(
            f"channels {channels} is not divisible by blocksize^2 ({rank})"
        )
    out_c = channels // rank
    if mode == "CRD":
        t = x.reshape(batch, out_c, blocksize, blocksize, height, width)
        t = t.transpose(0, 1, 4, 2, 5, 3)
    else:
        t = x.reshape(batch, blocksize, blocksize, out_c, height, width)
        t = t.transpose(0, 3, 4, 1, 5, 2)
    return np.ascontiguousarray(t.reshape(batch, out_c, height * blocksize, width * blocksize))


def nearest_upsample(x: np.ndarray, scale: int) -> np.ndarray:
    """整数倍の最近傍拡大。ORT Resize(nearest) と等価（テストで確認）。"""
    if scale <= 0:
        raise ValueError(f"invalid scale: {scale!r}")
    if scale == 1:
        return np.ascontiguousarray(x)
    return np.ascontiguousarray(
        np.repeat(np.repeat(x, scale, axis=2), scale, axis=3)
    )


def quantize_chw_into(dst_hwc: np.ndarray, src_chw: np.ndarray) -> None:
    """``clip(src * 255, 0, 255).astype(uint8)`` を HWC の ``dst_hwc`` へ直接書く。

    全体を結合してから量子化する従来手順と要素ごとに同じ演算で、出力は一致する。
    """
    buf = src_chw * 255.0
    np.clip(buf, 0.0, 255.0, out=buf)
    np.copyto(dst_hwc, buf.transpose(1, 2, 0), casting="unsafe")


def tail_quantize_into(
    dst_hwc: np.ndarray,
    body_out: np.ndarray,
    model_input: np.ndarray | None,
    manifest: dict[str, Any],
    box: tuple[int, int, int, int],
) -> None:
    """body 出力の ``box``（タイル座標 y0, y1, x0, x1）だけに tail 後処理と量子化を行う。

    ``tail_postprocess`` → 切り出し → ``clip(x * 255, 0, 255).astype(uint8)`` と
    要素ごとに同じ演算。切り出しを先に行い、拡大後の float 全体は作らない。
    ``dst_hwc`` は ``[(y1-y0)*r, (x1-x0)*r, C]`` の uint8。
    """
    r = int(manifest["blocksize"])
    mode = str(manifest["mode"])
    if mode not in ("CRD", "DCR"):
        raise ValueError(f"unknown pixel shuffle mode: {mode!r}")
    body = np.asarray(body_out)
    if body.ndim != 4 or body.shape[0] != 1:
        raise ValueError(f"tail_quantize_into needs [1, C, H, W], got {body.shape}")
    channels = int(body.shape[1])
    if channels % (r * r) != 0:
        raise ValueError(f"channels {channels} is not divisible by blocksize^2 ({r * r})")
    out_c = channels // (r * r)
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    if dst_hwc.shape != (h * r, w * r, out_c):
        raise ValueError(f"dst {dst_hwc.shape} != {(h * r, w * r, out_c)}")
    crop = body[0, :, y0:y1, x0:x1]

    def _shuffled(x: np.ndarray) -> np.ndarray:
        # [h, r, w, r, out_c] のビュー（pixel_shuffle と同じ並べ替え）。
        if mode == "CRD":
            return x.reshape(out_c, r, r, h, w).transpose(3, 1, 4, 2, 0)
        return x.reshape(r, r, out_c, h, w).transpose(3, 0, 4, 1, 2)

    if manifest.get("add_nearest_input"):
        if model_input is None:
            raise ValueError("manifest needs add_nearest_input but no model input was given")
        if int(manifest["scale"]) != r:
            raise ValueError("add_nearest_input needs scale == blocksize")
        source = np.asarray(model_input)[0, :, y0:y1, x0:x1]
        # 最近傍拡大は放送で表す（拡大済みの配列を作らない）。
        buf = _shuffled(np.ascontiguousarray(crop)) + source.transpose(1, 2, 0)[:, None, :, None, :]
        buf *= 255.0
    else:
        buf = _shuffled(crop * 255.0)
    np.clip(buf, 0.0, 255.0, out=buf)
    dst6 = dst_hwc.reshape(h, r, w, r, out_c)
    if np.shares_memory(dst6, dst_hwc):
        np.copyto(dst6, buf, casting="unsafe")
    else:  # reshape がコピーになった場合（通常は起きない）
        dst_hwc[...] = buf.reshape(h * r, w * r, out_c).astype(dst_hwc.dtype)


def tail_postprocess(
    body_out: np.ndarray,
    model_input: np.ndarray | None,
    manifest: dict[str, Any],
) -> np.ndarray:
    """body 出力（``[N, C, H, W]``）に tail 後処理を適用して元モデル出力相当を返す。"""
    blocksize = int(manifest["blocksize"])
    image = pixel_shuffle(np.asarray(body_out), blocksize, str(manifest["mode"]))
    if manifest.get("add_nearest_input"):
        if model_input is None:
            raise ValueError("manifest needs add_nearest_input but no model input was given")
        up = nearest_upsample(np.asarray(model_input), int(manifest["scale"]))
        if up.shape != image.shape:
            raise ValueError(f"upsampled input {up.shape} != shuffled {image.shape}")
        image = image + up
    return np.ascontiguousarray(image)

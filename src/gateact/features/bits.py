"""ビット列のパック/アンパック(LSB-first。Griffith のデバイスサンプル形式と同じ)。"""

from __future__ import annotations

import numpy as np


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """``[N, W]`` の 0/1 配列を ``[N, ceil(W/8)]`` の uint8 に LSB-first でパックする。

    ビット ``i`` はバイト ``i // 8`` のビット ``i % 8`` に入る。
    """
    arr = np.asarray(bits, dtype=np.uint8)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.packbits(arr, axis=1, bitorder="little")


def unpack_bits(packed: np.ndarray, width: int) -> np.ndarray:
    """``pack_bits`` の逆変換。``[N, width]`` の 0/1 uint8 を返す。"""
    arr = np.asarray(packed, dtype=np.uint8)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.unpackbits(arr, axis=1, count=width, bitorder="little")

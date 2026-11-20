"""文字 n-gram のハッシュによる固定長ビット特徴量(仕様 gateact-feat-v1)。

仕様(Python と C で同一):

- テキストを GateAct-norm-v1 で正規化し、前後に U+0002 / U+0003 を付ける。
- ``ngram_orders`` の各 n について、コードポイントの窓(n 文字)を UTF-8 にし、
  ``base = fnv1a32(bytes)`` を計算する。窓は n の昇順、位置の昇順に並ぶ。
- ``j = 0..num_hashes-1`` について ``idx = fmix32(base ^ (j * 0x9E3779B9 mod 2^32)) % width``
  のビットを 1 にする(出現の有無のみ。回数は数えない)。
- ビット i は特徴ベクトルの i 番目。パックするときは LSB-first(``features.bits``)。

Python 組み込みの ``hash()`` はプロセスごとにランダム化されるので使わない。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gateact.features.normalize import NORM_VERSION, normalize_codepoints

FEATURE_SPEC_VERSION = "gateact-feat-v1"
#: 正規化後のコードポイント数の上限(C 実装のバッファ。超えたら ValueError / GFEAT_ERR_TOO_LONG)
MAX_CODEPOINTS = 254
BOS, EOS = 0x02, 0x03
_MASK = 0xFFFFFFFF
_GOLDEN = 0x9E3779B9


def fnv1a32(data: bytes) -> int:
    """FNV-1a 32bit。"""
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & _MASK
    return h


def fmix32(h: int) -> int:
    """MurmurHash3 の最終ミキサ(32bit)。下位ビットの偏りを消すために使う。"""
    h &= _MASK
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _MASK
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _MASK
    h ^= h >> 16
    return h


@dataclass(frozen=True)
class FeatureConfig:
    """特徴量の設定。

    Attributes:
        width: ビット幅。
        ngram_orders: 使う n-gram の n(昇順)。
        num_hashes: 1つの n-gram が立てるビット数。
    """

    width: int = 512
    ngram_orders: tuple[int, ...] = (1, 2, 3)
    num_hashes: int = 1

    def __post_init__(self) -> None:
        """値の範囲を検証する。"""
        if self.width <= 0 or self.num_hashes <= 0:
            raise ValueError("width and num_hashes must be positive")
        if list(self.ngram_orders) != sorted(set(self.ngram_orders)) or min(self.ngram_orders) < 1:
            raise ValueError("ngram_orders must be strictly increasing positive integers")
        if max(self.ngram_orders) > 8:
            raise ValueError("ngram order > 8 is not supported by the C implementation")

    def to_dict(self) -> dict[str, Any]:
        """JSON 保存用の dict。仕様バージョンも含める。"""
        d = asdict(self)
        d["ngram_orders"] = list(self.ngram_orders)
        d["spec"] = FEATURE_SPEC_VERSION
        d["norm"] = NORM_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureConfig:
        """``to_dict`` の逆変換。仕様バージョンが違えばエラー。"""
        if d.get("spec", FEATURE_SPEC_VERSION) != FEATURE_SPEC_VERSION:
            raise ValueError(f"unsupported feature spec {d.get('spec')}")
        return cls(
            width=int(d["width"]),
            ngram_orders=tuple(int(n) for n in d["ngram_orders"]),
            num_hashes=int(d["num_hashes"]),
        )


def _windows(text: str, orders: Sequence[int]) -> Iterable[str]:
    norm = normalize_codepoints([ord(c) for c in text])
    if len(norm) > MAX_CODEPOINTS:
        raise ValueError(f"text too long after normalization: {len(norm)} > {MAX_CODEPOINTS}")
    cps = [BOS, *norm, EOS]
    s = "".join(map(chr, cps))
    for n in orders:
        for i in range(len(s) - n + 1):
            yield s[i : i + n]


def ngrams(text: str, orders: Sequence[int] = (1, 2, 3)) -> list[str]:
    """正規化と境界記号を付けた文字 n-gram を仕様の順に返す。"""
    return list(_windows(text, orders))


def feature_indices(text: str, cfg: FeatureConfig) -> list[int]:
    """立つビットの位置(重複あり、仕様の順)を返す。"""
    out: list[int] = []
    for gram in _windows(text, cfg.ngram_orders):
        base = fnv1a32(gram.encode("utf-8"))
        for j in range(cfg.num_hashes):
            out.append(fmix32(base ^ ((j * _GOLDEN) & _MASK)) % cfg.width)
    return out


def extract_bits(texts: Sequence[str], cfg: FeatureConfig) -> np.ndarray:
    """テキスト列を ``[N, width]`` の 0/1 uint8 配列にする。"""
    bits = np.zeros((len(texts), cfg.width), dtype=np.uint8)
    for i, t in enumerate(texts):
        bits[i, feature_indices(t, cfg)] = 1
    return bits

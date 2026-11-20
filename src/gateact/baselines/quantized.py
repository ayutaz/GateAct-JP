"""ESP32-S3 で同じ条件で比べるための、ベースラインの int8 量子化版(整数演算だけで推論)。

- bit-linear: ``score_k = bq_k + Σ_{bit_i=1} Wq[k, i]``(Wq は int8、per-tensor の対称スケール)
- n-gram: 埋め込み Eq(int8)と出力 Wq(int8)。平均の割り算を避けるため、
  ``score_k = Σ_d Wq[k, d] · Σ_g Eq[g, d] + bq_k · count`` を使う(元の logits を
  ``count / (sE · sW)`` 倍したもの。count > 0 なので argmax は変わらない)。
  n-gram の正規化は ESP32 で再現できる GateAct-norm-v1(元は NFKC + lower。データ上は
  空白の扱いだけが違い、90,448 文中 27 文で差がある)。

C 実装(``firmware/common/gateact_baselines.c``)はこの整数演算とスコアが完全一致する。
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gateact.baselines.bitlinear import BitLinearClassifier
from gateact.baselines.ngram import NgramClassifier
from gateact.features.normalize import normalize_text


def _sym_scale(w: np.ndarray) -> float:
    m = float(np.abs(w).max())
    return m / 127.0 if m > 0 else 1.0


def _q8(w: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8)


@dataclass
class QuantBitLinear:
    """int8 の bit-linear。"""

    weight: np.ndarray  # [K, W] int8
    bias: np.ndarray  # [K] int32
    scale: float

    def scores(self, bits: np.ndarray) -> np.ndarray:
        """``[N, W]`` の 0/1 から ``[N, K]`` の int64 スコア。"""
        b = np.asarray(bits, dtype=np.int64)
        return b @ self.weight.astype(np.int64).T + self.bias.astype(np.int64)


def quantize_bitlinear(clf: BitLinearClassifier) -> QuantBitLinear:
    """学習済み bit-linear を int8 にする(バイアスは同じスケールの int32)。"""
    w = clf.net.weight.detach().cpu().numpy()
    b = clf.net.bias.detach().cpu().numpy()
    s = _sym_scale(w)
    return QuantBitLinear(weight=_q8(w, s), bias=np.round(b / s).astype(np.int32), scale=s)


def gateact_ngram_ids(text: str, buckets: int, n_max: int = 3) -> list[int]:
    """GateAct-norm-v1 で正規化した文の文字 1〜n_max-gram の CRC-32 バケット。"""
    s = "\x02" + normalize_text(text) + "\x03"
    grams = [s[i : i + n] for n in range(1, n_max + 1) for i in range(len(s) - n + 1)]
    return [zlib.crc32(g.encode("utf-8")) % buckets for g in grams]


@dataclass
class QuantNgram:
    """int8 の n-gram 分類器。"""

    embedding: np.ndarray  # [B, D] int8
    weight: np.ndarray  # [K, D] int8
    bias: np.ndarray  # [K] int32(count を掛けて足す)
    buckets: int

    def _hsum(self, text: str) -> tuple[np.ndarray, int]:
        ids = gateact_ngram_ids(text, self.buckets)
        return self.embedding[ids].astype(np.int64).sum(axis=0), len(ids)

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        """``[N, K]`` の int64 スコア。"""
        out = np.zeros((len(texts), self.weight.shape[0]), dtype=np.int64)
        w = self.weight.astype(np.int64)
        for i, t in enumerate(texts):
            h, count = self._hsum(t)
            out[i] = w @ h + self.bias.astype(np.int64) * count
        return out

    def max_abs_score(self, texts: Sequence[str]) -> int:
        """int32 で足りるかの確認用(途中の和も含めた最大絶対値)。"""
        m = 0
        w = self.weight.astype(np.int64)
        for t in texts:
            h, count = self._hsum(t)
            terms = np.abs(w) @ np.abs(h) + np.abs(self.bias.astype(np.int64)) * count
            m = max(m, int(terms.max()), int(np.abs(h).max()))
        return m


def quantize_ngram(clf: NgramClassifier) -> QuantNgram:
    """学習済み n-gram 分類器を int8 にする。"""
    e = clf.net.embed.weight.detach().cpu().numpy()
    w = clf.net.out.weight.detach().cpu().numpy()
    b = clf.net.out.bias.detach().cpu().numpy()
    se, sw = _sym_scale(e), _sym_scale(w)
    return QuantNgram(
        embedding=_q8(e, se),
        weight=_q8(w, sw),
        bias=np.round(b / (se * sw)).astype(np.int32),
        buckets=clf.buckets,
    )

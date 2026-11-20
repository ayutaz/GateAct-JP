"""ベースライン B: JapaneseTinyAgentLM ``jtalm.model.classifier.NgramClassifier`` 相当の分類器。

JapaneseTinyAgentLM(Apache-2.0)の文書・仕様に従って再実装した(コードはコピーしていない):
NFKC + 小文字化した文の前後に U+0002 / U+0003 を付け、文字 1〜3-gram を CRC-32 で
``buckets`` 個に hash し、EmbeddingBag(mean, ``dim``)→ Linear。既定値は原実装と同じ
(buckets 65,536 / dim 40 / epochs 10 / lr 0.01 / batch 128)。クラスは GateAct-JP の固定クラス。
"""

from __future__ import annotations

import unicodedata
import zlib
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from gateact.baselines.torch_common import train_loop
from gateact.data.checks import Labeled
from gateact.data.labels import LabelSpec
from gateact.utils import seed_everything


def ngram_ids(text: str, buckets: int, n_max: int = 3) -> list[int]:
    """文字 1〜n_max-gram の CRC-32 バケット ID(原実装と同じ順序)。"""
    s = "\x02" + unicodedata.normalize("NFKC", text).lower() + "\x03"
    grams = [s[i : i + n] for n in range(1, n_max + 1) for i in range(len(s) - n + 1)]
    return [zlib.crc32(g.encode("utf-8")) % buckets for g in grams]


class _Net(nn.Module):
    def __init__(self, buckets: int, dim: int, n_classes: int) -> None:
        super().__init__()
        self.embed = nn.EmbeddingBag(buckets, dim, mode="mean")
        self.out = nn.Linear(dim, n_classes)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.out(self.embed(ids, offsets))


class NgramClassifier:
    """fastText 型の n-gram 分類器。"""

    name = "ngram"

    def __init__(
        self,
        spec: LabelSpec,
        *,
        buckets: int = 65536,
        dim: int = 40,
        epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 128,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        """ハイパーパラメータを受け取る(既定値は JapaneseTinyAgentLM と同じ)。"""
        self.spec = spec
        self.buckets, self.dim = buckets, dim
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed
        self.verbose = verbose
        seed_everything(seed)
        self.net = _Net(buckets, dim, spec.num_classes)

    def _batch(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        grams = [ngram_ids(t, self.buckets) for t in texts]
        offsets = torch.tensor([0] + [len(g) for g in grams[:-1]]).cumsum(0)
        return torch.tensor([i for g in grams for i in g]), offsets

    @torch.no_grad()
    def predict(self, texts: Sequence[str]) -> list[str]:
        """クラス名を予測する。"""
        self.net.eval()
        out: list[str] = []
        for i in range(0, len(texts), 2048):
            logits = self.net(*self._batch(texts[i : i + 2048]))
            out += [self.spec.names[j] for j in logits.argmax(-1).tolist()]
        return out

    def fit(self, train: Sequence[Labeled], dev: Sequence[Labeled]) -> dict[str, Any]:
        """学習し、dev の正解率が最良の epoch を採る。"""
        seed_everything(self.seed)
        texts = [d.example.text for d in train]
        y = torch.tensor([self.spec.index(d.label) for d in train])
        dev_texts = [d.example.text for d in dev]
        dev_y = [d.label for d in dev]

        def dev_eval() -> float:
            pred = self.predict(dev_texts)
            return sum(p == t for p, t in zip(pred, dev_y, strict=True)) / max(len(dev_y), 1)

        log = train_loop(
            self.net,
            lambda idx: self._batch([texts[j] for j in idx]),
            y,
            dev_eval,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            seed=self.seed,
            log_prefix="ngram ",
            verbose=self.verbose,
        )
        return {"training_log": log}

    @property
    def num_parameters(self) -> int:
        """パラメータ数。"""
        return sum(p.numel() for p in self.net.parameters())

    def describe(self) -> dict[str, Any]:
        """設定とサイズ(f32 の重みのバイト数)。"""
        return {
            "model": "char 1-3-gram EmbeddingBag + Linear (JapaneseTinyAgentLM NgramClassifier)",
            "buckets": self.buckets,
            "dim": self.dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "params": self.num_parameters,
            "weight_bytes_f32": self.num_parameters * 4,
        }

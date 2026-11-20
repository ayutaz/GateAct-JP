"""追加の比較対象: GateAct の特徴ビット(Griffith と同じ入力)に線形層をかけた分類器。

入力が同じなので、Griffith の LUT 回路と「特徴量の情報量」を切り分けて比べられる。
推論は「立っているビットの重みの和」なので、int8 量子化すれば組み込みでも小さい。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from gateact.baselines.torch_common import train_loop
from gateact.data.checks import Labeled
from gateact.data.labels import LabelSpec
from gateact.features.hashing import FeatureConfig, extract_bits
from gateact.utils import seed_everything


class BitLinearClassifier:
    """特徴ビット → Linear の分類器。"""

    name = "bitlinear"

    def __init__(
        self,
        spec: LabelSpec,
        features: FeatureConfig,
        *,
        epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        """設定を受け取る。"""
        self.spec, self.features = spec, features
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed
        self.weight_decay = weight_decay
        self.verbose = verbose
        seed_everything(seed)
        self.net = nn.Linear(features.width, spec.num_classes)

    @torch.no_grad()
    def predict_bits(self, bits: np.ndarray) -> list[str]:
        """特徴ビットからクラス名を予測する。"""
        self.net.eval()
        logits = self.net(torch.from_numpy(bits.astype(np.float32)))
        return [self.spec.names[j] for j in logits.argmax(-1).tolist()]

    def predict(self, texts: Sequence[str]) -> list[str]:
        """クラス名を予測する。"""
        return self.predict_bits(extract_bits(texts, self.features))

    def fit(self, train: Sequence[Labeled], dev: Sequence[Labeled]) -> dict[str, Any]:
        """学習し、dev の正解率が最良の epoch を採る。"""
        seed_everything(self.seed)
        x = torch.from_numpy(
            extract_bits([d.example.text for d in train], self.features).astype(np.float32)
        )
        y = torch.tensor([self.spec.index(d.label) for d in train])
        dev_bits = extract_bits([d.example.text for d in dev], self.features)
        dev_y = [d.label for d in dev]

        def dev_eval() -> float:
            pred = self.predict_bits(dev_bits)
            return sum(p == t for p, t in zip(pred, dev_y, strict=True)) / max(len(dev_y), 1)

        log = train_loop(
            self.net,
            lambda idx: (x[list(idx)],),
            y,
            dev_eval,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            seed=self.seed,
            log_prefix="bitlinear ",
            verbose=self.verbose,
        )
        return {"training_log": log}

    @property
    def num_parameters(self) -> int:
        """パラメータ数。"""
        return sum(p.numel() for p in self.net.parameters())

    def describe(self) -> dict[str, Any]:
        """設定とサイズ。"""
        return {
            "model": "feature bits + Linear",
            "features": self.features.to_dict(),
            "epochs": self.epochs,
            "lr": self.lr,
            "seed": self.seed,
            "params": self.num_parameters,
            "weight_bytes_f32": self.num_parameters * 4,
            "weight_bytes_int8": self.num_parameters,
        }

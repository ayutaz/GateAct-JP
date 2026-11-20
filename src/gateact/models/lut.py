"""Griffith の LUT 層で構成する Action 分類モデル。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch.nn as nn
from griffith_logic import GroupSum, LutLayer


@dataclass(frozen=True)
class LutModelConfig:
    """LUT 分類モデルの構成。

    Attributes:
        input_bits: 入力ビット幅。
        hidden: 中間 LUT 層の出力ゲート数のリスト(層ごと)。
        output_gates: 最終 LUT 層のゲート数。``num_classes`` で割り切れる必要がある。
        num_classes: クラス数(GroupSum の k)。
        n_inputs: 各 LUT の入力数(2 = LUT2、4 = LUT4、6 = LUT6)。
        tau: GroupSum の温度。hard 推論ではスコア = 整数カウント / tau。
        connections: LutLayer の結線方式(Griffith の ``connections`` 引数)。
    """

    input_bits: int
    hidden: list[int] = field(default_factory=list)
    output_gates: int = 0
    num_classes: int = 2
    n_inputs: int = 2
    tau: float = 10.0
    connections: str = "random"

    def to_dict(self) -> dict[str, Any]:
        """JSON 保存用の dict を返す。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LutModelConfig:
        """``to_dict`` の逆変換。"""
        return cls(**{**d, "hidden": list(d.get("hidden", []))})


def build_lut_model(cfg: LutModelConfig) -> nn.Sequential:
    """構成から ``LutLayer`` × N + ``GroupSum`` のモデルを作る。

    Raises:
        ValueError: ``output_gates`` が ``num_classes`` で割り切れない場合。
    """
    if cfg.output_gates <= 0 or cfg.output_gates % cfg.num_classes != 0:
        raise ValueError(
            f"output_gates ({cfg.output_gates}) must be positive and divisible by "
            f"num_classes ({cfg.num_classes})"
        )
    widths = [cfg.input_bits, *cfg.hidden, cfg.output_gates]
    layers: list[nn.Module] = [
        LutLayer(a, b, n_inputs=cfg.n_inputs, connections=cfg.connections)
        for a, b in zip(widths[:-1], widths[1:], strict=True)
    ]
    layers.append(GroupSum(k=cfg.num_classes, tau=cfg.tau))
    return nn.Sequential(*layers)


def count_gates(cfg: LutModelConfig) -> int:
    """モデルの LUT ゲート総数(圧縮前)を返す。"""
    return sum(cfg.hidden) + cfg.output_gates

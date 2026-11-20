"""分類器の共通インターフェース。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from gateact.data.checks import Labeled


class Classifier(Protocol):
    """テキスト → クラス名の分類器。"""

    name: str

    def fit(self, train: Sequence[Labeled], dev: Sequence[Labeled]) -> dict[str, Any]:
        """学習し、学習ログ(JSON にできる dict)を返す。"""
        ...

    def predict(self, texts: Sequence[str]) -> list[str]:
        """クラス名を予測する。"""
        ...

    def describe(self) -> dict[str, Any]:
        """設定とサイズの情報。"""
        ...

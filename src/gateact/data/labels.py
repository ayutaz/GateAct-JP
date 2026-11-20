"""call 列 → 固定 Action クラスの対応(``configs/labels_*.json``)。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gateact.data.schema import Call, canonicalize


@dataclass(frozen=True)
class LabelResult:
    """1件の call 列の分類結果。

    Attributes:
        label: クラス名。対象外なら None(**no_action にはしない**)。
        reason: 対象外の理由(``multi_action`` / ``unsupported:<name>``)。
        arg_loss: クラスにしたことで捨てた引数があるか(正解の call が既定の call と違う)。
    """

    label: str | None
    reason: str | None = None
    arg_loss: bool = False

    @property
    def in_scope(self) -> bool:
        """分類の対象か。"""
        return self.label is not None


@dataclass(frozen=True)
class _ClassRule:
    name: str
    empty: bool
    call: str | None
    args: dict[str, Any]
    args_not: dict[str, Any]
    default_call: Call | None

    def matches(self, call: Call) -> bool:
        if self.empty or call["name"] != self.call:
            return False
        a = call["arguments"]
        if any(a.get(k) != v for k, v in self.args.items()):
            return False
        return not any(a.get(k) == v for k, v in self.args_not.items())


class LabelSpec:
    """固定クラスの定義。クラス ID は JSON の並び順(0 = no_action)。"""

    def __init__(self, version: str, rules: list[_ClassRule]) -> None:
        """``load`` を使うこと。"""
        self.version = version
        self._rules = rules
        self.names = [r.name for r in rules]
        self._index = {n: i for i, n in enumerate(self.names)}
        if len(self._index) != len(self.names):
            raise ValueError("duplicate class names")
        if not rules or not rules[0].empty:
            raise ValueError("class 0 must be the empty (no_action) class")

    @classmethod
    def load(cls, path: str | Path) -> LabelSpec:
        """JSON 定義ファイルを読み込む。"""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        rules = [
            _ClassRule(
                name=c["name"],
                empty=bool(c.get("empty", False)),
                call=c.get("call"),
                args=dict(c.get("args", {})),
                args_not=dict(c.get("args_not", {})),
                default_call=c.get("default_call"),
            )
            for c in d["classes"]
        ]
        return cls(d["version"], rules)

    @property
    def num_classes(self) -> int:
        """クラス数。"""
        return len(self.names)

    @property
    def no_action(self) -> str:
        """Action なしのクラス名。"""
        return self.names[0]

    def index(self, name: str) -> int:
        """クラス名 → ID。"""
        return self._index[name]

    def default_calls(self, name: str) -> list[Call]:
        """クラスを JSON の call 列に戻したもの(既定の引数)。"""
        rule = self._rules[self._index[name]]
        return [] if rule.default_call is None else canonicalize([rule.default_call])

    def assign(self, calls: list[Call]) -> LabelResult:
        """Call 列をクラスに割り当てる。対象外は ``label=None``。"""
        calls = canonicalize(calls)
        if not calls:
            return LabelResult(label=self.no_action)
        if len(calls) > 1:
            return LabelResult(label=None, reason="multi_action")
        call = calls[0]
        for rule in self._rules:
            if rule.matches(call):
                return LabelResult(label=rule.name, arg_loss=self.default_calls(rule.name) != calls)
        return LabelResult(label=None, reason=f"unsupported:{call['name']}")

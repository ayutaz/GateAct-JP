"""JapaneseTinyAgentLM-Action-Synth 形式の JSONL を厳格に読み込む。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gateact.data.schema import Call, canonicalize, parse_calls

REQUIRED_FIELDS = ("id", "input", "output", "category")


class DataError(ValueError):
    """データファイルの形式エラー(ファイル名と行番号を含む)。"""


@dataclass(frozen=True)
class Example:
    """1件の入力文と正解の call 列。

    Attributes:
        id: サンプル ID。
        text: 入力の日本語文。
        calls: 正規化済みの正解 call 列(``[]`` は Action なし)。
        category: データセット上の分類(single / multi_action / negation / correction / no_action)。
        pair_id: 対照ペアの ID(無ければ None)。
        generator: 文を書いたモデル名など(無ければ None)。
    """

    id: str
    text: str
    calls: list[Call]
    category: str
    pair_id: str | None = None
    generator: str | None = None


def load_jsonl(path: str | Path) -> list[Example]:
    """JSONL を読み込み、全行を検証して ``Example`` のリストを返す。

    Raises:
        DataError: 不正な JSON、必須フィールドの欠落、空の入力文、不正な ``output``。
    """
    path = Path(path)
    out: list[Example] = []
    with path.open(encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"{path.name} line {lineno}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataError(f"{where}: invalid JSON ({e})") from e
            if not isinstance(row, dict):
                raise DataError(f"{where}: row must be an object")
            missing = [k for k in REQUIRED_FIELDS if k not in row]
            if missing:
                raise DataError(f"{where}: missing field(s) {missing}: output/id/input/category")
            text = row["input"]
            if not isinstance(text, str) or not text.strip():
                raise DataError(f"{where}: empty or non-string 'input'")
            try:
                calls = parse_calls(row["output"])
            except (ValueError, TypeError) as e:
                raise DataError(f"{where}: invalid 'output' ({e})") from e
            out.append(
                Example(
                    id=str(row["id"]),
                    text=text,
                    calls=canonicalize(calls),
                    category=str(row["category"]),
                    pair_id=row.get("pair_id"),
                    generator=row.get("generator"),
                )
            )
    return out

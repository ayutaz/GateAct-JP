"""Action の call 列(JapaneseTinyAgentLM の Action schema v1 形式)の解析と正規形。

正規化の規則は JapaneseTinyAgentLM ``jtalm.action.schema.canonicalize`` の仕様に合わせる:
``look`` の ``direction == "center"`` は ``amount`` を無視するので ``normal`` にそろえる。
"""

from __future__ import annotations

import json
from typing import Any

Call = dict[str, Any]


def parse_calls(text: str) -> list[Call]:
    """JSON 文字列を call のリストとして解析する。

    Raises:
        ValueError: JSON でない、リストでない、各要素が ``name``(str) と
            ``arguments``(dict) を持たない場合。
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError("calls must be a JSON array")
    calls: list[Call] = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"call must have a string 'name': {item!r}")
        args = item.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError(f"'arguments' must be an object: {item!r}")
        calls.append({"name": item["name"], "arguments": dict(args)})
    return calls


def canonicalize(calls: list[Call]) -> list[Call]:
    """比較用に call 列を正規化する(順序は意味を持つので保つ)。"""
    out: list[Call] = []
    for call in calls:
        args = dict(call.get("arguments", {}))
        if call.get("name") == "look" and args.get("direction") == "center" and "amount" in args:
            args["amount"] = "normal"
        out.append({"name": call.get("name"), "arguments": args})
    return out


def canonical_json(calls: list[Call]) -> str:
    """正規化した call 列のコンパクトな JSON(キーはソート)を返す。"""
    return json.dumps(
        canonicalize(calls), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )

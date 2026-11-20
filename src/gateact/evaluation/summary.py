"""``results/**`` の結果 JSON を名前ごと(seed をまとめて)に集計する。"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: (表示名, 取り出し方)
METRICS: list[tuple[str, tuple[str, ...]]] = [
    ("exact_accuracy", ("test", "exact_accuracy")),
    ("macro_f1", ("test", "macro_f1")),
    ("false_action_rate", ("test", "false_action_rate")),
    ("negation_accuracy", ("test", "negation_accuracy")),
    ("no_action_precision", ("test", "no_action", "precision")),
    ("no_action_recall", ("test", "no_action", "recall")),
    ("json_exact", ("test", "json_exact")),
    ("full_test_json_exact", ("full_test_json_exact",)),
]
_SEED = re.compile(r"_s(\d+)$")


def load_results(dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    """ディレクトリ内の ``*_s<seed>.json`` を読む。"""
    out = []
    for d in dirs:
        for p in sorted(Path(d).glob("*_s*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            if "test" in r:
                r["_stem"] = _SEED.sub("", p.stem)
                out.append(r)
    return out


def _get(d: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return float(cur) if isinstance(cur, int | float) else None


def aggregate(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """名前ごとに各指標の平均・標準偏差(標本、n≥2 のとき)を出す。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[r["_stem"]].append(r)
    rows: dict[str, dict[str, Any]] = {}
    for name, rs in sorted(groups.items()):
        row: dict[str, Any] = {"seeds": sorted(int(r["meta"]["seed"]) for r in rs), "n": len(rs)}
        for metric, path in METRICS:
            vals = [v for r in rs if (v := _get(r, path)) is not None]
            row[metric] = {
                "mean": statistics.fmean(vals) if vals else None,
                "sd": statistics.stdev(vals) if len(vals) >= 2 else None,
                "values": vals,
            }
        rows[name] = row
    return rows


def _fmt(m: dict[str, Any]) -> str:
    if m["mean"] is None:
        return "—"
    s = f"{100 * m['mean']:.2f}"
    return s + (f" ± {100 * m['sd']:.2f}" if m["sd"] is not None else "")


def to_markdown(rows: dict[str, dict[str, Any]]) -> str:
    """集計を Markdown の表にする(値は %、平均 ± SD)。"""
    head = ["model", "seeds", *[m for m, _ in METRICS]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, row in rows.items():
        cells = [name, str(row["n"]), *[_fmt(row[m]) for m, _ in METRICS]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """サブコマンドを登録する。"""
    p = sub.add_parser("summarize", help="結果 JSON を seed をまとめて集計する")
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None, help="JSON の出力先(同名 .md も書く)")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """集計して表示(と保存)する。"""
    rows = aggregate(load_results(args.dirs))
    md = to_markdown(rows)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    return 0

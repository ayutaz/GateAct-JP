"""``gateact data-check``: データの検査と manifest の書き出し。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gateact.data.checks import build_manifest
from gateact.data.io import load_jsonl
from gateact.data.labels import LabelSpec
from gateact.data.source import DEFAULT_DIR, FILES, HF_REPO, HF_REVISION, LICENSE, SHA256


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """サブコマンドを登録する。"""
    p = sub.add_parser("data-check", help="データの検査と manifest の作成")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--labels", type=Path, default=Path("configs/labels_v1.json"))
    p.add_argument("--out", type=Path, default=Path("results/data"))
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """検査を実行し ``manifest.json`` を書く。SHA-256 が固定値と違えば失敗する。"""
    spec = LabelSpec.load(args.labels)
    files = {split: args.data_dir / name for split, name in FILES.items()}
    rows = {split: load_jsonl(path) for split, path in files.items()}
    manifest = build_manifest(files, rows, spec, revision=HF_REVISION, source=HF_REPO)
    manifest["license"] = LICENSE
    bad = [s for s, info in manifest["splits"].items() if info["sha256"] != SHA256[info["file"]]]
    manifest["sha256_matches_pinned"] = not bad
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, info in manifest["splits"].items():
        print(
            f"{split}: n={info['n']} coverage={info['coverage']:.3f} "
            f"dup_groups={info['duplicate_groups']} "
            f"conflicts={info['conflicting_duplicate_groups']}"
        )
    print(f"overlap: {manifest['overlap_normalized_text']}")
    print(f"wrote {out}")
    return 1 if bad else 0

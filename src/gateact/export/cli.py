"""``gateact export``: checkpoint を C に変換し、評価セット全体で全段の一致を検証する。"""

from __future__ import annotations

import argparse
from pathlib import Path

from gateact.data.labels import LabelSpec
from gateact.data.source import DEFAULT_DIR
from gateact.evaluation.runner import load_splits
from gateact.export.verify import export_and_verify


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """サブコマンドを登録する。"""
    p = sub.add_parser("export", help="checkpoint → IR → C と一致検証")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compress", default="fast", choices=["none", "fast", "max"])
    p.add_argument("--dense-flat", default="off", choices=["off", "on", "auto"])
    p.add_argument("--n-samples", type=int, default=64, help="samples.h に入れる件数")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--labels", type=Path, default=Path("configs/labels_v1.json"))
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """評価セット(対象内 + 対象外の全文)で照合し、不一致があれば終了コード 1。"""
    spec = LabelSpec.load(args.labels)
    splits, _ = load_splits(args.data_dir, spec)
    texts = [d.example.text for d in splits.test] + [e.text for e, _ in splits.test_out_of_scope]
    labels = [spec.index(d.label) for d in splits.test] + [-1] * len(splits.test_out_of_scope)
    rep = export_and_verify(
        args.checkpoint,
        texts,
        args.out,
        labels=labels,
        compress=args.compress,
        dense_flat=args.dense_flat,
        n_samples=args.n_samples,
    )
    p, s = rep["parity"], rep["sizes"]
    print(
        f"parity n={p['n']} torch/ir={p['torch_vs_ir_raw_mismatch']} "
        f"raw/comp={p['ir_raw_vs_ir_compressed_mismatch']} ir/c={p['ir_vs_c_mismatch']}"
    )
    print(
        f"gates {s['gates_raw']} -> {s['gates_compressed']}, ctx={s['ctx_bytes']}B, "
        f"model.c={s['model_c_bytes']}B"
    )
    bad = p["torch_vs_ir_raw_mismatch"] + p["ir_raw_vs_ir_compressed_mismatch"]
    return 1 if bad + p["ir_vs_c_mismatch"] else 0

"""``gateact baseline``: ベースライン分類器の学習と評価。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from gateact.baselines.base import Classifier
from gateact.baselines.bitlinear import BitLinearClassifier
from gateact.baselines.ngram import NgramClassifier
from gateact.baselines.rules import RuleClassifier
from gateact.data.labels import LabelSpec
from gateact.data.source import DEFAULT_DIR
from gateact.evaluation.runner import evaluate_classifier, load_splits, run_metadata, write_result
from gateact.features.hashing import FeatureConfig


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """サブコマンドを登録する。"""
    p = sub.add_parser("baseline", help="ベースライン分類器の学習と評価")
    p.add_argument("--kind", choices=["rules", "ngram", "bitlinear"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--width", type=int, default=512, help="bitlinear の特徴ビット幅")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--labels", type=Path, default=Path("configs/labels_v1.json"))
    p.add_argument("--dev-frac", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("results/baselines"))
    p.add_argument("--runs", type=Path, default=Path("runs/baselines"), help="重みの保存先")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(handler=run)


def _make(args: argparse.Namespace, spec: LabelSpec) -> Classifier:
    if args.kind == "rules":
        return RuleClassifier(spec)
    if args.kind == "ngram":
        return NgramClassifier(spec, epochs=args.epochs, seed=args.seed, verbose=args.verbose)
    return BitLinearClassifier(
        spec,
        FeatureConfig(width=args.width),
        epochs=args.epochs,
        seed=args.seed,
        verbose=args.verbose,
    )


def run(args: argparse.Namespace) -> int:
    """学習・評価して ``<out>/<kind>[_w<width>]_s<seed>.json`` を書く。"""
    spec = LabelSpec.load(args.labels)
    splits, data_sha = load_splits(args.data_dir, spec, dev_frac=args.dev_frac)
    clf = _make(args, spec)
    train_info = clf.fit(splits.train, splits.dev)
    ev = evaluate_classifier(clf, splits, spec, n_boot=args.n_boot)
    result: dict[str, Any] = {
        "kind": args.kind,
        "meta": run_metadata(args.seed, data_sha, spec),
        "model": clf.describe(),
        "train": train_info,
        "split_sizes": {
            "train": len(splits.train),
            "dev": len(splits.dev),
            "test": len(splits.test),
            "test_out_of_scope": len(splits.test_out_of_scope),
            "removed_leakage": splits.removed_leakage,
        },
        **ev,
    }
    stem = args.kind + (f"_w{args.width}" if args.kind == "bitlinear" else "")
    net = getattr(clf, "net", None)
    if net is not None:
        import torch

        ckpt = args.runs / f"{stem}_s{args.seed}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": args.kind, "model": clf.describe(), "state_dict": net.state_dict(),
                    "labels": list(spec.names)}, ckpt)  # fmt: skip
        result["checkpoint"] = str(ckpt)
    path = write_result(args.out / f"{stem}_s{args.seed}.json", result)
    t = ev["test"]
    print(
        f"{stem} seed={args.seed}: test exact={t['exact_accuracy']:.4f} "
        f"macroF1={t['macro_f1']:.4f} false_action={t['false_action_rate']:.4f} "
        f"json_exact={t['json_exact']:.4f} full_json_exact={ev['full_test_json_exact']:.4f}"
    )
    print(f"wrote {path}")
    return 0

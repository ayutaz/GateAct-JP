"""``gateact train``: 設定 JSON から Griffith LUT 分類器を学習・評価・保存する。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch.nn as nn

from gateact.data.checks import Labeled
from gateact.data.labels import LabelSpec
from gateact.data.source import DEFAULT_DIR
from gateact.evaluation.runner import evaluate_classifier, load_splits, run_metadata, write_result
from gateact.features.hashing import FeatureConfig, extract_bits
from gateact.models.lut import LutModelConfig, count_gates
from gateact.training.lut_trainer import (
    TrainConfig,
    hard_predict_ids,
    save_checkpoint,
    train_lut,
)


class LutClassifier:
    """学習済み LUT モデルを ``Classifier`` として使うアダプタ(推論は hard 回路)。"""

    name = "lut"

    def __init__(
        self, model: nn.Module, model_cfg: LutModelConfig, feat_cfg: FeatureConfig, spec: LabelSpec
    ) -> None:
        """モデルと設定を受け取る。"""
        self.model, self.model_cfg, self.feat_cfg, self.spec = model, model_cfg, feat_cfg, spec

    def fit(self, train: Sequence[Labeled], dev: Sequence[Labeled]) -> dict[str, Any]:
        """学習済みなので何もしない。"""
        return {}

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Hard 回路でクラス名を予測する。"""
        if not texts:
            return []
        ids = hard_predict_ids(self.model, extract_bits(texts, self.feat_cfg), self.model_cfg.tau)
        return [self.spec.names[i] for i in ids]

    def describe(self) -> dict[str, Any]:
        """設定とサイズ。"""
        return {
            "model": "Griffith LutLayer stack + GroupSum (hard inference)",
            "model_config": self.model_cfg.to_dict(),
            "features": self.feat_cfg.to_dict(),
            "gates_before_compression": count_gates(self.model_cfg),
            "trainable_params": sum(p.numel() for p in self.model.parameters()),
        }


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """サブコマンドを登録する。"""
    p = sub.add_parser("train", help="Griffith LUT 分類器の学習と評価")
    p.add_argument("config", type=Path, help="configs/lut/*.json")
    p.add_argument("--seed", type=int, default=None, help="設定の seed を上書き")
    p.add_argument("--epochs", type=int, default=None, help="設定の epochs を上書き")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--labels", type=Path, default=Path("configs/labels_v1.json"))
    p.add_argument("--dev-frac", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("results/lut"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(handler=run)


def load_config(path: Path) -> tuple[str, FeatureConfig, LutModelConfig, TrainConfig]:
    """設定 JSON(``name`` / ``features`` / ``model`` / ``train``)を読む。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    feat = FeatureConfig.from_dict({"num_hashes": 1, "ngram_orders": [1, 2, 3], **d["features"]})
    model = LutModelConfig.from_dict({"input_bits": feat.width, **d["model"]})
    train = TrainConfig(**d.get("train", {}))
    return d["name"], feat, model, train


def run(args: argparse.Namespace) -> int:
    """学習・評価し、checkpoint と結果 JSON を書く。"""
    name, feat, mcfg, tcfg = load_config(args.config)
    overrides = {k: v for k, v in (("seed", args.seed), ("epochs", args.epochs)) if v is not None}
    tcfg = TrainConfig(**{**tcfg.to_dict(), **overrides})
    spec = LabelSpec.load(args.labels)
    splits, data_sha = load_splits(args.data_dir, spec, dev_frac=args.dev_frac)
    out = train_lut(mcfg, feat, tcfg, splits.train, splits.dev, spec, verbose=args.verbose)
    stem = f"{name}_s{tcfg.seed}"
    meta = run_metadata(tcfg.seed, data_sha, spec)
    ckpt = save_checkpoint(args.runs / name / f"{stem}.pt", out.model, mcfg, feat, spec, meta=meta)
    clf = LutClassifier(out.model, mcfg, feat, spec)
    ev = evaluate_classifier(clf, splits, spec, n_boot=args.n_boot)
    result: dict[str, Any] = {
        "kind": "lut",
        "name": name,
        "config_file": str(args.config),
        "meta": meta,
        "model": clf.describe(),
        "train_config": tcfg.to_dict(),
        "train": {
            "training_log": out.log,
            "best_epoch": out.best_epoch,
            "best_dev_hard_acc": out.best_dev_hard_acc,
            "seconds": out.seconds,
        },
        "checkpoint": str(ckpt),
        **ev,
    }
    path = write_result(args.out / f"{stem}.json", result)
    t = ev["test"]
    print(
        f"{stem}: best_epoch={out.best_epoch} test exact={t['exact_accuracy']:.4f} "
        f"macroF1={t['macro_f1']:.4f} false_action={t['false_action_rate']:.4f} "
        f"({out.seconds:.0f}s)"
    )
    print(f"wrote {path}")
    return 0

"""分類器を共通の分割で評価し、再現情報付きの結果 JSON を作る。"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from gateact import __version__
from gateact.baselines.base import Classifier
from gateact.data.checks import Labeled, Splits, prepare_splits
from gateact.data.io import load_jsonl
from gateact.data.labels import LabelSpec
from gateact.data.source import FILES
from gateact.evaluation.metrics import bootstrap_ci, classification_report, out_of_scope_report
from gateact.utils import git_sha, sha256_file


def load_splits(
    data_dir: str | Path, spec: LabelSpec, *, dev_frac: float = 0.05
) -> tuple[Splits, dict[str, str]]:
    """``data_dir`` の train / validation から分割を作り、ファイルの SHA-256 も返す。

    HF の validation を評価(test)に使い、train から dev を切り出す。
    """
    d = Path(data_dir)
    files = {k: d / v for k, v in FILES.items()}
    splits = prepare_splits(
        load_jsonl(files["train"]), load_jsonl(files["validation"]), spec, dev_frac=dev_frac
    )
    return splits, {p.name: sha256_file(p) for p in files.values()}


def _report(
    clf: Classifier, rows: Sequence[Labeled], spec: LabelSpec, n_boot: int
) -> dict[str, Any]:
    texts = [d.example.text for d in rows]
    t0 = time.perf_counter()
    preds = clf.predict(texts)
    secs = time.perf_counter() - t0
    y = [d.label for d in rows]
    rep = classification_report(
        y,
        preds,
        spec,
        categories=[d.example.category for d in rows],
        true_calls=[d.example.calls for d in rows],
    )
    if n_boot > 0 and rows:
        rep["ci95"] = {
            m: bootstrap_ci(y, preds, spec, metric=m, n_boot=n_boot)
            for m in ("exact_accuracy", "macro_f1")
        }
    rep["pc_predict_ms_per_sample"] = 1000 * secs / max(len(rows), 1)
    rep["predictions"] = preds
    return rep


def evaluate_classifier(
    clf: Classifier, splits: Splits, spec: LabelSpec, *, n_boot: int = 1000
) -> dict[str, Any]:
    """Dev と test(対象内)、test の対象外サンプルで評価する。"""
    test = _report(clf, splits.test, spec, n_boot)
    oos_rows = splits.test_out_of_scope
    oos_preds = clf.predict([e.text for e, _ in oos_rows])
    oos = out_of_scope_report(oos_preds, [str(r.reason) for _, r in oos_rows], spec)
    n_all = len(splits.test) + len(oos_rows)
    return {
        "dev": {k: v for k, v in _report(clf, splits.dev, spec, 0).items() if k != "predictions"},
        "test": test,
        "test_out_of_scope": oos,
        "coverage": len(splits.test) / n_all if n_all else None,
        # 元の JSON 生成タスク(全 validation)に対する機能差: 対象外は必ず不正解
        "full_test_json_exact": test.get("json_exact_count", 0) / n_all if n_all else None,
        "n_test_all": n_all,
    }


def run_metadata(seed: int, data_sha256: dict[str, str], spec: LabelSpec) -> dict[str, Any]:
    """結果に埋め込む再現情報。"""
    return {
        "gateact_version": __version__,
        "git_sha": git_sha(),
        "seed": seed,
        "data_sha256": data_sha256,
        "label_spec": spec.version,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def write_result(path: str | Path, result: dict[str, Any]) -> Path:
    """結果 JSON を書く。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return p

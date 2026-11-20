"""固定 Action 分類の評価指標(定義は docs/evaluation.md)。

主指標は Exact Accuracy と Macro-F1。ロボットが誤って動く割合(False Action Rate)を
別に重視する。JapaneseTinyAgentLM の JSON 生成タスクとの機能差を見るため、予測クラスを
既定の引数で call 列に戻したときの一致率(``json_exact``)も出す。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from gateact.data.labels import LabelSpec
from gateact.data.schema import Call, canonicalize


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    spec: LabelSpec,
    *,
    categories: Sequence[str] | None = None,
    true_calls: Sequence[list[Call]] | None = None,
) -> dict[str, Any]:
    """対象内サンプルの分類結果を集計する。

    Args:
        y_true: 正解クラス名。
        y_pred: 予測クラス名。
        spec: クラス定義。
        categories: データセットのカテゴリ(否定表現の正解率などに使う)。
        true_calls: 元の正解 call 列(``json_exact`` の計算に使う)。

    Returns:
        JSON にできる dict。Macro-F1 は正解または予測に1件以上現れるクラスの平均。
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    n = len(y_true)
    na = spec.no_action
    k = spec.num_classes
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[spec.index(t), spec.index(p)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1s: list[float] = []
    present: list[str] = []
    for i, name in enumerate(spec.names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        p, r, f = _prf(tp, fp, fn)
        per_class[name] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
        if tp + fp + fn > 0:
            present.append(name)
            f1s.append(f)

    correct = int(np.trace(cm))
    true_na = [t == na for t in y_true]
    pred_na = [p == na for p in y_pred]
    n_true_na = sum(true_na)
    n_true_act = n - n_true_na
    false_action = sum(t and not p for t, p in zip(true_na, pred_na, strict=True))
    missed_action = sum((not t) and p for t, p in zip(true_na, pred_na, strict=True))
    na_tp = sum(t and p for t, p in zip(true_na, pred_na, strict=True))

    report: dict[str, Any] = {
        "n": n,
        "exact": correct,
        "exact_accuracy": _rate(correct, n),
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "macro_f1_classes": present,
        "per_class": per_class,
        "false_action_rate": _rate(false_action, n_true_na),
        "false_action": false_action,
        "missed_action_rate": _rate(missed_action, n_true_act),
        "no_action": {
            "precision": _rate(na_tp, sum(pred_na)),
            "recall": _rate(na_tp, n_true_na),
            "support": n_true_na,
        },
        "confusion": cm.tolist(),
        "labels": list(spec.names),
    }
    if categories is not None:
        groups: dict[str, list[bool]] = defaultdict(list)
        for c, t, p in zip(categories, y_true, y_pred, strict=True):
            groups[c].append(t == p)
        report["by_category"] = {
            c: {"n": len(v), "accuracy": _rate(sum(v), len(v))} for c, v in sorted(groups.items())
        }
        neg = groups.get("negation", [])
        report["negation_accuracy"] = _rate(sum(neg), len(neg))
    if true_calls is not None:
        ok = sum(
            spec.default_calls(p) == canonicalize(c)
            for p, c in zip(y_pred, true_calls, strict=True)
        )
        report["json_exact"] = _rate(ok, n)
        report["json_exact_count"] = ok
    return report


def out_of_scope_report(
    y_pred: Sequence[str], reasons: Sequence[str], spec: LabelSpec
) -> dict[str, Any]:
    """対象外サンプル(分類器には正解が無い)に対する予測の分布を集計する。"""
    by: dict[str, Counter[str]] = defaultdict(Counter)
    for p, r in zip(y_pred, reasons, strict=True):
        by[r][p] += 1
    n = len(y_pred)
    return {
        "n": n,
        "predicted_no_action_rate": _rate(sum(p == spec.no_action for p in y_pred), n),
        "by_reason": {
            r: {"n": sum(c.values()), "predictions": dict(c.most_common())}
            for r, c in sorted(by.items())
        },
    }


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    spec: LabelSpec,
    *,
    metric: str = "exact_accuracy",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """サンプルの重複抽出による指標の (1 - alpha) 信頼区間(percentile 法)。"""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        v = classification_report(yt[idx].tolist(), yp[idx].tolist(), spec)[metric]
        if v is not None:
            vals.append(float(v))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)

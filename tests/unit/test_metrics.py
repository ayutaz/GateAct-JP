from pathlib import Path

import pytest

from gateact.data.labels import LabelSpec
from gateact.evaluation.metrics import bootstrap_ci, classification_report, out_of_scope_report

ROOT = Path(__file__).resolve().parents[2]
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")


def test_perfect_predictions() -> None:
    y = ["no_action", "nod", "look_left", "no_action"]
    r = classification_report(y, y, SPEC)
    assert r["exact_accuracy"] == 1.0
    assert r["macro_f1"] == 1.0
    assert r["false_action_rate"] == 0.0
    assert r["no_action"]["precision"] == 1.0 and r["no_action"]["recall"] == 1.0


def test_false_action_and_missed_action() -> None:
    y_true = ["no_action", "no_action", "nod", "nod"]
    y_pred = ["nod", "no_action", "no_action", "nod"]
    r = classification_report(y_true, y_pred, SPEC)
    assert r["exact_accuracy"] == 0.5
    assert r["false_action_rate"] == 0.5  # 2 件の no_action のうち 1 件が動いた
    assert r["missed_action_rate"] == 0.5
    assert r["no_action"]["precision"] == 0.5
    assert r["per_class"]["nod"]["recall"] == 0.5


def test_macro_f1_only_over_present_classes() -> None:
    y_true = ["nod", "bow"]
    y_pred = ["nod", "nod"]
    r = classification_report(y_true, y_pred, SPEC)
    # nod: P=0.5 R=1 F1=2/3、bow: P=0 R=0 F1=0 → macro = 1/3
    assert r["macro_f1"] == pytest.approx(1 / 3)
    assert "look_up" not in r["macro_f1_classes"]


def test_category_and_negation_accuracy() -> None:
    y_true = ["no_action", "no_action", "nod"]
    y_pred = ["no_action", "nod", "nod"]
    cats = ["negation", "negation", "single"]
    r = classification_report(y_true, y_pred, SPEC, categories=cats)
    assert r["negation_accuracy"] == 0.5
    assert r["by_category"]["single"]["accuracy"] == 1.0


def test_json_exact_uses_default_args() -> None:
    y_true = ["look_left", "look_left"]
    y_pred = ["look_left", "look_left"]
    calls = [
        [{"name": "look", "arguments": {"direction": "left", "amount": "normal"}}],
        [{"name": "look", "arguments": {"direction": "left", "amount": "slight"}}],
    ]
    r = classification_report(y_true, y_pred, SPEC, true_calls=calls)
    assert r["exact_accuracy"] == 1.0
    assert r["json_exact"] == 0.5


def test_confusion_matrix_shape() -> None:
    r = classification_report(["nod"], ["bow"], SPEC)
    cm = r["confusion"]
    assert len(cm) == SPEC.num_classes and all(len(row) == SPEC.num_classes for row in cm)
    assert cm[SPEC.index("nod")][SPEC.index("bow")] == 1


def test_out_of_scope_report() -> None:
    r = out_of_scope_report(
        ["nod", "no_action", "nod"], ["multi_action", "unsupported:turn", "multi_action"], SPEC
    )
    assert r["n"] == 3
    assert r["predicted_no_action_rate"] == pytest.approx(1 / 3)
    assert r["by_reason"]["multi_action"]["n"] == 2


def test_bootstrap_ci_contains_point_estimate_and_is_seeded() -> None:
    y_true = ["nod"] * 50 + ["bow"] * 50
    y_pred = ["nod"] * 40 + ["bow"] * 10 + ["bow"] * 45 + ["nod"] * 5
    lo, hi = bootstrap_ci(y_true, y_pred, SPEC, metric="exact_accuracy", n_boot=200, seed=0)
    assert lo <= 0.85 <= hi
    assert (lo, hi) == bootstrap_ci(
        y_true, y_pred, SPEC, metric="exact_accuracy", n_boot=200, seed=0
    )

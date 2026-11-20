import json
from pathlib import Path

import pytest

from gateact.evaluation.summary import aggregate, load_results, to_markdown


def _write(d: Path, name: str, seed: int, acc: float) -> None:
    r = {"kind": "x", "name": name, "meta": {"seed": seed},
         "test": {"exact_accuracy": acc, "macro_f1": acc, "false_action_rate": 0.1, "json_exact": acc,
                  "negation_accuracy": 0.5, "no_action": {"precision": 1.0, "recall": 0.9}},
         "full_test_json_exact": acc / 2}  # fmt: skip
    (d / f"{name}_s{seed}.json").write_text(json.dumps(r))


def test_aggregate_mean_sd(tmp_path: Path) -> None:
    _write(tmp_path, "m", 0, 0.8)
    _write(tmp_path, "m", 1, 0.9)
    _write(tmp_path, "z", 0, 0.5)
    rows = aggregate(load_results([tmp_path]))
    m = rows["m"]
    assert m["seeds"] == [0, 1]
    assert m["exact_accuracy"]["mean"] == pytest.approx(0.85)
    assert m["exact_accuracy"]["sd"] == pytest.approx(0.0707, abs=1e-3)
    assert rows["z"]["exact_accuracy"]["sd"] is None
    md = to_markdown(rows)
    assert "| m | 2 | 85.00 ± 7.07 |" in md

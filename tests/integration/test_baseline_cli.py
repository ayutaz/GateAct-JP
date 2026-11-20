import json
import shutil
from pathlib import Path

import pytest

from gateact.cli import main

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    fx = ROOT / "tests" / "fixtures" / "action_mini.jsonl"
    shutil.copy(fx, d / "train.jsonl")
    shutil.copy(fx, d / "validation.jsonl")
    return d


@pytest.mark.parametrize("kind", ["rules", "ngram", "bitlinear"])
def test_baseline_cli_writes_result(kind: str, data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    rc = main(["baseline", "--kind", kind, "--epochs", "2", "--data-dir", str(data_dir),
               "--dev-frac", "0.0", "--n-boot", "10", "--out", str(out),
               "--runs", str(tmp_path / "runs")])  # fmt: skip
    assert rc == 0
    [path] = list(out.glob("*.json"))
    r = json.loads(path.read_text())
    assert r["kind"] == kind
    assert r["meta"]["data_sha256"]["train.jsonl"]
    # fixture は train=validation なので漏れ除去で学習データは空になる → 評価だけは動く
    assert r["split_sizes"]["removed_leakage"] == 24
    assert r["test"]["n"] == 20 and r["test_out_of_scope"]["n"] == 4
    assert 0 <= r["full_test_json_exact"] <= 1
    if kind != "rules":
        assert Path(r["checkpoint"]).exists()


def test_train_cli_writes_checkpoint_and_result(data_dir: Path, tmp_path: Path) -> None:
    out, runs = tmp_path / "out", tmp_path / "runs"
    rc = main(["train", str(ROOT / "tests/fixtures/lut_tiny.json"), "--data-dir", str(data_dir),
               "--dev-frac", "0.0", "--n-boot", "5", "--out", str(out), "--runs", str(runs)])  # fmt: skip
    assert rc == 0
    r = json.loads((out / "lut_tiny_s0.json").read_text())
    assert r["kind"] == "lut" and r["model"]["gates_before_compression"] == 512
    assert Path(r["checkpoint"]).exists()

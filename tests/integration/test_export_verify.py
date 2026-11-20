"""M3: 学習済み checkpoint → IR → 圧縮 → C の全段一致とサイズ測定。"""

import json
from pathlib import Path

import pytest

from gateact.data.checks import Labeled
from gateact.data.io import load_jsonl
from gateact.data.labels import LabelSpec
from gateact.export.verify import export_and_verify
from gateact.features.hashing import FeatureConfig
from gateact.models.lut import LutModelConfig
from gateact.training.lut_trainer import TrainConfig, save_checkpoint, train_lut

ROOT = Path(__file__).resolve().parents[2]
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rows = load_jsonl(ROOT / "tests" / "fixtures" / "action_mini.jsonl")
    data = [Labeled(r, SPEC.assign(r.calls)) for r in rows if SPEC.assign(r.calls).in_scope]
    feat = FeatureConfig(width=256)
    mcfg = LutModelConfig(input_bits=256, hidden=[256], output_gates=256, num_classes=16, tau=4.0,
                          connections="unique")  # fmt: skip
    out = train_lut(mcfg, feat, TrainConfig(epochs=5, lr=0.05, batch_size=8), data, data, SPEC)
    return save_checkpoint(tmp_path_factory.mktemp("ck") / "m.pt", out.model, mcfg, feat, SPEC)


@pytest.mark.needs_cc
@pytest.mark.parametrize("dense_flat", ["off", "on"])
def test_all_stages_match(checkpoint: Path, tmp_path: Path, dense_flat: str) -> None:
    texts = [r.text for r in load_jsonl(ROOT / "tests" / "fixtures" / "action_mini.jsonl")]
    texts += ["", "ＡＢＣ", "ｶﾞｷﾞ", "今日は晴れ"]
    report = export_and_verify(checkpoint, texts, tmp_path, n_samples=8, dense_flat=dense_flat)
    p = report["parity"]
    assert p["n"] == len(texts)
    assert p["torch_vs_ir_mismatch"] == 0
    assert p["torch_vs_ir_raw_mismatch"] == 0
    assert p["ir_raw_vs_ir_compressed_mismatch"] == 0
    assert p["ir_vs_c_mismatch"] == 0
    assert p["prediction_agreement"] == 1.0
    s = report["sizes"]
    assert s["gates_raw"] == 512
    assert 0 < s["gates_compressed"] <= 512
    assert s["ctx_bytes"] > 0 and s["model_c_bytes"] > 0 and s["host_object_text_bytes"] > 0
    assert (tmp_path / "model" / "samples.h").exists()
    saved = json.loads((tmp_path / "export_report.json").read_text())
    assert saved["parity"] == p


@pytest.mark.needs_cc
def test_lut6_model_exports_through_split(tmp_path: Path) -> None:
    rows = load_jsonl(ROOT / "tests" / "fixtures" / "action_mini.jsonl")
    data = [Labeled(r, SPEC.assign(r.calls)) for r in rows if SPEC.assign(r.calls).in_scope]
    feat = FeatureConfig(width=256)
    mcfg = LutModelConfig(input_bits=256, hidden=[128], output_gates=128, num_classes=16, tau=4.0,
                          n_inputs=6, parameterization="warp", connections="unique")  # fmt: skip
    out = train_lut(mcfg, feat, TrainConfig(epochs=3, lr=0.05, batch_size=8), data, data, SPEC)
    ck = save_checkpoint(tmp_path / "m6.pt", out.model, mcfg, feat, SPEC)
    texts = [r.text for r in rows]
    rep = export_and_verify(ck, texts, tmp_path / "o", n_samples=4)
    p = rep["parity"]
    assert p["torch_vs_ir_mismatch"] == 0 and p["torch_vs_ir_raw_mismatch"] == 0
    assert p["ir_raw_vs_ir_compressed_mismatch"] == 0 and p["ir_vs_c_mismatch"] == 0

from pathlib import Path

import numpy as np
import torch

from gateact.data.checks import Labeled
from gateact.data.io import load_jsonl
from gateact.data.labels import LabelSpec
from gateact.features.hashing import FeatureConfig
from gateact.models.lut import LutModelConfig
from gateact.training.lut_trainer import (
    TrainConfig,
    hard_predict_ids,
    load_checkpoint,
    save_checkpoint,
    train_lut,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")
FEAT = FeatureConfig(width=256)
MODEL = LutModelConfig(input_bits=256, hidden=[512], output_gates=512, num_classes=16, tau=4.0,
                       connections="unique")  # fmt: skip


def _data() -> list[Labeled]:
    rows = load_jsonl(ROOT / "tests" / "fixtures" / "action_mini.jsonl")
    return [Labeled(r, SPEC.assign(r.calls)) for r in rows if SPEC.assign(r.calls).in_scope]


def test_train_lut_logs_soft_and_hard_and_learns() -> None:
    data = _data()
    out = train_lut(
        MODEL, FEAT, TrainConfig(epochs=40, lr=0.05, batch_size=8, seed=0), data, data, SPEC
    )
    assert len(out.log) == 40
    assert {"train_loss", "dev_soft_acc", "dev_hard_acc"} <= set(out.log[0])
    assert out.best_dev_hard_acc == max(e["dev_hard_acc"] for e in out.log)
    assert out.best_dev_hard_acc > 0.5


def test_train_is_reproducible() -> None:
    data = _data()
    cfg = TrainConfig(epochs=3, lr=0.05, batch_size=8, seed=1)
    a = train_lut(MODEL, FEAT, cfg, data, data, SPEC)
    b = train_lut(MODEL, FEAT, cfg, data, data, SPEC)
    assert [e["train_loss"] for e in a.log] == [e["train_loss"] for e in b.log]


def test_checkpoint_roundtrip_preserves_hard_outputs(tmp_path: Path) -> None:
    data = _data()
    out = train_lut(MODEL, FEAT, TrainConfig(epochs=2, seed=0, batch_size=8), data, data, SPEC)
    path = save_checkpoint(tmp_path / "m.pt", out.model, MODEL, FEAT, SPEC, meta={"x": 1})
    model2, mcfg, fcfg, meta = load_checkpoint(path)
    assert mcfg == MODEL and fcfg == FEAT and meta["x"] == 1 and meta["label_spec"] == SPEC.version
    bits = np.random.default_rng(0).integers(0, 2, (10, 256), dtype=np.uint8)
    assert np.array_equal(
        hard_predict_ids(out.model, bits, MODEL.tau), hard_predict_ids(model2, bits, MODEL.tau)
    )


def test_hard_predict_ties_pick_lowest_index() -> None:
    class Const(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0, 3.0, 3.0]]).repeat(x.shape[0], 1)

    assert hard_predict_ids(Const(), np.zeros((2, 4), dtype=np.uint8), 1.0).tolist() == [1, 1]

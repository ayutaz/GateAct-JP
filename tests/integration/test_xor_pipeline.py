"""M0: Griffith XOR を学習 → hard 精度 → IR → C 生成 → Python/IR/C 一致。"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gateact.export.pipeline import (
    CModelRunner,
    export_bundle,
    simulate_bundle,
    torch_hard_scores,
)
from gateact.models.xor import XOR_CONFIG as CFG
from gateact.models.xor import XOR_X as X
from gateact.models.xor import XOR_Y as Y
from gateact.models.xor import train_xor


@pytest.fixture(scope="module")
def xor_model() -> torch.nn.Module:
    return train_xor()


def test_hard_accuracy_is_perfect(xor_model: torch.nn.Module) -> None:
    scores = torch_hard_scores(xor_model, X, tau=CFG.tau)
    assert np.array_equal(scores.argmax(1), Y)


def test_torch_hard_equals_ir_simulate(xor_model: torch.nn.Module) -> None:
    bundle = export_bundle(xor_model, input_width=2, name="xor", compress="none")
    assert np.array_equal(torch_hard_scores(xor_model, X, tau=CFG.tau), simulate_bundle(bundle, X))


def test_compression_is_lossless(xor_model: torch.nn.Module) -> None:
    raw = export_bundle(xor_model, input_width=2, name="xor", compress="none")
    comp = export_bundle(xor_model, input_width=2, name="xor", compress="fast")
    all_inputs = X
    assert np.array_equal(simulate_bundle(raw, all_inputs), simulate_bundle(comp, all_inputs))


@pytest.mark.needs_cc
def test_c_equals_ir_simulate(xor_model: torch.nn.Module, tmp_path: Path) -> None:
    bundle = export_bundle(xor_model, input_width=2, name="xor", compress="fast")
    runner = CModelRunner.build(bundle, tmp_path)
    assert np.array_equal(runner.run(X), simulate_bundle(bundle, X))

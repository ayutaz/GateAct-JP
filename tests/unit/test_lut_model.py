import pytest
import torch

from gateact.models.lut import LutModelConfig, build_lut_model
from gateact.utils import seed_everything


def test_output_shape_matches_num_classes() -> None:
    cfg = LutModelConfig(input_bits=16, hidden=[32], output_gates=32, num_classes=4, n_inputs=2)
    model = build_lut_model(cfg)
    y = model(torch.rand(5, 16))
    assert y.shape == (5, 4)


def test_output_gates_must_be_divisible_by_classes() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_lut_model(LutModelConfig(input_bits=8, hidden=[16], output_gates=30, num_classes=4))


def test_backward_updates_parameters() -> None:
    seed_everything(0)
    cfg = LutModelConfig(input_bits=8, hidden=[16], output_gates=16, num_classes=2)
    model = build_lut_model(cfg)
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = torch.nn.functional.cross_entropy(model(torch.rand(4, 8)), torch.tensor([0, 1, 0, 1]))
    loss.backward()
    opt.step()
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))


def test_hard_eval_outputs_integer_counts_over_tau() -> None:
    cfg = LutModelConfig(input_bits=8, hidden=[16], output_gates=16, num_classes=2, tau=4.0)
    model = build_lut_model(cfg).eval()
    with torch.no_grad():
        y = model(torch.randint(0, 2, (6, 8)).float()) * cfg.tau
    assert torch.allclose(y, y.round())


def test_seed_everything_is_reproducible() -> None:
    cfg = LutModelConfig(input_bits=8, hidden=[16], output_gates=16, num_classes=2)
    seed_everything(123)
    a = [p.detach().clone() for p in build_lut_model(cfg).parameters()]
    seed_everything(123)
    b = [p.detach().clone() for p in build_lut_model(cfg).parameters()]
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))

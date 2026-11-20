import numpy as np
import torch
from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.ir.serialize import ir_from_dict, ir_to_dict
from griffith_logic.ir.simulate import simulate

from gateact.export.lut6 import has_wide_luts, split_wide_luts
from gateact.utils import seed_everything


def _random_lut6_ir(seed: int) -> dict:
    seed_everything(seed)
    m = torch.nn.Sequential(
        LutLayer(32, 48, n_inputs=6, parameterization="warp"),
        LutLayer(48, 32, n_inputs=6, parameterization="warp"),
        GroupSum(k=4, tau=2.0),
    )
    for p in m.parameters():  # ランダムな真理値表になるよう重みを乱す
        torch.nn.init.normal_(p, std=3.0)
    return ir_to_dict(export_ir(m.eval(), input_width=32, name="t"))


def test_split_is_exact_on_all_layers() -> None:
    for seed in range(3):
        d = _random_lut6_ir(seed)
        assert has_wide_luts(d)
        s = split_wide_luts(d)
        assert not has_wide_luts(s)
        assert all(n["params"]["n_inputs"] <= 5 for n in s["nodes"] if n["type"] == "lut_layer")
        x = np.random.default_rng(seed).integers(0, 2, (512, 32), dtype=np.uint8)
        a = simulate(ir_from_dict(d), x)
        b = simulate(ir_from_dict(s), x)
        assert np.array_equal(a, b)


def test_split_structure() -> None:
    s = split_wide_luts(_random_lut6_ir(0))
    luts = [n for n in s["nodes"] if n["type"] == "lut_layer"]
    assert [(n["params"]["n_inputs"], n["params"]["out_width"]) for n in luts] == [
        (5, 144),
        (3, 48),
        (5, 96),
        (3, 32),
    ]
    ids = [n["id"] for n in s["nodes"]]
    chain = [(e["src"], e["dst"]) for e in s["edges"]]
    assert chain == list(zip(["x", *ids], [*ids, "y"], strict=True))


def test_ir_without_wide_luts_is_unchanged() -> None:
    seed_everything(0)
    m = torch.nn.Sequential(LutLayer(8, 8, n_inputs=4, parameterization="warp"), GroupSum(k=2))
    d = ir_to_dict(export_ir(m.eval(), input_width=8, name="t"))
    assert split_wide_luts(d) == d

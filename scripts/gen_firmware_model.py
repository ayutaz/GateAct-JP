"""ファームウェア用のモデル C ソースと samples.h を生成する。

uv run python scripts/gen_firmware_model.py xor
uv run python scripts/gen_firmware_model.py bundle --bundle B --samples S --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gateact.export.codegen import write_c_model, write_samples_header
from gateact.export.pipeline import export_bundle, simulate_bundle
from gateact.models.xor import XOR_X, XOR_Y, train_xor

ROOT = Path(__file__).resolve().parents[1]


def gen_xor(out: Path) -> None:
    bundle = export_bundle(train_xor(), input_width=2, name="xor", compress="fast")
    write_c_model(bundle, out)
    write_samples_header(
        out / "samples.h",
        XOR_X,
        labels=XOR_Y.tolist(),
        expected_scores=simulate_bundle(bundle, XOR_X),
    )
    print(f"wrote {out}")


def gen_gateact_smoke(out: Path) -> None:
    import tempfile

    from gateact.data.checks import Labeled
    from gateact.data.io import load_jsonl
    from gateact.data.labels import LabelSpec
    from gateact.export.verify import export_and_verify
    from gateact.features.hashing import FeatureConfig
    from gateact.models.lut import LutModelConfig
    from gateact.training.lut_trainer import TrainConfig, save_checkpoint, train_lut

    spec = LabelSpec.load(ROOT / "configs/labels_v1.json")
    rows = load_jsonl(ROOT / "tests/fixtures/action_mini.jsonl")
    data = [Labeled(r, spec.assign(r.calls)) for r in rows if spec.assign(r.calls).in_scope]
    feat = FeatureConfig(width=512)
    mcfg = LutModelConfig(
        input_bits=512, hidden=[1024], output_gates=1024, num_classes=16, tau=8.0,
        connections="unique",
    )  # fmt: skip
    res = train_lut(mcfg, feat, TrainConfig(epochs=10, lr=0.05, batch_size=8), data, data, spec)
    with tempfile.TemporaryDirectory() as tmp:
        ck = save_checkpoint(Path(tmp) / "gateact_smoke.pt", res.model, mcfg, feat, spec)
        rep = export_and_verify(
            ck,
            [d.example.text for d in data],
            out,
            labels=[spec.index(d.label) for d in data],
            n_samples=len(data),
        )
    print(f"wrote {out / 'model'} parity={rep['parity']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("xor")
    x.add_argument("--out", type=Path, default=ROOT / "firmware/esp32s3/xor/main/model")
    g = sub.add_parser("gateact-smoke")
    g.add_argument("--out", type=Path, default=ROOT / "firmware/esp32s3/gateact/main")
    args = ap.parse_args()
    if args.cmd == "xor":
        gen_xor(args.out)
    elif args.cmd == "gateact-smoke":
        gen_gateact_smoke(args.out)


if __name__ == "__main__":
    main()

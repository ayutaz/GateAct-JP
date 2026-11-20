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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("xor")
    x.add_argument("--out", type=Path, default=ROOT / "firmware/esp32s3/xor/main/model")
    args = ap.parse_args()
    if args.cmd == "xor":
        gen_xor(args.out)


if __name__ == "__main__":
    main()

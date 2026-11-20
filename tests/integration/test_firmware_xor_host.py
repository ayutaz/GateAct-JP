"""M0: ESP32-S3 用 XOR ファームウェアの main.c をホストでビルドし、IR と一致することを確認する。"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gateact.export.codegen import write_c_model, write_samples_header
from gateact.export.pipeline import export_bundle, simulate_bundle
from gateact.models.xor import XOR_X as X
from gateact.models.xor import XOR_Y as Y
from gateact.models.xor import train_xor

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT / "firmware" / "esp32s3" / "xor" / "main"


def test_write_c_model_emits_self_contained_sources(tmp_path: Path) -> None:
    bundle = export_bundle(train_xor(), input_width=2, name="xor", compress="fast")
    files = write_c_model(bundle, tmp_path)
    for name in ("model.h", "model.c", "mdl_kernels.h", "mdl_kernels.c", "mdl_simd.h"):
        assert (tmp_path / name).is_file()
        assert tmp_path / name in files
    info = json.loads((tmp_path / "model_info.json").read_text())
    assert info["input_width"] == 2 and info["output_width"] == 2


def test_samples_header_packs_bits_lsb_first(tmp_path: Path) -> None:
    bits = np.zeros((1, 10), dtype=np.uint8)
    bits[0, [0, 3, 9]] = 1
    path = write_samples_header(
        tmp_path / "samples.h", bits, labels=[1], expected_scores=np.array([[2, 5]])
    )
    text = path.read_text()
    assert "#define SAMPLE_COUNT 1" in text
    assert "{0x09, 0x02}" in text  # bit0+bit3 -> 0x09, bit9 -> byte1 bit1 -> 0x02
    assert "{2, 5}" in text


@pytest.mark.needs_cc
def test_xor_firmware_main_runs_on_host(tmp_path: Path) -> None:
    bundle = export_bundle(train_xor(), input_width=2, name="xor", compress="fast")
    model_dir = tmp_path / "model"
    write_c_model(bundle, model_dir)
    expected = simulate_bundle(bundle, X)
    write_samples_header(model_dir / "samples.h", X, labels=Y.tolist(), expected_scores=expected)
    exe = tmp_path / "host_xor"
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc is not None
    subprocess.run(
        [cc, "-std=c99", "-O2", "-Wall", "-Werror", "-DHOST_BUILD", "-DMDL_NO_DEFAULT_CTX",
         f"-I{model_dir}", str(FW / "main.c"), str(model_dir / "model.c"),
         str(model_dir / "mdl_kernels.c"), "-o", str(exe)],
        check=True,
    )  # fmt: skip
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    result = json.loads(next(ln for ln in out.splitlines() if ln.startswith("RESULT "))[7:])
    assert result["n"] == 4
    assert result["score_mismatch"] == 0
    assert result["correct"] == 4

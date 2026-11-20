"""ファームウェアの bit-linear / n-gram 版(GA_MODEL_KIND 2/3)がホストで Python と一致すること。"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from gateact.baselines.bitlinear import BitLinearClassifier
from gateact.baselines.export import export_baseline_firmware
from gateact.baselines.ngram import NgramClassifier
from gateact.benchmarks.device import HostProcess, query, read_boot
from gateact.data.checks import Labeled
from gateact.data.io import load_jsonl
from gateact.data.labels import LabelSpec
from gateact.features.hashing import FeatureConfig

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT / "firmware" / "esp32s3" / "gateact" / "main"
COMMON = ROOT / "firmware" / "common"
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")

pytestmark = pytest.mark.needs_cc


def _data() -> list[Labeled]:
    rows = load_jsonl(ROOT / "tests" / "fixtures" / "action_mini.jsonl")
    return [Labeled(r, SPEC.assign(r.calls)) for r in rows if SPEC.assign(r.calls).in_scope]


@pytest.mark.parametrize("kind", ["bitlinear", "ngram"])
def test_baseline_firmware_host(kind: str, tmp_path: Path) -> None:
    data = _data()
    clf = (BitLinearClassifier(SPEC, FeatureConfig(width=512), epochs=20, lr=0.05) if kind == "bitlinear"
           else NgramClassifier(SPEC, buckets=4096, dim=16, epochs=20, lr=0.05))  # fmt: skip
    clf.fit(data, data)
    ck = tmp_path / "b.pt"
    torch.save({"kind": kind, "model": clf.describe(), "state_dict": clf.net.state_dict(),
                "labels": SPEC.names}, ck)  # fmt: skip
    texts = [d.example.text for d in data]
    rep = export_baseline_firmware(
        ck, SPEC, texts, tmp_path, labels=[SPEC.index(d.label) for d in data]
    )
    m = tmp_path / "model"
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc
    exe = tmp_path / "fw"
    subprocess.run(
        [cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-DHOST_BUILD", "-DBENCH_ITERS=100",
         f"-I{COMMON}", f"-I{m}", str(FW / "main.c"), str(COMMON / "gateact_features.c"),
         str(COMMON / "gateact_baselines.c"), "-o", str(exe)],
        check=True,
    )  # fmt: skip
    t = HostProcess(exe)
    try:
        boot = read_boot(t, timeout=30)
        res = query(t, texts)
    finally:
        t.close()
    st = boot.records["SELFTEST"]
    assert st["score_mismatch"] == 0 and st["feature_mismatch"] == 0 and st["errors"] == 0
    assert boot.records["INFO"]["kind"] == (2 if kind == "bitlinear" else 3)
    assert np.array_equal(np.array([r["scores"] for r in res]), rep["scores"])

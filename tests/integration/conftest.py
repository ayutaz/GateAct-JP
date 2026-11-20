"""統合テスト共通の fixture。"""

import shutil
import subprocess
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
FW = ROOT / "firmware" / "esp32s3" / "gateact" / "main"
COMMON = ROOT / "firmware" / "common"
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")


@pytest.fixture(scope="session")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, list[str]]:
    tmp = tmp_path_factory.mktemp("fw")
    rows = load_jsonl(ROOT / "tests" / "fixtures" / "action_mini.jsonl")
    data = [Labeled(r, SPEC.assign(r.calls)) for r in rows if SPEC.assign(r.calls).in_scope]
    feat = FeatureConfig(width=256)
    mcfg = LutModelConfig(input_bits=256, hidden=[256], output_gates=256, num_classes=16, tau=4.0,
                          connections="unique")  # fmt: skip
    out = train_lut(mcfg, feat, TrainConfig(epochs=5, lr=0.05, batch_size=8), data, data, SPEC)
    ck = save_checkpoint(tmp / "m.pt", out.model, mcfg, feat, SPEC)
    texts = [r.text for r in rows]
    labels = [SPEC.index(d.label) for d in data]
    export_and_verify(ck, [d.example.text for d in data], tmp, labels=labels, n_samples=len(data))
    exe = tmp / "gateact_host"
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc
    m = tmp / "model"
    subprocess.run(
        [cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-DHOST_BUILD", "-DMDL_NO_DEFAULT_CTX",
         "-DBENCH_ITERS=200", f"-I{COMMON}", f"-I{m}", str(FW / "main.c"),
         str(COMMON / "gateact_features.c"), str(m / "model.c"), str(m / "mdl_kernels.c"),
         "-o", str(exe)],
        check=True,
    )  # fmt: skip
    return exe, ck, texts

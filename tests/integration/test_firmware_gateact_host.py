"""M4: ESP32-S3 ファームウェア main.c をホストでビルドし、テキスト → Action が Python と一致すること。"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gateact.data.checks import Labeled
from gateact.data.io import load_jsonl
from gateact.data.labels import LabelSpec
from gateact.export.pipeline import export_bundle, simulate_bundle
from gateact.export.verify import export_and_verify
from gateact.features.hashing import FeatureConfig, extract_bits
from gateact.models.lut import LutModelConfig
from gateact.training.lut_trainer import TrainConfig, load_checkpoint, save_checkpoint, train_lut

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT / "firmware" / "esp32s3" / "gateact" / "main"
COMMON = ROOT / "firmware" / "common"
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")
pytestmark = pytest.mark.needs_cc


@pytest.fixture(scope="module")
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


def _run(exe: Path, lines: list[bytes]) -> dict[str, list[dict]]:
    out = subprocess.run([str(exe)], input=b"\n".join(lines) + b"\n", capture_output=True, check=True)
    tags: dict[str, list[dict]] = {}
    for ln in out.stdout.decode().splitlines():
        tag, _, body = ln.partition(" ")
        if body.startswith("{"):
            tags.setdefault(tag, []).append(json.loads(body))
    return tags


def test_selftest_and_bench(built: tuple[Path, Path, list[str]]) -> None:
    exe, _, _ = built
    tags = _run(exe, [])
    st = tags["SELFTEST"][0]
    assert st["errors"] == 0 and st["feature_mismatch"] == 0 and st["score_mismatch"] == 0
    assert st["n"] == st["labeled"] == 20
    b = tags["BENCH"][0]
    assert b["iters"] == 200 and b["total"]["p50"] <= b["total"]["p99"]
    assert tags["INFO"][0]["width"] == 256 and tags["INFO"][0]["classes"] == 16


def test_text_to_action_matches_python(built: tuple[Path, Path, list[str]]) -> None:
    exe, ck, texts = built
    texts = texts + ["ＡＢＣ　ｶﾞｷﾞ", "😀", " 右を 見て "]
    tags = _run(exe, [t.encode() for t in texts])
    results = tags["RESULT"]
    assert len(results) == len(texts)
    model, mcfg, fcfg, meta = load_checkpoint(ck)
    bundle = export_bundle(model, input_width=fcfg.width, name="g", compress="fast")
    expected = simulate_bundle(bundle, extract_bits(texts, fcfg))
    for r, exp in zip(results, expected, strict=True):
        assert r["ok"] is True
        assert r["scores"] == exp.tolist()
        assert r["id"] == int(np.argmax(exp))
        assert r["action"] == meta["labels"][r["id"]]


def test_invalid_utf8_line_is_rejected(built: tuple[Path, Path, list[str]]) -> None:
    exe, _, _ = built
    r = _run(exe, [b"\xff\xfe"])["RESULT"][0]
    assert r == {"ok": False, "error": -1, "bytes": 2}

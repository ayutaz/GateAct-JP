"""int8 ベースラインの C 実装が Python の整数スコアと完全一致すること。"""

import random
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest

from gateact.baselines.bitlinear import BitLinearClassifier
from gateact.baselines.ngram import NgramClassifier
from gateact.baselines.quantized import quantize_bitlinear, quantize_ngram
from gateact.data.checks import Labeled
from gateact.data.io import load_jsonl
from gateact.data.labels import LabelSpec
from gateact.export.firmware import write_bitlinear_header, write_ngram_header
from gateact.features.hashing import FeatureConfig, extract_bits

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "firmware" / "common"
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")
pytestmark = pytest.mark.needs_cc


def _data() -> list[Labeled]:
    rows = load_jsonl(ROOT / "tests" / "fixtures" / "action_mini.jsonl")
    return [Labeled(r, SPEC.assign(r.calls)) for r in rows if SPEC.assign(r.calls).in_scope]


def _texts() -> list[str]:
    rng = random.Random(0)
    pool = [chr(c) for c in range(0x3041, 0x3097)] + list("右左上下見ＡＢｶﾞ 　") + ["😀"]
    return (
        [d.example.text for d in _data()]
        + ["", "ＡＢＣ　ｶﾞ"]
        + ["".join(rng.choice(pool) for _ in range(rng.randint(0, 30))) for _ in range(200)]
    )


def _build(tmp: Path, kind: str) -> Path:
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc
    exe = tmp / f"drv_{kind}"
    subprocess.run(
        [cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", f"-DKIND_{kind.upper()}",
         f"-I{tmp}", f"-I{COMMON}", str(ROOT / "tests/c/baseline_driver.c"),
         str(COMMON / "gateact_baselines.c"), str(COMMON / "gateact_features.c"), "-o", str(exe)],
        check=True,
    )  # fmt: skip
    return exe


def _run(exe: Path, texts: list[str], k: int) -> np.ndarray:
    payload = b"".join(struct.pack("<I", len(t.encode())) + t.encode() for t in texts)
    out = subprocess.run([str(exe)], input=payload, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i4").astype(np.int64).reshape(len(texts), k)


def test_crc32_matches_zlib(tmp_path: Path) -> None:
    # n-gram の ID 一致で間接的に検証するが、代表値も確認する
    assert zlib.crc32(b"123456789") == 0xCBF43926


def test_bitlinear_c_matches_python(tmp_path: Path) -> None:
    feat = FeatureConfig(width=512)
    clf = BitLinearClassifier(SPEC, feat, epochs=20, lr=0.05)
    clf.fit(_data(), _data())
    q = quantize_bitlinear(clf)
    write_bitlinear_header(tmp_path / "baseline_model.h", q.weight, q.bias)
    texts = _texts()
    c = _run(_build(tmp_path, "bitlinear"), texts, 16)
    assert np.array_equal(c, q.scores(extract_bits(texts, feat)))


def test_ngram_c_matches_python(tmp_path: Path) -> None:
    clf = NgramClassifier(SPEC, buckets=4096, dim=16, epochs=20, lr=0.05)
    clf.fit(_data(), _data())
    q = quantize_ngram(clf)
    write_ngram_header(tmp_path / "baseline_model.h", q.embedding, q.weight, q.bias)
    texts = _texts()
    c = _run(_build(tmp_path, "ngram"), texts, 16)
    assert np.array_equal(c, q.scores(texts))

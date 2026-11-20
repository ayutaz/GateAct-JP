"""M4: ESP32-S3 ファームウェア main.c をホストでビルドし、テキスト → Action が Python と一致すること。"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gateact.export.pipeline import export_bundle, simulate_bundle
from gateact.features.hashing import extract_bits
from gateact.training.lut_trainer import load_checkpoint

pytestmark = pytest.mark.needs_cc


def _run(exe: Path, lines: list[bytes]) -> dict[str, list[dict]]:
    out = subprocess.run(
        [str(exe)], input=b"\n".join(lines) + b"\n", capture_output=True, check=True
    )
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

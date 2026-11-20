from pathlib import Path

import numpy as np

from gateact.baselines.bitlinear import BitLinearClassifier
from gateact.baselines.ngram import NgramClassifier
from gateact.baselines.quantized import (
    gateact_ngram_ids,
    quantize_bitlinear,
    quantize_ngram,
)
from gateact.data.checks import Labeled
from gateact.data.io import load_jsonl
from gateact.data.labels import LabelSpec
from gateact.features.hashing import FeatureConfig, extract_bits

ROOT = Path(__file__).resolve().parents[2]
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")


def _data() -> list[Labeled]:
    rows = load_jsonl(ROOT / "tests" / "fixtures" / "action_mini.jsonl")
    return [Labeled(r, SPEC.assign(r.calls)) for r in rows if SPEC.assign(r.calls).in_scope]


def test_gateact_ngram_ids_use_gateact_norm() -> None:
    import zlib

    s = "\x02ab c\x03"  # 空白の連続は1つ、前後は除く
    grams = [s[i : i + n] for n in (1, 2, 3) for i in range(len(s) - n + 1)]
    assert gateact_ngram_ids("  ＡＢ   c ", 1000) == [zlib.crc32(g.encode()) % 1000 for g in grams]


def test_quantized_bitlinear_is_integer_and_agrees() -> None:
    data = _data()
    feat = FeatureConfig(width=256)
    clf = BitLinearClassifier(SPEC, feat, epochs=60, lr=0.05)
    clf.fit(data, data)
    q = quantize_bitlinear(clf)
    assert q.weight.dtype == np.int8 and q.bias.dtype == np.int32
    bits = extract_bits([d.example.text for d in data], feat)
    s = q.scores(bits)
    assert s.dtype == np.int64
    float_pred = clf.predict_bits(bits)
    agree = np.mean([SPEC.names[i] == p for i, p in zip(s.argmax(1), float_pred, strict=True)])
    assert agree >= 0.9


def test_quantized_ngram_is_integer_and_agrees() -> None:
    data = _data()
    clf = NgramClassifier(SPEC, buckets=4096, dim=16, epochs=30, lr=0.05)
    clf.fit(data, data)
    q = quantize_ngram(clf)
    assert q.embedding.dtype == np.int8 and q.weight.dtype == np.int8
    texts = [d.example.text for d in data]
    s = q.scores(texts)
    agree = np.mean(
        [SPEC.names[i] == p for i, p in zip(s.argmax(1), clf.predict(texts), strict=True)]
    )
    assert agree >= 0.9
    assert q.max_abs_score(texts) < 2**31

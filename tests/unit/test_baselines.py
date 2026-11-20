from pathlib import Path

import pytest

from gateact.baselines.bitlinear import BitLinearClassifier
from gateact.baselines.ngram import NgramClassifier, ngram_ids
from gateact.baselines.rules import RuleClassifier
from gateact.data.checks import Labeled
from gateact.data.io import load_jsonl
from gateact.data.labels import LabelSpec
from gateact.features.hashing import FeatureConfig

ROOT = Path(__file__).resolve().parents[2]
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")


def _fixture() -> list[Labeled]:
    rows = load_jsonl(ROOT / "tests" / "fixtures" / "action_mini.jsonl")
    out = []
    for r in rows:
        res = SPEC.assign(r.calls)
        if res.in_scope:
            out.append(Labeled(r, res))
    return out


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("右を見て", "look_right"),
        ("ちょっと左を向いて", "look_left"),
        ("上を見上げて", "look_up"),
        ("下を向いて", "look_down"),
        ("正面を向いて", "look_center"),
        ("うなずいて", "nod"),
        ("首を横に振って", "shake"),
        ("おじぎして", "bow"),
        ("にっこり笑って", "expr_happy"),
        ("びっくりして", "expr_surprised"),
        ("悲しい顔をして", "expr_sad"),
        ("ライトを青くして", "led_on"),
        ("ライトを消して", "led_off"),
        ("音量を上げて", "volume_up"),
        ("音を小さくして", "volume_down"),
        ("右を見ないで", "no_action"),
        ("今日はいい天気だね", "no_action"),
        ("左じゃなくて右を見て", "look_right"),
    ],
)
def test_rules(text: str, label: str) -> None:
    assert RuleClassifier(SPEC).predict([text]) == [label]


def test_ngram_ids_match_jtalm_spec() -> None:
    # NFKC + lower、前後に \x02 / \x03、1〜3-gram、CRC-32 % buckets
    import zlib

    ids = ngram_ids("Ａb", buckets=1000)
    s = "\x02ab\x03"
    grams = [s[i : i + n] for n in (1, 2, 3) for i in range(len(s) - n + 1)]
    assert ids == [zlib.crc32(g.encode()) % 1000 for g in grams]


def test_ngram_classifier_fits_fixture() -> None:
    data = _fixture()
    clf = NgramClassifier(SPEC, buckets=4096, dim=16, epochs=30, lr=0.05, seed=0)
    clf.fit(data, data)
    preds = clf.predict([d.example.text for d in data])
    acc = sum(p == d.label for p, d in zip(preds, data, strict=True)) / len(data)
    assert acc >= 0.9
    assert clf.num_parameters == 4096 * 16 + 16 * 16 + 16


def test_bitlinear_classifier_fits_fixture() -> None:
    data = _fixture()
    clf = BitLinearClassifier(SPEC, FeatureConfig(width=512), epochs=60, lr=0.05, seed=0)
    clf.fit(data, data)
    preds = clf.predict([d.example.text for d in data])
    acc = sum(p == d.label for p, d in zip(preds, data, strict=True)) / len(data)
    assert acc >= 0.9

from pathlib import Path

from gateact.data.checks import (
    build_manifest,
    conflicting_duplicates,
    dev_split,
    find_duplicates,
    prepare_splits,
    split_overlap,
)
from gateact.data.io import Example, load_jsonl
from gateact.data.labels import LabelSpec

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "action_mini.jsonl"
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")
NOD = [{"name": "nod", "arguments": {"count": 1}}]


def ex(i: str, text: str, calls: list | None = None, cat: str = "single") -> Example:
    return Example(id=i, text=text, calls=calls if calls is not None else NOD, category=cat)


def test_find_duplicates_uses_normalized_text() -> None:
    rows = [ex("a", "うなずいて"), ex("b", "うなずいて "), ex("c", "おじぎ")]
    groups = find_duplicates(rows)
    assert groups == [["a", "b"]]


def test_conflicting_duplicates_detects_different_outputs() -> None:
    rows = [ex("a", "右"), ex("b", "右", [])]
    assert conflicting_duplicates(rows) == [["a", "b"]]
    assert conflicting_duplicates([ex("a", "右"), ex("b", "右")]) == []


def test_split_overlap() -> None:
    assert split_overlap([ex("a", "ＡＢ"), ex("b", "x")], [ex("c", "ab")]) == {"ab"}


def test_dev_split_is_deterministic_and_groups_same_text() -> None:
    rows = [ex(str(i), f"文{i % 50}") for i in range(500)]
    tr1, dv1 = dev_split(rows, frac=0.2)
    tr2, dv2 = dev_split(list(reversed(rows)), frac=0.2)
    assert {r.id for r in dv1} == {r.id for r in dv2}
    assert {r.text for r in tr1}.isdisjoint({r.text for r in dv1})
    assert 0.05 < len(dv1) / len(rows) < 0.4


def test_prepare_splits_removes_leakage_and_out_of_scope() -> None:
    train = [
        ex("t1", "うなずいて"),
        ex("t2", "右を見て", []),
        ex(
            "t3",
            "おじぎ",
            [{"name": "bow", "arguments": {}}, {"name": "nod", "arguments": {"count": 1}}],
        ),
    ]
    test = [ex("v1", "うなずいて")]
    s = prepare_splits(train, test, SPEC, dev_frac=0.0)
    assert [r.example.id for r in s.train] == ["t2"]
    assert s.removed_leakage == 1
    assert s.out_of_scope["train"] == {"multi_action": 1}
    assert [r.label for r in s.test] == ["nod"]


def test_manifest_contains_hashes_counts_and_coverage() -> None:
    rows = load_jsonl(FIXTURE)
    m = build_manifest({"train": FIXTURE}, {"train": rows}, SPEC, revision="abc")
    t = m["splits"]["train"]
    assert t["n"] == 24 and len(t["sha256"]) == 64
    assert m["revision"] == "abc"
    assert 0 < t["coverage"] < 1
    assert t["labels"]["no_action"] == 3
    assert t["out_of_scope"]["multi_action"] == 1
    assert sum(t["labels"].values()) + sum(t["out_of_scope"].values()) == 24

import json
from pathlib import Path

import pytest

from gateact.data.io import DataError, Example, load_jsonl
from gateact.data.labels import LabelSpec
from gateact.data.schema import canonical_json, parse_calls

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "action_mini.jsonl"
SPEC = LabelSpec.load(ROOT / "configs" / "labels_v1.json")


def _write(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _row(**kw: object) -> str:
    base = {"id": "a", "input": "右を見て", "output": "[]", "category": "no_action"}
    base.update(kw)
    return json.dumps(base, ensure_ascii=False)


# --- 読み込み -----------------------------------------------------------------


def test_load_fixture() -> None:
    rows = load_jsonl(FIXTURE)
    assert len(rows) == 24
    assert isinstance(rows[0], Example)
    assert rows[0].text == "右を見て"
    assert rows[0].calls == [
        {"name": "look", "arguments": {"amount": "normal", "direction": "right"}}
    ]


def test_invalid_json_line_is_rejected_with_line_number(tmp_path: Path) -> None:
    p = _write(tmp_path, [_row(), "{not json"])
    with pytest.raises(DataError, match="line 2"):
        load_jsonl(p)


def test_missing_field_is_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, [json.dumps({"id": "a", "input": "x", "category": "single"})])
    with pytest.raises(DataError, match="output"):
        load_jsonl(p)


def test_invalid_output_json_is_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, [_row(output="[{bad")])
    with pytest.raises(DataError, match="output"):
        load_jsonl(p)


def test_empty_input_text_is_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, [_row(input="")])
    with pytest.raises(DataError, match="input"):
        load_jsonl(p)


# --- schema --------------------------------------------------------------------


def test_parse_calls_requires_list_of_named_calls() -> None:
    assert parse_calls("[]") == []
    with pytest.raises(ValueError):
        parse_calls('{"name": "nod"}')
    with pytest.raises(ValueError):
        parse_calls('[{"arguments": {}}]')


def test_canonical_json_center_amount_is_normalized() -> None:
    a = [{"name": "look", "arguments": {"direction": "center", "amount": "large"}}]
    b = [{"arguments": {"amount": "normal", "direction": "center"}, "name": "look"}]
    assert canonical_json(a) == canonical_json(b)


# --- ラベル --------------------------------------------------------------------


def test_spec_has_16_classes_with_no_action_first() -> None:
    assert SPEC.num_classes == 16
    assert SPEC.names[0] == "no_action"


def test_empty_calls_is_no_action() -> None:
    r = SPEC.assign([])
    assert r.label == "no_action" and r.in_scope


def test_single_look_right_ignores_amount_but_records_arg_loss() -> None:
    r = SPEC.assign([{"name": "look", "arguments": {"direction": "right", "amount": "slight"}}])
    assert r.label == "look_right"
    assert r.arg_loss is True


def test_default_args_have_no_arg_loss() -> None:
    r = SPEC.assign([{"name": "nod", "arguments": {"count": 1}}])
    assert r.label == "nod" and r.arg_loss is False


def test_led_any_color_is_on_and_off_is_off() -> None:
    assert SPEC.assign([{"name": "set_led", "arguments": {"color": "blue"}}]).label == "led_on"
    assert SPEC.assign([{"name": "set_led", "arguments": {"color": "off"}}]).label == "led_off"


@pytest.mark.parametrize(
    ("calls", "reason"),
    [
        ([{"name": "look", "arguments": {"direction": "up_right", "amount": "normal"}}],
         "unsupported:look"),
        ([{"name": "set_volume", "arguments": {"level": 50}}], "unsupported:set_volume"),
        ([{"name": "nod", "arguments": {"count": 1}}, {"name": "bow", "arguments": {}}],
         "multi_action"),
    ],
)  # fmt: skip
def test_out_of_scope_is_never_no_action(calls: list, reason: str) -> None:
    r = SPEC.assign(calls)
    assert r.label is None
    assert not r.in_scope
    assert r.reason == reason


def test_label_ids_roundtrip() -> None:
    for i, name in enumerate(SPEC.names):
        assert SPEC.index(name) == i

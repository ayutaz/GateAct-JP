"""デバイスとの対話(scripts/esp32_bench.py が使う)を、ホストビルドのファームウェアで検証する。"""

import pytest

from gateact.benchmarks.device import HostProcess, compare_results, parse_line, query, read_boot

pytestmark = pytest.mark.needs_cc


def test_parse_line() -> None:
    assert parse_line('BENCH {"a": 1}') == ("BENCH", {"a": 1})
    assert parse_line("READY") == ("READY", None)
    assert parse_line("I (123) boot: x") == ("I (123) boot: x", None)


def test_device_session_with_host_build(built: tuple) -> None:
    exe, ck, texts = built
    from gateact.export.pipeline import export_bundle, simulate_bundle
    from gateact.features.hashing import extract_bits
    from gateact.training.lut_trainer import load_checkpoint

    t = HostProcess(exe)
    try:
        boot = read_boot(t, timeout=30)
        assert boot.ready
        assert boot.records["SELFTEST"]["score_mismatch"] == 0
        results = query(t, texts[:5])
    finally:
        t.close()
    model, mcfg, fcfg, _ = load_checkpoint(ck)
    bundle = export_bundle(model, input_width=fcfg.width, name="g", compress="fast")
    exp = simulate_bundle(bundle, extract_bits(texts[:5], fcfg))
    assert compare_results(results, exp.tolist()) == {"n": 5, "score_mismatch": 0}

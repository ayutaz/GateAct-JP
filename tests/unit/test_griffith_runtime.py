from pathlib import Path

import pytest

from gateact.export import griffith_runtime as gr


def test_resolved_runtime_has_verified_files() -> None:
    root = gr.runtime_dir()
    for rel in gr.RUNTIME_FILES:
        assert (root / rel).is_file()


def test_tampered_runtime_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = gr.runtime_dir()
    for rel in gr.RUNTIME_FILES:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((src / rel).read_bytes())
    (tmp_path / "src" / "mdl_kernels.c").write_text("/* tampered */\n")
    monkeypatch.setenv("GATEACT_GRIFFITH_RUNTIME", str(tmp_path))
    with pytest.raises(gr.RuntimeVerificationError):
        gr.runtime_dir()


def test_patch_points_griffith_verify_to_runtime() -> None:
    import griffith_logic.deploy.verify as v

    gr.patch_griffith_runtime()
    assert v.RUNTIME_C == gr.runtime_dir()

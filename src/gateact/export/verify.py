"""学習済み checkpoint を IR → (圧縮) → C に変換し、全段の整数一致とサイズを測る。

照合の段:
    PyTorch hard 推論 == IR simulate(圧縮前) == IR simulate(圧縮後) == 生成 C(ホスト)

入力は Python の特徴量(``extract_bits``)。C 特徴量との一致は
``tests/integration/test_features_c_parity.py`` が別に保証し、テキストから Action までの
C 単体の一致は M4 のファームウェア(ホストビルド)で確認する。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gateact.export.codegen import write_samples_header
from gateact.export.pipeline import (
    CModelRunner,
    export_bundle,
    simulate_bundle,
    torch_hard_scores,
)
from gateact.features.hashing import extract_bits
from gateact.training.lut_trainer import load_checkpoint


def _mismatch(a: np.ndarray, b: np.ndarray) -> int:
    return int((a != b).any(axis=1).sum())


def _ir_gates(bundle: dict[str, Any]) -> int:
    return sum(
        int(n["params"].get("out_width", 0))
        for n in bundle["ir"]["nodes"]
        if n["type"] == "lut_layer"
    )


def _host_object_size(c_file: Path, include: Path) -> dict[str, int]:
    """ホストで ``-O2`` コンパイルしたオブジェクトのサイズ(参考値。実機値は idf.py size)。"""
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        return {}
    obj = c_file.with_suffix(".o")
    subprocess.run(
        [cc, "-std=c99", "-O2", "-DMDL_NO_DEFAULT_CTX", f"-I{include}", "-c", str(c_file),
         "-o", str(obj)],
        check=True,
        capture_output=True,
    )  # fmt: skip
    out: dict[str, int] = {"host_object_file_bytes": obj.stat().st_size}
    size_tool = shutil.which("size")
    if size_tool:
        lines = subprocess.run([size_tool, str(obj)], capture_output=True, text=True).stdout
        rows = [ln.split() for ln in lines.strip().splitlines()]
        if len(rows) >= 2 and rows[1] and rows[1][0].isdigit():
            # GNU: text data bss ... / macOS: __TEXT __DATA __OBJC others ...(先頭列がコード+定数)
            out["host_object_text_bytes"] = int(rows[1][0])
    obj.unlink()
    return out


def export_and_verify(
    checkpoint: str | Path,
    texts: Sequence[str],
    out_dir: str | Path,
    *,
    labels: Sequence[int] | None = None,
    compress: str = "fast",
    dense_flat: str = "off",
    n_samples: int = 64,
) -> dict[str, Any]:
    """Checkpoint を変換・照合し、ファームウェア用の C 一式と報告を ``out_dir`` に書く。

    Args:
        checkpoint: ``save_checkpoint`` の出力。
        texts: 照合に使う入力文。
        out_dir: 出力先(``model/`` に C 一式と ``samples.h``、``export_report.json``)。
        labels: ``texts`` の正解クラス ID(``samples.h`` に入れる。無ければ -1)。
        compress: Griffith の回路圧縮 level(``none`` / ``fast`` / ``max``)。
        dense_flat: C 生成で dense-flat を使うか(``off`` / ``on`` / ``auto``)。
        n_samples: ``samples.h`` に入れる先頭のサンプル数。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, mcfg, fcfg, meta = load_checkpoint(checkpoint)
    bits = extract_bits(list(texts), fcfg)

    torch_scores = torch_hard_scores(model, bits, tau=mcfg.tau)
    raw = export_bundle(model, input_width=fcfg.width, name="gateact", compress="none",
                        labels=meta.get("labels"))  # fmt: skip
    comp = export_bundle(model, input_width=fcfg.width, name="gateact", compress=compress,
                         labels=meta.get("labels"))  # fmt: skip
    ir_raw = simulate_bundle(raw, bits)
    ir_comp = simulate_bundle(comp, bits)
    model_dir = out / "model"
    runner = CModelRunner.build(comp, model_dir, dense_flat=dense_flat)
    c_scores = runner.run(bits)

    n_s = min(n_samples, len(texts))
    write_samples_header(
        model_dir / "samples.h",
        bits[:n_s],
        labels=list(labels[:n_s]) if labels is not None else None,
        expected_scores=ir_comp[:n_s],
    )
    header = (model_dir / "model.h").read_text(encoding="utf-8")
    ctx = re.search(r"#define MDL_CTX_BYTES (\d+)", header)
    raw_json = json.dumps(raw["ir"], separators=(",", ":"))
    comp_json = json.dumps(comp["ir"], separators=(",", ":"))
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "compress": compress,
        "dense_flat": dense_flat,
        "parity": {
            "n": len(texts),
            "torch_vs_ir_raw_mismatch": _mismatch(torch_scores, ir_raw),
            "ir_raw_vs_ir_compressed_mismatch": _mismatch(ir_raw, ir_comp),
            "ir_vs_c_mismatch": _mismatch(ir_comp, c_scores),
            "prediction_agreement": float(
                (torch_scores.argmax(1) == c_scores.argmax(1)).mean() if len(texts) else 1.0
            ),
        },
        "sizes": {
            "gates_raw": _ir_gates(raw),
            "gates_compressed": _ir_gates(comp),
            "compress_info": (comp.get("about") or {}).get("compress"),
            "ir_json_bytes_raw": len(raw_json),
            "ir_json_bytes_compressed": len(comp_json),
            "model_c_bytes": (model_dir / "model.c").stat().st_size,
            "ctx_bytes": int(ctx.group(1)) if ctx else None,
            **_host_object_size(model_dir / "model.c", model_dir),
        },
        "model_config": mcfg.to_dict(),
        "features": fcfg.to_dict(),
        "labels": meta.get("labels"),
    }
    (out / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return report

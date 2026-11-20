"""バンドルから自己完結した C99 ソース一式(モデル + Griffith ランタイム + サンプル)を書き出す。

ESP-IDF のファームウェアとホストの一致検証の両方が、このディレクトリをそのまま使う。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from griffith_logic.deploy.cbackend import generate_c

from gateact.export.griffith_runtime import GRIFFITH_SHA, runtime_dir
from gateact.export.pipeline import bundle_output_width
from gateact.features.bits import pack_bits

_RUNTIME_COPY = {
    "include/mdl_kernels.h": "mdl_kernels.h",
    "include/mdl_simd.h": "mdl_simd.h",
    "src/mdl_kernels.c": "mdl_kernels.c",
}


def write_c_model(
    bundle: dict[str, Any], out_dir: str | Path, *, name: str = "model", dense_flat: str = "off"
) -> list[Path]:
    """バンドルを C99 に変換し、ランタイムと ``model_info.json`` と一緒に書き出す。

    Args:
        bundle: ``export_bundle`` が返すバンドル。
        out_dir: 出力ディレクトリ(無ければ作る)。
        name: 生成するファイル名の stem(``model`` → ``model.h`` / ``model.c``)。
        dense_flat: Griffith の dense-flat(平坦ネットリスト ``lut_flat``)を使うか
            (``off`` / ``on`` / ``auto``)。バンドル自体は変えず、C 生成用の複製だけを変える。

    Returns:
        書き出したファイルのパス。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    decision: dict[str, Any] | None = None
    c_bundle = bundle
    if dense_flat != "off":
        from griffith_logic.deploy.flatten_dense import resolve_dense_flat

        c_bundle, decision = resolve_dense_flat(bundle, mode=dense_flat)
    files = generate_c(c_bundle, name=name, dense_flat_decision=decision)
    for filename, text in files.items():
        path = out / filename
        path.write_text(text, encoding="utf-8")
        written.append(path)
    rt = runtime_dir()
    for src, dst in _RUNTIME_COPY.items():
        shutil.copyfile(rt / src, out / dst)
        written.append(out / dst)
    ir = bundle["ir"]
    info = {
        "name": name,
        "input_width": int(ir["inputs"][0]["width"]),
        "output_width": bundle_output_width(bundle),
        "labels": bundle.get("labels"),
        "griffith_sha": GRIFFITH_SHA,
        "compress": (bundle.get("about") or {}).get("compress"),
        "dense_flat": dense_flat,
        "dense_flat_decision": decision,
    }
    info_path = out / "model_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written.append(info_path)
    return written


def _c_array(values: Sequence[int], fmt: str = "{}") -> str:
    return "{" + ", ".join(fmt.format(int(v)) for v in values) + "}"


def write_samples_header(
    path: str | Path,
    bits: np.ndarray,
    *,
    labels: Sequence[int] | None = None,
    expected_scores: np.ndarray | None = None,
) -> Path:
    """デバイス上で検証するサンプル(パック済み入力・正解・期待スコア)の C ヘッダを書く。

    入力は LSB-first でパックする(ビット i はバイト ``i // 8`` のビット ``i % 8``)。

    Args:
        path: 出力先(``samples.h``)。
        bits: ``[N, W]`` の 0/1 配列。
        labels: 正解クラス(無ければ -1)。
        expected_scores: ``[N, K]`` の IR simulate のスコア(無ければ照合しない)。
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n, width = bits.shape
    packed = pack_bits(bits)
    lines = [
        "/* 自動生成 — gateact.export.codegen.write_samples_header。編集しない。 */",
        "#ifndef GATEACT_SAMPLES_H",
        "#define GATEACT_SAMPLES_H",
        "#include <stdint.h>",
        f"#define SAMPLE_COUNT {n}",
        f"#define SAMPLE_INPUT_WIDTH {width}",
        f"#define SAMPLE_PACKED_BYTES {packed.shape[1]}",
        f"#define SAMPLE_HAS_EXPECTED {0 if expected_scores is None else 1}",
        "static const uint8_t SAMPLE_BITS[SAMPLE_COUNT][SAMPLE_PACKED_BYTES] = {",
        *(f"  {_c_array(row, '0x{:02x}')}," for row in packed),
        "};",
        "static const int16_t SAMPLE_LABEL[SAMPLE_COUNT] = "
        + _c_array(labels if labels is not None else [-1] * n)
        + ";",
    ]
    if expected_scores is not None:
        exp = np.asarray(expected_scores, dtype=np.int64)
        lines += [
            f"#define SAMPLE_OUTPUT_WIDTH {exp.shape[1]}",
            "static const int32_t SAMPLE_EXPECTED[SAMPLE_COUNT][SAMPLE_OUTPUT_WIDTH] = {",
            *(f"  {_c_array(row)}," for row in exp),
            "};",
        ]
    lines.append("#endif /* GATEACT_SAMPLES_H */")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

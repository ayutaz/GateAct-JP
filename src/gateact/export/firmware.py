"""ファームウェア(firmware/esp32s3/gateact)用の生成ヘッダ。

- ``gateact_model_config.h``: 特徴量の設定とクラス名(C の特徴量・出力が Python と同じ設定で動く)
- ``text_samples.h``: 自己検査とベンチマークに使う UTF-8 テキスト、正解、期待する特徴ビットと
  IR simulate のスコア
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from gateact.features.bits import pack_bits
from gateact.features.hashing import FEATURE_SPEC_VERSION, FeatureConfig


def c_string(data: bytes) -> str:
    """バイト列を C の文字列リテラルにする(英数字以外は 3 桁の8進エスケープ)。"""
    out = []
    for b in data:
        ch = chr(b)
        out.append(ch if ch.isascii() and ch.isalnum() else f"\\{b:03o}")
    return '"' + "".join(out) + '"'


def write_model_config_header(
    path: str | Path,
    feat: FeatureConfig,
    labels: Sequence[str],
    *,
    model_name: str,
    gates: int,
    kind: int = 1,
) -> Path:
    """``gateact_model_config.h`` を書く(kind: 1 = LUT、2 = bit-linear、3 = n-gram)。"""
    mask = sum(1 << (n - 1) for n in feat.ngram_orders)
    names = ", ".join(c_string(x.encode()) for x in labels)
    text = f"""/* 自動生成 — gateact.export.firmware。編集しない。 */
#ifndef GATEACT_MODEL_CONFIG_H
#define GATEACT_MODEL_CONFIG_H
#define GA_MODEL_KIND {kind}
#define GA_FEATURE_SPEC {c_string(FEATURE_SPEC_VERSION.encode())}
#define GA_FEAT_WIDTH {feat.width}u
#define GA_FEAT_NGRAM_MASK {mask}u
#define GA_FEAT_NUM_HASHES {feat.num_hashes}u
#define GA_MODEL_NAME {c_string(model_name.encode())}
#define GA_MODEL_GATES {gates}
#define GA_NUM_LABELS {len(labels)}
static const char *const GA_LABELS[GA_NUM_LABELS] = {{{names}}};
#endif /* GATEACT_MODEL_CONFIG_H */
"""
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    return p


def write_text_samples_header(
    path: str | Path,
    texts: Sequence[str],
    labels: Sequence[int],
    bits: np.ndarray,
    scores: np.ndarray,
) -> Path:
    """``text_samples.h`` を書く(テキスト・正解・期待ビット(パック済み)・期待スコア)。"""
    n = len(texts)
    packed = pack_bits(bits) if n else np.zeros((0, 1), dtype=np.uint8)
    raw = [t.encode("utf-8") for t in texts]
    lines = [
        "/* 自動生成 — gateact.export.firmware。編集しない。 */",
        "#ifndef GATEACT_TEXT_SAMPLES_H",
        "#define GATEACT_TEXT_SAMPLES_H",
        "#include <stdint.h>",
        f"#define TS_COUNT {n}",
        f"#define TS_PACKED_BYTES {packed.shape[1]}",
        f"#define TS_NUM_SCORES {scores.shape[1] if n else 1}",
        "static const char *const TS_TEXT[TS_COUNT] = {",
        *(f"  {c_string(r)}," for r in raw),
        "};",
        "static const uint16_t TS_LEN[TS_COUNT] = {" + ", ".join(str(len(r)) for r in raw) + "};",
        "static const int16_t TS_LABEL[TS_COUNT] = {" + ", ".join(map(str, labels)) + "};",
        "static const uint8_t TS_BITS[TS_COUNT][TS_PACKED_BYTES] = {",
        *("  {" + ", ".join(f"0x{v:02x}" for v in row) + "}," for row in packed),
        "};",
        "static const int32_t TS_SCORES[TS_COUNT][TS_NUM_SCORES] = {",
        *("  {" + ", ".join(str(int(v)) for v in row) + "}," for row in scores),
        "};",
        "#endif /* GATEACT_TEXT_SAMPLES_H */",
    ]
    p = Path(path)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _c_int_array(name: str, ctype: str, values: np.ndarray, per_line: int = 32) -> str:
    flat = np.asarray(values).reshape(-1)
    body = ",\n".join(
        "  " + ",".join(str(int(v)) for v in flat[i : i + per_line])
        for i in range(0, len(flat), per_line)
    )
    return f"static const {ctype} {name}[{len(flat)}] = {{\n{body}\n}};\n"


def write_bitlinear_header(path: str | Path, weight: np.ndarray, bias: np.ndarray) -> Path:
    """int8 bit-linear の重みヘッダ(``baseline_model.h``)を書く。"""
    k, w = weight.shape
    text = (
        "/* 自動生成 — gateact.export.firmware。編集しない。 */\n"
        "#ifndef GATEACT_BASELINE_MODEL_H\n#define GATEACT_BASELINE_MODEL_H\n#include <stdint.h>\n"
        f"#define GB_CLASSES {k}\n#define GB_WIDTH {w}\n"
        + _c_int_array("GB_WEIGHT", "int8_t", weight)
        + _c_int_array("GB_BIAS", "int32_t", bias)
        + "#endif\n"
    )
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    return p


def write_ngram_header(
    path: str | Path, embedding: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> Path:
    """int8 n-gram 分類器の重みヘッダ(``baseline_model.h``)を書く。"""
    b, d = embedding.shape
    k = weight.shape[0]
    text = (
        "/* 自動生成 — gateact.export.firmware。編集しない。 */\n"
        "#ifndef GATEACT_BASELINE_MODEL_H\n#define GATEACT_BASELINE_MODEL_H\n#include <stdint.h>\n"
        f"#define GB_CLASSES {k}\n#define GB_BUCKETS {b}\n#define GB_DIM {d}\n"
        + _c_int_array("GB_EMBEDDING", "int8_t", embedding)
        + _c_int_array("GB_WEIGHT", "int8_t", weight)
        + _c_int_array("GB_BIAS", "int32_t", bias)
        + "#endif\n"
    )
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    return p

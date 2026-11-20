"""ベースラインの checkpoint を読み、int8 量子化してファームウェア用の一式を書く。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gateact.baselines.bitlinear import BitLinearClassifier
from gateact.baselines.ngram import NgramClassifier
from gateact.baselines.quantized import quantize_bitlinear, quantize_ngram
from gateact.data.labels import LabelSpec
from gateact.export.firmware import (
    write_bitlinear_header,
    write_model_config_header,
    write_ngram_header,
    write_text_samples_header,
)
from gateact.features.hashing import FeatureConfig, extract_bits

KIND_ID = {"bitlinear": 2, "ngram": 3}


def load_baseline(path: str | Path, spec: LabelSpec) -> BitLinearClassifier | NgramClassifier:
    """``gateact baseline`` が保存した重みから分類器を作り直す。"""
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    desc = ck["model"]
    clf: BitLinearClassifier | NgramClassifier
    if ck["kind"] == "bitlinear":
        clf = BitLinearClassifier(spec, FeatureConfig.from_dict(desc["features"]))
    elif ck["kind"] == "ngram":
        clf = NgramClassifier(spec, buckets=desc["buckets"], dim=desc["dim"])
    else:
        raise ValueError(f"unsupported baseline kind {ck['kind']!r}")
    clf.net.load_state_dict(ck["state_dict"])
    return clf


def export_baseline_firmware(
    checkpoint: str | Path,
    spec: LabelSpec,
    texts: Sequence[str],
    out_dir: str | Path,
    *,
    labels: Sequence[int] | None = None,
    n_samples: int = 64,
) -> dict[str, Any]:
    """int8 化した重み・設定・サンプルを ``out_dir/model`` に書き、量子化後の予測を返す。"""
    clf = load_baseline(checkpoint, spec)
    model_dir = Path(out_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("model.c", "model.h", "mdl_kernels.c"):
        (model_dir / stale).unlink(missing_ok=True)
    texts = list(texts)
    n_s = min(n_samples, len(texts))
    if isinstance(clf, BitLinearClassifier):
        feat = clf.features
        qb = quantize_bitlinear(clf)
        bits = extract_bits(texts, feat)
        scores = qb.scores(bits)
        write_bitlinear_header(model_dir / "baseline_model.h", qb.weight, qb.bias)
        model_bytes = qb.weight.nbytes + qb.bias.nbytes
        kind = "bitlinear"
    else:
        feat = FeatureConfig(width=8)  # n-gram は特徴ビットを使わない(設定は形式上のもの)
        qn = quantize_ngram(clf)
        scores = qn.scores(texts)
        bits = np.zeros((len(texts), feat.width), dtype=np.uint8)
        write_ngram_header(model_dir / "baseline_model.h", qn.embedding, qn.weight, qn.bias)
        model_bytes = qn.embedding.nbytes + qn.weight.nbytes + qn.bias.nbytes
        kind = "ngram"
    sample_labels = list(labels[:n_s]) if labels is not None else [-1] * n_s
    write_text_samples_header(
        model_dir / "text_samples.h", texts[:n_s], sample_labels, bits[:n_s], scores[:n_s]
    )
    write_model_config_header(
        model_dir / "gateact_model_config.h",
        feat,
        list(spec.names),
        model_name=Path(checkpoint).stem,
        gates=0,
        kind=KIND_ID[kind],
    )
    report = {
        "checkpoint": str(checkpoint),
        "kind": kind,
        "quantization": "int8 symmetric per-tensor",
        "model_bytes": int(model_bytes),
        "n": len(texts),
    }
    (Path(out_dir) / "export_report.json").write_text(json.dumps(report, indent=1) + "\n")
    report["scores"] = scores
    return report

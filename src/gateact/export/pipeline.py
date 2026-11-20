"""学習済み Griffith モデル → IR → バンドル → C の変換と、各段の整数一致検証。

真実源は Griffith IR(``griffith_logic.ir.simulate``)。PyTorch hard 推論・生成 C・
ESP32-S3 の出力はすべて IR simulate と整数で一致しなければならない。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.compress import apply_compress, attach_compress_info
from griffith_logic.deploy.verify import build_c_model, run_c_model_packed
from griffith_logic.export import export_ir
from griffith_logic.ir import validate
from griffith_logic.ir.serialize import ir_from_dict, ir_to_dict
from griffith_logic.ir.simulate import simulate


def torch_hard_scores(model: nn.Module, bits: np.ndarray, *, tau: float) -> np.ndarray:
    """PyTorch の hard(eval)推論スコアを整数カウントで返す。

    GroupSum の出力は ``count / tau`` なので ``tau`` を掛けて丸める。

    Args:
        model: ``eval()`` 済みでなくてもよい(内部で eval にする)。
        bits: ``[N, input_bits]`` の 0/1 配列。
        tau: GroupSum の温度。
    """
    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(np.asarray(bits, dtype=np.float32)))
    return torch.round(y * tau).to(torch.int64).numpy()


def export_bundle(
    model: nn.Module,
    *,
    input_width: int,
    name: str,
    labels: list[str] | None = None,
    compress: str = "fast",
) -> dict[str, Any]:
    """モデルを IR に変換し、(任意で回路圧縮して)配布バンドルを作る。

    Args:
        model: 学習済みの ``nn.Sequential``(LutLayer + GroupSum)。
        input_width: 入力ビット幅。
        name: IR の metadata.name。
        labels: クラス名(バンドルに記録される)。
        compress: ``"none"`` または Griffith ``apply_compress`` の level(``"fast"`` 等)。

    Returns:
        ``griffith_logic.deploy.build_bundle`` の返すバンドル dict(検証済み)。
    """
    model.eval()
    ir = export_ir(model, input_width=input_width, name=name)
    validate(ir, target="hdl")
    doc: dict[str, Any] = {"ir": ir_to_dict(ir)}
    if labels is not None:
        doc["labels"] = list(labels)
    if compress != "none":
        doc, info, _ = apply_compress(doc, level=compress)
        doc = attach_compress_info(doc, info)
    return build_bundle(doc, verify=True)


def simulate_bundle(bundle: dict[str, Any], bits: np.ndarray) -> np.ndarray:
    """バンドルの IR を Griffith の参照シミュレータで実行し、整数スコアを返す。"""
    graph = ir_from_dict(bundle["ir"])
    return simulate(graph, np.asarray(bits, dtype=np.uint8)).astype(np.int64)


def bundle_output_width(bundle: dict[str, Any]) -> int:
    """バンドルの出力幅(クラス数)を返す。"""
    return int(ir_from_dict(bundle["ir"]).outputs[0].width)


@dataclass
class CModelRunner:
    """生成 C をホストでビルドし、標準入出力経由で実行するランナー。"""

    exe: Path
    n_out: int

    @classmethod
    def build(cls, bundle: dict[str, Any], work_dir: str | Path) -> CModelRunner:
        """バンドルから C を生成・コンパイルする(``cc -std=c99 -O2``)。"""
        from gateact.export.griffith_runtime import patch_griffith_runtime

        patch_griffith_runtime()
        exe = build_c_model(bundle, Path(work_dir))
        return cls(exe=exe, n_out=bundle_output_width(bundle))

    def run(self, bits: np.ndarray) -> np.ndarray:
        """``[N, W]`` の 0/1 入力に対する ``[N, n_out]`` の整数スコアを返す。"""
        from griffith_logic.deploy.verify import PACKED_EXE_NAME

        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        packed_exe = self.exe.parent / (PACKED_EXE_NAME + self.exe.suffix)
        if packed_exe.exists():
            return run_c_model_packed(packed_exe, bits, self.n_out)
        from griffith_logic.deploy.verify import run_c_model

        return np.stack([run_c_model(self.exe, row.tobytes(), self.n_out) for row in bits])

"""学習済み Griffith モデル → IR → バンドル → C の変換と、各段の整数一致検証。

真実源は Griffith IR(``griffith_logic.ir.simulate``)。PyTorch hard 推論・生成 C・
ESP32-S3 の出力はすべて IR simulate と整数で一致しなければならない。
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.compress import apply_compress, attach_compress_info
from griffith_logic.export import export_ir
from griffith_logic.ir import validate
from griffith_logic.ir.serialize import ir_from_dict, ir_to_dict
from griffith_logic.ir.simulate import simulate

_COMMON = Path(__file__).resolve().parents[3] / "firmware" / "common"


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
    """``write_c_model`` の出力(ファームウェアと同じファイル)をホストでビルドして実行する。"""

    exe: Path
    input_width: int
    n_out: int

    @classmethod
    def build(
        cls,
        bundle: dict[str, Any],
        work_dir: str | Path,
        *,
        cc: str | None = None,
        dense_flat: str = "off",
    ) -> CModelRunner:
        """C を生成し ``cc -std=c99 -O2`` でバッチ推論ドライバとリンクする。"""
        from gateact.export.codegen import write_c_model

        work = Path(work_dir)
        write_c_model(bundle, work, dense_flat=dense_flat)
        compiler = cc or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
        if compiler is None:
            raise RuntimeError("C compiler (cc/clang/gcc) not found")
        exe = work / "host_model_driver"
        subprocess.run(
            [compiler, "-std=c99", "-O2", "-DMDL_NO_DEFAULT_CTX", f"-I{work}",
             str(_COMMON / "host_model_driver.c"), str(work / "model.c"),
             str(work / "mdl_kernels.c"), "-o", str(exe)],
            check=True,
            capture_output=True,
        )  # fmt: skip
        width = int(bundle["ir"]["inputs"][0]["width"])
        return cls(exe=exe, input_width=width, n_out=bundle_output_width(bundle))

    def run(self, bits: np.ndarray) -> np.ndarray:
        """``[N, W]`` の 0/1 入力に対する ``[N, n_out]`` の整数スコアを返す。"""
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        if bits.ndim != 2 or bits.shape[1] != self.input_width:
            raise ValueError(f"expected [N, {self.input_width}] bits")
        payload = struct.pack("<I", bits.shape[0]) + bits.tobytes()
        proc = subprocess.run([str(self.exe)], input=payload, capture_output=True, check=True)
        out = np.frombuffer(proc.stdout, dtype="<i4").astype(np.int64)
        return out.reshape(bits.shape[0], self.n_out)

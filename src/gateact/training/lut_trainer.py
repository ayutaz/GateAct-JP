"""Griffith LUT 分類器の学習ループ。

soft(連続緩和)で学習し、各 epoch の終わりに dev で soft と **hard(離散化した回路)** の
両方の正解率を測る。採用する epoch は dev の hard 正解率で選ぶ(デプロイするのは hard 回路)。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gateact.data.checks import Labeled
from gateact.data.labels import LabelSpec
from gateact.features.hashing import FeatureConfig, extract_bits
from gateact.models.lut import LutModelConfig, build_lut_model
from gateact.utils import seed_everything


@dataclass(frozen=True)
class TrainConfig:
    """学習の設定。"""

    epochs: int = 30
    lr: float = 0.01
    batch_size: int = 256
    seed: int = 0
    device: str = "cpu"
    weight_decay: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON 保存用。"""
        return asdict(self)


@dataclass
class TrainResult:
    """学習の結果。``model`` は dev の hard 正解率が最良の epoch の重み(CPU, eval)。"""

    model: nn.Module
    log: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = 0
    best_dev_hard_acc: float = 0.0
    seconds: float = 0.0


def _to_tensor(bits: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(bits.astype(np.float32)).to(device)


@torch.no_grad()
def _scores(model: nn.Module, x: torch.Tensor, *, hard: bool, batch: int = 4096) -> torch.Tensor:
    model.train(not hard)
    out = [model(x[i : i + batch]) for i in range(0, len(x), batch)]
    return torch.cat(out) if out else torch.empty(0)


def hard_predict_ids(model: nn.Module, bits: np.ndarray, tau: float) -> np.ndarray:
    """Hard 回路の予測クラス ID。同点は小さい ID を採る(C の argmax と同じ規則)。

    スコアを整数カウント(``× tau`` して丸め)に戻してから argmax するので、
    浮動小数点の誤差で C と食い違うことがない。
    """
    device = next(iter(model.parameters()), torch.empty(0)).device
    s = _scores(model, _to_tensor(bits, str(device)), hard=True)
    counts = torch.round(s * tau).to(torch.int64).cpu().numpy()
    return counts.argmax(axis=1)


def train_lut(
    model_cfg: LutModelConfig,
    feat_cfg: FeatureConfig,
    train_cfg: TrainConfig,
    train: Sequence[Labeled],
    dev: Sequence[Labeled],
    spec: LabelSpec,
    *,
    verbose: bool = False,
) -> TrainResult:
    """LUT 分類器を学習する。"""
    if model_cfg.input_bits != feat_cfg.width or model_cfg.num_classes != spec.num_classes:
        raise ValueError("model input_bits / num_classes must match features / label spec")
    seed_everything(train_cfg.seed)
    dev_name = train_cfg.device
    model = build_lut_model(model_cfg).to(dev_name)
    x = _to_tensor(extract_bits([d.example.text for d in train], feat_cfg), dev_name)
    y = torch.tensor([spec.index(d.label) for d in train], device=dev_name)
    dev_bits = extract_bits([d.example.text for d in dev], feat_cfg)
    dx = _to_tensor(dev_bits, dev_name)
    dy = np.array([spec.index(d.label) for d in dev])
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    gen = torch.Generator().manual_seed(train_cfg.seed)
    result = TrainResult(model=model)
    best_state: dict[str, torch.Tensor] | None = None
    t_start = time.perf_counter()
    for epoch in range(1, train_cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(y), generator=gen).to(dev_name)
        total = 0.0
        for i in range(0, len(y), train_cfg.batch_size):
            idx = perm[i : i + train_cfg.batch_size]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        soft_acc = hard_acc = 0.0
        if len(dy):
            soft = _scores(model, dx, hard=False).argmax(1).cpu().numpy()
            hard = hard_predict_ids(model, dev_bits, model_cfg.tau)
            soft_acc, hard_acc = float((soft == dy).mean()), float((hard == dy).mean())
        entry = {
            "epoch": epoch,
            "train_loss": total / max(len(y), 1),
            "dev_soft_acc": soft_acc,
            "dev_hard_acc": hard_acc,
            "seconds": time.perf_counter() - t0,
        }
        result.log.append(entry)
        if verbose:
            print(
                f"epoch {epoch}: loss {entry['train_loss']:.4f} dev soft "
                f"{entry['dev_soft_acc']:.4f} hard {entry['dev_hard_acc']:.4f} "
                f"({entry['seconds']:.1f}s)",
                flush=True,
            )
        if best_state is None or entry["dev_hard_acc"] > result.best_dev_hard_acc:
            result.best_dev_hard_acc = entry["dev_hard_acc"]
            result.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model = model.cpu()
    if best_state is not None:
        model.load_state_dict(best_state)
    result.model = model.eval()
    result.seconds = time.perf_counter() - t_start
    return result


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_cfg: LutModelConfig,
    feat_cfg: FeatureConfig,
    spec: LabelSpec,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """重みと、再構築に必要な設定を1ファイルに保存する。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_config": model_cfg.to_dict(),
            "feature_config": feat_cfg.to_dict(),
            "labels": list(spec.names),
            "meta": {**(meta or {}), "label_spec": spec.version},
        },
        p,
    )
    return p


def load_checkpoint(
    path: str | Path,
) -> tuple[nn.Module, LutModelConfig, FeatureConfig, dict[str, Any]]:
    """``save_checkpoint`` の逆。モデルは eval 状態で返す。"""
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    mcfg = LutModelConfig.from_dict(ck["model_config"])
    fcfg = FeatureConfig.from_dict(ck["feature_config"])
    model = build_lut_model(mcfg)
    model.load_state_dict(ck["state_dict"])
    meta = dict(ck["meta"])
    meta["labels"] = ck["labels"]
    return model.eval(), mcfg, fcfg, meta

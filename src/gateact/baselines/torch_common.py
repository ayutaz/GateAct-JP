"""PyTorch 分類器の共通の学習ループ(dev の正解率で epoch を選ぶ)。"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

Batch = tuple[torch.Tensor, ...]


def train_loop(
    model: torch.nn.Module,
    make_batch: Callable[[Sequence[int]], Batch],
    y: torch.Tensor,
    dev_eval: Callable[[], float],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    log_prefix: str = "",
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Adam + cross entropy で学習し、dev 正解率が最良の重みを読み戻す。"""
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = list(range(len(y)))
    best, best_state, log = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        total = 0.0
        t0 = time.perf_counter()
        for i in range(0, len(order), batch_size):
            part = order[i : i + batch_size]
            inputs = make_batch(part)
            loss = F.cross_entropy(model(*inputs), y[part])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(part)
        acc = dev_eval()
        entry = {
            "epoch": epoch,
            "train_loss": total / max(len(order), 1),
            "dev_exact": acc,
            "seconds": time.perf_counter() - t0,
        }
        log.append(entry)
        if verbose:
            print(f"{log_prefix}epoch {epoch}: loss {entry['train_loss']:.4f} dev {acc:.4f}")
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return log

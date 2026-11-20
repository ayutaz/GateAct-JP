"""共通ユーティリティ(乱数シード固定・ハッシュ・実行環境の記録)。"""

from __future__ import annotations

import hashlib
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Python / NumPy / PyTorch の乱数シードをまとめて固定する。

    Args:
        seed: 乱数シード。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: str | Path) -> str:
    """ファイルの SHA-256 を16進文字列で返す。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(cwd: str | Path | None = None) -> str | None:
    """現在の Git commit SHA を返す(取得できなければ None)。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None

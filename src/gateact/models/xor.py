"""M0 の動作確認用 XOR モデル(Griffith README の 30 秒例と同じ構成)。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gateact.models.lut import LutModelConfig, build_lut_model
from gateact.utils import seed_everything

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
XOR_Y = np.array([0, 1, 1, 0], dtype=np.int64)
XOR_CONFIG = LutModelConfig(input_bits=2, hidden=[16], output_gates=16, num_classes=2, tau=2.0)


def train_xor(seed: int = 0, steps: int = 500, lr: float = 0.05) -> nn.Module:
    """XOR を学習し、``eval()`` 済みのモデルを返す。"""
    seed_everything(seed)
    model = build_lut_model(XOR_CONFIG)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.from_numpy(XOR_X).float()
    y = torch.from_numpy(XOR_Y)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    return model.eval()

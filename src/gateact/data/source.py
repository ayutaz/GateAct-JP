"""学習データの出典(固定 revision と SHA-256)。"""

from __future__ import annotations

from pathlib import Path

HF_REPO = "japanese-data-analyze/JapaneseTinyAgentLM-Action-Synth"
HF_REVISION = "6ac72256456d8e76fca4708823c853438331619f"
LICENSE = "Apache-2.0"
FILES: dict[str, str] = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
}
SHA256: dict[str, str] = {
    "train.jsonl": "570ae9e0f0516b8a9201efbaf664c368a4cce1d41da0aab0559a6b46ffb5f306",
    "validation.jsonl": "795e853da8f99298259b8b9a9c0d3bac9be34ae5f8fb701cf101dd1a74dd3333",
}
DEFAULT_DIR = Path("data/raw/action_synth")

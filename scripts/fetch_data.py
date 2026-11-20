"""HF から JapaneseTinyAgentLM-Action-Synth を固定 revision で取得し、SHA-256 を検証する。

事前に `uvx --from huggingface_hub hf auth login` と、データセットページでの
利用申請(manual gate)が必要。
トークンはファイルに書かず、huggingface_hub の既定の保存場所から読む。

    uv run --group data python scripts/fetch_data.py [--out data/raw/action_synth]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from gateact.data.source import DEFAULT_DIR, FILES, HF_REPO, HF_REVISION, SHA256
from gateact.utils import sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        dst = args.out / name
        if not dst.exists():
            from huggingface_hub import hf_hub_download

            src = hf_hub_download(HF_REPO, name, repo_type="dataset", revision=HF_REVISION)
            shutil.copyfile(src, dst)
        digest = sha256_file(dst)
        if digest != SHA256[name]:
            raise SystemExit(f"SHA-256 mismatch for {dst}: {digest} != {SHA256[name]}")
        print(f"ok {dst} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

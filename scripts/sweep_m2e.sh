#!/usr/bin/env bash
# M2 改善実験 第 2 回(1 seed、10 epoch): tau と第 1 回で効いた変更の組み合わせ。ログは runs/sweep_m2e.log
set -euo pipefail
cd "$(dirname "$0")/.."
for c in lut6warp_w512_tau4 lut6warp_w512h2_tau8 lut6warp_w512_2x8k_tau8; do
  uv run gateact train "configs/lut/$c.json" --verbose
done

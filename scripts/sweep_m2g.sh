#!/usr/bin/env bash
# M2 改善実験 第 4 回: tau 4 を 20 epoch に延ばす(3 seed)。ログは runs/sweep_m2g.log
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  uv run gateact train configs/lut/lut6warp_w512_tau4_e20.json --seed "$seed" --verbose
done

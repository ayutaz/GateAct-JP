#!/usr/bin/env bash
# M2 改善実験 第 3 回: tau 2(1 seed)と採用候補 2 つの追加 seed。ログは runs/sweep_m2f.log
set -euo pipefail
cd "$(dirname "$0")/.."
uv run gateact train configs/lut/lut6warp_w512_tau2.json --verbose
for seed in 1 2; do
  uv run gateact train configs/lut/lut6warp_w512_tau4.json --seed "$seed" --verbose
done
for seed in 1 2; do
  uv run gateact train configs/lut/lut6warp_w512_2x8k_tau8.json --seed "$seed" --verbose
done

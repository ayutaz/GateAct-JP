#!/usr/bin/env bash
# M2: 本命の LUT6 warp 512 bit を追加 seed で学習する(s0 は sweep_m2a で実施済み)。
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 1 2; do
  uv run gateact train configs/lut/lut6warp_w512_2x4k.json --seed "$seed" --verbose
done

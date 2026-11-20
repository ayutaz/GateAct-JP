#!/usr/bin/env bash
# M2 初回スイープ(1 seed, 10 epoch)。結果は results/lut/、ログは runs/sweep_m2a.log
set -euo pipefail
for w in 1024 2048; do uv run gateact baseline --kind bitlinear --width "$w" --seed 0; done
for c in lut2_w512_2x4k lut2_w512_2x4k_lr05 lut4warp_w512_2x4k lut6warp_w512_2x4k lut2_w512_3x8k lut2_w1024_3x8k; do
  uv run gateact train "configs/lut/$c.json" --verbose
done

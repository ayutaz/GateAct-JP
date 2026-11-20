#!/usr/bin/env bash
# M2 改善実験(1 seed、10 epoch): 入力の密度・規模・学習設定を 1 つずつ変える。
# 基準は lut6warp_w512_2x4k(test 91.21 ± 0.38%、3 seed)。ログは runs/sweep_m2d.log
set -euo pipefail
cd "$(dirname "$0")/.."
for c in lut6warp_w256_2x4k lut6warp_w512h2_2x4k lut6warp_w512_tau32 lut6warp_w512_tau8 \
         lut6warp_w512_lr03 lut6warp_w512_3x4k lut6warp_w512_2x8k; do
  uv run gateact train "configs/lut/$c.json" --verbose
done

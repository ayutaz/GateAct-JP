#!/usr/bin/env bash
# M2 第2スイープ(1 seed、20 epoch): LUT4/LUT6 warp × 入力幅。結果は results/lut/、ログは runs/sweep_m2b.log
set -euo pipefail
# ベースラインの重みを保存し直す(ファームウェア用。結果は同じ seed で再計算)
uv run gateact baseline --kind ngram --seed 0
uv run gateact baseline --kind bitlinear --width 512 --seed 0
for c in lut6warp_w1024_2x4k lut6warp_w2048_2x4k lut4warp_w1024_2x4k; do
  uv run gateact train "configs/lut/$c.json" --verbose
done

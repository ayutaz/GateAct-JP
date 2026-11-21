# GateAct-JP

**日本語の指示文からロボットの Action を判定する、論理ゲートだけでできた超軽量分類器。**
[Griffith](https://github.com/tsukushi-gate/Griffith)(Differentiable Logic Gate Network)で
ゼロから学習し、論理回路 → C99 → ESP32-S3 単体で推論します。
[JapaneseTinyAgentLM](https://github.com/ayutaz/JapaneseTinyAgentLM) の Action schema と
データ・評価方法を出発点にし、既存の分類器と精度・サイズ・速度・メモリを実測で比較します。

> 研究段階。進捗は [`docs/plan.md`](docs/plan.md)、実験記録は [`docs/research.md`](docs/research.md)。
> 数値はすべて実測で、PC / ホスト C / ESP32-S3 実機を区別して書きます。**ESP32-S3 実機での
> 速度と一致率はまだ測っていません**(ボード未接続)。

## 何をするか

```
UTF-8 の文 ──▶ 正規化(GateAct-norm-v1)──▶ 文字 n-gram を FNV-1a でハッシュ(512 bit)
           ──▶ Griffith LUT 回路(LUT6 × 4,096 + 4,096、圧縮後 28,178 ゲート)
           ──▶ 16 クラスのスコア(整数)──▶ argmax ──▶ {"action": "look_left", ...}
```

16 クラス: `no_action`、`look_up/down/left/right/center`、`nod`、`shake`、`bow`、
`expr_happy/surprised/sad`、`led_on/off`、`volume_up/down`(`configs/labels_v1.json`)。
複数 Action の文など 16 クラスで表せない文は「対象外」として別集計します(validation の 62.2% が対象内)。

## 結果(2026-11-21 時点)

### 精度(PC、HF validation の対象内 2,826 件、3 seed の平均 ± SD)

| モデル | Exact | Macro-F1 | False Action ↓ | 否定表現 | no_action Precision |
|---|---:|---:|---:|---:|---:|
| ルール(キーワード) | 83.44 | 72.87 | 7.20 | 91.99 | 81.91 |
| n-gram 分類器(JapaneseTinyAgentLM 相当) | **98.76 ± 0.07** | **97.76 ± 0.21** | **0.73 ± 0.00** | **99.29 ± 0.18** | **98.84 ± 0.14** |
| bit-linear(同じ 512 bit 特徴 + 線形層) | 93.54 ± 0.02 | 92.20 ± 0.66 | 4.10 ± 0.29 | 93.71 ± 0.51 | 94.06 ± 0.37 |
| **GateAct LUT6 tau 4**(28,178 ゲート) | 93.97 ± 0.34 | 92.15 ± 0.10 | 2.31 ± 0.30 | 96.86 ± 0.27 | 93.41 ± 0.47 |
| **GateAct LUT6 2 倍 + tau 8**(54,716 ゲート) | 94.99 ± 0.13 | 93.05 ± 0.89 | 2.34 ± 0.14 | 96.86 ± 0.84 | 94.62 ± 0.36 |

単位は %。ルールは決定的なので 1 回。全指標は [`results/summary.md`](results/summary.md)。
ハイパーパラメータは dev(train の 5%)だけで選んだ。

### ESP32-S3 ファームウェアのサイズ(`idf.py size` 実測、書き込みなし)

| モデル | イメージ | `.rodata` | DIRAM 使用 |
|---|---:|---:|---:|
| n-gram int8 | 2,833,744 B | 2,672,096 B | 102,151 B |
| bit-linear int8 | 224,064 B | 62,320 B | 98,407 B |
| GateAct LUT6 tau 4 | 555,664 B | 391,328 B | 122,503 B |
| GateAct LUT6 2 倍 + tau 8 | 869,936 B | 705,600 B | 144,943 B |

int8 量子化による精度の低下は無かった(n-gram 98.83 → 98.87%、bit-linear 93.52 → 93.60%、seed 0)。

### 推論時間(ホスト C、Apple M1。**ESP32-S3 の値ではない**)

| モデル | 1 文あたり P50(特徴量 + モデル) |
|---|---:|
| n-gram int8 | 0.6–1.4 µs |
| bit-linear int8 | 0.9–1.7 µs |
| GateAct LUT6 tau 4 | 約 41 µs |
| GateAct LUT6 2 倍 + tau 8 | 約 77 µs |

### 結論(現時点)

- **論理ゲート回路は、同じ特徴量の線形モデル(bit-linear)と同等以上の精度**(93.97% / 94.99%)で、
  誤作動(False Action 2.3% 対 4.1%)と否定表現(96.9% 対 93.7%)では明確に良い。
- **既存の n-gram 分類器には及ばない**(98.76%、差 3.8〜4.8 point)。サイズは n-gram の約 1/5〜1/3。
- **速度は負け**。入力は疎(512 bit 中 約 48 bit)で、線形モデルは立っているビットだけを足せば
  よいが、回路は全ゲートを毎回評価するので PC 上で 30〜100 倍遅い。ESP32-S3 実機は未測定。
- 効いたのは GroupSum の温度 tau(16 → 4 で +2.8 point)とゲート数。最初の設定(tau 16)では
  91.21% で bit-linear に負けていた。LUT2(difflogic の 2 入力ゲート)は 71% にとどまった。
- Griffith の C 生成は LUT6 に未対応なので、LUT5 + LUT3 への厳密な分割を実装した
  ([`src/gateact/export/lut6.py`](src/gateact/export/lut6.py))。

詳細と経緯(失敗した条件を含む)は [`docs/research.md`](docs/research.md)、評価の定義は
[`docs/evaluation.md`](docs/evaluation.md)。

## 一致検証

PyTorch hard 推論 == Griffith IR == LUT6 分割後 IR == 圧縮 IR == 生成 C を、評価セット全
4,546 文で整数スコアまで照合しています(不一致 0)。特徴量は Python と C で全文一致。
ファームウェアをホストビルドし、テキスト入力から Action までの出力も Python と照合します。

## 再現手順

```bash
uv sync --locked
make check                                   # lint + 型 + テスト(データ不要、CI と同じ)

# データ(HF の利用承認と `hf auth login` が必要。data/ は Git 管理外)
make data

# 学習と評価(PC。LUT6 は 1 seed 約 17 分、Apple M1 CPU)
uv run gateact baseline --kind rules
uv run gateact baseline --kind ngram --seed 0
uv run gateact train configs/lut/lut6warp_w512_tau4.json --seed 0
uv run gateact summarize results/baselines results/lut --out results/summary.json

# C 生成と全段の一致検証 → ファームウェア
uv run gateact export runs/lut6warp_w512_tau4/lut6warp_w512_tau4_s0.pt --out runs/export/lut6
uv run python scripts/host_bench.py runs/export/lut6/model --out results/host/lut6.json
. ~/esp/esp-idf/export.sh
cd firmware/esp32s3/gateact && idf.py -D MODEL_DIR=$PWD/../../../runs/export/lut6/model build
```

実機での計測は [`docs/hardware.md`](docs/hardware.md)(`scripts/esp32_bench.py`)。

## 構成

| パス | 内容 |
|---|---|
| `src/gateact/features/` | 正規化(GateAct-norm-v1)と n-gram ハッシュ特徴量(C 実装と同一仕様) |
| `src/gateact/models/`, `training/` | Griffith LUT モデルと学習(dev の hard 精度で epoch 選択) |
| `src/gateact/baselines/` | ルール、n-gram、bit-linear、int8 量子化 |
| `src/gateact/export/` | IR → 圧縮 → C99、LUT6 分割、全段の一致検証、ファームウェア用ヘッダ |
| `src/gateact/evaluation/` | 指標、評価ランナー、集計 |
| `firmware/` | 共通 C(特徴量・ベースライン)と ESP32-S3 ファームウェア(ESP-IDF v5.5) |
| `configs/`, `results/` | 実験設定と実測結果(JSON) |

## English summary

GateAct-JP trains a tiny Japanese text → robot-action classifier (16 classes) made only of
lookup-table logic gates (Griffith DLGN), exports it to dependency-free C99, and builds it for
the ESP32-S3. It is compared on the same data, classes and splits against a keyword rule
baseline, the n-gram classifier of JapaneseTinyAgentLM and a bit-linear model.

**Results so far:** after tuning the GroupSum temperature (tau 16 → 4), the LUT6 gate circuit
reaches 93.97 ± 0.34% exact accuracy (3 seeds, 28k gates) and 94.99 ± 0.13% with twice the
gates. That matches or beats a linear model on the same 512-bit features (93.54%) with a lower
false-action rate (2.3% vs 4.1%), but stays below the n-gram classifier (98.76%). Firmware images
are 556 KB / 870 KB versus 224 KB (linear) and 2.83 MB (n-gram). In host C the circuit is 30–100×
slower (≈41–77 µs vs ≈1 µs on Apple M1) because sparse inputs let linear models touch only the
set bits while the circuit evaluates every gate. On-device ESP32-S3 latency has not been measured
yet. All torch → IR → C stages are verified bit-exact on all 4,546 evaluation sentences.

## License

Apache-2.0([LICENSE](LICENSE)、第三者コンポーネントは [NOTICE](NOTICE))。
学習データ(Hugging Face `japanese-data-analyze/JapaneseTinyAgentLM-Action-Synth`)は再配布しません。

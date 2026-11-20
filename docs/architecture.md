# アーキテクチャ

```
日本語テキスト(UTF-8)
  │  GateAct-norm-v1(NFKC の部分集合 + 小文字化 + 空白の圧縮)   Python: features/normalize.py
  │  文字 n-gram(前後に U+0002 / U+0003)                        C:      firmware/common/gateact_features.c
  │  FNV-1a 32bit → fmix32 → mod width で 1 ビット                 (Python と C はビット単位で一致)
  ▼
固定長ビット列(例: 512 bit)
  │  Griffith LutLayer × L(LUT2 / LUT4 / LUT6)                   学習: PyTorch(soft) → eval で hard
  │  GroupSum(16 クラス、整数カウント)                            IR → 回路圧縮 → C99(Griffith deploy)
  ▼
16 クラスの整数スコア → argmax(同点は小さい ID)→ Action
```

## 特徴量 gateact-feat-v1

| 項目 | 仕様 |
|---|---|
| 正規化 | GateAct-norm-v1(下記) |
| 境界 | 先頭に U+0002、末尾に U+0003 を付けたコードポイント列 |
| n-gram | `ngram_orders`(既定 1, 2, 3)の昇順、位置の昇順 |
| ハッシュ | `base = fnv1a32(utf8(gram))`、`idx_j = fmix32(base ^ (j·0x9E3779B9)) % width`(j < `num_hashes`) |
| 値 | 出現の有無(0/1)。回数は数えない |
| ビット順 | ビット i = 特徴ベクトルの i 番目。パック時は LSB-first(バイト i/8 のビット i%8) |
| 長さ上限 | 正規化後 254 コードポイント(超えたら Python は ValueError、C は `GFEAT_ERR_TOO_LONG`) |
| UTF-8 | 厳格(overlong / surrogate / U+10FFFF 超 / 途中切れは C で `GFEAT_ERR_UTF8`。Python の strict decoder と同じ受理範囲) |

Python の組み込み `hash()` はプロセスごとに乱数化されるため使わない。

### GateAct-norm-v1 と NFKC の判断

完全な NFKC は表が数十 KB になり ESP32 では重い。データ(JapaneseTinyAgentLM-Action-Synth v1.1)を
調べると **90,448 文のうち NFKC で変わる文は 0 件**(作成時に NFKC 済み)だった。一方、実機に
人が打ち込む文では全角英数字・半角カナが混ざりうるので、NFKC のうち日本語入力で現れやすい部分
だけを実装した。

1. 全角 ASCII(U+FF01〜FF5E)→ 半角、和字間隔 U+3000 → 空白
2. 半角カタカナ(U+FF61〜FF9F)→ 全角(NFKC と同じ対応表)
3. かな + 結合濁点/半濁点 → 合成済みの文字(NFC の表、58 組)
4. ASCII A〜Z → a〜z
5. ASCII 空白の連続 → 空白 1 つ、前後を除く

1〜3 は対象範囲の全文字で `unicodedata.normalize("NFKC")` と一致することをテストで確認している
(`tests/unit/test_features.py`)。それ以外(丸数字、単位記号、互換漢字など)は変換しない。
**学習と ESP32 で同じ正規化を使う**(PC 実験用に別の前処理を使わない)。

C の表は `scripts/gen_c_tables.py` で Python から生成し、生成物が古いとテストが落ちる。

## モデル

`gateact.models.lut.build_lut_model`:
`LutLayer(in, h1) → … → LutLayer(h_{L-1}, out) → GroupSum(k=16, tau)`。
`output_gates` はクラス数で割り切れる必要がある(Griffith `GroupSum` の制約。
`build_lut_model` が `ValueError` を出す)。

- 学習は soft(ゲートの確率的な混合)、評価・デプロイは hard(`model.eval()` で各ゲートを
  argmax の 1 つの論理関数に確定)。**採用 epoch は dev の hard 正解率で選ぶ。**
- hard のスコアは `GroupSum` の出力 × tau を丸めた整数カウント。IR simulate・C と整数で比べる。

## エクスポートと一致検証

`gateact export <ckpt>`(`gateact.export.verify.export_and_verify`):

```
PyTorch hard  ==  IR simulate(圧縮前)  ==  IR simulate(圧縮後, Griffith sweep)  ==  生成 C(ホスト)
```

を評価セット全文で照合し、`export_report.json` にサイズ(ゲート数、IR、`model.c`、ctx)と
一緒に書く。生成 C はファームウェアに載せるファイルそのもの(`write_c_model`)を
`firmware/common/host_model_driver.c` とリンクして実行する。

## ファームウェア(ESP32-S3)

`firmware/esp32s3/gateact/main/main.c`。同じソースをホスト(`-DHOST_BUILD`)でもビルドする。

| 出力 | 内容 |
|---|---|
| `INFO` | IDF 版、CPU 周波数、flash / PSRAM、heap、モデルと特徴量の設定 |
| `SELFTEST` | 同梱テキスト(`text_samples.h`)で C の特徴ビットとスコアが Python と一致するか、全スコアの FNV-1a |
| `BENCH` | `BENCH_ITERS`(既定 2,000)回の前処理 / 推論 / 全体の µs(mean, min, P50, P95, P99, max)。CPU サイクルカウンタで計測 |
| `MEM` | 計測後の free heap / 最小 free heap / 最大ブロック / ctx |
| `RESULT` | シリアルの 1 行ごとに `{"id", "action", "scores", "us": {...}}` |

`firmware/esp32s3/xor` は M0 の動作確認用(Griffith 公式の XOR を ESP32-S3 向けにしたもの)。

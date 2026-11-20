# 評価

## 比較の枠組み

| 区分 | 対象 | 条件 |
|---|---|---|
| **主比較**(固定クラス分類器どうし) | A. ルール、B. n-gram 分類器(JapaneseTinyAgentLM `NgramClassifier` 相当)、bit-linear、C. Griffith LUT | 同じデータ・同じ 16 クラス・同じ分割・同じ評価コード |
| 参考比較 | D. JapaneseTinyAgentLM 3M(JSON を生成する Transformer) | 出力の形が違う(可変長の call 列、数値引数)。分類器と同じ表に並べない |
| PC と実機 | PC(Apple M1、Python / ホスト C)と ESP32-S3 実機 | 分けて書く。実機の速度比較は同じボード・同じファームウェア(`GA_MODEL_KIND` の切り替え) |

## データと分割

`docs/data.md`。test = HF validation の対象内 2,826 件(モデル選択に使わない)、dev = train の 5%。

## 指標(`gateact.evaluation.metrics`)

| 指標 | 定義 |
|---|---|
| Exact Accuracy | 予測クラス = 正解クラスの割合 |
| Macro-F1 | 正解または予測に 1 件以上現れたクラスの F1 の平均 |
| Precision / Recall | クラスごと(`per_class`) |
| **False Action Rate** | 正解が no_action の文のうち、何かの Action を予測した割合(誤って動く) |
| Missed Action Rate | 正解が Action の文のうち、no_action を予測した割合 |
| no_action Precision / Recall | no_action クラスの P / R |
| 否定表現の正解率 | データセットのカテゴリ `negation` の文の Exact |
| カテゴリ別 | single / negation / correction / no_action(対象内のみ) |
| カバー率 | 対象内 / 全体(validation で 62.2%) |
| `json_exact` | 予測クラスを既定の引数で call 列に戻したとき、元の正解 JSON と一致する割合(対象内) |
| `full_test_json_exact` | 上を validation 全 4,546 件に対して計算(対象外は必ず不正解)。元の JSON 生成タスクとの機能差 |
| 対象外への予測 | 対象外の文(複数 Action など)に何を予測したかの分布(`test_out_of_scope`) |
| 95% 区間 | test のサンプルを 1,000 回重複抽出した percentile 区間(Exact、Macro-F1) |

**注意**: False Action Rate は「何でも no_action と答える」モデルほど低くなる。no_action の
Precision、Missed Action Rate と必ず一緒に見る。

## 推論の定義

- Griffith LUT: **hard 回路**(`model.eval()`)の整数スコアの argmax。同点は小さいクラス ID
  (C と同じ規則)。soft の精度は学習の記録(`dev_soft_acc`)にだけ残す。
- ベースライン: PC の結果は f32。実機の比較では int8 量子化版を使い、量子化後の精度を別に記録する
  (`gateact export-baseline --report`)。

## 一致検証

| 段 | 方法 |
|---|---|
| 特徴量 Python ⇔ C | `tests/integration/test_features_c_parity.py`(validation 全文 + 境界ケース + ランダム) |
| PyTorch hard ⇔ IR ⇔ 圧縮 IR ⇔ 生成 C | `gateact export`(評価セット全文)、`tests/integration/test_export_verify.py` |
| テキスト → Action(ファームウェアのホストビルド) | `tests/integration/test_firmware_*_host.py` |
| ESP32-S3 実機 | `SELFTEST`(同梱サンプル)と `scripts/esp32_bench.py`(評価セット全文をシリアルで照合) |

## 実行

```bash
uv run gateact baseline --kind ngram --seed 0          # results/baselines/ngram_s0.json
uv run gateact train configs/lut/<name>.json --seed 0  # results/lut/<name>_s0.json, runs/<name>/
uv run gateact summarize results/baselines results/lut --out results/summary.json
```

## 結果

`results/summary.md`(`gateact summarize` の出力)と `docs/research.md` を参照。

# データ

## 出典とライセンス

| 項目 | 値 |
|---|---|
| データセット | [`japanese-data-analyze/JapaneseTinyAgentLM-Action-Synth`](https://huggingface.co/datasets/japanese-data-analyze/JapaneseTinyAgentLM-Action-Synth) |
| revision | `6ac72256456d8e76fca4708823c853438331619f`(`gateact.data.source` で固定) |
| ライセンス | Apache-2.0(2026-10-20 に CC BY-SA 4.0 から変更) |
| アクセス | 利用申請制(manual gate)。`hf auth login` が必要 |
| 件数 | train 85,902 / validation 4,546(公開情報と一致) |
| SHA-256 | train `570ae9e0…b5f306` / validation `795e853d…dd3333`(全桁は `results/data/manifest.json`) |

**データ本体は Git に入れない**(`/data/` は `.gitignore`)。コミットするのは manifest(件数・
SHA-256・集計)だけ。CI はデータもトークンも使わず、自作の fixture(`tests/fixtures/`)で動く。

```bash
uvx --from huggingface_hub hf auth login     # 初回のみ
make data                                    # 取得 + SHA-256 検証 + 検査 + manifest
```

## 形式

1 行 1 件の JSONL: `id`, `input`(日本語の文), `output`(Action の call 列の JSON 文字列。
`[]` は Action なし), `category`(single / multi_action / negation / correction / no_action),
`language`, `pair_id`, `generator`, `relabeled`。`gateact.data.io.load_jsonl` は不正な JSON、
必須フィールドの欠落、空の入力、不正な `output` を行番号付きで拒否する。

## 検査結果(`gateact data-check`、2026-11-20)

| 分割 | 件数 | 対象内(カバー率) | 重複グループ | ラベル矛盾 |
|---|---:|---:|---:|---:|
| train | 85,902 | 53,115(61.8%) | 0 | 0 |
| validation | 4,546 | 2,826(62.2%) | 0 | 0 |

- 正規化テキストの train / validation の重なり: **0 件**。
- 全文が NFKC 済み(NFKC で変わる文 0 件)。UTF-8 で最大 180 バイト。

## 固定クラス(`configs/labels_v1.json`)

単一の call を `name` と主要な列挙引数で 16 クラスに分ける。`[]` は `no_action`。

| クラス | 対応 | train | validation |
|---|---|---:|---:|
| no_action | `[]` | 27,783 | 1,513 |
| look_up / down / left / right / center | `look` の direction(程度・角度は捨てる) | 4,200 / 3,156 / 3,092 / 2,903 / 1,394 | 192 / 181 / 167 / 151 / 74 |
| nod / shake / bow | 回数は捨てる | 3,469 / 261 / 123 | 167 / 10 / 6 |
| expr_happy / surprised / sad | `set_expression` | 1,772 / 1,253 / 971 | 98 / 54 / 61 |
| led_on / led_off | `set_led`(off 以外の色は on) | 1,939 / 191 | 103 / 5 |
| volume_up / volume_down | `adjust_volume` の direction | 255 / 353 | 15 / 29 |

### 対象外(分類から除外し、別集計する。**no_action にはしない**)

| 理由 | train | validation |
|---|---:|---:|
| multi_action(2 つの call) | 24,202 | 1,265 |
| unsupported:turn | 3,013 | 162 |
| unsupported:look(斜め方向) | 2,940 | 150 |
| unsupported:set_expression(neutral / angry / sleepy / doubt) | 963 | 49 |
| unsupported:set_volume / set_brightness / adjust_brightness | 661 / 469 / 539 | 41 / 21 / 32 |

捨てた引数(程度・回数・LED の色など)があるサンプルは対象内の 23% ある(`arg_loss_rate`)。
これを元の JSON に戻したときの一致率は `json_exact` として別に出す(評価の節)。

## 分割

- **test = HF の validation(4,546 件)**。モデル選択には使わない。
- train から、正規化テキストの SHA-256 で決定的に 5% を **dev** に切り出す(同じ文は同じ側。
  salt `gateact-dev-v1`)。epoch とハイパーパラメータの選択は dev だけで行う。
- test と同じ正規化テキストを train から除く処理を入れている(今のデータでは 0 件)。
- 対象内の件数: train 50,502 / dev 2,613 / test 2,826(test の対象外 1,720)。

## fixture

`tests/fixtures/action_mini.jsonl`(24 件)は開発者が書いた文で、第三者の文を含まない。

# 開発計画と進捗

Claude Code はこの表の上から、未完了(`[ ]`)で依存が満たされたタスクを1つずつ進める。
完了したら `[x]` にし、日付と根拠(テスト名・results のパス)を書く。外部要因で止まる
ものは `BLOCKED:` を付けて次へ進む。

凡例: `[x]` 完了 / `[ ]` 未着手 / `[~]` 進行中 / `BLOCKED` 外部要因で停止

## 前提・確定事項(2026-11-20 調査)

- GitHub `ayutaz/GateAct-JP` は**未作成**(`gh repo view` が not found)。remote なし。
  指示どおり勝手に作成しない → ローカルブランチで開発し、作成されたら push / PR。
- Griffith: `e0675f4e36596954aa424d522e0ee966bba9a3e1`(v0.1.0、2026-11-20)を uv Git 依存で固定。
  Python ≥3.11、torch ≥2.8。`LutLayer`/`GroupSum`/`export_ir`/`deploy.build_bundle`/
  `deploy.verify.build_c_model`/`ir.simulate` を使う。
- JapaneseTinyAgentLM: `e32b6c15734c02eb31fd51d0128e53057d045c8a` を参照(コードはコピーしない)。
- データ: HF `japanese-data-analyze/JapaneseTinyAgentLM-Action-Synth` revision
  `6ac72256456d8e76fca4708823c853438331619f`、Apache-2.0、manual gate。`ayousanz` で
  アクセス確認済み。train 85,902 / validation 4,546。
- ESP-IDF v5.5 が `~/esp/esp-idf` に導入済み。ESP32-S3 のシリアルポートは 2026-11-20 時点で未接続。
- Python 3.12(uv 管理)。

## M0: 開発環境と CI

- [x] リポジトリ・Git・ツール確認(2026-11-20)
- [x] uv プロジェクト(`pyproject.toml`、`uv.lock`、Python 3.12、Griffith SHA 固定)
- [x] ハーネス(`CLAUDE.md`、本計画、`Makefile`、`.claude/`)
- [x] pytest / ruff / mypy 導入と最小テスト(2026-11-20、`tests/unit/test_lut_model.py`)
- [x] Griffith XOR: 学習 → hard 精度 → IR → C 生成 → Python/IR/C 一致テスト(2026-11-20、
  `tests/integration/test_xor_pipeline.py`。Griffith wheel に C ランタイムが無い問題を
  `gateact.export.griffith_runtime` で回避 → `docs/development.md`)
- [x] GitHub Actions `ci.yml` / `firmware.yml` / `release.yml`(2026-11-20、actionlint 通過。
  remote 未作成のため GitHub 上では未実行)
- [x] ESP32-S3 向け XOR ファームウェアのビルド(2026-11-20、`make firmware-xor`、ESP-IDF v5.5、
  イメージ 197,895 B、`results/m0/firmware_xor_esp32s3_size.json`。ホストビルドで4/4一致。
  実機実行はボード未接続のため未実施)

## M1: データ・特徴量・ベースライン

- [x] データ取得スクリプト + 検査(JSON 不正・欠損・重複・train/val 分離)+ manifest
  (2026-11-20、`make data`、`results/data/manifest.json`。重複 0・矛盾 0・重なり 0)
- [x] 固定クラス設計(`configs/labels_v1.json`、16 クラス)と対象外の別集計(カバー率 train 61.8% / val 62.2%)
- [x] テキスト正規化(GateAct-norm-v1)と FNV-1a ハッシュ n-gram ビット特徴量(Python)
- [x] 特徴量の C99 実装と Python/C ビット一致テスト(validation 全 4,546 文で一致、ASan/UBSan 確認)
- [x] 評価指標(Exact/Macro-F1/P/R/False Action/no-action P/R/否定/カバー率/json_exact/bootstrap CI)
- [x] ベースライン A: ルールベース(test exact 83.44%)
- [x] ベースライン B: n-gram 分類器(JapaneseTinyAgentLM `NgramClassifier` 相当、3 seed 98.76 ± 0.07%)
- [x] 追加比較: 同じ特徴ビットの線形分類器 bit-linear(w512: 93.54 ± 0.02%、w1024 95.97%、w2048 96.85%)
- [x] ベースライン評価結果を `results/baselines/` に保存

## M2: Griffith モデル

- [x] LUT2 モデル構築・学習ループ・checkpoint・soft/hard 評価(dev の hard 精度で epoch 選択)
- [x] 1 seed で学習 → PC 精度測定(LUT2 2×4k: 71.37% — 不十分。LUT4 warp: 86.84%)
- [x] 入力幅 / ゲート数 / LUT2 vs LUT4 vs LUT6 の小規模比較(sweep_m2a/m2b。最良 LUT6 warp 512 bit: 90.80%。入力を広げると下がる)
- [x] 3 seed 以上で比較(LUT6 warp 512 bit: 91.21 ± 0.38%。docs/research.md)
- [x] 改善実験(sweep_m2d/e/f): tau 4 で 93.97 ± 0.34%、ゲート 2 倍 + tau 8 で 94.99 ± 0.13%(3 seed)

## M3: C コード生成

- [x] IR export・回路圧縮・バンドル・C99 生成(`gateact export`、dense / dense-flat)
- [x] PyTorch hard / IR / C の完全一致テスト(圧縮前後含む。`tests/integration/test_export_verify.py`)
- [x] サイズ測定(ゲート数・IR・C・.rodata): LUT6 512 bit s0 と int8 n-gram / bit-linear の ESP32-S3 イメージ(docs/research.md)。dense-flat は LUT6 には適用されない(Griffith の判定「対象の2入力lut_layer列が無い」、出力は off と同一)

## M4: ESP32-S3 実機評価

- [x] ファームウェア(シリアル UTF-8 入力 → 特徴量 → 推論 → JSON 出力 + 時間)
- [x] ベンチマークモード(同梱サンプル、既定 2,000 回、P50/P95/P99)
- [x] ホストビルドで Python と同じ出力を確認(LUT / bit-linear / n-gram)
- [x] ホスト C の推論時間(`scripts/host_bench.py`、PC の参考値。`results/host/`)
- [ ] BLOCKED(ボード未接続): 実機での書き込み・一致率・Flash/RAM/heap・レイテンシ
  (`docs/hardware.md`、`scripts/esp32_bench.py` は準備済み)
- [x] n-gram / bit-linear の int8 C 実装とファームウェア切り替え(GA_MODEL_KIND 2/3)

## M5: 研究評価・公開準備

- [x] 全ベースライン比較表、統計的ばらつき(`results/summary.md`、README)
- [x] README(日本語 + 英語概要)、再現手順(実機の数値は M4 後に追記)
- [ ] GitHub Release 準備(remote 作成後)
- [x] 次の研究課題(docs/research.md)

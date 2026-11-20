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

- [ ] データ取得スクリプト + 検査(JSON 不正・欠損・重複・train/val 分離)+ manifest
- [ ] 固定クラス設計(`configs/labels_v1.json`)と対象外の別集計
- [ ] テキスト正規化(GateAct-norm-v1)と FNV-1a ハッシュ n-gram ビット特徴量(Python)
- [ ] 特徴量の C99 実装と Python/C ビット一致テスト
- [ ] 評価指標(Exact/Macro-F1/P/R/False Action/no-action P/R/否定/カバー率)
- [ ] ベースライン A: ルールベース
- [ ] ベースライン B: n-gram 分類器(JapaneseTinyAgentLM `NgramClassifier` 相当)
- [ ] ベースライン評価結果を `results/baselines/` に保存

## M2: Griffith モデル

- [ ] LUT2 モデル構築・学習ループ・checkpoint・soft/hard 評価
- [ ] 1 seed で学習 → PC 精度測定
- [ ] 入力幅 / ゲート数 / LUT2 vs LUT4 の小規模比較
- [ ] 3 seed 以上で比較

## M3: C コード生成

- [ ] IR export・回路圧縮・バンドル・C99 生成
- [ ] PyTorch hard / IR / C の完全一致テスト(圧縮前後含む)
- [ ] サイズ測定(ゲート数・IR・C・.rodata)

## M4: ESP32-S3 実機評価

- [ ] ファームウェア(シリアル UTF-8 入力 → 特徴量 → 推論 → JSON 出力 + 時間)
- [ ] ベンチマークモード(同梱サンプル、1,000 回以上、P50/P95/P99)
- [ ] ホストビルドで実機と同じ出力を確認
- [ ] 実機: 書き込み・一致率・Flash/RAM/heap・レイテンシ(要ボード接続)
- [ ] n-gram 分類器の ESP32-S3 実装(同一ボードでの速度比較)

## M5: 研究評価・公開準備

- [ ] 全ベースライン比較表、統計的ばらつき
- [ ] README(日本語 + 英語概要)、docs 整備、再現手順
- [ ] GitHub Release 準備(remote 作成後)
- [ ] 次の研究課題

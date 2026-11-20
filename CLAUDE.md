# CLAUDE.md — GateAct-JP 開発ルール

Claude Code がこのリポジトリで自律的に開発を進めるためのルール。**作業の起点は必ず
[`docs/plan.md`](docs/plan.md)**(マイルストーンとタスクの進捗表)。

## プロジェクト目的

Griffith(Differentiable Logic Gate Network)で日本語テキスト → ロボット Action の固定クラス
分類器をゼロから学習し、論理回路 → C99 → ESP32-S3 単体で推論する。既存の
JapaneseTinyAgentLM の `NgramClassifier`・ルールベース・3M Transformer と公平に比較し、
**否定的な結果も含めて**実測で報告する。

## 自律開発ループ(ハーネス)

1. `docs/plan.md` を読み、未完了で依存が満たされた最初のタスクを選ぶ(`/next` コマンド)。
2. TDD で実装する(下記)。
3. `make check` を通す(lint / format / pytest)。C や firmware を触ったら `make c-parity` /
   `make firmware` も。
4. `docs/plan.md` のチェックを更新し、実測値は `results/` に JSON、要約は docs に書く。
5. 機能単位でコミットする(細かすぎず、1機能 = 1〜数コミット)。
6. 外部要因(実機なし、アクセス権なし)で止まるタスクは plan に「BLOCKED: 理由」を書いて次へ。

人に確認が必要なのは: GitHub リポジトリ作成・push、外部への公開(PyPI/HF/Release)、
実機への書き込み前のボード確認、課金が発生する操作。

## uv(必須)

```bash
uv sync --locked          # 環境構築(CI と同じ)
uv run pytest             # 実行は必ず uv run
uv add <pkg>              # 依存追加(uv.lock も更新される) / 開発用は uv add --group dev
uv build                  # wheel / sdist
```

`pip install` と `requirements.txt` は使わない。Python は 3.12 に固定(`.python-version`)。
Griffith は `pyproject.toml` の `[tool.uv.sources]` で **Git commit SHA 固定**。更新するときは
SHA を変え `uv lock` し、`make check` と `make c-parity` を通してから別コミットにする。

## TDD 手順

1. 期待する振る舞いを決める → 2. 失敗する pytest を書く → 3. **期待した理由で失敗すること**を
確認 → 4. 最小実装 → 5. 通す → 6. リファクタ → 7. 再実行 → 8. コミット。

## コマンド

| コマンド | 内容 |
|---|---|
| `make check` | `uv sync --locked` + ruff check + ruff format --check + pytest(CI と同じ) |
| `make test` / `make cov` | pytest / カバレッジ付き |
| `make fmt` | ruff format + ruff check --fix |
| `make c-parity` | 特徴量 C 実装・生成モデル C の Python 一致検証(cc 必須) |
| `make data` | HF からデータ取得 + 検査 + manifest(要 `hf auth login` と利用承認) |
| `make train-smoke` | 小さな fixture で学習 → export → C 一致まで(CI と同じ) |
| `make firmware` | ESP-IDF で ESP32-S3 ファームウェアをビルド(要 `~/esp/esp-idf`) |

## コーディング規約

- 型ヒント必須、公開 API には docstring(Google 形式・日本語可)。ruff(`D` 含む)に従う。
- 乱数は `gateact.utils.seed_everything(seed)` で固定。実験設定は `configs/*.json` に保存し、
  結果 JSON に設定・git SHA・データ SHA-256 を埋め込む。
- Python の組み込み `hash()` を特徴量に使わない(FNV-1a 32bit 固定、`docs/architecture.md`)。
- 特徴量の仕様を変えたら Python/C 両方とテスト、仕様バージョン(`FEATURE_SPEC_VERSION`)を更新。

## Git 運用

- `main` へ直接コミットしない。`feat/...` `fix/...` `docs/...` ブランチで作業し、マイルストーン
  完了時に main へ fast-forward マージ(remote ができたら PR ベース)。
- `main` への force push 禁止。既存履歴を書き換えない。
- コミットメッセージは英語 Conventional Commits(`feat:`, `fix:`, `test:`, `docs:`, `ci:`)。

## CI/CD

- `.github/workflows/ci.yml`: PR と main push。Ubuntu(主)+ macOS。lint・test・cov・build・
  train smoke・C parity。重い学習・実機測定は CI で実行しない。
- `firmware.yml`: ESP-IDF コンテナで ESP32-S3 ビルド(書き込みなし)、サイズを Artifact に。
- `release.yml`: `v*` タグで全チェック → wheel/sdist + 成果物を GitHub Release に添付。
  PyPI / Hugging Face への公開は自動化しない(明示的な手順のみ)。
- 外部 Action は commit SHA で固定、`permissions` は最小。

## データ管理

- 学習データ(HF `japanese-data-analyze/JapaneseTinyAgentLM-Action-Synth`、Apache-2.0、
  manual gate)は `data/raw/`(Git 管理外)に置く。コミットするのは manifest(件数・SHA-256・
  revision)と集計値だけ。
- `tests/fixtures/` には自作の小さな合成 fixture のみ(Apache-2.0)。第三者の文を入れない。
- HF トークンや `.env` をコミットしない。CI はトークンなしで動くこと。

## 評価・実測のルール

- **実測していない数値を書かない**。モック・ホスト実行の値を実機値として書かない
  (PC / ホスト C / ESP32-S3 実機を必ず区別して表記)。
- soft 精度ではなく **hard(離散化後)精度**を主指標にする。
- 主比較(固定クラス分類器どうし: ルール / n-gram / Griffith)と参考比較(3M Transformer の
  JSON 生成)を分ける。対象外サンプル(複数 Action 等)は別集計してカバー率を必ず記録。
- 複数 seed(≥3)で平均 ± SD。失敗した実験も `results/` と `docs/research.md` に残す。

## ESP32-S3 開発手順

```bash
. ~/esp/esp-idf/export.sh                 # ESP-IDF v5.5(パイプに繋がず source)
cd firmware/esp32s3/gateact
idf.py set-target esp32s3 && idf.py build
ls /dev/cu.usbmodem* /dev/cu.usbserial*   # ボード確認(書き込み前に必ず)
idf.py -p <PORT> flash
uv run python scripts/esp32_bench.py --port <PORT>   # 非対話でログ取得・集計
```

`idf.py monitor` は対話端末が必要なので使わない。

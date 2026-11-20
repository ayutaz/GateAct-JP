---
description: docs/plan.md の次の未完了タスクを TDD で進め、検証・コミット・計画更新まで行う
---

GateAct-JP の自律開発ループを1周進める。CLAUDE.md のルールに従うこと。

1. `docs/plan.md` を読み、未完了(`[ ]`)で依存が満たされた最初のタスクを選ぶ。
   `$ARGUMENTS` があればそのタスク/マイルストーンを優先する。
2. 必要なら関連コード・docs・Griffith のソース(`.venv` 内 `griffith_logic`)を読んで仕様を確認する。
   API を推測しない。
3. TDD: 失敗するテスト → 失敗理由の確認 → 最小実装 → 通す → リファクタ。
4. `make check` を通す。C/firmware を触ったら `make c-parity` / `make firmware` も。
5. 学習・評価を実行したら結果 JSON を `results/` に保存し、実測値だけを docs に反映する。
6. `docs/plan.md` を更新(`[x]`、日付、根拠)。外部要因で止まるなら `BLOCKED: 理由`。
7. 機能単位でコミット(Conventional Commits、英語)。
8. 実装内容・テスト結果・実測値・変更ファイル・Git 状態・次の作業を簡潔に報告する。
   時間が許す限り次のタスクへ続ける。

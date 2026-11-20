# Contributing / 貢献ガイド

GateAct-JP への貢献を歓迎します。Contributions are welcome.

1. Issue で目的を共有する(バグ報告 / 実験提案のテンプレートあり)。
2. `uv sync --locked` で環境を作り、ブランチを切る(`feat/...`, `fix/...`)。
3. **TDD**: 失敗するテストを先に書き、`make check` を通す。
4. 実測値は `results/` に JSON で残し、実測していない数値をドキュメントに書かない。
5. データセットの本文・トークン・非公開データをコミットしない(`docs/data.md`)。
6. PR では変更内容・テスト結果・(該当すれば)実測環境を書く。

依存管理は uv のみ(`pip install` / `requirements.txt` は使わない)。ライセンスは Apache-2.0 で、
貢献物も Apache-2.0 で提供されたものとみなします。

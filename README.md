# GateAct-JP

**日本語の指示文からロボットの Action を判定する、論理ゲートだけでできた超軽量分類器。**
[Griffith](https://github.com/tsukushi-gate/Griffith)(Differentiable Logic Gate Network)で
ゼロから学習し、論理回路 → C99 → ESP32-S3 単体で推論します。
[JapaneseTinyAgentLM](https://github.com/ayutaz/JapaneseTinyAgentLM) の Action schema と
データ・評価方法を出発点にし、既存の分類器と精度・サイズ・速度・メモリを実測で比較します。

> 開発中(M0)。進捗は [`docs/plan.md`](docs/plan.md)。数値はすべて実測のみを記載します。

## English summary

GateAct-JP trains a tiny Japanese text → robot-action classifier made only of lookup-table logic
gates (Griffith DLGN), exports it to dependency-free C99, and runs it on an ESP32-S3. It is
compared fairly (same data, classes and splits) against a keyword rule baseline and the
n-gram classifier of JapaneseTinyAgentLM, including negative results.

## Quick start

```bash
uv sync --locked
make check                 # lint + tests (includes Griffith XOR → IR → C parity)
make firmware-xor          # ESP32-S3 build (needs ESP-IDF v5.5)
```

## License

Apache-2.0([LICENSE](LICENSE)、第三者コンポーネントは [NOTICE](NOTICE))。

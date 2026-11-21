# Changelog

形式は [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/)、バージョンは
[Semantic Versioning](https://semver.org/lang/ja/) に従う。数値はすべて実測(`docs/research.md`)。

## [0.1.0] - 未リリース

最初の公開版(研究段階)。**ESP32-S3 実機での計測はまだ含まない**(ファームウェアのビルドとサイズのみ)。

### 追加

- 日本語テキスト → 16 クラスの Action 分類のデータ処理(`labels_v1`、対象外サンプルの別集計、
  train/dev/test 分割と重複・漏洩検査、HF データの revision と SHA-256 固定)。
- 正規化 GateAct-norm-v1 と文字 n-gram ハッシュ特徴量 gateact-feat-v1(Python と C99 で完全一致)。
- Griffith LUT モデルの学習(dev の hard 精度で epoch 選択)と評価(Exact、Macro-F1、
  False/Missed Action Rate、否定表現、json_exact、95% 区間、複数 seed の集計)。
- ベースライン: ルール、n-gram 分類器(JapaneseTinyAgentLM 相当)、bit-linear、int8 量子化。
- IR → 回路圧縮 → C99 の変換と、PyTorch hard / IR / 分割 IR / 圧縮 IR / 生成 C の全段一致検証。
- LUT6 層を厳密に等価な LUT5 + LUT3 に分割する IR 変換(Griffith の C 生成が LUT6 に未対応のため)。
- ESP32-S3 ファームウェア(シリアル UTF-8 入力 → JSON 出力、SELFTEST / BENCH / MEM、
  LUT・bit-linear・n-gram の切り替え)と、ホストビルドでの一致検証。
- CI(lint / test / build、Ubuntu + macOS)、ESP-IDF でのファームウェアビルド、下書き Release。

### 結果(PC、test 2,826 件、3 seed)

- GateAct LUT6 tau 4: 93.97 ± 0.34%、2 倍 + tau 8: 94.99 ± 0.13%。
- bit-linear 93.54 ± 0.02%、n-gram 98.76 ± 0.07%、ルール 83.44%。

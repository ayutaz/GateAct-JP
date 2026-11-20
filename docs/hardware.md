# ESP32-S3 実機手順

## 前提

- ESP-IDF v5.5(`~/esp/esp-idf`、Python venv は `~/.espressif/python_env/idf5.5_py*_env`)
- ボード: ESP32-S3。既定の設定は M5Stack CoreS3 と同じ(240MHz、D-cache 64KB / 64B line、
  I-cache 32KB、`-O2` 相当、コンソールはネイティブ USB-Serial/JTAG、quad PSRAM は見つかれば使う)。
  UART の開発ボード(DevKitC など)では `idf.py menuconfig` でコンソールを UART に変える。
- flash 4MB 以上(パーティション: アプリ 3.9MB。`firmware/esp32s3/gateact/partitions.csv`)

## 1. モデルを生成する

```bash
# Griffith LUT 分類器(GA_MODEL_KIND=1)
uv run gateact export runs/<name>/<name>_s0.pt --out firmware/esp32s3/gateact/main --n-samples 200
# 比較用: int8 bit-linear(2)/ int8 n-gram(3)
uv run gateact export-baseline runs/baselines/ngram_s0.pt --out firmware/esp32s3/gateact/main \
    --n-samples 200 --report results/device/ngram_int8_s0.json
```

`main/model/` は生成物(Git 管理外)。`text_samples.h` に評価セットの先頭 N 文と、Python で
計算した特徴ビット・スコアが入り、実機の SELFTEST で照合する。

## 2. ビルド

```bash
make firmware          # set-target esp32s3 → build → size
```

## 3. 書き込み(**必ずボードを確認してから**)

```bash
ls /dev/cu.usbmodem* /dev/cu.usbserial*        # ポートの確認
. ~/esp/esp-idf/export.sh
# 既存のファームウェアがあるボードでは、先に flash 全体を退避する(16MB の例)
esptool.py --chip esp32s3 -p <PORT> -b 921600 read_flash 0 0x1000000 backup_<board>.bin
cd firmware/esp32s3/gateact && idf.py -p <PORT> flash
```

## 4. 計測

```bash
uv run --group device python scripts/esp32_bench.py --port <PORT> \
    --checkpoint runs/<name>/<name>_s0.pt --out results/device/<name>_esp32s3.json
```

- 起動ログ: `INFO`(CPU 周波数、flash、PSRAM、heap)、`SELFTEST`(C の特徴ビットとスコアが
  Python と一致するか)、`BENCH`(既定 2,000 回の前処理 / 推論 / 全体の P50 / P95 / P99、µs)、`MEM`
- 評価セット全文(4,546 文)をシリアルで送り、実機のスコアを IR simulate と照合する
  (`query.score_mismatch` = 0 が完全一致)。実機の予測で分類指標も出す。
- 同じ手順をホストビルドで回すと(`--host-exe`)、実機との比較ができる。

`idf.py monitor` は対話端末が要るので使わない。

## 実測の状況

| 項目 | 状態 |
|---|---|
| ESP32-S3 向けビルド(XOR / GateAct smoke) | 済(2026-11-20、ESP-IDF v5.5) |
| ホストビルドでのテキスト → Action の一致 | 済(テストで自動確認) |
| 実機での実行・計測 | **未実施**(2026-11-20 時点でボード未接続) |

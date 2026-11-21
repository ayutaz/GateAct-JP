# 開発ガイド

## 環境構築

```bash
uv sync --locked      # Python 3.12 + 依存(Griffith は commit SHA 固定)
make check            # lint + format + pytest(CI と同じ)
```

- macOS(Apple Silicon)は PyPI の torch、Linux(CI)は `download.pytorch.org/whl/cpu` の
  CPU 版 torch を使う(`pyproject.toml` の `[tool.uv.sources]`)。
- C の一致検証には `cc`(clang / gcc)が必要。無い環境では `needs_cc` テストが skip される。

## Griffith の固定と、ランタイム取得のパッチ

`griffith-logic` は `tsukushi-gate/Griffith@e0675f4e36596954aa424d522e0ee966bba9a3e1`(v0.1.0)
に固定している。

**既知の問題([tsukushi-gate/Griffith#6](https://github.com/tsukushi-gate/Griffith/issues/6) で報告済み)**: v0.1.0 の wheel には C ランタイム(`runtime/c/`)が含まれず、
`griffith_logic.deploy.verify.RUNTIME_C` がパッケージの外(リポジトリのルート)を指すため、
インストールした環境では `build_c_model()` と `griffith deploy codegen-c` のランタイムコピーが
`no such file or directory: .../site-packages/runtime/c/src/mdl_kernels.c` で失敗する。

GateAct-JP 側の対処(`src/gateact/export/griffith_runtime.py`):

- Griffith のコードは変更しない。固定 commit のソースアーカイブ
  (`codeload.github.com/tsukushi-gate/Griffith/tar.gz/<SHA>`)から `runtime/c` の4ファイルだけを
  `~/.cache/gateact/griffith-runtime/<SHA>/c/` に展開し、**SHA-256 を固定値で検証**する。
- C をビルドする直前に `RUNTIME_C` をそのパスへ差し替える(`patch_griffith_runtime()`)。
- オフライン環境では `GATEACT_GRIFFITH_RUNTIME=<runtime/c のパス>` で指定できる(同じく検証する)。
- ファームウェアには `write_c_model()` が `mdl_kernels.{c,h}` / `mdl_simd.h` を生成ディレクトリへ
  コピーする(生成物は Git 管理外。Griffith の Apache-2.0 のまま)。

Griffith を更新するときは、`pyproject.toml` の `rev` と `griffith_runtime.py` の
`GRIFFITH_SHA` / `RUNTIME_FILES` の SHA-256 を同時に更新し、`make check` を通す。

## ESP32-S3

```bash
make firmware-xor     # XOR モデル生成 → idf.py set-target esp32s3 → build → size
```

ESP-IDF v5.5 を `~/esp/esp-idf` に置き、IDF の Python venv(`~/.espressif/python_env/idf5.5_py*_env`)
を使う。書き込みと実機計測の手順は [`hardware.md`](hardware.md)(M4 で追加)。

## ブランチとコミット

`CLAUDE.md` の Git 運用を参照。マイルストーン単位で `feat/m*-...` ブランチを切る。

## LUT6 の C 生成(上流に機能要望 [tsukushi-gate/Griffith#7](https://github.com/tsukushi-gate/Griffith/issues/7))

Griffith の C 生成は `n_inputs > 5` の lut_layer を拒否する。学習した LUT6 層は
`src/gateact/export/lut6.py` で LUT5 + LUT3 に厳密に分割してから C を生成する(`gateact export` が自動で行う)。

## リリース手順(人が実行する)

1. `pyproject.toml` の `version` と `CHANGELOG.md` を更新し、`uv lock`、`make check` を通して PR で main に入れる。
2. main で `git tag -a vX.Y.Z -m "GateAct-JP vX.Y.Z"` → `git push origin vX.Y.Z`。
3. `release.yml` が CI とファームウェアのビルドを通したあと、GitHub Release を**下書き**で作る
   (wheel / sdist、ファームウェアとサイズ、`results/` と `configs/`、`uv.lock`)。
4. 下書きの成果物と注記(実機で未計測の項目など)を確認してから、人が公開する。
   PyPI / Hugging Face には公開しない。

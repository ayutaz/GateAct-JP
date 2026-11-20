/*
 * GateAct-JP テキスト特徴量(gateact-feat-v1 / gateact-norm-v1)の C99 実装。
 * Python 実装 src/gateact/features/{normalize,hashing}.py とビット単位で一致する
 * (tests/integration/test_features_c_parity.py)。
 *
 * 依存は <stdint.h> / <stddef.h> / <string.h>(memset)のみ。ヒープ確保なし、浮動小数点なし。
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef GATEACT_FEATURES_H
#define GATEACT_FEATURES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GFEAT_SPEC_VERSION "gateact-feat-v1"
/* 正規化後のコードポイント数の上限(Python の MAX_CODEPOINTS と同じ) */
#define GFEAT_MAX_CODEPOINTS 254
#define GFEAT_MAX_ORDER 8

enum {
  GFEAT_OK = 0,
  GFEAT_ERR_UTF8 = -1,     /* 不正な UTF-8(overlong / surrogate / 範囲外 / 途中切れ) */
  GFEAT_ERR_TOO_LONG = -2, /* 正規化後 GFEAT_MAX_CODEPOINTS 超 */
  GFEAT_ERR_CONFIG = -3
};

typedef struct {
  uint32_t width;      /* 出力ビット幅 */
  uint32_t ngram_mask; /* bit (n-1) が立っていれば n-gram を使う(n <= 8) */
  uint32_t num_hashes; /* 1 n-gram あたりのビット数 */
} gfeat_config;

uint32_t gfeat_fnv1a32(const uint8_t *data, size_t len);
uint32_t gfeat_fmix32(uint32_t h);

/* UTF-8 を正規化したコードポイント列にする。out は GFEAT_MAX_CODEPOINTS+1 要素以上。 */
int gfeat_normalize(const uint8_t *utf8, size_t len, uint32_t *out, size_t *n_out);

/* 特徴ビット(0/1 を1バイトずつ、width バイト)を書く。Griffith の mdl_infer_ctx の入力形式。 */
int gfeat_extract(const uint8_t *utf8, size_t len, const gfeat_config *cfg, uint8_t *bits);

#ifdef __cplusplus
}
#endif

#endif /* GATEACT_FEATURES_H */

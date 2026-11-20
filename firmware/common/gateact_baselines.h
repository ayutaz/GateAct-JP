/*
 * 比較用ベースライン(int8 量子化)の C99 実装。Python: src/gateact/baselines/quantized.py
 *  - bit-linear: score_k = b_k + Σ_{bit_i=1} W[k][i]
 *  - n-gram:     score_k = Σ_d W[k][d]·H[d] + b_k·count、H = Σ_g E[crc32(g) % B]
 *                (g は GateAct-norm-v1 で正規化した文の文字 1〜3-gram、前後に U+0002/U+0003)
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef GATEACT_BASELINES_H
#define GATEACT_BASELINES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t gb_crc32(const uint8_t *data, size_t len);

void gb_bitlinear(const uint8_t *bits, uint32_t width, uint32_t n_classes, const int8_t *weight,
                  const int32_t *bias, int32_t *scores);

/* gram ID を ids に書き、個数を *n_ids に返す(ids は 3 * (GFEAT_MAX_CODEPOINTS + 2) 要素以上)。
 * 戻り値は gfeat_normalize と同じエラーコード。 */
int gb_ngram_ids(const uint8_t *utf8, size_t len, uint32_t buckets, uint32_t *ids, size_t *n_ids);

/* hsum は dim 要素の作業領域 */
void gb_ngram_scores(const uint32_t *ids, size_t n_ids, uint32_t dim, uint32_t n_classes,
                     const int8_t *embedding, const int8_t *weight, const int32_t *bias,
                     int32_t *hsum, int32_t *scores);

#ifdef __cplusplus
}
#endif

#endif /* GATEACT_BASELINES_H */

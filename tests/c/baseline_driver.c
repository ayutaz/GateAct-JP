/* テスト用: stdin = {len(u32 LE), bytes}* → stdout = 各テキストの GB_CLASSES 個の int32 LE スコア
 * (エラー時は全要素 INT32_MIN)。-DKIND_BITLINEAR / -DKIND_NGRAM でモデルを選ぶ。 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "baseline_model.h"
#include "gateact_baselines.h"
#include "gateact_features.h"

int main(void) {
  uint8_t lb[4];
  int32_t scores[GB_CLASSES];
  while (fread(lb, 1, 4, stdin) == 4) {
    uint32_t len = (uint32_t)lb[0] | ((uint32_t)lb[1] << 8) | ((uint32_t)lb[2] << 16) |
                   ((uint32_t)lb[3] << 24);
    uint8_t *text = malloc(len ? len : 1);
    if (len && fread(text, 1, len, stdin) != len) return 3;
    int rc;
#if defined(KIND_BITLINEAR)
    static uint8_t bits[GB_WIDTH];
    gfeat_config cfg = {GB_WIDTH, 7u, 1u};
    rc = gfeat_extract(text, len, &cfg, bits);
    if (rc == 0) gb_bitlinear(bits, GB_WIDTH, GB_CLASSES, GB_WEIGHT, GB_BIAS, scores);
#else
    static uint32_t ids[3 * (GFEAT_MAX_CODEPOINTS + 2)];
    static int32_t hsum[GB_DIM];
    size_t n = 0;
    rc = gb_ngram_ids(text, len, GB_BUCKETS, ids, &n);
    if (rc == 0)
      gb_ngram_scores(ids, n, GB_DIM, GB_CLASSES, GB_EMBEDDING, GB_WEIGHT, GB_BIAS, hsum, scores);
#endif
    for (int k = 0; k < GB_CLASSES; ++k) {
      uint32_t u = rc == 0 ? (uint32_t)scores[k] : 0x80000000u;
      uint8_t b[4] = {(uint8_t)u, (uint8_t)(u >> 8), (uint8_t)(u >> 16), (uint8_t)(u >> 24)};
      fwrite(b, 1, 4, stdout);
    }
    free(text);
  }
  return 0;
}

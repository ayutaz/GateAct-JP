/* ホスト検証用ドライバ: stdin = n(u32 LE) + n × MDL_INPUT_WIDTH バイト(0/1)
 * stdout = n × MDL_OUTPUT_WIDTH の int32 LE スコア。生成 model.h の mdl_infer_ctx を使う。
 * SPDX-License-Identifier: Apache-2.0 */
#include <stdint.h>
#include <stdio.h>

#include "model.h"

static uint32_t g_work[(MDL_CTX_BYTES + 3) / 4 + 1];

int main(void) {
  uint8_t nb[4];
  if (fread(nb, 1, 4, stdin) != 4) return 2;
  uint32_t n = (uint32_t)nb[0] | ((uint32_t)nb[1] << 8) | ((uint32_t)nb[2] << 16) |
               ((uint32_t)nb[3] << 24);
  mdl_ctx ctx;
  if (mdl_ctx_init(&ctx, g_work, sizeof(g_work)) != 0) return 3;
  static uint8_t bits[MDL_INPUT_WIDTH];
  static int32_t scores[MDL_OUTPUT_WIDTH];
  for (uint32_t i = 0; i < n; ++i) {
    if (fread(bits, 1, MDL_INPUT_WIDTH, stdin) != MDL_INPUT_WIDTH) return 4;
    if (mdl_infer_ctx(&ctx, bits, scores) != 0) return 5;
    for (int k = 0; k < MDL_OUTPUT_WIDTH; ++k) {
      uint32_t u = (uint32_t)scores[k];
      uint8_t b[4] = {(uint8_t)u, (uint8_t)(u >> 8), (uint8_t)(u >> 16), (uint8_t)(u >> 24)};
      fwrite(b, 1, 4, stdout);
    }
  }
  return 0;
}

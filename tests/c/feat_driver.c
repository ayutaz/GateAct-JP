/* テスト用ドライバ: stdin = [width,mask,num_hashes](u32 LE) + {len(u32 LE), bytes}*
 * stdout = レコードごとに status(i32 LE) + LSB-first パック済みビット。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gateact_features.h"

static int read_u32(uint32_t *v) {
  uint8_t b[4];
  if (fread(b, 1, 4, stdin) != 4) return 0;
  *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
  return 1;
}

static void write_i32(int32_t v) {
  uint32_t u = (uint32_t)v;
  uint8_t b[4] = {(uint8_t)u, (uint8_t)(u >> 8), (uint8_t)(u >> 16), (uint8_t)(u >> 24)};
  fwrite(b, 1, 4, stdout);
}

int main(void) {
  gfeat_config cfg;
  if (!read_u32(&cfg.width) || !read_u32(&cfg.ngram_mask) || !read_u32(&cfg.num_hashes)) return 2;
  uint8_t *bits = malloc(cfg.width);
  size_t nbytes = (cfg.width + 7) / 8;
  uint8_t *packed = malloc(nbytes);
  uint32_t len;
  while (read_u32(&len)) {
    uint8_t *text = malloc(len ? len : 1);
    if (len && fread(text, 1, len, stdin) != len) return 3;
    int rc = gfeat_extract(text, len, &cfg, bits);
    memset(packed, 0, nbytes);
    if (rc == GFEAT_OK)
      for (uint32_t i = 0; i < cfg.width; ++i)
        if (bits[i]) packed[i >> 3] |= (uint8_t)(1u << (i & 7));
    write_i32(rc);
    fwrite(packed, 1, nbytes, stdout);
    free(text);
  }
  free(bits);
  free(packed);
  return 0;
}

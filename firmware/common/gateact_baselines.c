/* SPDX-License-Identifier: Apache-2.0 */
#include "gateact_baselines.h"

#include <string.h>

#include "gateact_features.h"

static uint32_t g_crc_table[256];
static int g_crc_ready;

static void crc_init(void) {
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    g_crc_table[i] = c;
  }
  g_crc_ready = 1;
}

uint32_t gb_crc32(const uint8_t *data, size_t len) {
  if (!g_crc_ready) crc_init();
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) c = g_crc_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

void gb_bitlinear(const uint8_t *bits, uint32_t width, uint32_t n_classes, const int8_t *weight,
                  const int32_t *bias, int32_t *scores) {
  for (uint32_t k = 0; k < n_classes; ++k) scores[k] = bias[k];
  for (uint32_t i = 0; i < width; ++i) {
    if (!bits[i]) continue;
    for (uint32_t k = 0; k < n_classes; ++k) scores[k] += weight[k * width + i];
  }
}

static size_t enc(uint32_t cp, uint8_t *o) {
  if (cp < 0x80u) {
    o[0] = (uint8_t)cp;
    return 1;
  }
  if (cp < 0x800u) {
    o[0] = (uint8_t)(0xC0u | (cp >> 6));
    o[1] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 2;
  }
  if (cp < 0x10000u) {
    o[0] = (uint8_t)(0xE0u | (cp >> 12));
    o[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
    o[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 3;
  }
  o[0] = (uint8_t)(0xF0u | (cp >> 18));
  o[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
  o[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
  o[3] = (uint8_t)(0x80u | (cp & 0x3Fu));
  return 4;
}

int gb_ngram_ids(const uint8_t *utf8, size_t len, uint32_t buckets, uint32_t *ids, size_t *n_ids) {
  uint32_t seq[GFEAT_MAX_CODEPOINTS + 3];
  size_t n = 0;
  int rc = gfeat_normalize(utf8, len, seq + 1, &n);
  if (rc != GFEAT_OK) return rc;
  seq[0] = 0x02u;
  seq[n + 1] = 0x03u;
  size_t m = n + 2, out = 0;
  uint8_t buf[12];
  for (size_t order = 1; order <= 3; ++order) {
    for (size_t i = 0; i + order <= m; ++i) {
      size_t k = 0;
      for (size_t j = 0; j < order; ++j) k += enc(seq[i + j], buf + k);
      ids[out++] = gb_crc32(buf, k) % buckets;
    }
  }
  *n_ids = out;
  return GFEAT_OK;
}

void gb_ngram_scores(const uint32_t *ids, size_t n_ids, uint32_t dim, uint32_t n_classes,
                     const int8_t *embedding, const int8_t *weight, const int32_t *bias,
                     int32_t *hsum, int32_t *scores) {
  memset(hsum, 0, dim * sizeof(int32_t));
  for (size_t g = 0; g < n_ids; ++g) {
    const int8_t *row = embedding + (size_t)ids[g] * dim;
    for (uint32_t d = 0; d < dim; ++d) hsum[d] += row[d];
  }
  for (uint32_t k = 0; k < n_classes; ++k) {
    int32_t s = bias[k] * (int32_t)n_ids;
    const int8_t *wk = weight + (size_t)k * dim;
    for (uint32_t d = 0; d < dim; ++d) s += (int32_t)wk[d] * hsum[d];
    scores[k] = s;
  }
}

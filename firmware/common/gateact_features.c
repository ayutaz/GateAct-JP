/* SPDX-License-Identifier: Apache-2.0 — 仕様は gateact_features.h と src/gateact/features/ */
#include "gateact_features.h"

#include <string.h>

#include "gateact_norm_tables.h"

#define BOS 0x02u
#define EOS 0x03u
#define GOLDEN 0x9E3779B9u

uint32_t gfeat_fnv1a32(const uint8_t *data, size_t len) {
  uint32_t h = 0x811C9DC5u;
  for (size_t i = 0; i < len; ++i) {
    h ^= data[i];
    h *= 0x01000193u;
  }
  return h;
}

uint32_t gfeat_fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;
  return h;
}

/* 厳格な UTF-8 デコード(Python の bytes.decode('utf-8') と同じ受理範囲)。
 * 成功なら消費バイト数(1..4)、失敗なら 0。 */
static size_t decode_utf8(const uint8_t *s, size_t len, uint32_t *cp) {
  uint8_t b0 = s[0];
  if (b0 < 0x80) {
    *cp = b0;
    return 1;
  }
  size_t n;
  uint32_t c, min;
  if (b0 >= 0xC2 && b0 <= 0xDF) {
    n = 2; c = b0 & 0x1Fu; min = 0x80;
  } else if (b0 >= 0xE0 && b0 <= 0xEF) {
    n = 3; c = b0 & 0x0Fu; min = 0x800;
  } else if (b0 >= 0xF0 && b0 <= 0xF4) {
    n = 4; c = b0 & 0x07u; min = 0x10000;
  } else {
    return 0;
  }
  if (len < n) return 0;
  for (size_t i = 1; i < n; ++i) {
    if ((s[i] & 0xC0u) != 0x80u) return 0;
    c = (c << 6) | (s[i] & 0x3Fu);
  }
  if (c < min || c > 0x10FFFFu || (c >= 0xD800u && c <= 0xDFFFu)) return 0;
  *cp = c;
  return n;
}

static uint32_t map_cp(uint32_t cp) {
  if (cp >= 0xFF01u && cp <= 0xFF5Eu) {
    cp -= 0xFEE0u;
  } else if (cp == 0x3000u) {
    cp = 0x20u;
  } else if (cp >= 0xFF61u && cp <= 0xFF9Fu) {
    cp = GFEAT_HALFWIDTH_KANA[cp - 0xFF61u];
  }
  if (cp >= 0x41u && cp <= 0x5Au) cp += 0x20u;
  return cp;
}

static uint32_t compose(uint32_t base, uint32_t mark) {
  if (mark != 0x3099u && mark != 0x309Au) return 0;
  for (int i = 0; i < GFEAT_N_COMPOSE; ++i)
    if (GFEAT_COMPOSE[i][0] == base && GFEAT_COMPOSE[i][1] == mark) return GFEAT_COMPOSE[i][2];
  return 0;
}

static int is_ascii_space(uint32_t cp) { return cp == 0x20u || (cp >= 0x09u && cp <= 0x0Du); }

int gfeat_normalize(const uint8_t *utf8, size_t len, uint32_t *out, size_t *n_out) {
  size_t n = 0, pos = 0;
  while (pos < len) {
    uint32_t cp;
    size_t used = decode_utf8(utf8 + pos, len - pos, &cp);
    if (used == 0) return GFEAT_ERR_UTF8;
    pos += used;
    cp = map_cp(cp);
    if (n > 0) {
      uint32_t c = compose(out[n - 1], cp);
      if (c) {
        out[n - 1] = c;
        continue;
      }
    }
    if (is_ascii_space(cp)) {
      if (n == 0 || out[n - 1] == 0x20u) continue;
    }
    /* 末尾の空白1つは後で除くので、一時的に上限+1まで許す */
    if (n >= GFEAT_MAX_CODEPOINTS + 1) return GFEAT_ERR_TOO_LONG;
    out[n++] = is_ascii_space(cp) ? 0x20u : cp;
  }
  if (n > 0 && out[n - 1] == 0x20u) --n;
  if (n > GFEAT_MAX_CODEPOINTS) return GFEAT_ERR_TOO_LONG;
  *n_out = n;
  return GFEAT_OK;
}

static size_t encode_utf8(uint32_t cp, uint8_t *o) {
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

int gfeat_extract(const uint8_t *utf8, size_t len, const gfeat_config *cfg, uint8_t *bits) {
  if (cfg == NULL || cfg->width == 0 || cfg->num_hashes == 0 || cfg->ngram_mask == 0 ||
      (cfg->ngram_mask >> GFEAT_MAX_ORDER) != 0)
    return GFEAT_ERR_CONFIG;
  /* seq = BOS + 正規化テキスト + EOS(正規化は seq+1 に書く) */
  uint32_t seq[GFEAT_MAX_CODEPOINTS + 3];
  size_t n = 0;
  int rc = gfeat_normalize(utf8, len, seq + 1, &n);
  if (rc != GFEAT_OK) return rc;
  seq[0] = BOS;
  seq[n + 1] = EOS;
  size_t m = n + 2;

  memset(bits, 0, cfg->width);
  uint8_t buf[GFEAT_MAX_ORDER * 4];
  for (uint32_t order = 1; order <= GFEAT_MAX_ORDER; ++order) {
    if (!(cfg->ngram_mask & (1u << (order - 1)))) continue;
    if (m < order) continue;
    for (size_t i = 0; i + order <= m; ++i) {
      size_t k = 0;
      for (uint32_t j = 0; j < order; ++j) k += encode_utf8(seq[i + j], buf + k);
      uint32_t base = gfeat_fnv1a32(buf, k);
      for (uint32_t j = 0; j < cfg->num_hashes; ++j)
        bits[gfeat_fmix32(base ^ (j * GOLDEN)) % cfg->width] = 1;
    }
  }
  return GFEAT_OK;
}

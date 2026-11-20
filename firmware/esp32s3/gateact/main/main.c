/*
 * GateAct-JP firmware: UTF-8 日本語テキスト → gateact-feat-v1 特徴量 → Griffith 論理回路 → Action。
 *
 * 起動すると次の順に1行 JSON を出力する(すべて先頭に種別のタグが付く):
 *   INFO     {...}  CPU 周波数・flash・heap・PSRAM・モデルの情報
 *   SELFTEST {...}  同梱テキストで C の特徴ビットとスコアが Python(IR simulate)と一致するか
 *   BENCH    {...}  BENCH_ITERS 回の前処理/推論/全体の時間(P50/P95/P99、µs)
 *   MEM      {...}  計測後の heap
 *   READY
 * 以後、1行(UTF-8、改行まで)を受け取るたびに RESULT {...} を返す。
 *
 * -DHOST_BUILD でホスト(cc)でもビルドでき、stdin の各行に同じ RESULT を返す。
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gateact_features.h"
#include "gateact_model_config.h"
#include "text_samples.h"

/* GA_MODEL_KIND: 1 = Griffith LUT 回路、2 = int8 bit-linear、3 = int8 n-gram(gateact_model_config.h) */
#if GA_MODEL_KIND == 1
#include "model.h"
#define NUM_OUT MDL_OUTPUT_WIDTH
#define MODEL_STATIC_BYTES MDL_CTX_BYTES
#else
#include "baseline_model.h"
#include "gateact_baselines.h"
#define NUM_OUT GB_CLASSES
#define MODEL_STATIC_BYTES 0
#endif

#ifndef BENCH_ITERS
#define BENCH_ITERS 2000
#endif
#define LINE_MAX_BYTES 1024

#if GA_MODEL_KIND == 1 && (MDL_INPUT_WIDTH != GA_FEAT_WIDTH || MDL_OUTPUT_WIDTH != GA_NUM_LABELS)
#error "model.h と gateact_model_config.h の幅が一致しない"
#endif
#if GA_MODEL_KIND == 2 && (GB_WIDTH != GA_FEAT_WIDTH || GB_CLASSES != GA_NUM_LABELS)
#error "baseline_model.h と gateact_model_config.h の幅が一致しない"
#endif

/* ---- プラットフォーム ---------------------------------------------------------------- */
#ifdef HOST_BUILD
#include <time.h>
static uint64_t ticks(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static double ticks_per_us(void) { return 1000.0; }
#define PLATFORM "host"
#else
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_chip_info.h"
#include "esp_clk_tree.h"
#include "esp_cpu.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_psram.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
static uint32_t g_cpu_hz;
/* esp_cpu_get_cycle_count は 32bit(240MHz で約 17.9 秒で一周)。区間計測にだけ使う */
static uint64_t ticks(void) { return (uint64_t)esp_cpu_get_cycle_count(); }
static double ticks_per_us(void) { return (double)g_cpu_hz / 1e6; }
#define PLATFORM "esp32s3"
#endif

#if GA_MODEL_KIND == 1
static uint32_t g_work[(MDL_CTX_BYTES + 3) / 4 + 1];
static mdl_ctx g_ctx;
#elif GA_MODEL_KIND == 3
static uint32_t g_ids[3 * (GFEAT_MAX_CODEPOINTS + 2)];
static int32_t g_hsum[GB_DIM];
#endif
#if GA_MODEL_KIND != 3
static uint8_t g_bits[GA_FEAT_WIDTH];
static const gfeat_config g_feat = {GA_FEAT_WIDTH, GA_FEAT_NGRAM_MASK, GA_FEAT_NUM_HASHES};
#endif
static int32_t g_scores[NUM_OUT];

static int argmax(const int32_t *s, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (s[i] > s[best]) best = i;
  return best;
}

/* 1件の推論。区間ごとの tick を返す(rc != 0 なら特徴量のエラー) */
typedef struct {
  uint32_t feat, model, total;
} timing_t;

static int infer(const uint8_t *text, size_t len, int *pred, timing_t *t) {
  uint64_t t0 = ticks();
#if GA_MODEL_KIND == 3
  size_t n_ids = 0;
  int rc = gb_ngram_ids(text, len, GB_BUCKETS, g_ids, &n_ids);
#else
  int rc = gfeat_extract(text, len, &g_feat, g_bits);
#endif
  uint64_t t1 = ticks();
  if (rc != GFEAT_OK) return rc;
#if GA_MODEL_KIND == 1
  mdl_infer_ctx(&g_ctx, g_bits, g_scores);
#elif GA_MODEL_KIND == 2
  gb_bitlinear(g_bits, GB_WIDTH, GB_CLASSES, GB_WEIGHT, GB_BIAS, g_scores);
#else
  gb_ngram_scores(g_ids, n_ids, GB_DIM, GB_CLASSES, GB_EMBEDDING, GB_WEIGHT, GB_BIAS, g_hsum,
                  g_scores);
#endif
  *pred = argmax(g_scores, NUM_OUT);
  uint64_t t2 = ticks();
  t->feat = (uint32_t)(t1 - t0);
  t->model = (uint32_t)(t2 - t1);
  t->total = (uint32_t)(t2 - t0);
  return 0;
}

/* ---- 出力 ------------------------------------------------------------------------- */
static void print_info(void) {
#ifdef HOST_BUILD
  printf("INFO {\"platform\":\"host\"");
#else
  esp_chip_info_t chip;
  esp_chip_info(&chip);
  uint32_t flash = 0;
  esp_flash_get_size(NULL, &flash);
  size_t psram = 0;
#if CONFIG_SPIRAM
  psram = esp_psram_is_initialized() ? esp_psram_get_size() : 0;
#endif
  printf("INFO {\"platform\":\"esp32s3\",\"idf\":\"%s\",\"cores\":%d,\"revision\":%d,"
         "\"cpu_hz\":%" PRIu32 ",\"flash_bytes\":%" PRIu32 ",\"psram_bytes\":%u,"
         "\"free_heap\":%u,\"free_internal\":%u,\"min_free_heap\":%u",
         esp_get_idf_version(), chip.cores, chip.revision, g_cpu_hz, flash, (unsigned)psram,
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
         (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT));
#endif
  printf(",\"model\":\"%s\",\"kind\":%d,\"gates\":%d,\"feature_spec\":\"%s\",\"width\":%u,"
         "\"ngram_mask\":%u,\"num_hashes\":%u,\"classes\":%d,\"ctx_bytes\":%d,"
         "\"samples\":%d,\"bench_iters\":%d}\n",
         GA_MODEL_NAME, GA_MODEL_KIND, GA_MODEL_GATES, GA_FEATURE_SPEC, (unsigned)GA_FEAT_WIDTH,
         (unsigned)GA_FEAT_NGRAM_MASK, (unsigned)GA_FEAT_NUM_HASHES, GA_NUM_LABELS,
         MODEL_STATIC_BYTES, TS_COUNT, BENCH_ITERS);
}

static int selftest(void) {
  int feat_mismatch = 0, score_mismatch = 0, correct = 0, labeled = 0, errors = 0;
  uint32_t hash = 0x811C9DC5u; /* 全スコアの FNV-1a(実機とホストの比較用) */
  for (int i = 0; i < TS_COUNT; ++i) {
    int pred = -1;
    timing_t t;
    if (infer((const uint8_t *)TS_TEXT[i], TS_LEN[i], &pred, &t) != 0) {
      ++errors;
      continue;
    }
#if GA_MODEL_KIND != 3
    for (int b = 0; b < TS_PACKED_BYTES; ++b) {
      uint8_t packed = 0;
      for (int k = 0; k < 8 && b * 8 + k < (int)GA_FEAT_WIDTH; ++k)
        packed |= (uint8_t)(g_bits[b * 8 + k] << k);
      if (packed != TS_BITS[i][b]) {
        ++feat_mismatch;
        break;
      }
    }
#endif
    for (int k = 0; k < NUM_OUT; ++k) {
      if (g_scores[k] != TS_SCORES[i][k]) {
        ++score_mismatch;
        break;
      }
    }
    for (int k = 0; k < NUM_OUT; ++k) {
      uint32_t u = (uint32_t)g_scores[k];
      for (int s = 0; s < 32; s += 8) hash = (hash ^ ((u >> s) & 0xFFu)) * 0x01000193u;
    }
    if (TS_LABEL[i] >= 0) {
      ++labeled;
      correct += (pred == TS_LABEL[i]);
    }
  }
  printf("SELFTEST {\"n\":%d,\"errors\":%d,\"feature_mismatch\":%d,\"score_mismatch\":%d,"
         "\"labeled\":%d,\"correct\":%d,\"score_hash\":\"%08" PRIx32 "\"}\n",
         TS_COUNT, errors, feat_mismatch, score_mismatch, labeled, correct, hash);
  return errors + feat_mismatch + score_mismatch;
}

static int cmp_u32(const void *a, const void *b) {
  uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  return (x > y) - (x < y);
}

static uint32_t g_t_feat[BENCH_ITERS], g_t_model[BENCH_ITERS], g_t_total[BENCH_ITERS];

static void print_stats(const char *name, uint32_t *v, int n, int last) {
  qsort(v, (size_t)n, sizeof(uint32_t), cmp_u32);
  double sum = 0;
  for (int i = 0; i < n; ++i) sum += v[i];
  double k = ticks_per_us();
#define PCT(p) ((double)v[(int)((double)(n - 1) * (p) + 0.5)] / k)
  printf("\"%s\":{\"mean\":%.3f,\"min\":%.3f,\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f,\"max\":%.3f}%s",
         name, sum / n / k, (double)v[0] / k, PCT(0.50), PCT(0.95), PCT(0.99),
         (double)v[n - 1] / k, last ? "" : ",");
#undef PCT
}

static void bench(void) {
  if (TS_COUNT == 0) return;
  int pred;
  timing_t t;
  for (int i = 0; i < TS_COUNT; ++i) infer((const uint8_t *)TS_TEXT[i], TS_LEN[i], &pred, &t);
  for (int it = 0; it < BENCH_ITERS; ++it) {
    int i = it % TS_COUNT;
    infer((const uint8_t *)TS_TEXT[i], TS_LEN[i], &pred, &t);
    g_t_feat[it] = t.feat;
    g_t_model[it] = t.model;
    g_t_total[it] = t.total;
  }
  printf("BENCH {\"platform\":\"%s\",\"iters\":%d,\"unit\":\"us\",", PLATFORM, BENCH_ITERS);
  print_stats("features", g_t_feat, BENCH_ITERS, 0);
  print_stats("model", g_t_model, BENCH_ITERS, 0);
  print_stats("total", g_t_total, BENCH_ITERS, 1);
  printf("}\n");
}

static void print_mem(void) {
#ifndef HOST_BUILD
  printf("MEM {\"free_heap\":%u,\"free_internal\":%u,\"min_free_heap\":%u,"
         "\"largest_free_block\":%u,\"static_ctx_bytes\":%u}\n",
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
         (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT),
         (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT), (unsigned)MODEL_STATIC_BYTES);
#endif
}

static void handle_line(char *line) {
  size_t len = strlen(line);
  while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
  int pred = -1;
  timing_t t;
  int rc = infer((const uint8_t *)line, len, &pred, &t);
  if (rc != 0) {
    printf("RESULT {\"ok\":false,\"error\":%d,\"bytes\":%u}\n", rc, (unsigned)len);
    return;
  }
  double k = ticks_per_us();
  printf("RESULT {\"ok\":true,\"id\":%d,\"action\":\"%s\",\"bytes\":%u,\"scores\":[", pred,
         GA_LABELS[pred], (unsigned)len);
  for (int i = 0; i < NUM_OUT; ++i) printf(i ? ",%ld" : "%ld", (long)g_scores[i]);
  printf("],\"us\":{\"features\":%.3f,\"model\":%.3f,\"total\":%.3f}}\n", t.feat / k,
         t.model / k, t.total / k);
  fflush(stdout);
}

static char g_line[LINE_MAX_BYTES];

static int setup(void) {
#if GA_MODEL_KIND == 1
  if (mdl_ctx_init(&g_ctx, g_work, sizeof(g_work)) != 0) {
    printf("ERROR {\"msg\":\"mdl_ctx_init failed\"}\n");
    return 1;
  }
#endif
  print_info();
  int bad = selftest();
  bench();
  print_mem();
  printf("READY\n");
  fflush(stdout);
  return bad;
}

#ifdef HOST_BUILD
int main(void) {
  int bad = setup();
  while (fgets(g_line, sizeof g_line, stdin)) handle_line(g_line);
  return bad != 0;
}
#else
static void console_init(void) {
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
  usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
  usb_serial_jtag_driver_install(&cfg);
  usb_serial_jtag_vfs_use_driver();
  usb_serial_jtag_vfs_set_rx_line_endings(ESP_LINE_ENDINGS_LF);
#elif CONFIG_ESP_CONSOLE_UART
  uart_driver_install(CONFIG_ESP_CONSOLE_UART_NUM, 2048, 0, 0, NULL, 0);
  uart_vfs_dev_use_driver(CONFIG_ESP_CONSOLE_UART_NUM);
  uart_vfs_dev_port_set_rx_line_endings(CONFIG_ESP_CONSOLE_UART_NUM, ESP_LINE_ENDINGS_LF);
#endif
  setvbuf(stdin, NULL, _IONBF, 0);
}

void app_main(void) {
  esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU, ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED, &g_cpu_hz);
  console_init();
  vTaskDelay(pdMS_TO_TICKS(1500)); /* ホスト側がポートを開くのを待つ */
  setup();
  for (;;) {
    if (fgets(g_line, sizeof g_line, stdin)) {
      handle_line(g_line);
    } else {
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }
}
#endif

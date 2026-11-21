/*
 * GateAct-JP M0: Griffith XOR on ESP32-S3.
 *
 * samples.h の全サンプルを推論し、IR simulate の期待スコアと整数照合して結果を出力する。
 * 同じソースを -DHOST_BUILD でホスト(cc)ビルドでき、実機とホストの出力を比較できる。
 *
 * 出力:
 *   SAMPLE {"i":0,"pred":0,"label":0,"scores":[..],"ok":1}
 *   RESULT {"n":4,"correct":4,"score_mismatch":0,"cycles_per_infer":...}
 *   DONE
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/* glibc は -std=c99 だと clock_gettime を宣言しない(macOS は _POSIX_C_SOURCE を付けると
 * clock_gettime_nsec_np が消えるので付けない) */
#if defined(HOST_BUILD) && !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdint.h>
#include <stdio.h>

#include "model.h"
#include "samples.h"

#ifdef HOST_BUILD
#include <time.h>
static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#else
#include "esp_cpu.h"
#include "esp_clk_tree.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
static uint32_t cpu_hz(void) {
  uint32_t hz = 0;
  esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU, ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED, &hz);
  return hz;
}
#endif

static uint32_t g_work[(MDL_CTX_BYTES + 3) / 4 + 1];
static uint8_t g_bits[MDL_INPUT_WIDTH];
static int32_t g_scores[MDL_OUTPUT_WIDTH];

static void unpack(const uint8_t *packed) {
  for (int i = 0; i < MDL_INPUT_WIDTH; ++i) g_bits[i] = (packed[i >> 3] >> (i & 7)) & 1u;
}

static int argmax(const int32_t *s, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (s[i] > s[best]) best = i;
  return best;
}

static int run_all(void) {
  mdl_ctx ctx;
  if (mdl_ctx_init(&ctx, g_work, sizeof(g_work)) != 0) {
    printf("ERROR {\"msg\":\"mdl_ctx_init failed\"}\n");
    return 1;
  }
  int correct = 0, mismatch = 0;
  for (int i = 0; i < SAMPLE_COUNT; ++i) {
    unpack(SAMPLE_BITS[i]);
    mdl_infer_ctx(&ctx, g_bits, g_scores);
    int pred = argmax(g_scores, MDL_OUTPUT_WIDTH);
    int ok = 1;
#if SAMPLE_HAS_EXPECTED
    for (int k = 0; k < MDL_OUTPUT_WIDTH; ++k)
      if (g_scores[k] != SAMPLE_EXPECTED[i][k]) ok = 0;
#endif
    mismatch += !ok;
    correct += (pred == SAMPLE_LABEL[i]);
    printf("SAMPLE {\"i\":%d,\"pred\":%d,\"label\":%d,\"scores\":[", i, pred, SAMPLE_LABEL[i]);
    for (int k = 0; k < MDL_OUTPUT_WIDTH; ++k) printf(k ? ",%ld" : "%ld", (long)g_scores[k]);
    printf("],\"ok\":%d}\n", ok);
  }

  /* 計測: 全サンプルを REPS 周推論した平均 */
  enum { REPS = 1000 };
#ifdef HOST_BUILD
  uint64_t t0 = now_ns();
#else
  uint32_t c0 = esp_cpu_get_cycle_count();
#endif
  for (int r = 0; r < REPS; ++r)
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
      unpack(SAMPLE_BITS[i]);
      mdl_infer_ctx(&ctx, g_bits, g_scores);
    }
#ifdef HOST_BUILD
  double ns = (double)(now_ns() - t0) / (REPS * SAMPLE_COUNT);
  printf("RESULT {\"platform\":\"host\",\"n\":%d,\"correct\":%d,\"score_mismatch\":%d,"
         "\"ns_per_infer\":%.1f}\n", SAMPLE_COUNT, correct, mismatch, ns);
#else
  uint32_t cycles = esp_cpu_get_cycle_count() - c0;
  double per = (double)cycles / (REPS * SAMPLE_COUNT);
  uint32_t hz = cpu_hz();
  printf("RESULT {\"platform\":\"esp32s3\",\"n\":%d,\"correct\":%d,\"score_mismatch\":%d,"
         "\"cpu_hz\":%lu,\"cycles_per_infer\":%.1f,\"ns_per_infer\":%.1f}\n",
         SAMPLE_COUNT, correct, mismatch, (unsigned long)hz, per, per * 1e9 / (double)hz);
#endif
  printf("DONE\n");
  return mismatch != 0;
}

#ifdef HOST_BUILD
int main(void) { return run_all(); }
#else
void app_main(void) {
  vTaskDelay(pdMS_TO_TICKS(500));
  run_all();
  for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
#endif

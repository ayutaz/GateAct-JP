# main/model/ は `uv run gateact export <ckpt> --out firmware/esp32s3/gateact/main` の生成物(Git 管理外)。
if(NOT DEFINED MODEL_DIR)
  set(MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/model)
endif()
if(NOT EXISTS ${MODEL_DIR}/model.c OR NOT EXISTS ${MODEL_DIR}/text_samples.h)
  message(FATAL_ERROR "${MODEL_DIR} に生成モデルが無い。`uv run gateact export ...` か "
                      "`uv run python scripts/gen_firmware_model.py gateact-smoke` を先に実行する")
endif()
set(COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../common)
idf_component_register(
  SRCS "main.c" "${COMMON_DIR}/gateact_features.c" "${MODEL_DIR}/model.c" "${MODEL_DIR}/mdl_kernels.c"
  INCLUDE_DIRS "." "${COMMON_DIR}" "${MODEL_DIR}"
  PRIV_REQUIRES esp_driver_uart esp_driver_usb_serial_jtag esp_timer spi_flash esp_psram esp_hw_support
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE MDL_NO_DEFAULT_CTX)

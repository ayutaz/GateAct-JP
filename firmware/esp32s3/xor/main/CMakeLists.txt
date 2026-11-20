# main/model/ は `uv run gateact export-c` / scripts/gen_firmware_model.py の生成物(Git 管理外)。
# -DMODEL_DIR=<dir> で別ディレクトリを指定できる。
if(NOT DEFINED MODEL_DIR)
  set(MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/model)
endif()
if(NOT EXISTS ${MODEL_DIR}/model.c)
  message(FATAL_ERROR "${MODEL_DIR}/model.c が無い。先に `uv run python scripts/gen_firmware_model.py xor` を実行する")
endif()
idf_component_register(
  SRCS "main.c" "${MODEL_DIR}/model.c" "${MODEL_DIR}/mdl_kernels.c"
  INCLUDE_DIRS "." "${MODEL_DIR}"
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE MDL_NO_DEFAULT_CTX)

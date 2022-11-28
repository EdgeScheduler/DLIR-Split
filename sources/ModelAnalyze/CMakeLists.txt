include_directories("${DLIR_SPLIT_SOURCE_DIR}/include")
aux_source_directory(. MODEL_ANALYZE_SRCS)

add_library(model_analyze ${MODEL_ANALYZE_SRCS})
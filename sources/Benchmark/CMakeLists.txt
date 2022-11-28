include_directories("${DLIR_SPLIT_SOURCE_DIR}/include")
aux_source_directory(. BENCHMARK_SRCS)

add_library(benchmark ${BENCHMARK_SRCS})
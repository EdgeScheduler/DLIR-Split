include_directories("${DLIR_SPLIT_SOURCE_DIR}/include" "${DLIR_SPLIT_SOURCE_DIR}/third_party/include")
aux_source_directory(. UTILS_SRCS)

add_library(utils ${UTILS_SRCS})
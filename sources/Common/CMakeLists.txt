include_directories("${DLIR_SPLIT_SOURCE_DIR}/include")
aux_source_directory(. COMMON_SRCS)

add_library(common ${COMMON_SRCS})
include_directories("${DLIR_SPLIT_SOURCE_DIR}/include")
aux_source_directory(. TENSOR_SRCS)

add_library(tensor ${TENSOR_SRCS})
cmake_minimum_required(VERSION 3.0.0)

# set(COMPILE_MODE "DLIR_MODE")     # set target version

set(PROJECT_NAME "DLIR_SPLIT")
project(${PROJECT_NAME})
set(target "DLIR-SPLIT")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set binary path
if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    SET(EXECUTABLE_OUTPUT_PATH "${DLIR_SPLIT_SOURCE_DIR}/bin/debug")
elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    SET(EXECUTABLE_OUTPUT_PATH "${DLIR_SPLIT_SOURCE_DIR}/bin/release")
endif()

add_definitions(-DONNX_ML=1)
add_definitions(-DONNX_NAMESPACE=onnx)

include_directories("include" "third_party/include" "libs")
link_directories("third_party/lib")

add_subdirectory("sources/Common" "Libs/Common")
add_subdirectory("sources/Benchmark" "Libs/Benchmark")
add_subdirectory("sources/ModelAnalyze" "Libs/ModelAnalyze")
add_subdirectory("sources/Tensor" "Libs/Tensor")
add_subdirectory("sources/Utils" "Libs/Utils")

aux_source_directory("entrance/Split" DLIR_SPLIT_SRCS)

add_executable(${target} ${DLIR_SPLIT_SRCS})
target_link_libraries(${target} model_analyze utils benchmark tensor common onnx_shared pthread protobuf stdc++fs onnxruntime)



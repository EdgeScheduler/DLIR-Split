#ifndef __EVALUATE_MODELS_H__
#define __EVALUATE_MODELS_H__

#include <stdio.h>
#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Utils/helper.h"
namespace evam
{
    /// @brief
    /// @param model_name
    /// @param driver
    /// @param test_count
    /// @param default_batchsize
    /// @return
    float TimeEvaluateChildModels_impl(std::string model_name, int child_num = -1, std::string GPU_tag = "default", bool enable_bench_cache=false, int test_count = 3, int default_batchsize = 15);

    float TimeEvaluateChildModels_impl(std::string model_name, std::filesystem::path model_path, std::string key, std::string GPU_tag = "default", bool enable_bench_cache=false, int test_count = 3, int default_batchsize = 15);

    /// @brief
    /// @param model_name
    /// @param child_num
    /// @param test_count
    /// @param default_batchsize
    /// @return
    nlohmann::json TimeEvaluateChildModels(std::string model_name, std::string GPU_tag = "default", bool enable_bench_cache=false, int test_count = 5, int default_batchsize = 15);

    /// @brief
    /// @param model_name
    /// @param file_name
    /// @return
    float EvalStdCurrentModelSplit(std::string model_name, std::string file_name = "", std::string GPU_tag = "default", bool enable_bench_cache=false);
}

#endif // __EVALUATE_MODELS_H__
#ifndef __EVALUATE_MODELS_H__
#define __EVALUATE_MODELS_H__

#include <stdio.h>
#include<filesystem>
#include <string>
#include <vector>
#include<nlohmann/json.hpp>
namespace evam
{
    /// @brief 
    /// @param model_name 
    /// @param driver 
    /// @param test_count 
    /// @param default_batchsize 
    /// @return 
    float TimeEvaluateChildModels_impl(std::string model_name, int child_num=-1, int test_count=5, int default_batchsize=15);

    /// @brief 
    /// @param model_name 
    /// @param child_num 
    /// @param test_count 
    /// @param default_batchsize 
    /// @return 
    nlohmann::json TimeEvaluateChildModels(std::string model_name, int child_num=-1, int test_count=5, int default_batchsize=15);

    /// @brief 
    /// @param model_name 
    /// @param onnx_path 
    /// @param file_name 
    void EvalCurrentModelSplit(std::string model_name, std::string file_name="");
}

#endif // __EVALUATE_MODELS_H__
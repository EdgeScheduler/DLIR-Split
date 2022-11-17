#ifndef __EVALUATE_MODELS_H__
#define __EVALUATE_MODELS_H__

#include <nlohmann/json.hpp>

namespace evam
{
    /// @brief 
    /// @param model_name 
    /// @param driver 
    /// @param test_count 
    /// @param default_batchsize 
    /// @return 
    nlohmann::json TimeEvaluateChildModels(std::string &model_name, std::vector<std::string> &driver, int test_count=20, int default_batchsize=15);
}

#endif // __EVALUATE_MODELS_H__
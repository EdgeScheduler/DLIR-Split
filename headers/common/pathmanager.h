#ifndef PATHManager_H
#define PATHManager_H

#include<iostream>
#include<filesystem>

/// @brief manager paths related to onnx-models
class OnnxPathManager
{
public:
    /// @brief get onnx save root path
    /// @return "$ProjectRoot/$OnnxsRoot/"
    static std::filesystem::path GetOnnxRootFold();
    /// @brief get raw onnx-model save path
    /// @param model_name name of model
    /// @return "$ProjectRoot/$OnnxsRoot/$model_name/$model_name.onnx"
    static std::filesystem::path GetModelSavePath(std::string model_name);

    /// @brief get raw onnx-model params describe-file save path
    /// @param model_name name of model
    /// @return "$ProjectRoot/$OnnxsRoot/$model_name/$model_name.json"
    static std::filesystem::path GetModelParamsSavePath(std::string model_name);

    /// @brief get childs onnx-model save path
    /// @param model_name name of model
    /// @param idx idx of child-model, start from 0. idx<0 means raw-model
    /// @return "$ProjectRoot/$OnnxsRoot/$model_name/childs/$idx/$model_name-$idx.onnx"
    static std::filesystem::path GetChildModelSavePath(std::string model_name, int idx=-1);
    /// @brief get childs onnx-model params describe-file  save path
    /// @param model_name name of model
    /// @param idx idx of child-model, start from 0. idx<0 means raw-model
    /// @return "$ProjectRoot/$OnnxsRoot/$model_name/childs/$idx/$model_name-$idx-params.json"
    static std::filesystem::path GetChildModelParamsSavePath(std::string model_name, int idx=-1);
    /// @brief get sum params describe-file save path
    /// @param model_name name of model
    /// @return "$ProjectRoot/$OnnxsRoot/$model_name/childs/$model_name-params.json"
    static std::filesystem::path GetChildModelSumParamsSavePath(std::string model_name);
    /// @brief get cache describe-file save path
    /// @param model_name name of model
    /// @return "$ProjectRoot/$OnnxsRoot/$model_name/childs/cache.json"
    static std::filesystem::path GetChildModelSumCacheSavePath(std::string model_name);
private:
    /// @brief onnx fold relative-path
    static std::filesystem::path onnxRootFold;
};

class BenchmarkPathManager
{
private:
    static std::filesystem::path benchmarkRootFold;
};

/// @brief manager Project Root-Path
class RootPathManager
{
public:
    /// @brief get executor-file path, get project root path instead while developing.
    /// @return "$project"
    static std::filesystem::path GetRunRootFold();
private:
    static std::filesystem::path projectRootFold;
    
};




#endif // !PATHManager_H
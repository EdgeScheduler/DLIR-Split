#ifndef MODELANALYZER_H
#define MODELANALYZER_H

#include <iostream>
#include <filesystem>
#include <cstdio>
#include <nlohmann/json.hpp>
#include "../Common/PathManager.h"
#include "../../library/onnx.proto3.pb.h"

class GraphNode
{

};


/// @brief
class ModelAnalyzer
{
public:
    /// @brief 
    /// @param model_name 
    /// @param onnx_path 
    ModelAnalyzer(std::string model_name, std::filesystem::path onnx_path);

    /// @brief 
    /// @param onnx_path 
    /// @return 
    onnx::ModelProto onnx_load(); 

    /// @brief 
    /// @return 
    bool Init();

//     /// @brief 
//     /// @param enable 
//     void SetEnableCache(bool enable = true);

//     /// @brief 
//     /// @param node 
//     /// @return 
//     bool EnableStart(GraphNode node);

//     /// @brief 
//     /// @return 
//     nlohmann::json LoadCache();

//     /// @brief 
//     void RuntimeAnalyze();

//     /// @brief 
//     /// @param raw_onnx_path 
//     /// @param new_onnx_path 
//     /// @param new_onnx_param_path 
//     /// @param start_node 
//     /// @param end_node 
//     /// @param print_error 
//     /// @return 
//     nlohmann::json ExtractModelByNode(std::filesystem::path raw_onnx_path, std::filesystem::path new_onnx_path, std::filesystem::path new_onnx_param_path,
//                                       GraphNode start_node, GraphNode end_node, bool print_error = true);

//     /// @brief 
//     void RecordDependency();

//     /// @brief 
//     /// @param childs 
//     /// @return 
//     nlohmann::json SplitAndStoreChilds(std::vector<GraphNode> childs);

//     /// @brief 
//     /// @param onnx_path 
//     /// @param params_path 
//     /// @param default_batch 
//     /// @return 
//     nlohmann::json CreateParamsInfo(std::filesystem::path onnx_path, std::filesystem::path params_path, int default_batch = 15);

//     /// @brief 
//     /// @return 
//     std::vector<GraphNode> GetConvergeNodes();

//     /// @brief 
//     /// @return 
//     std::vector<GraphNode> GetAllNodes();

private:

    /// @brief 
    std::string modelName;

    /// @brief 
    std::string onnxPath;

    /// @brief 
    OnnxPathManager manager;

    /// @brief 
    std::vector<GraphNode> nodes;

    /// @brief 
    GraphNode start_node;

    /// @brief 
    std::vector<std::string> params;

    /// @brief 
    bool use_cache;
};

#endif // !MODELANALYZER_H
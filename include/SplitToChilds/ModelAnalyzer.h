// #ifndef MODELANALYZER_H
// #define MODELANALYZER_H

// #include <iostream>
// #include <filesystem>
// #include <cstdio>
// #include "GraphNode.h"
// #include <nlohmann/json.hpp>


// /// @brief 
// class ModelAnalyzer
// {
//     public:
//         ModelAnalyzer(std::string model_name, std::filesystem::path onnx_path);
//         bool Init();
//         void SetEnableCache(bool enable = true);
//         bool EnableStart(GraphNode node);
//         nlohmann::json LoadCache();
//         void RuntimeAnalyze();
//         nlohmann::json ExtractModelByNode(std::filesystem::path raw_onnx_path, std::filesystem::path new_onnx_path, std::filesystem::path new_onnx_param_path,
//         GraphNode start_node, GraphNode end_node, bool print_error = true);
//         void RecordDependency();
//         nlohmann::json SplitAndStoreChilds(std::vector<GraphNode> childs);
//         nlohmann::json CreateParamsInfo(std::filesystem::path onnx_path, std::filesystem::path params_path, int default_batch = 15);
//         std::vector<GraphNode> GetConvergeNodes();
//         std::vector<GraphNode> GetAllNodes();

//     private:
//         std::vector<GraphNode> nodes;
//         GraphNode start_node;
//         std::vector<std::string> params;
//         bool use_cache;
// };

// #endif // !MODELANALYZER_H
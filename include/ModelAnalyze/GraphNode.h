#ifndef __GRAPHNODE_H__
#define __GRAPHNODE_H__

#include <iostream>
#include <filesystem>
#include <cstdio>
#include <set>
#include <vector>
#include <iterator>
#include <nlohmann/json.hpp>
#include "Common/PathManager.h"
#include "onnx/shape_inference/implementation.h"

class GraphNode
{
public:
    /// @brief
    GraphNode();

    /// @brief
    /// @param node
    /// @param TotalParams
    /// @param idx
    GraphNode(onnx::NodeProto node, std::set<std::string> TotalParams = std::set<std::string>(), int index = -1);

    /// @brief
    /// @param node
    GraphNode(const GraphNode &node);

    /// @brief
    /// @param node
    /// @return
    bool operator==(GraphNode &node);

    /// @brief return true if this node is converge-node for total model-graph
    /// @return
    bool IsConvergeNode();

// private:
    std::string name;
    std::string type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> dependencies_inputs;
    std::vector<std::string> dependencies_outputs;
    std::set<std::string> params;
    nlohmann::json input_info;
    int idx;
};

#endif // __GRAPHNODE_H__
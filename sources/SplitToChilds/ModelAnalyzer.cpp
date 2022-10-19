#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "../../include/SplitToChilds/ModelAnalyzer.h"
#include "../../include/Common/PathManager.h"
#include "../../library/onnx.proto3.pb.h"
#include "../../include/Utils/OnnxUtil.h"
#include "../../include/Common/JsonSerializer.h"

// some tool function

void print_dim(const ::onnx::TensorShapeProto_Dimension &dim)
{
    switch (dim.value_case())
    {
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
        std::cout << dim.dim_param();
        break;
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
        std::cout << dim.dim_value();
        break;
    default:
        assert(false && "should never happen");
    }
}

void print_io_info(const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &info)
{
    for (auto input_data : info)
    {
        auto shape = input_data.type().tensor_type().shape();
        std::cout << "  " << input_data.name() << ":";
        std::cout << "[";
        if (shape.dim_size() != 0)
        {
            int size = shape.dim_size();
            for (int i = 0; i < size - 1; ++i)
            {
                print_dim(shape.dim(i));
                std::cout << ", ";
            }
            print_dim(shape.dim(size - 1));
        }
        std::cout << "]\n";
    }
}

// GraphNode
GraphNode::GraphNode()
{
    this->name = "";
    this->type = "";
    this->inputs = std::vector<std::string>();
    this->outputs = std::vector<std::string>();
    this->dependencies_inputs = std::vector<std::string>();
    this->dependencies_outputs = std::vector<std::string>();
    this->params = std::set<std::string>();
    this->idx = 0;
}

GraphNode::GraphNode(onnx::NodeProto node, std::set<std::string> TotalParams, int index)
{
    this->name = node.name();
    this->type = node.op_type();
    this->inputs = std::vector<std::string>();
    this->outputs = std::vector<std::string>();
    this->dependencies_inputs = std::vector<std::string>();
    this->dependencies_outputs = std::vector<std::string>();
    this->params = std::set<std::string>();
    this->idx = index;

    int output_size = node.output_size();
    for (auto &output : node.output())
    {
        this->outputs.emplace_back(output);
    }

    for (auto &input_name : node.input())
    {
        if (std::find(TotalParams.begin(), TotalParams.end(), input_name) != TotalParams.end())
        {
            this->params.emplace(input_name);
        }
        else
        {
            this->inputs.emplace_back(input_name);
        }
    }
}

GraphNode::GraphNode(const GraphNode &node)
{
    this->name = node.name;
    this->type = node.type;
    this->inputs = node.inputs;
    this->outputs = node.outputs;
    this->dependencies_inputs = node.dependencies_inputs;
    this->dependencies_outputs = node.dependencies_outputs;
    this->params = node.params;
    this->idx = node.idx;
}

bool GraphNode::operator==(GraphNode &node)
{
    if (this->name == node.name && this->type == node.type && this->inputs == node.inputs && this->outputs == node.outputs &&
        this->dependencies_inputs == node.dependencies_inputs && this->dependencies_outputs == node.dependencies_outputs && this->params == node.params && this->idx == node.idx)
        return true;
    return false;
}

bool GraphNode::IsConvergeNode()
{
    return this->dependencies_inputs.size() < 2 ? true : false;
}

// ModelAnalyzer

ModelAnalyzer::ModelAnalyzer(std::string model_name, const std::filesystem::path &onnx_path)
{
    this->modelName = model_name;
    this->manager = OnnxPathManager();
    this->use_cache = true;
    this->start_node = GraphNode();

    if (onnx_path.empty())
    {
        this->onnxPath = manager.GetModelSavePath(this->modelName);
    }
    else
        this->onnxPath = onnx_path;

    if (!this->Init())
        return;
}

bool ModelAnalyzer::Init()
{

    onnx::ModelProto model;
    try
    {
        model = onnxUtil.load(onnxPath);
        auto graph = model.graph();

        for (auto &data : graph.initializer())
            this->params.emplace(data.name());

        int node_size = graph.node_size();
        for (int i = 0; i < node_size; i++)
            this->nodes.emplace_back(GraphNode(graph.node(i), params, i));
        this->start_node = this->nodes[0];

        RecordDependency();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: fail to init model-analyzer" << std::endl;
        std::cerr << e.what() << std::endl;
        return false;
    }
    return true;
}

void ModelAnalyzer::SetEnableCache(bool enable)
{
    this->use_cache = true;
}

bool ModelAnalyzer::EnableStart(GraphNode &node)
{
    if (node == this->nodes[0] || node.idx > this->start_node.idx)
        return true;
    return false;
}

const std::filesystem::path &ModelAnalyzer::GetModelPath() const
{
    return this->onnxPath;
}

nlohmann::json ModelAnalyzer::LoadCache()
{
    try
    {
        return JsonSerializer::LoadJson(OnnxPathManager::GetChildModelSumCacheSavePath(this->modelName));
    }
    catch(const std::exception& e)
    {
        std::cerr << "Warning" << e.what() << '\n';
        return nlohmann::json({});
    }
}

// nlohmann::json ModelAnalyzer::ExtractModelByNode(std::filesystem::path raw_onnx_path, std::filesystem::path new_onnx_path, std::filesystem::path new_onnx_param_path,
//                                             GraphNode start_node, GraphNode end_node, bool print_error)
// {
//     try
//     {
//         /* code */
//     }
//     catch(const std::exception& e)
//     {
//         std::cerr << e.what() << '\n';
//     }
    
// }

void ModelAnalyzer::RecordDependency()
{
    std::set<std::string> dependency;
    int node_size = nodes.size();
    for (int idx = node_size - 1; idx >= 0; idx--)
    {
        if (idx == node_size - 1)
            for (auto &out : nodes[idx].outputs)
                if (std::find(params.begin(), params.end(), out) == params.end())
                    this->nodes[idx].dependencies_outputs.emplace_back(out);

        for (auto &input_name : nodes[idx].inputs)
        {
            dependency.emplace(input_name);
        }

        for (auto &output_name : nodes[idx].outputs)
        {
            dependency.erase(output_name);
        }

        nodes[idx].dependencies_inputs.assign(dependency.begin(), dependency.end());

        if (idx > 0)
            nodes[idx - 1].dependencies_outputs = nodes[idx].dependencies_inputs;
    }
}

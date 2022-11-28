#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <iterator>
#include "openGA.hpp"
#include "onnx/shape_inference/implementation.h"
#include "Benchmark/evaluate_models.h"
#include "ModelAnalyze/ModelAnalyzer.h"
#include "Common/JsonSerializer.h"
#include "Common/PathManager.h"
#include "Common/PathManager.h"
#include "Utils/helper.h"
#include "Utils/UniformOptimizer.h"

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

    try
    {
        if (!this->Init())
            throw -1;
    }
    catch (int e)
    {
        std::cerr << "Error while initiating analyzer." << '\n';
    }
}

bool ModelAnalyzer::Init()
{

    onnx::ModelProto model;
    try
    {
        model = onnxUtil::load(onnxPath);
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
    catch (const std::exception &e)
    {
        std::cerr << "Warning" << e.what() << '\n';
        return nlohmann::json({});
    }
}

void ModelAnalyzer::ExtractModelByNodeWithWrite(nlohmann::json* value,std::filesystem::path raw_onnx_path, std::filesystem::path new_onnx_path, std::filesystem::path new_onnx_param_path,GraphNode* start_node, GraphNode* end_node, bool print_error)
{
    *value=ExtractModelByNode(raw_onnx_path,new_onnx_path,new_onnx_param_path,*start_node,*end_node,print_error);
    (*value)["from"] = start_node->idx;
    (*value)["to"] = end_node->idx;
}

nlohmann::json ModelAnalyzer::ExtractModelByNode(std::filesystem::path raw_onnx_path, std::filesystem::path new_onnx_path, std::filesystem::path new_onnx_param_path,GraphNode& start_node, GraphNode& end_node, bool print_error)
{

    // c++!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    onnxUtil::extract_model(raw_onnx_path, new_onnx_path, start_node.dependencies_inputs, end_node.dependencies_outputs);

    //
    //
    // python
    //
    //
    // Py_Initialize();
    // PyObject *pModule = PyImport_ImportModule("onnx");
    // PyObject *pFunc = PyObject_GetAttrString(pModule, "utils.extract_model");
    // PyObject *pArgs = PyTuple_New(4);
    // PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", raw_onnx_path.c_str()));
    // PyTuple_SetItem(pArgs, 1, Py_BuildValue("s", new_onnx_path.c_str()));

    // int input_size = start_node.dependencies_inputs.size();
    // int output_size = end_node.dependencies_outputs.size();

    // PyObject *inputs = PyList_New(input_size);
    // PyObject *outputs = PyList_New(output_size);
    // int i = 0;
    // std::vector<std::string>::iterator iList;
    // for (i = 0, iList = start_node.dependencies_inputs.begin(); iList != start_node.dependencies_inputs.end(); ++iList, ++i)
    // {
    //     PyList_SetItem(inputs, i, PyBytes_FromString((*iList).c_str()));
    // }

    // for (i = 0, iList = start_node.dependencies_inputs.begin(); iList != start_node.dependencies_inputs.end(); ++iList, ++i)
    // {
    //     PyList_SetItem(outputs, i, PyBytes_FromString((*iList).c_str()));
    // }
    // PyTuple_SetItem(pArgs, 2, inputs);
    // PyTuple_SetItem(pArgs, 3, outputs);
    // // PyObject* pReturn = PyEval_CallObject(pFunc, pArgs);
    // PyEval_CallObject(pFunc, pArgs);
    // Py_Finalize();

    return CreateParamsInfo(new_onnx_path, new_onnx_param_path);
}

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

nlohmann::json ModelAnalyzer::SplitAndStoreChilds(std::vector<GraphNode> input_childs)
{
    std::filesystem::remove_all(OnnxPathManager::GetOnnxRootFold() / this->getName() / "childs/");
    nlohmann::json total_param;
    std::vector<GraphNode> childs = std::vector<GraphNode>();
    for (auto &child : input_childs)
        if (EnableStart(child))
            childs.emplace_back(child);
    sort(childs.begin(), childs.end(), [](GraphNode x, GraphNode y)
         { return x.idx <= y.idx; });
    if (childs.size() < 1 || childs[0].idx != 0)
        childs.emplace(childs.begin(), nodes[0]);

    std::vector<GraphNode> childs_ = std::vector<GraphNode>();
    int tmp = -1;
    for (auto &child : childs)
    {
        if (child.idx > tmp)
        {
            childs_.emplace_back(child);
            tmp = child.idx;
        }
    }
    childs = childs_;

    nlohmann::json info = CreateParamsInfo(onnxPath, OnnxPathManager::GetModelParamsSavePath(modelName));
    info["from"] = nodes[0].idx;
    info["to"] = nodes.back().idx;
    total_param["-1"] = info;

    int childs_size = childs.size();
    std::vector<nlohmann::json> infos(childs_size);
    std::vector<std::shared_ptr<std::thread>> threads;
    for (int child_idx = 0; child_idx < childs_size; child_idx++)
    {
        int start_index=childs[child_idx].idx;
        int end_index=nodes.back().idx;
        if (child_idx + 1 < childs.size())
            end_index = childs[child_idx + 1].idx - 1;

        std::cout << modelName << "-" << child_idx << " ==|> " << nodes[start_index].name << " --> " << nodes[end_index].name << std::endl;

        std::filesystem::path child_onnx_path = OnnxPathManager::GetChildModelSavePath(modelName, child_idx);
        std::filesystem::path child_params_path = OnnxPathManager::GetChildModelParamsSavePath(modelName, child_idx);

        //this->ExtractModelByNodeWithWrite(infos[child_idx],onnxPath, child_onnx_path, child_params_path, start_node, end_node);

        threads.push_back(std::make_shared<std::thread>(&ModelAnalyzer::ExtractModelByNodeWithWrite,this, &infos[child_idx],onnxPath, child_onnx_path, child_params_path, &nodes[start_index], &nodes[end_index], true));
    }

    for(auto &th: threads)
    {
        th->join();
    }
    std::cout << "end split"<<std::endl;

    for(int child_idx = 0; child_idx < childs_size; child_idx++)
    {
        total_param[std::to_string(child_idx)] = infos[child_idx];
    }

    JsonSerializer::StoreJson(total_param, OnnxPathManager::GetChildModelSumParamsSavePath(modelName));

    return total_param;
}

nlohmann::json ModelAnalyzer::CreateParamsInfo(std::filesystem::path onnx_path, std::filesystem::path params_path, int default_batch)
{
    onnx::ModelProto model = onnxUtil::load(onnx_path);
    onnx::GraphProto graph = model.graph();

    // for (auto&data: graph.input())
    // {
    //     std::cout<<data.name()<<"DDDSDSDS"<<std::endl;
    // }

    nlohmann::json params_dict;
    params_dict["model_path"] = onnx_path;

    std::set<std::string> weight_params;
    for (auto &v : graph.initializer())
        weight_params.emplace(v.name());

    auto CreateParamsDict = [weight_params, default_batch, params_path](nlohmann::json params_dict, google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> tennsors) -> nlohmann::json
    {
        std::vector<nlohmann::json> k_data = std::vector<nlohmann::json>();
        for (auto &m : tennsors)
        {
            nlohmann::json param;

            if (std::find(weight_params.begin(), weight_params.end(), m.name()) != weight_params.end())
                continue;

            param["type"] = m.type().tensor_type().elem_type();
            param["name"] = m.name();

            std::vector<int> shape_list = std::vector<int>();
            int mul_value = 1;
            google::protobuf::RepeatedPtrField<onnx::TensorShapeProto_Dimension> dim = m.type().tensor_type().shape().dim();
            for (auto &v : dim)
            {
                shape_list.emplace_back((typeid(v.dim_value()) == typeid(int64_t) && v.dim_value() > 0) ? v.dim_value() : -1);
                mul_value *= ((typeid(v.dim_value()) == typeid(int) && v.dim_value() > 0) ? v.dim_value() : default_batch);
            }

            param["shape"] = shape_list;

            k_data.emplace_back(param);

            // cost部分

            // double cost = 0.0;
            // if (param["type"].get<std::string>().find("int") != std::string::npos)
            // {
            //     cost = mul_value * 4;
            // } else
            // {
            //     cost = mul_value * 4;
            // }
            // param["cost"] = cost / (1024 * 1024);
            // params_dict[k]["cost"] += cost;
            // params_dict[k]["data"]
        }
        return k_data;
    };

    params_dict["input"]["data"] = CreateParamsDict(params_dict, graph.input());
    params_dict["output"]["data"] = CreateParamsDict(params_dict, graph.output());
    JsonSerializer::StoreJson(params_dict, params_path, true);

    return params_dict;
}

bool ModelAnalyzer::UniformSplit(int count)
{
    if(count>this->size())
    {
        return false;
    }

    UniformOptimizer::optimize(*this);
    return true;
}

std::vector<GraphNode> ModelAnalyzer::GetConvergeNodes()
{
    std::vector<GraphNode> result = std::vector<GraphNode>();
    for (auto &node : nodes)
        if (node.IsConvergeNode())
            result.emplace_back(node);
    return result;
}

const std::vector<GraphNode> &ModelAnalyzer::GetAllNodes() const
{
    return nodes;
}

GraphNode &ModelAnalyzer::operator[](int i)
{
    return nodes[i];
}

ModelAnalyzer::iterator ModelAnalyzer::begin()
{
    return ModelAnalyzer::iterator(nodes.data());
}

ModelAnalyzer::iterator ModelAnalyzer::end()
{
    return ModelAnalyzer::iterator(&nodes.back() + 1);
}

int ModelAnalyzer::size() const
{
    return nodes.size();
}

std::string ModelAnalyzer::getName()
{
    return this->modelName;
}

// Operator overload

std::ostream &operator<<(std::ostream &os, const GraphNode &node)
{
    os << "id=" << node.idx << ", name=" << node.name << ", inputs=" << node.inputs << ", outputs=" << node.outputs << ", dependencies_inputs=" << node.dependencies_inputs << ", dependencies_outputs=" << node.dependencies_outputs << std::endl;
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    if (!v.empty())
    {
        os << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, ", "));
        os << "\b\b]";
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const ModelAnalyzer &analyzer)
{
    for (auto &node : analyzer.GetAllNodes())
    {
        os << node << std::endl;
    }
    return os;
}

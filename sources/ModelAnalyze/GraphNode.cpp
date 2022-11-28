#include "ModelAnalyze/GraphNode.h"

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
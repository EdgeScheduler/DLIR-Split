#include "../../include/tensor/ModelTensorsInfo.h"
#include <iostream>
// TensorsInfo
TensorsInfo::TensorsInfo(const std::vector<std::string> &labels, const std::vector<std::vector<int64_t>> &shapes, const std::vector<ONNXTensorElementDataType> &types)
{
    this->SetTensorsInfo(labels, shapes, types);
}

int TensorsInfo::SetTensorsInfo(const std::vector<std::string> &labels, const std::vector<std::vector<int64_t>> &shapes, const std::vector<ONNXTensorElementDataType> &types)
{
    this->tensors.clear();
    int length = labels.size() < shapes.size() ? labels.size() : shapes.size();

    for (int i = 0; i < length; i++)
    {
        this->AppendTensorInfo(labels[i], shapes[i], types[i]);
    }

    return length;
}

void TensorsInfo::AppendTensorInfo(const char *label, const std::vector<int64_t> &shape, const ONNXTensorElementDataType &type)
{
    this->tensors.push_back(ValueInfo(label, shape, type));
}

void TensorsInfo::AppendTensorInfo(const std::string &label, const std::vector<int64_t> &shape, const ONNXTensorElementDataType &type)
{
    this->tensors.push_back(ValueInfo(label, shape, type));
}

std::vector<std::string> TensorsInfo::GetLabels()
{
    std::vector<std::string> labels;
    for (auto iter = this->tensors.begin(); iter < this->tensors.end(); iter++)
    {
        labels.push_back(iter->GetName());
    }
    return labels;
}

std::vector<std::vector<int64_t>> TensorsInfo::GetShapes()
{
    std::vector<std::vector<int64_t>> shapes;
    for (auto iter = this->tensors.begin(); iter < this->tensors.end(); iter++)
    {
        shapes.push_back(iter->GetShape());
    }
    return shapes;
}

std::vector<ONNXTensorElementDataType> TensorsInfo::GetTypes()
{
    std::vector<ONNXTensorElementDataType> types;
    for (auto iter = this->tensors.begin(); iter < this->tensors.end(); iter++)
    {
        types.push_back(iter->GetType());
    }
    return types;
}

int TensorsInfo::GetTensorsCount()
{
    return this->tensors.size();
}

std::vector<ValueInfo> TensorsInfo::GetAllTensors()
{
    return this->tensors;
}

std::ostream &operator<<(std::ostream &out, const TensorsInfo &value)
{
    for (ValueInfo tensor : value.tensors)
    {
        out << tensor << std::endl;
    }

    return out;
}

// ModelInfo
ModelInfo::ModelInfo(const Ort::Session &session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < session.GetInputCount(); i++)
    {
        // need to get shape
        this->input.AppendTensorInfo(session.GetInputNameAllocated(i, allocator).get(), session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape(), session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
    }

    for (int i = 0; i < session.GetOutputCount(); i++)
    {
        // need to get shape
        this->output.AppendTensorInfo(session.GetOutputNameAllocated(i, allocator).get(), session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape(), session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
    }
}

TensorsInfo ModelInfo::GetInput()
{
    return this->input;
}

TensorsInfo ModelInfo::GetOutput()
{
    return this->output;
}

std::ostream &operator<<(std::ostream &out, const ModelInfo &value)
{
    out << "input:" << std::endl;
    out << value.input << std::endl;
    out << "output:" << std::endl;
    ;
    out << value.output;

    return out;
}

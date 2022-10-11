#include "../../include/tensor/ValueInfo.h"

namespace OnnxValueType
{
    std::string OnnxTypeToString(const ONNXTensorElementDataType type)
    {
        switch (type)
        {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return "uint16_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return "int16_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return "string";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "double";

        default:
            return "unknown";
        }
    }
}

ValueInfo::ValueInfo(const std::string &name, const std::vector<int64_t> &shapes, const ONNXTensorElementDataType &type)
{
    this->name = name;
    this->shapes = shapes;
    this->type = type;
}

ValueInfo::ValueInfo(const char *name, const std::vector<int64_t> &shapes, const ONNXTensorElementDataType &type)
{
    this->name = std::string(name);
    this->shapes = shapes;
    this->type = type;
}

std::string ValueInfo::GetName() const
{
    return this->name;
}

std::vector<int64_t> ValueInfo::GetShape() const
{
    return this->shapes;
}

ONNXTensorElementDataType ValueInfo::GetType() const
{
    return this->type;
}

std::string ValueInfo::GetTypeString() const
{
    return OnnxValueType::OnnxTypeToString(this->type);
}

std::size_t ValueInfo::GetDimSize() const
{
    return this->shapes.size();
}

std::ostream &operator<<(std::ostream &out, const ValueInfo &value)
{
    out << value.name << ": [";
    for (auto iter = value.shapes.begin(); iter < value.shapes.end(); iter++)
    {
        if (iter == value.shapes.begin())
        {
            out << *iter;
        }
        else
        {
            out << ", " << *iter;
        }
    }
    out << "], type=" << value.GetTypeString();

    return out;
}
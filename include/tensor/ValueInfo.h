#ifndef TENSORINFO_H
#define TENSORINFO_H

#include <vector>
#include <string>
#include <ostream>
#include <onnxruntime_cxx_api.h>

namespace OnnxValueType
{
    std::string OnnxTypeToString(const ONNXTensorElementDataType type);
}

class ValueInfo
{
public:
    ValueInfo(const std::string &name, const std::vector<int64_t> &shapes, const ONNXTensorElementDataType &type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ValueInfo(const char *name, const std::vector<int64_t> &shapes, const ONNXTensorElementDataType &type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    /// @brief get input/output name in model
    /// @return name
    virtual std::string GetName() const;

    /// @brief get get input/output shapes in model
    /// @return shapes
    virtual std::vector<int64_t> GetShape() const;
    /// @brief get get input/output dim-size in model
    /// @return dim size
    virtual std::size_t GetDimSize() const;
    /// @brief get raw element data-type
    /// @return ONNXTensorElementDataType
    virtual ONNXTensorElementDataType GetType() const;
    /// @brief get element data-type
    /// @return std::string
    virtual std::string GetTypeString() const;
    friend std::ostream &operator<<(std::ostream &out, const ValueInfo &value);

private:
    std::vector<int64_t> shapes;
    std::string name;
    ONNXTensorElementDataType type;
};

#endif // !TENSORINFO_H

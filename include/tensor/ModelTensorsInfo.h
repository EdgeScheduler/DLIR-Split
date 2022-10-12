#ifndef MODELTTENSORSINFO_H
#define MODELTTENSORSINFO_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "ValueInfo.h"

/// @brief a group of tensor-describe
class TensorsInfo : public virtual Serializability
{
public:
    TensorsInfo(){};
    TensorsInfo(const nlohmann::json &json);
    TensorsInfo(const std::vector<std::string> &labels, const std::vector<std::vector<int64_t>> &shapes, const std::vector<ONNXTensorElementDataType> &types);

    /// @brief serial to json object
    /// @return one nlohmann::json object. if value not valid, it will be nullptr.
    nlohmann::json ToJson() const;

    /// @brief load class to json
    /// @param nlohmann::json  object
    virtual void LoadFromJson(const nlohmann::json &json);

    /// @brief set labels and shapes for object
    /// @param labels labels name
    /// @param shapes shapes
    /// @param type data types
    /// @return how many item was set.
    int SetTensorsInfo(const std::vector<std::string> &labels, const std::vector<std::vector<int64_t>> &shapes, const std::vector<ONNXTensorElementDataType> &types);

    /// @brief get all labels
    /// @return
    std::vector<std::string> GetLabels() const;

    /// @brief get all shapes
    /// @return
    std::vector<std::vector<int64_t>> GetShapes() const;

    /// @brief get all types
    /// @return
    std::vector<ONNXTensorElementDataType> GetTypes() const;

    /// @brief get raw ValueInfos
    /// @return
    std::vector<ValueInfo> GetAllTensors() const;

    /// @brief get how many items here
    /// @return count
    int GetTensorsCount() const;

    /// @brief add tensor to object
    /// @param label label name
    /// @param shape shape
    /// @param type data type
    /// @return
    void AppendTensorInfo(const char *label, const std::vector<int64_t> &shape, const ONNXTensorElementDataType &type);
    
    /// @brief add tensor to object
    /// @param label label name
    /// @param shape shape
    /// @param type data type
    /// @return
    void AppendTensorInfo(const std::string &label, const std::vector<int64_t> &shape, const ONNXTensorElementDataType &type);

    friend std::ostream &operator<<(std::ostream &out, const TensorsInfo &value);

private:
    std::vector<ValueInfo> tensors;
};

/// @brief describe for model input-output
class ModelInfo : public virtual Serializability
{
public:
    ModelInfo(){};
    ModelInfo(const nlohmann::json &json);
    ModelInfo(const Ort::Session &session);
    /// @brief serial to json object
    /// @return one nlohmann::json object.
    virtual nlohmann::json ToJson() const;
    /// @brief load class to json
    /// @param nlohmann::json  object
    virtual void LoadFromJson(const nlohmann::json &json);
    TensorsInfo GetInput() const;
    TensorsInfo GetOutput() const;
    friend std::ostream &operator<<(std::ostream &out, const ModelInfo &value);

private:
    TensorsInfo input;
    TensorsInfo output;
};

#endif // !MODELTTENSORSINFO_H
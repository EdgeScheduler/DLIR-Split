#ifndef __TASK_H__
#define __TASK_H__

#include <time.h>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include "../tensor/ModelTensorsInfo.h"
#include "../tensor/TensorValue.h"
#include "../tensor/ValueInfo.h"

class Task
{
public:
    Task(std::string tag, ModelInfo *modelInfo = nullptr);
    /// @brief set model-infos
    /// @param modelInfo
    void SetModelInfo(ModelInfo *modelInfo);

    /// @brief get how much time cost by task
    /// @return
    float TimeCost();

    /// @brief Set model-input-value (data)
    /// @param tensors
    void SetInputs(std::map<std::string, TensorValue<float>> &datas);

    /// @brief Get model-input reference (data)
    /// @return
    const std::vector<TensorValue<float>> &GetInputs();

    /// @brief record model inference resul (model-output)
    /// @param tensors
    void SetOutputs(std::vector<Ort::Value> &tensors);

    /// @brief Get model-output reference
    /// @return
    const std::vector<TensorValue<float>> &GetOutputs();

public:
    std::vector<TensorValue<float>> Inputs;
    std::vector<TensorValue<float>> Outputs;

    // runtime args
public:
    Ort::Session *_session;
    std::vector<const char *> *_input_labels;
    std::vector<const char *> *_output_labels;
    std::vector<Ort::Value> _input_datas;

private:
    clock_t startTime;
    clock_t endTime;
    std::string tag;
    ModelInfo *modelInfo;
};

#endif // __TASK_H__
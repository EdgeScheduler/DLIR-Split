#include <cassert>

#include "../../include/GPUAllocator/Task.h"

Task::Task(std::string tag, ModelInfo *modelInfo) : tag(tag), modelInfo(modelInfo) {}

void Task::SetModelInfo(ModelInfo *modelInfo)
{
    this->modelInfo = modelInfo;
}

float Task::TimeCost()
{
    return double(endTime - startTime) / CLOCKS_PER_SEC * 1000.0;
}

void Task::SetInputs(std::map<std::string, TensorValue<float>> &datas)
{
    for (const ValueInfo &info : this->modelInfo->GetInput().GetAllTensors())
    {
        this->Inputs.emplace_back(datas.at(info.GetName())); //
    }

    for (const ValueInfo &info : this->modelInfo->GetOutput().GetAllTensors())
    {
        this->Outputs.push_back(TensorValue(info, false));
    }

    for (TensorValue<float> &value : this->Inputs)
    {
        this->_input_datas.push_back(value);
    }

    this->startTime = clock();
}

void Task::SetOutputs(std::vector<Ort::Value> &tensors)
{
    this->endTime = clock();
    for (int i = 0; i < tensors.size(); i++)
    {
        this->Outputs[i].RecordOrtValue(tensors[i]);
    }
}

const std::vector<TensorValue<float>> &Task::GetInputs()
{
    return this->Inputs;
}

const std::vector<TensorValue<float>> &Task::GetOutputs()
{
    return this->Outputs;
}

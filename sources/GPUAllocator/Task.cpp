#include <cassert>

#include "../../include/GPUAllocator/Task.h"

Task::Task(std::string modelName, ModelInfo *modelInfo,std::string tag) : modelName(modelName),tag(tag), modelInfo(modelInfo) {}

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

void Task::RecordTimeCosts(clock_t cost)
{
    this->timeCosts.push_back(cost);
}

std::vector<clock_t> &Task::GetTimeCosts()
{
    return this->timeCosts;
}

std::vector<float> Task::GetTimeCostsByMs()
{
    std::vector<float> result(timeCosts.size());
    for (int i = 0; i < timeCosts.size(); i++)
    {
        result[i] = double(timeCosts[i]) / CLOCKS_PER_SEC * 1000.0;
    }

    return result;
}

std::string& Task::GetTag()
{
    return this->tag;
}


std::string& Task::GetModelName()
{
    return this->modelName;
}

clock_t Task::GetStartTime()
{
    return this->startTime;
}

clock_t Task::GetEndTime()
{
    return this->endTime;
}

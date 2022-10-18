#include "../../include/GPUAllocator/ExecutorManager.h"
#include "../../include/Common/Drivers.h"
#include <iostream>

ExecutorManager::ExecutorManager() : environment(ORT_LOGGING_LEVEL_WARNING, "test")
{
    sessionOption.SetIntraOpNumThreads(1);
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    sessionOption.AppendExecutionProvider_CUDA(Drivers::GPU_CUDA::GPU0);
}

ExecutorManager::~ExecutorManager()
{
}

void ExecutorManager::RunExecutor(std::string model_name)
{
    this->executorCount++; // it means no matter thread create ok, give a token_id.
    std::shared_ptr<ExecutorDescribe> executorDescribe = std::make_shared<ExecutorDescribe>();
    executorDescribe->executor = std::make_shared<ModelExecutor>(model_name, &sessionOption, &environment, executorCount, &tokenManager, &gpuMutex, &dealTask);
    executorDescribe->executorID = executorCount;
    executorDescribe->modelName = model_name;
    std::thread runThread(&ModelExecutor::RunCycle, executorDescribe->executor);
    executorDescribe->threadHandle = std::move(runThread);

    this->executorMap.insert(std::pair<std::string, std::shared_ptr<ExecutorDescribe>>(model_name, executorDescribe));
}

void ExecutorManager::AddTask(std::string model_name, std::map<std::string, TensorValue<float>> &datas)
{
    auto iter = this->executorMap.find(model_name);
    if (iter == this->executorMap.end())
    {
        std::cout << "no such model-executor run here." << std::endl;
        return;
    }

    iter->second->executor->AddTask(datas);
}

std::map<std::string, std::shared_ptr<ExecutorDescribe>> &ExecutorManager::GetExecutorInformation()
{
    return this->executorMap;
}

std::vector<std::thread *> ExecutorManager::GetAllThreads()
{
    std::vector<std::thread *> result;
    for (auto iter = executorMap.begin(); iter != executorMap.end(); iter++)
    {
        result.push_back(&iter->second->threadHandle);
    }
    return result;
}

void ExecutorManager::Join()
{
    std::vector<std::thread *> threads = this->GetAllThreads();
    for (auto &thread : threads)
    {
        thread->join();
    }
}

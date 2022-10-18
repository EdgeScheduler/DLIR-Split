#include <iostream>
#include <nlohmann/json.hpp>
#include "../../include/GPUAllocator/ExecutorManager.h"
#include "../../include/Tensor/TensorValue.hpp"
#include "../../include/Common/JsonSerializer.h"
#include "../../include/Tensor/ModelInputCreator.h"
#include "../../include/Common/PathManager.h"
using namespace std;

int main()
{
    ExecutorManager executorManager;

    nlohmann::json vgg19 = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath("vgg19"));
    ModelInfo vgg19Info(vgg19);
    ModelInputCreator vgg19Creator(vgg19Info.GetInput());
    executorManager.RunExecutor("vgg19");       // 1

    nlohmann::json resnet50 = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath("resnet50"));
    ModelInfo resnet50Info(resnet50);
    ModelInputCreator resnet50Creator(resnet50Info.GetInput());
    executorManager.RunExecutor("resnet50");    // 2

    nlohmann::json googlenet = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath("googlenet"));
    ModelInfo googlenetInfo(googlenet);
    ModelInputCreator googlenetCreator(googlenetInfo.GetInput());
    executorManager.RunExecutor("googlenet");   // 3

    for (int i = 0; i < 2; i++)
    {
        auto data = vgg19Creator.CreateInput();
        executorManager.AddTask("vgg19", data);
    }

    for (int i = 0; i < 2; i++)
    {
        auto data = resnet50Creator.CreateInput();
        executorManager.AddTask("resnet50", data);
    }

    for (int i = 0; i < 2; i++)
    {
        auto data = googlenetCreator.CreateInput();
        executorManager.AddTask("googlenet", data);
    }

    for (int token : {3, 2, 3, 2, 2, 1, 1, 2, 1, 1, 1, 1})
    {
        executorManager.Grant(token);
    }

    auto executorInfo = executorManager.GetExecutorInformation();
    for (auto iter = executorInfo.begin(); iter != executorInfo.end(); iter++)
    {
        auto &results = iter->second->executor->GetResultQueue();
        for (int i = 0; i < 2; i++)
        {
            auto result= results.Pop();
            cout<<result.GetTag()<<"-"<<i<<": "<<result.TimeCost()<<"ms"<<endl;
            
            for(auto& value:result.GetOutputs())
            {
                value.Print();
            }
        }
    }

    executorManager.Join();
    return 0;
}
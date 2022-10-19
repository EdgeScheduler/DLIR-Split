#include <iostream>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "../include/GPUAllocator/ExecutorManager.h"
#include "../include/Tensor/TensorValue.hpp"
#include "../include/Common/JsonSerializer.h"
#include "../include/Tensor/ModelInputCreator.h"
#include "../include/Common/PathManager.h"
using namespace std;

// g++ -DALLOW_GPU_PARALLEL for parallel
int main(int argc, char *argv[])
{
    int tokens[] = {3, 3, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1};
    if (argc >= 13)
    {
        for (int i = 0; i < 12; i++)
        {
            tokens[i] = atoi(argv[i + 1]);
        }
    }

    ExecutorManager executorManager;

    nlohmann::json vgg19 = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath("vgg19"));
    ModelInfo vgg19Info(vgg19);
    ModelInputCreator vgg19Creator(vgg19Info.GetInput());
    executorManager.RunExecutor("vgg19"); // 1

    nlohmann::json resnet50 = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath("resnet50"));
    ModelInfo resnet50Info(resnet50);
    ModelInputCreator resnet50Creator(resnet50Info.GetInput());
    executorManager.RunExecutor("resnet50"); // 2

    nlohmann::json googlenet = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath("googlenet"));
    ModelInfo googlenetInfo(googlenet);
    ModelInputCreator googlenetCreator(googlenetInfo.GetInput());
    executorManager.RunExecutor("googlenet"); // 3

    std::vector<std::map<std::string, TensorValue<float>>> vgg19Datas;
    std::vector<std::map<std::string, TensorValue<float>>> resnet50Datas;
    std::vector<std::map<std::string, TensorValue<float>>> googlenetDatas;
    for (int i = 0; i < 2; i++)
    {
        vgg19Datas.push_back(vgg19Creator.CreateInput());
    }
    for (int i = 0; i < 2; i++)
    {
        resnet50Datas.push_back(resnet50Creator.CreateInput());
    }
    for (int i = 0; i < 2; i++)
    {
        googlenetDatas.push_back(googlenetCreator.CreateInput());
    }

    for (auto &data : vgg19Datas)
    {
        executorManager.AddTask("vgg19", data);
    }
    for (auto &data : resnet50Datas)
    {
        executorManager.AddTask("resnet50", data);
    }
    for (auto &data : googlenetDatas)
    {
        executorManager.AddTask("googlenet", data);
    }

#ifndef ALLOW_GPU_PARALLEL
    cout << "use token:";
    for (int token : tokens)
    {
        cout << token << " ";
        executorManager.Grant(token);
    }
    cout << endl;
#else
    cout << "tokens deprecated, you have allow GPU parallel."<<endl;
#endif // !1

    auto executorInfo = executorManager.GetExecutorInformation();
    for (auto iter = executorInfo.begin(); iter != executorInfo.end(); iter++)
    {
        auto &results = iter->second->executor->GetResultQueue();
        for (int i = 0; i < 2; i++)
        {
            auto result = results.Pop();
            cout << result.GetModelName() << "-" << i << ": " << result.TimeCost() << "ms start=" << result.GetStartTime() << "," << result.GetEndTime() << endl;
            for (auto t : result.GetTimeCostsByMs())
            {
                cout << t << " ";
            }
            cout << endl
                 << endl;
        }
    }

    executorManager.Join();
    return 0;
}

// 3 3   2 2   2 2   1 1 1   1 1 1
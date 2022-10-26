#include <iostream>
#include <cstdlib>
#include <thread>
#include <ctime>
#include <nlohmann/json.hpp>
#include "../include/GPUAllocator/ExecutorManager.h"
#include "../include/Tensor/TensorValue.hpp"
#include "../include/Common/JsonSerializer.h"
#include "../include/Tensor/ModelInputCreator.h"
#include "../include/Common/PathManager.h"
#include "../include/Random/UniformRandom.h"

using DatasType = std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>>;

// g++ -DALLOW_GPU_PARALLEL for parallel
void ReqestGenerate(ExecutorManager& executorManager, std::vector<std::pair<std::string, ModelInputCreator>>& inputCreators,int count);
void ReplyGather(int count);

int main()
{
    int dataCount=1000;
    ExecutorManager executorManager;
    std::vector<std::pair<std::string, ModelInputCreator>> inputCreators;

    for(auto model_name: {"vgg19","resnet50","googlenet"})
    {
        nlohmann::json json = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath(model_name));
        ModelInfo info(json);
        ModelInputCreator creator(info.GetInput());
        executorManager.RunExecutor(model_name);
        inputCreators.push_back(std::pair<std::string, ModelInputCreator>(model_name,creator));
    }
    
    std::cout<<"wait system to init..."<<std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::thread reqestGenerateThread(ReqestGenerate, executorManager, inputCreators,dataCount);
    std::thread replyGatherThread(ReplyGather,dataCount);




//     std::vector<DatasType> vgg19Datas;
//     std::vector<DatasType> resnet50Datas;
//     std::vector<DatasType> googlenetDatas;
//     for (int i = 0; i < 2; i++)
//     {
//         vgg19Datas.push_back(vgg19Creator.CreateInput());
//     }
//     for (int i = 0; i < 2; i++)
//     {
//         resnet50Datas.push_back(resnet50Creator.CreateInput());
//     }
//     for (int i = 0; i < 2; i++)
//     {
//         googlenetDatas.push_back(googlenetCreator.CreateInput());
//     }

//     for (int i = 0; i < 2; i++)
//     {
//         executorManager.AddTask("vgg19", vgg19Datas[i], std::to_string(i));
//     }
//     for (int i = 0; i < 2; i++)
//     {
//         executorManager.AddTask("resnet50", resnet50Datas[i], std::to_string(i));
//     }
//     for (int i = 0; i < 2; i++)
//     {
//         executorManager.AddTask("googlenet", googlenetDatas[i], std::to_string(i));
//     }

// #ifndef ALLOW_GPU_PARALLEL
//     // cout << "use token:";
//     // for (int token : tokens)
//     // {
//     //     cout << token << " ";
//     //     executorManager.Grant(token);
//     // }
//     // cout << endl;
// #else
//     cout << "tokens deprecated, you have allow GPU parallel." << endl;
// #endif // !1

//     auto executorInfo = executorManager.GetExecutorInformation();
//     for (auto iter = executorInfo.begin(); iter != executorInfo.end(); iter++)
//     {
//         auto &results = iter->second->executor->GetResultQueue();
//         for (int i = 0; i < 2; i++)
//         {
//             auto result = results.Pop();
//             // cout << result->GetModelName() << "-" << i << ": " << result->TimeCost() << "ms start=" << result->GetStartTime() << "," << result->GetEndTime() << endl;
//             // for (auto t : result->GetTimeCostsByMs())
//             // {
//             //     cout << t << " ";
//             // }

//             // cout << setw(4) << result->GetDescribe();
//             nlohmann::json describe = result->GetDescribe();
//             cout << describe["model_name"].get<std::string>() << "-" << describe["tag"].get<std::string>() << ":" << endl;
//             cout << "total(ms): " << describe["total_cost_by_ms"].get<float>() << " wait (ms): " << describe["wait_cost"].get<float>() << " execute (ms): " << describe["execute_cost"].get<float>() << endl
//                  << endl;
//         }
//     }

    executorManager.Join();
    return 0;
}

void ReqestGenerate(ExecutorManager& executorManager, std::vector<std::pair<std::string, ModelInputCreator>>& inputCreators, int count)
{
    
    std::cout<<"run request generate with possion."<<std::endl;
    for(int i=0;i<count;i++)
    {

        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}

void ReplyGather(int count)
{
    std::cout<<"run reply gather."<<std::endl;
}

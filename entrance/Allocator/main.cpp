#include <iostream>
#include <cstdlib>
#include <thread>
#include <ctime>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "../../include/GPUAllocator/ExecutorManager.h"
#include "../../include/Tensor/TensorValue.hpp"
#include "../../include/Common/JsonSerializer.h"
#include "../../include/Tensor/ModelInputCreator.h"
#include "../../include/Common/PathManager.h"
#include "../../include/Random/UniformRandom.h"
#include "../../include/Random/PossionRandom.h"

using DatasType = std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>>;

// g++ -DALLOW_GPU_PARALLEL for parallel
void ReqestGenerate(ExecutorManager* executorManager, std::vector<std::pair<std::string, ModelInputCreator>>* inputCreators,int count,float lambda=30);
void ReplyGather(ExecutorManager *executorManager,int count);

int main(int argc, char *argv[])
{
    int dataCount=1000;
    float lambda=40;
    if (argc >= 2)
    {
        dataCount=atoi(argv[1]);
    }
    if (argc >= 3)
    {
        lambda=atoi(argv[2]);
    }

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
    
    std::thread reqestGenerateThread(ReqestGenerate, &executorManager, &inputCreators,dataCount,lambda);
    ReplyGather(&executorManager,dataCount);
    
    // std::cout<<"process to end."<<std::endl;
    // exit(0);
    reqestGenerateThread.join();
    executorManager.Join();
    
    return 0;
}

void ReqestGenerate(ExecutorManager* executorManager, std::vector<std::pair<std::string, ModelInputCreator>>* inputCreators, int count,float lambda)
{
    PossionRandom possionRandom;
    UniformRandom uniformRandom;
    std::pair<std::string, ModelInputCreator>* creator=nullptr;
    std::cout<<"run request generate with possion, and start to prepare random input."<<std::endl;
    
    std::vector<std::pair<std::string, DatasType>> datas(count);
    for(int i=0;i<count;i++)
    {
        float u=uniformRandom.Random();
        if(u<1.0/3)
        {
            creator=&(*inputCreators)[0];
        }
        else if(u<2.0/3)
        {
            creator=&(*inputCreators)[1];
        }
        else
        {
            creator=&(*inputCreators)[2];
        }
        datas[i]=std::pair<std::string, DatasType>(creator->first,creator->second.CreateInput());
    }

    std::cout<<"generate request ok, wait system to init..."<<std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));

    std::cout<<"start to send request("<<count<<")."<<std::endl;
    for(int i=0;i<count;i++)
    {
        executorManager->AddTask(datas[i].first, datas[i].second,std::to_string(i));
        std::this_thread::sleep_for(std::chrono::milliseconds((int)possionRandom.Random(lambda)));
    }
    std::cout<<"end send request."<<std::endl;
}

void ReplyGather(ExecutorManager* executorManager,int count)
{
    std::cout<<"run reply gather."<<std::endl;
#ifdef ALLOW_GPU_PARALLEL
    auto saveFold=RootPathManager::GetRunRootFold() / "data"/"raw";
#else
    std::filesystem::path saveFold=RootPathManager::GetRunRootFold() / "data"/"allocator";
#endif
    std::filesystem::create_directories(saveFold);
    auto &applyQueue=executorManager->GetApplyQueue();
    for(int i=0;i<count;i++)
    {
        auto task=applyQueue.Pop();
        JsonSerializer::StoreJson(task->GetDescribe(),saveFold/(std::to_string(i)+".json"));
    }
    std::cout<<"all apply for "<<count<<" task received."<<std::endl;
}

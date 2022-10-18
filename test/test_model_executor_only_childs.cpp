#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <time.h>
#include <map>
#include <onnxruntime_cxx_api.h>
#include "../include/Common/Drivers.h"
#include "../include/Tensor/ValueInfo.h"
#include "../include/Tensor/TensorValue.hpp"
#include "../include/Tensor/ModelTensorsInfo.h"
#include "../include/Common/PathManager.h"
#include "../include/GPUAllocator/Task.h"
#include "../include/GPUAllocator/ModelExecutor.h"
#include "../include/Tensor/ModelInputCreator.h"
using namespace std;

// g++ -DALLOW_GPU_Parallel for parallel

// 41.6 28.3ms 24.3ms 18.5ms
int main(int argc, char *argv[])
{
    std::string model_name = "vgg19";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test"); // log id: "test"
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AppendExecutionProvider_CUDA(Drivers::GPU_CUDA::GPU0);

    std::filesystem::path model_path;
    model_path = OnnxPathManager::GetChildModelSavePath(model_name);
    Ort::Session session(env, model_path.c_str(), session_options);

    ModelInfo modelInfo(session);
    ModelInputCreator inputCreator(modelInfo.GetInput());

    ModelExecutor executor(model_name, &session_options, &env, 1, nullptr, nullptr, nullptr);

    // executor.AddTask(data);

    for (int i = 0; i < 4; i++)
    {
        auto data = inputCreator.CreateInput();
        executor.AddTask(data);
        executor.RunOnce();
        executor.RunOnce();
        executor.RunOnce();

        Task task = executor.GetResultQueue().Pop();

        cout << "childs-" << i << ": " << task.TimeCost() << "ms" << endl;
        // for (auto &value : task.GetTimeCostsByMs())
        // {
        //     cout << "--:" << value << endl;
        // }

        for (auto &value : task.GetOutputs())
        {
            value.Print();
        }
        cout << endl;
    }

    return 0;
}
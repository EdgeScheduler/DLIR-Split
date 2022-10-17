#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <time.h>
#include <map>
#include <onnxruntime_cxx_api.h>
#include "../include/common/Drivers.h"
#include "../include/tensor/ValueInfo.h"
#include "../include/tensor/TensorValue.h"
#include "../include/tensor/ModelTensorsInfo.h"
#include "../include/common/PathManager.h"
#include "../include/GPUAllocator/task.h"
#include "../include/GPUAllocator/ModelExecutor.h"
using namespace std;

// 41.6 28.3ms 24.3ms 18.5ms
int main(int argc, char *argv[])
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test"); // log id: "test"
    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AppendExecutionProvider_CUDA(Drivers::GPU_CUDA::GPU0);

    std::filesystem::path model_path;
    model_path = OnnxPathManager::GetChildModelSavePath("vgg19");
    Ort::Session session(env, model_path.c_str(), session_options);

    ModelInfo modelInfo(session);

    //  create input and output for bench
    vector<TensorValue<float>> input_tensors;
    vector<const char *> input_labels;
    for (const ValueInfo &info : modelInfo.GetInput().GetAllTensors())
    {
        input_tensors.push_back(TensorValue(info, true));
        input_labels.push_back(info.GetName().c_str());
    }

    vector<Ort::Value> input_values;
    for (auto &tensor : input_tensors)
    {
        input_values.push_back(tensor);
    }

    vector<TensorValue<float>> output_tensors;
    vector<const char *> output_labels;
    for (const ValueInfo &info : modelInfo.GetOutput().GetAllTensors())
    {
        output_tensors.push_back(TensorValue(info, false));
        output_labels.push_back(info.GetName().c_str());
    }

    // run raw:

    for (int i = 0; i < 5; i++)
    {
        clock_t start = clock();
        vector<Ort::Value> output_values = session.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), output_labels.data(), output_labels.size());
        cout << "raw-" << i << ": "  << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << "ms)." << endl;

        if (i == 0)
        {
            for (int i = 0; i < output_values.size(); i++)
            {
                output_tensors[i].RecordOrtValue(output_values[i]);
            }

            for (auto &tensor : output_tensors)
            {
                tensor.Print();
            }
        }
    }
    cout << "input:" << endl;
    std::map<std::string, TensorValue<float>> data;
    for (int i = 0; i < input_labels.size(); i++)
    {
        data.insert(pair<std::string, TensorValue<float>>(input_labels[i], input_tensors[i]));
    }

    ModelExecutor executor("vgg19", &session_options, &env, 1, nullptr);

    // executor.AddTask(data);

    for (int i = 0; i < 5; i++)
    {
        executor.AddTask(data);
        cout << "run task: " << i << endl;
        executor.RunOnce();
        executor.RunOnce();
        executor.RunOnce();

        Task task = std::move(executor.GetResultQueue().front());
        executor.GetResultQueue().pop();

        cout << "childs-" << i << ": " << task.TimeCost() << "ms" << endl;
        if (i == 0)
        {
            for (auto &value : task.GetOutputs())
            {
                value.Print();
            }
        }
    }

    return 0;
}
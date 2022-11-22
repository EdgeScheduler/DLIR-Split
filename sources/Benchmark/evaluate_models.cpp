#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <time.h>
#include <onnxruntime_cxx_api.h>
#include "Common/Drivers.h"
#include "Tensor/ValueInfo.h"
#include "Tensor/TensorValue.hpp"
#include "Tensor/ModelTensorsInfo.h"
#include "Common/PathManager.h"
#include "Common/JsonSerializer.h"
#include "Benchmark/evaluate_models.h"
using namespace std;


namespace evam
{
    float TimeEvaluateChildModels_impl(std::string model_name, int child_num, int test_count, int default_batchsize)
    {
        std::filesystem::path model_path;
        model_path = OnnxPathManager::GetChildModelSavePath(model_name, child_num);

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "evaluate"); // log id: "test"

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        session_options.AppendExecutionProvider_CUDA(Drivers::GPU_CUDA::GPU1);
        Ort::Session session(env, model_path.c_str(), session_options);

        ModelInfo modelInfo(session);
        // cout << modelInfo << endl;
        // cout << modelInfo << endl;

        // cout << "input:" << endl;
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

        // cout << "input values:" << endl;
        // for (auto &tensor : input_tensors)
        // {
        //     tensor.Print();
        // }

        clock_t start = clock();
        vector<Ort::Value> output_values = session.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), output_labels.data(), output_labels.size());
        
        // cout << endl
        //     << "run-0(" << setiosflags(ios::fixed) << setprecision(2) << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << "ms)." << endl;
        // print with TensorValue
        for (int i = 0; i < output_values.size(); i++)
        {
            output_tensors[i].RecordOrtValue(output_values[i]);
        }

        // for (auto &tensor : output_tensors)
        // {
        //     tensor.Print();
        // }

        float time_cost = 0;
        int cold_num = 2;
        // start to test run time
        clock_t end;
        for (int i = 0; i < test_count; i++)
        {
            // for (auto &tensor : input_tensors)
            // {
            //     tensor.Random();
            // }
            start = clock();
            vector<Ort::Value> output_values = session.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), output_labels.data(), output_labels.size());
            // cout << "run-" << i << "(" << setiosflags(ios::fixed) << setprecision(2) << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << "ms)."
            //     << "=> [" << setprecision(6) << *output_values[0].GetTensorMutableData<float>() << " ...]" << endl;
            end = clock();
            if(i >= cold_num)
            {
                time_cost += (end - start) * 1000.0 / CLOCKS_PER_SEC;
            }
            // release memory
            for (auto &value : output_values)
            {
                Ort::OrtRelease(value.release());
            }
        }
        
        float result = time_cost / (test_count - cold_num);
        return result;
    }

    nlohmann::json TimeEvaluateChildModels(std::string model_name, int test_count, int default_batchsize)
    {
        nlohmann::json result;
        float raw = TimeEvaluateChildModels_impl(model_name, -1, test_count, default_batchsize);
        // raw
        result["raw"]["avg"] = raw;

        //childs
        nlohmann::json params_dict = JsonSerializer::LoadJson(OnnxPathManager::GetChildModelSumParamsSavePath(model_name));
        float total = 0;
        float tmp;
        std::vector<float> childs;
        for(int idx = 0; idx < params_dict.size() - 1; idx++)
        {
            tmp = TimeEvaluateChildModels_impl(model_name, idx, test_count, default_batchsize);
            result["childs"][to_string(idx)]["avg"] = tmp;
            total += tmp;
            childs.push_back(tmp);
        }

        std::cout << "Raw: " << raw << std::endl << "Split: " << total << std::endl << "Improve: " << (raw - total) / raw << std::endl;

        float avg = total / params_dict.size();
        float var = 0;
        for (auto &data : childs)
        {
            std::cout<< "|| " << data << " "; 
            var += pow(data - avg, 2);
        }
        float sigma = sqrt(var / childs.size());
        std::cout << std::endl << "sigma: " << sigma << std::endl;
        result["childs"]["Std"] = sigma;
        result["childs"]["Improve"] = (raw - total) / raw;
        result["childs"]["total_time"] = total;
        return result;
    }

    // float TimeVarEvaluateChildModels(std::string model_name, int test_count, int default_batchsize)
    // {
    //     nlohmann::json params_dict = JsonSerializer::LoadJson(OnnxPathManager::GetChildModelSumParamsSavePath(model_name));
    //     float total = 0;
    //     float tmp;
    //     std::vector<float> childs;
    //     for(int idx = 0; idx < params_dict.size() - 1; idx++)
    //     {
    //         tmp = TimeEvaluateChildModels_impl(model_name, idx, test_count, default_batchsize);
    //         total += tmp;
    //         childs.push_back(tmp);
    //     }

    //     // std::cout << "Raw: " << raw << std::endl << "Split: " << total << std::endl << "Improve: " << (raw - total) / raw << std::endl;

    //     float avg = total / params_dict.size();
    //     float var = 0;
    //     for (auto &data : childs)
    //     {
    //         var += pow(data - avg, 2);
    //     }

    //     std::filesystem::path save_fold = "Benchmark/data/timecost/" + model_name + "/optimize/";
    //     if(!std::filesystem::exists(save_fold))
    //         std::filesystem::create_directories(save_fold);
    //     nlohmann::json time_evaluate_dict = TimeEvaluateChildModels(model_name);
    //     JsonSerializer::StoreJson(time_evaluate_dict, save_fold += (file_name.empty() ? "data.json" : file_name));

    //     return var;
    // }



    void EvalCurrentModelSplit(std::string model_name, std::string file_name)
    {
        std::filesystem::path save_fold = "Benchmark/data/timecost/" + model_name + "/GPU/";
        if(!std::filesystem::exists(save_fold))
            std::filesystem::create_directories(save_fold);
        nlohmann::json time_evaluate_dict = TimeEvaluateChildModels(model_name);
        JsonSerializer::StoreJson(time_evaluate_dict, save_fold += (file_name.empty() ? "data.json" : file_name));
    }

    float EvalStdCurrentModelSplit(std::string model_name, std::string file_name)
    {
        std::filesystem::path save_fold = "Benchmark/data/timecost/" + model_name + "/optimize/";
        if(!std::filesystem::exists(save_fold))
            std::filesystem::create_directories(save_fold);
        nlohmann::json time_evaluate_dict = TimeEvaluateChildModels(model_name);
        JsonSerializer::StoreJsonAppend(time_evaluate_dict, save_fold += (file_name.empty() ? "data.json" : file_name));
        return time_evaluate_dict["childs"]["Std"].get<float>();
    }

    
}
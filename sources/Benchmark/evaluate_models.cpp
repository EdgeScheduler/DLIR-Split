#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <mutex>
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
    float TimeEvaluateChildModels_impl(std::string model_name, std::filesystem::path model_path, std::string key, std::string GPU_tag, bool enable_bench_cache, int test_count, int default_batchsize)
    {
        static std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        static nlohmann::json cache = [&]()
        {
            if (std::filesystem::exists(BenchmarkPathManager::GetModelTimeBenchmarkCacheSavePath(model_name, GPU_tag)))
            {
                return JsonSerializer::LoadJson(BenchmarkPathManager::GetModelTimeBenchmarkCacheSavePath(model_name, GPU_tag));
            }
            else
            {
                return nlohmann::json({});
            }
        }();

        if (cache.contains(key) && enable_bench_cache)
        {
            return cache[key].get<float>();
        }

        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "evaluate"); // log id: "test"

        static Ort::SessionOptions session_options;
        static bool init_flag = [&]()
        {
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
            session_options.AppendExecutionProvider_CUDA(Drivers::GPU_CUDA::GPU0);
            return true;
        }();
        
        Ort::Session session(env, model_path.c_str(), session_options);
        // std::cout<<model_name<<", "<< model_path.c_str() << ", "<<key<<std::endl;

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

        vector<const char *> output_labels;
        for (const ValueInfo &info : modelInfo.GetOutput().GetAllTensors())
        {
            output_labels.push_back(info.GetName().c_str());
        }

        float time_cost = 0;
        int cold_num = 3;
        test_count+=cold_num;
        // start to test run time
        // std::cout<<key<<"=>";
        for (int i = 0; i < test_count; i++)
        {
            clock_t start = clock();
            vector<Ort::Value> output_values = session.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), output_labels.data(), output_labels.size());
            clock_t end = clock();

            std::cout<<(end - start) * 1000.0/CLOCKS_PER_SEC<<" ";
            if (i >= cold_num)
            {
                time_cost += (end - start) * 1000.0;
            }
            // release memory
            for (auto &value : output_values)
            {
                Ort::OrtRelease(value.release());
            }
        }
        std::cout<<std::endl;

        float result = time_cost / (test_count - cold_num) /  CLOCKS_PER_SEC;

        cache[key] = result;
        if(enable_bench_cache)
        {
            JsonSerializer::StoreJson(cache, BenchmarkPathManager::GetModelTimeBenchmarkCacheSavePath(model_name, GPU_tag), true);
        }

        lock.unlock();
        return result;
    }

    float TimeEvaluateChildModels_impl(std::string model_name, int child_num, std::string GPU_tag, bool enable_bench_cache, int test_count, int default_batchsize)
    {
        nlohmann::json child_summary = JsonSerializer::LoadJson(OnnxPathManager::GetChildModelSumParamsSavePath(model_name));
        std::string key = std::to_string(child_summary[std::to_string(child_num)]["from"].get<int>()) + "-" + std::to_string(child_summary[std::to_string(child_num)]["to"].get<int>());
        
        return TimeEvaluateChildModels_impl(model_name,OnnxPathManager::GetChildModelSavePath(model_name, child_num), key, GPU_tag, enable_bench_cache, test_count, default_batchsize);
    }

    nlohmann::json TimeEvaluateChildModels(std::string model_name, std::string GPU_tag, bool enable_bench_cache, int test_count, int default_batchsize)
    {
        nlohmann::json result;
        nlohmann::json child_summary = JsonSerializer::LoadJson(OnnxPathManager::GetChildModelSumParamsSavePath(model_name));
        
        float raw = TimeEvaluateChildModels_impl(model_name, -1, GPU_tag,enable_bench_cache, test_count, default_batchsize);
        // raw
        result["raw"]["avg"] = raw;
        result["raw"]["from"] = child_summary["-1"]["from"].get<int>();
        result["raw"]["to"] = child_summary["-1"]["to"].get<int>();

        // childs
        nlohmann::json params_dict = JsonSerializer::LoadJson(OnnxPathManager::GetChildModelSumParamsSavePath(model_name));
        float total = 0;
        float tmp;
        std::vector<float> childs;
        for (int idx = 0; idx < params_dict.size() - 1; idx++)
        {
            tmp = TimeEvaluateChildModels_impl(model_name, idx, GPU_tag,enable_bench_cache, test_count, default_batchsize);
            result["childs"][to_string(idx)]["avg"] = tmp;
            result["childs"][to_string(idx)]["from"] = child_summary[to_string(idx)]["from"].get<int>();
            result["childs"][to_string(idx)]["to"] = child_summary[to_string(idx)]["to"].get<int>();

            total += tmp;
            childs.push_back(tmp);
        }

        std::cout << "Raw: " << raw << std::endl
                  << "Split: " << total << std::endl
                  << "OverHead: " << (total - raw) / raw << std::endl;

        float avg = total / params_dict.size();
        float var = 0;
        for (auto &data : childs)
        {
            std::cout << "|| " << data << " ";
            var += pow(data - avg, 2);
        }
        float sigma = sqrt(var / childs.size());
        std::cout << std::endl
                  << "sigma: " << sigma << std::endl;
        result["childs"]["Std"] = sigma;
        result["childs"]["overhead"] = (total-raw) / raw;
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

    float EvalStdCurrentModelSplit(std::string model_name, std::string file_name, std::string GPU_tag, bool enable_bench_cache)
    {
        std::filesystem::path record_path = BenchmarkPathManager::GetModelSplitRecordJsonSavePath(model_name);
        static nlohmann::json cache = [=]()
        {
            if (std::filesystem::exists(record_path))
            {
                return JsonSerializer::LoadJson(record_path);
            }
            else
            {
                return nlohmann::json::array();
            }
        }();

        nlohmann::json time_evaluate_dict = TimeEvaluateChildModels(model_name, GPU_tag, enable_bench_cache);

        cache.push_back(time_evaluate_dict);
        JsonSerializer::StoreJson(cache, record_path, true);
        return time_evaluate_dict["childs"]["Std"].get<float>();
    }
}
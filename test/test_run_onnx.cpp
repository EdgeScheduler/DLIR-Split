#include <assert.h>
#include <vector>
#include<string>
#include <iostream>
#include<memory>
#include <onnxruntime_cxx_api.h>
#include "../include/common/Drivers.h"
#include"../include/tensor/ValueInfo.h"
#include"../include/tensor/ModelTensorsInfo.h"
using namespace std;

int main(int argc, char *argv[])
{
    const char *test_onnx = "resnet50-5.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test"); // log id: "test"

    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    // session_options.AppendExecutionProvider_CUDA(Drivers::GPU_CUDA::GPU0);
    Ort::Session session(env, test_onnx, session_options);

    ModelInfo modelInfo(session);
    cout<<modelInfo;

    
    

    // // generate input
    // std::vector<int64_t> input_dim = {15,3,224,224};
    // size_t input_tensor_size = 15*3*224*224;
    // std::vector<float> input_tensor_values(input_tensor_size);
    // for (unsigned int i = 0; i < input_tensor_size; i++)
    //     input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    // // create input tensor object from data values
    // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_dim.data(), input_dim.size());
    // assert(input_tensor.IsTensor());

    // std::vector<int64_t> input_mask_node_dims = {1, 20, 4};
    // size_t input_mask_tensor_size = 1 * 20 * 4;
    // std::vector<float> input_mask_tensor_values(input_mask_tensor_size);
    // for (unsigned int i = 0; i < input_mask_tensor_size; i++)
    //     input_mask_tensor_values[i] = (float)i / (input_mask_tensor_size + 1);
    // // create input tensor object from data values
    // auto mask_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Ort::Value input_mask_tensor = Ort::Value::CreateTensor<float>(mask_memory_info, input_mask_tensor_values.data(), input_mask_tensor_size, input_mask_node_dims.data(), 3);
    // assert(input_mask_tensor.IsTensor());

    // std::vector<Ort::Value> ort_inputs;
    // ort_inputs.push_back(std::move(input_tensor));
    // ort_inputs.push_back(std::move(input_mask_tensor));
    // // score model & input tensor, get back output tensor
    // auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 2);

    // // Get pointer to output tensor float values
    // float *floatarr = output_tensors[0].GetTensorMutableData<float>();
    // float *floatarr_mask = output_tensors[1].GetTensorMutableData<float>();

    // printf("Done!\n");
    return 0;
}

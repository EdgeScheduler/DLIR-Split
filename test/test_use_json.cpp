#include <nlohmann/json.hpp>
#include <iostream>
#include"../include/tensor/ModelTensorsInfo.h"
using namespace std;

int main()
{
    nlohmann::json config_json = nlohmann::json::parse(R"({"input":{"data":[{"name":"onnx::Conv_327","shape":[15,64,56,56],"type":"float32"},{"name":"input.8","shape":[15,64,56,56],"type":"float32"}]},"output":{"data":[{"name":"input.24","shape":[15,64,56,56],"type":"float32"},{"name":"input.8","shape":[15,64,56,56],"type":"float32"}]}})"); //构建json对象
    ModelInfo modelInfo(config_json);
    cout<<modelInfo<<endl;
    

    return 0;
}
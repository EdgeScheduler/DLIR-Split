#include <filesystem>
#include <iostream>
#include "../include/SplitToChilds/ModelAnalyzer.h"
#include "../library/onnx.proto3.pb.h"
using namespace std;

int main()
{

    // static filesystem::path p = "./";
    // string modelname = "resnet";
    // ModelAnalyzer a = ModelAnalyzer(modelname, p);

    // ModelAnalyzer analyzer = ModelAnalyzer("resnet50", "../Onnxs/resnet50/resnet50.onnx");
    // onnx::ModelProto model = analyzer.onnx_load();
    
    std::cout << "end" << std::endl;
    

    return 0;
}
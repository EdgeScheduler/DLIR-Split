#include <filesystem>
#include <iostream>
#include "../include/SplitToChilds/ModelAnalyzer.h"
#include "../library/onnx.proto3.pb.h"
#include "../include/utils/OnnxUtil.h"
using namespace std;

int main()
{

    // static filesystem::path p = "./";
    // string modelname = "resnet";
    // ModelAnalyzer a = ModelAnalyzer(modelname, p);

    ModelAnalyzer analyzer = ModelAnalyzer("vgg19");
    onnx::ModelProto model = onnxUtil.load(analyzer.getModelPath());
    auto graph = model.graph();

    for(auto data: graph.initializer())
    {
        std::cout<<data.name()<<","<<std::endl;
    }
    // std::cout<<graph.initializer().data()<<std::endl; //要用tensorproto输出
    
    std::cout << "test end" << std::endl;
    

    return 0;
}
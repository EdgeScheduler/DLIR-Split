#include <filesystem>
#include <iostream>
#include "../include/SplitToChilds/ModelAnalyzer.h"
#include "../library/onnx.proto3.pb.h"
#include "../include/Utils/OnnxUtil.h"
using namespace std;

int main()
{

    // static filesystem::path p = "./";
    // string modelname = "resnet";
    // ModelAnalyzer a = ModelAnalyzer(modelname, p);

    ModelAnalyzer analyzer = ModelAnalyzer("vgg19");
    onnx::ModelProto model = onnxUtil.load(analyzer.GetModelPath());
    auto graph = model.graph();
    


    // initializer
    // std::cout<<graph.initializer_size()<<std::endl;
    // for(auto data: graph.initializer())
    // {
    //     std::cout<<data.name()<<","<<std::endl;
    // }

    // node
    // for(auto data: graph.node())
    // {
    //     for(auto inp: data.input())
    //     {
    //         std::cout<<inp<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    for(auto data: graph.node())
    {
        std::cout<<data.name()<<std::endl;
    }

    
    std::cout << "test end" << std::endl;


    return 0;
}
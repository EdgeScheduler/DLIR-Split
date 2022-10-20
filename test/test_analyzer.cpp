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

    // graph
    // std::cout<<graph.initializer_size()<<std::endl;
    for(auto data: graph.input())
    {
        for(auto i: data.type().tensor_type().shape().dim())
        std::cout<< i.dim_value() <<","<<std::endl;
    }

    // node
    // for (auto data : graph.node())
    // {
    //     for (auto inp : data.output())
    //     {
    //         std::cout << inp << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    // for(auto data: graph.node())
    // {
    //     std::cout<<data.output(0)<<std::endl;
    // }

    std::cout << "test end" << std::endl;

    return 0;
}

/*

googlenet-0: 89.791ms start=6292953,6382744
89.728

googlenet-1: 149.102ms start=6304128,6453230
70.425

resnet50-0: 101.449ms start=6281131,6382580
56.379 44.987

resnet50-1: 168.004ms start=6285126,6453130
33.103 37.418

vgg19-0: 107.863ms start=6266010,6373873
34.241 27.407 46.191

vgg19-1: 202.407ms start=6273386,6475793
32.661 48.727 20.489

*/
#include <iostream>
#include <fstream>
#include <cassert>
#include "../../include/SplitToChilds/ModelAnalyzer.h"
#include "../../include/common/PathManager.h"
#include "../../library/onnx.proto3.pb.h"

// some tool function

void print_dim(const ::onnx::TensorShapeProto_Dimension &dim)
{
    switch (dim.value_case())
    {
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
        std::cout << dim.dim_param();
        break;
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
        std::cout << dim.dim_value();
        break;
    default:
        assert(false && "should never happen");
    }
}

void print_io_info(const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &info)
{
    for (auto input_data : info)
    {
        auto shape = input_data.type().tensor_type().shape();
        std::cout << "  " << input_data.name() << ":";
        std::cout << "[";
        if (shape.dim_size() != 0)
        {
            int size = shape.dim_size();
            for (int i = 0; i < size - 1; ++i)
            {
                print_dim(shape.dim(i));
                std::cout << ", ";
            }
            print_dim(shape.dim(size - 1));
        }
        std::cout << "]\n";
    }
}



// ModelAnalyzer

ModelAnalyzer::ModelAnalyzer(std::string model_name, std::filesystem::path onnx_path)
{
    this->modelName = model_name;
    this->manager = OnnxPathManager();
    this->use_cache = true;

    if (onnx_path.empty())
    {
        this->onnxPath = manager.GetModelSavePath(this->modelName);
    }
    else
        this->onnxPath = onnx_path;

    if (!this->Init())
        return;
}

onnx::ModelProto ModelAnalyzer::onnx_load()
{
    std::ifstream input(this -> onnxPath, std::ios::ate | std::ios::binary); // open file and move current position in file to the end

    std::cout<<input.fail()<<std::endl;

    std::streamsize size = input.tellg(); // get current position in file

    input.seekg(0, std::ios::beg);        // move to start of file

    std::vector<char> buffer(size);
    input.read(buffer.data(), size); // read raw data

    onnx::ModelProto model;
    model.ParseFromArray(buffer.data(), size); // parse protobuf

    auto graph = model.graph();

    std::cout << "graph inputs:\n";
    print_io_info(graph.input());

    std::cout << "graph outputs:\n";
    print_io_info(graph.output());

    return model;
}

bool ModelAnalyzer::Init()
{
    // try
    // {
        
    //     /* code */
    //     // std::ifstream input(onnxpath,std::ios::ate | std::ios::binary);
    // }
    // catch (const std::exception &e)
    // {
    //     std::cerr << e.what() << '\n';
    // }
    return true;
}


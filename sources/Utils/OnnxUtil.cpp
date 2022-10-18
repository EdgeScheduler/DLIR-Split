#include "../../include/Utils/OnnxUtil.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

onnx::ModelProto OnnxUtil::load(const std::filesystem::path &onnx_path)
{
    onnx::ModelProto model;
    // onnx::GraphProto graph;

    std::ifstream input(onnx_path, std::ios::in | std::ios::binary); // open file and move current position in file to the end

    bool isSuccess = !input.fail(); 

    try
    {
        if(!isSuccess)
            throw -1;

        google::protobuf::io::IstreamInputStream rawInput(&input);
        google::protobuf::io::CodedInputStream coded_input(&rawInput);

        coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max() / 4); //cancel the limit

        model.ParseFromCodedStream(&coded_input);

        auto graph = model.graph();
    }
    catch(int e)
    {
        std::cerr << "Error while loading onnx model." << '\n';
    }

    return model;
}
#include <fstream>
#include <iostream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/zero_copy_stream.h>
// #include "onnx.proto3.pb.h"

// bug ↓
#include "onnx/shape_inference/implementation.h"
//


// 在目录下放入vgg19.onnx，再把bug部分注释掉，可以正常运行，报warning是正常情况
// g++ -std=c++17 -o test ./*.c* -lpthread -lprotobuf && ./test


int main(void) {

        onnx::ModelProto model;
        std::ifstream in("vgg19.onnx", std::ios_base::binary);

        google::protobuf::io::IstreamInputStream rawInput(&in);
        google::protobuf::io::CodedInputStream coded_input(&rawInput);
        coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max() - 1); // cancel the limit
        model.ParseFromCodedStream(&coded_input);
        in.close();

        // bug ↓
        // onnx::shape_inference::InferShapes(model);
        //

        std::cout<<model.graph().input().size()<<"\n";
}
#ifndef __HELPER_H__
#define __HELPER_H__

// #include "../../library/onnx.proto3.pb.h"
// #include <onnx/onnx.pb.h>
#include "../../library/onnx/onnx.pb.h"
#include <iostream>
#include <fstream>
#include <filesystem>

// #include <nlohmann/json.hpp>
// #include "../../library/onnx.proto3.pb.h"

// namespace nlohmann
// {

// 
namespace onnxUtil
{
    /// @brief 
    /// @param input_path 
    /// @param output_path 
    /// @param input_names 
    /// @param output_names 
    void extract_model(std::filesystem::path input_path, std::filesystem::path output_path, std::vector<std::string> input_names, std::vector<std::string> output_names);

    /// @brief 
    /// @param nodes 
    /// @param name 
    /// @param inputs 
    /// @param outputs 
    /// @param initializer 
    /// @param doc_string 
    /// @param value_info 
    /// @param sparse_initializer 
    /// @return 
    onnx::GraphProto make_graph(std::vector<onnx::NodeProto> nodes, std::string name, std::vector<onnx::ValueInfoProto> inputs, std::vector<onnx::ValueInfoProto> outputs,
    std::vector<onnx::TensorProto> initializer=std::vector<onnx::TensorProto>(), std::string doc_string="", std::vector<onnx::ValueInfoProto> value_info=std::vector<onnx::ValueInfoProto>(),
    std::vector<onnx::SparseTensorProto> sparse_initializer = std::vector<onnx::SparseTensorProto>());

    /// @brief 
    /// @param graph 
    /// @param ir_version 
    /// @param opset_imports 
    /// @param producer_name 
    /// @param local_functions 
    /// @return 
    onnx::ModelProto make_model(onnx::GraphProto graph, int ir_version, google::protobuf::RepeatedPtrField<onnx::OperatorSetIdProto> opset_imports, std::vector<onnx::FunctionProto> local_functions);

    // /// @brief 
    // /// @param model 
    // /// @param check_type 
    // /// @param strict_mode 
    // /// @param data_prop 
    // /// @return 
    // onnx::ModelProto infer_shapes(onnx::ModelProto model, bool check_type=false, bool strict_mode=false, bool data_prop=false);

    /// @brief 
    /// @param onnx_path 
    /// @return 
    onnx::ModelProto load(const std::filesystem::path &onnx_path="");
}




#endif // __HELPER_H__
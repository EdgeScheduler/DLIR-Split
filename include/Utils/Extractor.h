#ifndef __EXTRACTOR_H__
#define __EXTRACTOR_H__

// #include "../../library/onnx.proto3.pb.h"
// #include <onnx/onnx.pb.h>
#include "../../library/onnx/onnx.pb.h"
#include <nlohmann/json.hpp>

class Extractor
{
    public:
        /// @brief 
        /// @param model 
        Extractor(onnx::ModelProto model);

        /// @brief 
        /// @param input_names 
        /// @param output_names 
        /// @return 
        onnx::ModelProto extract_model(std::vector<std::string> input_names, std::vector<std::string> output_names);

        /// @brief 
        /// @param objs 
        /// @return 
        std::map<std::string, onnx::ValueInfoProto> _build_name2obj_dict(google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> objs);

        /// @brief 
        /// @param objs 
        /// @return 
        std::map<std::string, onnx::TensorProto> _build_name2obj_dict(google::protobuf::RepeatedPtrField<onnx::TensorProto> objs);

        /// @brief 
        /// @param original_io 
        /// @param io_names_to_extract 
        std::vector<onnx::ValueInfoProto> _collect_new_io_core(google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> original_io, std::vector<std::string> io_names_to_extract);

        /// @brief 
        /// @param names 
        /// @return 
        std::vector<onnx::ValueInfoProto> _collect_new_inputs(std::vector<std::string> names);

        /// @brief 
        /// @param names 
        /// @return 
        std::vector<onnx::ValueInfoProto> _collect_new_outputs(std::vector<std::string> names);

        /// @brief 
        /// @param node_output_name 
        /// @param graph_input_names 
        /// @param reachable_nodes 
        void _dfs_search_reachable_nodes(std::string node_output_name, std::vector<std::string> graph_input_names, std::vector<onnx::NodeProto> &reachable_nodes);


        /// @brief 
        /// @param input_names 
        /// @param output_names 
        /// @return 
        std::vector<onnx::NodeProto> _collect_reachable_nodes(std::vector<std::string> input_names, std::vector<std::string> output_names);

        /// @brief 
        /// @param nodes 
        /// @return 
        std::vector<onnx::TensorProto> _collect_reachable_tensors_initializer(std::vector<onnx::NodeProto> nodes);

        /// @brief 
        /// @param nodes 
        /// @return 
        std::vector<onnx::ValueInfoProto> _collect_reachable_tensors_value_info(std::vector<onnx::NodeProto> nodes);

        /// @brief 
        /// @param nodes 
        /// @param referred_local_functions 
        /// @return 
        std::vector<onnx::NodeProto> find_referred_funcs(std::vector<onnx::NodeProto> nodes, std::vector<onnx::FunctionProto> referred_local_functions);

        /// @brief 
        /// @param nodes 
        /// @return 
        std::vector<onnx::FunctionProto> _collect_referred_local_functions(std::vector<onnx::NodeProto> nodes);

        /// @brief 
        /// @param nodes 
        /// @param inputs 
        /// @param outputs 
        /// @param initializer 
        /// @param value_info 
        /// @param local_funcitons 
        /// @return 
        onnx::ModelProto _make_model(std::vector<onnx::NodeProto> nodes, std::vector<onnx::ValueInfoProto> inputs, std::vector<onnx::ValueInfoProto> outputs, 
        std::vector<onnx::TensorProto> initializer, std::vector<onnx::ValueInfoProto> value_info, std::vector<onnx::FunctionProto> local_funcitons);

        
    private:
        onnx::ModelProto model;
        onnx::GraphProto graph;
        std::map<std::string, onnx::TensorProto> wmap;
        std::map<std::string, onnx::ValueInfoProto> vimap;


};












#endif // __EXTRACTOR_H__
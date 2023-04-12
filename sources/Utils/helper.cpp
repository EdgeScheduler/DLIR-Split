#include "Utils/helper.h"
#include "Utils/Extractor.h"
// #include "../../library/onnx/onnx.pb.h"
#include "onnx/shape_inference/implementation.h"
// #include <onnx/onnx.pb.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "onnx/checker.h"
#include <fstream>

namespace onnxUtil
{
    onnx::GraphProto make_graph(std::vector<onnx::NodeProto> nodes, std::string name, std::vector<onnx::ValueInfoProto> inputs, std::vector<onnx::ValueInfoProto> outputs,
                                std::vector<onnx::TensorProto> initializer, std::string doc_string, std::vector<onnx::ValueInfoProto> value_info,
                                std::vector<onnx::SparseTensorProto> sparse_initializer)
    {
        onnx::GraphProto graph = onnx::GraphProto();

        graph.mutable_node()->CopyFrom(google::protobuf::RepeatedPtrField<onnx::NodeProto>(nodes.begin(), nodes.end()));
        graph.set_name(name);
        graph.mutable_input()->CopyFrom(google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>(inputs.begin(), inputs.end()));
        graph.mutable_output()->CopyFrom(google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>(outputs.begin(), outputs.end()));
        graph.mutable_initializer()->CopyFrom(google::protobuf::RepeatedPtrField<onnx::TensorProto>(initializer.begin(), initializer.end()));
        graph.mutable_sparse_initializer()->CopyFrom(google::protobuf::RepeatedPtrField<onnx::SparseTensorProto>(sparse_initializer.begin(), sparse_initializer.end()));
        graph.mutable_value_info()->CopyFrom(google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>(value_info.begin(), value_info.end()));
        if (!doc_string.empty())
            graph.set_doc_string(doc_string);

        return graph;
    }

    void extract_model(std::filesystem::path input_path, std::filesystem::path output_path, std::vector<std::string> input_names, std::vector<std::string> output_names)
    {
        try
        {
            if (!std::filesystem::exists(input_path))
                throw input_path;
            if (output_path.empty())
                throw output_path;
            if (output_names.empty())
                throw output_names;
        }
        catch (const std::filesystem::path path)
        {
            std::cerr << "Invalid Input or Output path!" << '\n';
        }
        catch (const std::string names)
        {
            std::cerr << "Output tensor names shall not be empty!" << '\n';
        }
        onnx::checker::check_model(input_path);
        onnx::ModelProto model = onnxUtil::load(input_path);

        Extractor e = Extractor(model);
        onnx::ModelProto extracted = e.extract_model(input_names, output_names);

        std::fstream output(output_path, std::ios::out | std::ios::trunc | std::ios::binary);
        std::string model_string;
        try
        {
            extracted.SerializeToString(&model_string);
            output << model_string;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
        bool check_model = false;
        if (check_model)
            onnx::checker::check_model(output_path);
    }

    onnx::ModelProto make_model(onnx::GraphProto input_graph, int ir_version, google::protobuf::RepeatedPtrField<onnx::OperatorSetIdProto> input_opset_imports,
                                std::vector<onnx::FunctionProto> local_functions)
    {
        onnx::ModelProto model = onnx::ModelProto();
        // onnx::GraphProto graph = model.graph();

        model.set_ir_version(onnx::IR_VERSION);
        model.mutable_graph()->CopyFrom(input_graph);

        // if (input_opset_imports.size() > 0)
        // {
        model.mutable_opset_import()->CopyFrom(input_opset_imports);
        // }
        // else{
        //     onnx::OperatorSetIdProto* imp = model.add_opset_import();
        //     imp->version = OpSchemaRegistry::DomainToVersionRange::Instance().Map()["ai.onnx.ml"][1];
        // }

        google::protobuf::RepeatedPtrField<onnx::FunctionProto> model_functions = model.functions();
        model.mutable_functions()->CopyFrom(google::protobuf::RepeatedPtrField<onnx::FunctionProto>(local_functions.begin(), local_functions.end()));
        model.set_ir_version(ir_version);
        model.set_producer_name("onnx.utils.extract_model");
        return model;
    }

    // onnx::ModelProto infer_shapes(onnx::ModelProto model, bool check_type, bool strict_mode, bool data_prop)
    // {
    //     // onnx::ModelProto proto{};
    //     // ONNX_NAMESPACE::ParseProtoFromPyBytes(&proto, bytes);
    //     ShapeInferenceOptions options{check_type, strict_mode == true ? 1 : 0, data_prop};
    //     shape_inference::InferShapes(model, OpSchemaRegistry::Instance(), options);
    //     std::string out;
    //     proto.SerializeToString(&out);
    //     return out;
    // }

    onnx::ModelProto load(const std::filesystem::path &onnx_path)
    {
        onnx::ModelProto model;
        // onnx::GraphProto graph;

        std::ifstream input(onnx_path, std::ios::in | std::ios::binary); // open file and move current position in file to the end

        bool isSuccess = !input.fail();

        try
        {
            if (!isSuccess)
                throw -1;

            google::protobuf::io::IstreamInputStream rawInput(&input);
            google::protobuf::io::CodedInputStream coded_input(&rawInput);

            coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max() - 1); // cancel the limit

            model.ParseFromCodedStream(&coded_input);
        }
        catch (int e)
        {
            std::cerr << "Error while loading onnx model." << '\n';
        }

        return model;
    }

    ONNXTensorElementDataType IntToOnnxType(const int type)
    {
        switch (type)
        {
        case 1:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

        case 2:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;

        case 3:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;

        case 4:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;

        case 5:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;

        case 6:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;

        case 7:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

        // case 8:
        //     return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;

        // case 9:
        //     return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;

        case 10:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        
        case 11:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;

        case 12:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;

        case 13:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;

        case 14:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;

        case 15:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;

        case 16:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;

        default:
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        }
    }

    void tensor_transform(std::vector<std::shared_ptr<TensorValueObject>> &tensors, const ValueInfo &info)
    {
        switch (info.GetType())
        {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            tensors.push_back(std::make_shared<TensorValue<int8_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            tensors.push_back(std::make_shared<TensorValue<int16_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            tensors.push_back(std::make_shared<TensorValue<int32_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            tensors.push_back(std::make_shared<TensorValue<int64_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            tensors.push_back(std::make_shared<TensorValue<float_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            tensors.push_back(std::make_shared<TensorValue<uint8_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            tensors.push_back(std::make_shared<TensorValue<uint16_t>>(info, true));
            break;
        // case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        //     tensors.push_back(std::make_shared<TensorValue<std::string>>(info, true));
            // break;
        // case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        //     tensors.push_back(std::make_shared<TensorValue<int8_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            tensors.push_back(std::make_shared<TensorValue<float_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            tensors.push_back(std::make_shared<TensorValue<double_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            tensors.push_back(std::make_shared<TensorValue<uint32_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            tensors.push_back(std::make_shared<TensorValue<uint64_t>>(info, true));
            break;
        default:
            tensors.push_back(std::make_shared<TensorValue<float>>(info, true));
        }
    }
}

// namespace helper
// {

// }
#include "Utils/Extractor.h"
#include "Utils/helper.h"
// #include "../../library/shapeinfer.h"
#include "onnx/shape_inference/implementation.h"
#include <set>
#include <algorithm>

Extractor::Extractor(onnx::ModelProto model)
{
    this -> model = model;
    onnx::shape_inference::InferShapes(this->model);
    graph = this -> model.graph();
    wmap = _build_name2obj_dict(graph.initializer());
    vimap = _build_name2obj_dict(graph.value_info());
}

onnx::ModelProto Extractor::extract_model(std::vector<std::string> input_names, std::vector<std::string> output_names)
{
    std::vector<onnx::ValueInfoProto> inputs = _collect_new_inputs(input_names);
    std::vector<onnx::ValueInfoProto> outputs = _collect_new_outputs(output_names);
    std::vector<onnx::NodeProto> nodes = _collect_reachable_nodes(input_names, output_names);
    std::vector<onnx::TensorProto> initializer = _collect_reachable_tensors_initializer(nodes);
    std::vector<onnx::ValueInfoProto> value_info = _collect_reachable_tensors_value_info(nodes);
    std::vector<onnx::FunctionProto> local_functions = _collect_referred_local_functions(nodes);
    onnx::ModelProto model = _make_model(nodes, inputs, outputs, initializer, value_info, local_functions);
    // for( auto &o:outputs)
    // {
    //     std::cout<<o.name()<<"; ";
    // }
    // std::cout<<"____"<<std::endl;
    return model;
}

std::map<std::string, onnx::ValueInfoProto> Extractor::_build_name2obj_dict(google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> objs)
{
    std::map<std::string, onnx::ValueInfoProto> result = std::map<std::string, onnx::ValueInfoProto>();
    for (auto &obj : objs)
    {
        // std::cout<<"name: "<<obj.name()<<"; ";
        if (obj.name() != "")
            result.emplace(obj.name(), obj);
    }
    // std::cout<<"___"<<std::endl;
    return result;
}

std::map<std::string, onnx::TensorProto> Extractor::_build_name2obj_dict(google::protobuf::RepeatedPtrField<onnx::TensorProto> objs)
{
    std::map<std::string, onnx::TensorProto> result = std::map<std::string, onnx::TensorProto>();
    for (auto &obj : objs)
    {
        result.emplace(obj.name(), obj);
    }
    return result;
}

std::vector<onnx::ValueInfoProto> Extractor::_collect_new_io_core(google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> original_io, std::vector<std::string> io_names_to_extract)
{
    std::map<std::string, onnx::ValueInfoProto> original_io_map = _build_name2obj_dict(original_io);

    std::set<std::string> original_io_names = std::set<std::string>();
    for (std::map<std::string, onnx::ValueInfoProto>::iterator it = original_io_map.begin(); it != original_io_map.end(); ++it)
        original_io_names.emplace(it->first);

    std::set<std::string> s_io_names_to_extract = std::set<std::string>();
    for (std::vector<std::string>::iterator it = io_names_to_extract.begin(); it != io_names_to_extract.end(); ++it)
        s_io_names_to_extract.emplace(*it); 

    std::set<std::string> io_names_to_keep = std::set<std::string>();
    
    std::set_intersection(s_io_names_to_extract.begin(), s_io_names_to_extract.end(), original_io_names.begin(), original_io_names.end(), std::inserter(io_names_to_keep, io_names_to_keep.begin()));

    std::set<std::string> new_io_names_to_add = std::set<std::string>();
    std::set_difference(s_io_names_to_extract.begin(), s_io_names_to_extract.end(), original_io_names.begin(), original_io_names.end(), std::inserter(new_io_names_to_add, new_io_names_to_add.begin()));

    google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> new_io_tensors = google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>();

    std::vector<onnx::ValueInfoProto> new_io_vec = std::vector<onnx::ValueInfoProto>();

    // for(auto&data:original_io_map)
    // {
    //     std::cout<<"QQQQQQ"<<data.first<<std::endl;
    // }
    // for(auto&data:vimap)
    // {
    //     std::cout<<"WWWW"<<data.first<<std::endl;
    // }

    for (auto &name : io_names_to_keep)
    {
        // std::cout<<"originalmap: "<<name<<"; ";
        new_io_vec.emplace_back(original_io_map[name]);
    }
    // std::cout<<"___"<<std::endl;
    for (auto &name : new_io_names_to_add)
    {// {   std::cout<<"vmap: "<<name<<"; ";
        new_io_vec.emplace_back(vimap[name]);
    }
    // std::cout<<"____"<<std::endl;
    new_io_tensors = google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>(new_io_vec.begin(), new_io_vec.end());

    std::map<std::string, onnx::ValueInfoProto> new_io_tensors_map = _build_name2obj_dict(new_io_tensors);
    std::vector<onnx::ValueInfoProto> result = std::vector<onnx::ValueInfoProto>();
    for (auto &name : io_names_to_extract)
    {   
        // std::cout<<"name: "<<new_io_tensors_map[name].name()<<"; ";
        if (new_io_tensors_map[name].name() != "")
            result.emplace_back(new_io_tensors_map[name]);
    }
    // std::cout<<"____"<<std::endl;
    return result;
}

std::vector<onnx::ValueInfoProto> Extractor::_collect_new_inputs(std::vector<std::string> names)
{
    return _collect_new_io_core(graph.input(), names);
}

std::vector<onnx::ValueInfoProto> Extractor::_collect_new_outputs(std::vector<std::string> names)
{
    return _collect_new_io_core(graph.output(), names);
}

void Extractor::_dfs_search_reachable_nodes(std::string node_output_name, std::vector<std::string> graph_input_names, std::vector<onnx::NodeProto> &reachable_nodes)
{
    if (std::find(graph_input_names.begin(), graph_input_names.end(), node_output_name) != graph_input_names.end())
        return;

    for (auto &node : graph.node())
    {

        if (std::find(node.output().begin(), node.output().end(), node_output_name) == node.output().end())
            continue;
        // if (std::find(reachable_nodes.begin(), reachable_nodes.end(), node) != reachable_nodes.end())
        //     continue;
        bool flag = false;
        for(int i = 0; i < reachable_nodes.size(); i++)
        {   
            if (reachable_nodes[i].name() == node.name())
            {
                flag = true;
                break;
            }
        }
        if (flag == true)
            continue;
        reachable_nodes.emplace_back(node);

        for(auto &name: node.input())
        {
            _dfs_search_reachable_nodes(name, graph_input_names, reachable_nodes);
        }
    }
}

std::vector<onnx::NodeProto> Extractor::_collect_reachable_nodes(std::vector<std::string> input_names, std::vector<std::string> output_names)
{
    std::vector<onnx::NodeProto> reachable_nodes = std::vector<onnx::NodeProto>();
    for(auto &name: output_names)
        _dfs_search_reachable_nodes(name, input_names, reachable_nodes);

    std::vector<onnx::NodeProto> nodes = std::vector<onnx::NodeProto>();
    
    for(auto &n: graph.node())
        for(int i = 0; i < reachable_nodes.size(); i++)
            if (reachable_nodes[i].name() == n.name())
                nodes.emplace_back(n);
    return nodes;
}

std::vector<onnx::TensorProto> Extractor::_collect_reachable_tensors_initializer(std::vector<onnx::NodeProto> nodes)
{
    std::set<std::string> all_tensors_name = std::set<std::string>();
    for (auto &node: nodes)
    {
        for(auto &name: node.input())
            all_tensors_name.emplace(name);
        for(auto &name: node.output())
            all_tensors_name.emplace(name);
    }

    std::string t;
    std::vector<onnx::TensorProto> initializer = std::vector<onnx::TensorProto>();
    for (std::map<std::string, onnx::TensorProto>::iterator it = wmap.begin(); it != wmap.end(); ++it)
    {
        t = it->first;
        if (std::find(all_tensors_name.begin(), all_tensors_name.end(), t) != all_tensors_name.end())
            initializer.emplace_back(wmap[t]);
    }

    assert(graph.sparse_initializer().size() == 0);

    return initializer;
}

std::vector<onnx::ValueInfoProto> Extractor::_collect_reachable_tensors_value_info(std::vector<onnx::NodeProto> nodes)
{
    std::set<std::string> all_tensors_name = std::set<std::string>();
    for (auto &node: nodes)
    {
        for(auto &name: node.input())
            all_tensors_name.emplace(name);
        for(auto &name: node.output())
            all_tensors_name.emplace(name);
    }

    std::string t;
    std::vector<onnx::ValueInfoProto> value_info = std::vector<onnx::ValueInfoProto>();
    for (std::map<std::string, onnx::ValueInfoProto>::iterator it = vimap.begin(); it != vimap.end(); ++it)
    {
        t = it->first;
        if (std::find(all_tensors_name.begin(), all_tensors_name.end(), t) != all_tensors_name.end())
            value_info.emplace_back(vimap[t]);
    }

    assert(graph.quantization_annotation().size() == 0);

    return value_info;
}

std::vector<onnx::NodeProto> Extractor::find_referred_funcs(std::vector<onnx::NodeProto> nodes, std::vector<onnx::FunctionProto> referred_local_functions)
{
    std::vector<onnx::NodeProto> new_nodes = std::vector<onnx::NodeProto>();
    for (auto &node: nodes)
    {
        if (model.functions().size() > 0)
        {
            for (auto &f: model.functions())
            {
                if (f.name() == node.op_type() && f.domain() == node.domain())
                {
                    for (int i = 0; i < referred_local_functions.size(); i++)
                    {
                        if (&referred_local_functions[i] == &f)
                            continue;
                        referred_local_functions.emplace_back(f);
                        new_nodes.insert(new_nodes.end(), f.node().begin(), f.node().end());
                    }
                }
            }
        }
    }
    return new_nodes;
}

std::vector<onnx::FunctionProto> Extractor::_collect_referred_local_functions(std::vector<onnx::NodeProto> nodes)
{
    std::vector<onnx::FunctionProto> referred_local_functions = std::vector<onnx::FunctionProto>();
    std::vector<onnx::NodeProto> new_nodes = find_referred_funcs(nodes, referred_local_functions);
    while(new_nodes.size() > 0)
        new_nodes = find_referred_funcs(new_nodes, referred_local_functions);

    return referred_local_functions;
}

onnx::ModelProto Extractor::_make_model(std::vector<onnx::NodeProto> nodes, std::vector<onnx::ValueInfoProto> inputs, std::vector<onnx::ValueInfoProto> outputs, 
        std::vector<onnx::TensorProto> initializer, std::vector<onnx::ValueInfoProto> value_info, std::vector<onnx::FunctionProto> local_functions)
{
    std::string name = "Extracted from {" + graph.name() +"}";
    onnx::GraphProto graph = onnxUtil::make_graph(nodes, name, inputs, outputs, initializer, "", value_info);

    return onnxUtil::make_model(graph, model.ir_version(), model.opset_import(), local_functions);
}
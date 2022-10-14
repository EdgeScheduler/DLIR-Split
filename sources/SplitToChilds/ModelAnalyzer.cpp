#include "../../include/SplitToChilds/ModelAnalyzer.h"
#include "../common/PathManager.cpp"

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

bool ModelAnalyzer::Init()
{
    try
    {
        /* code */
        // std::ifstream input(onnxpath,std::ios::ate | std::ios::binary);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}

#include "../../include/GPUAllocator/ModelExecutor.h"
#include "../../include/common/JsonSerializer.h"
#include "../../include/common/PathManager.h"

ModelExecutor::ModelExecutor(std::string model_name, Ort::SessionOptions *session_opt, Ort::Env *env) : modelName(model_name), sessionOption(session_opt), onnxruntimeEnv(env), todo(0), modelCount(0)
{
    std::filesystem::path rawModelPath = OnnxPathManager::GetModelSavePath(modelName);
    Ort::Session rawSession(*onnxruntimeEnv, rawModelPath.c_str(), *sessionOption);
    this->rawModelInfo = ModelInfo(rawSession, rawModelPath);

    std::filesystem::path modelSumParamsPath = OnnxPathManager::GetChildModelSumParamsSavePath(modelName);
    nlohmann::json json = JsonSerializer::LoadJson(modelSumParamsPath);
    int start = 0;
    while (json.contains(std::to_string(start)))
    {
        std::filesystem::path model_path = OnnxPathManager::GetChildModelSavePath(modelName, start);
        Ort::Session session(*onnxruntimeEnv, model_path.c_str(), *sessionOption);

        this->modelInfos.push_back(ModelInfo(session, model_path));
        this->sessions.push_back(std::move(session));
        this->modelCount += 1;
    }
}

void ModelExecutor::ToNext()
{
    this->todo = (this->todo + 1) % len(this->modelCount);
}

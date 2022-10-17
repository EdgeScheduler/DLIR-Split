#ifndef __MODELEXECUTOR_H__
#define __MODELEXECUTOR_H__

#include <onnxruntime_cxx_api.h>
#include <vector>
#include "../tensor/ModelTensorsInfo.h"

class ModelExecutor
{
public:
    ModelExecutor(std::string model_name,Ort::SessionOptions* session_opt,Ort::Env* env);

private:
    int modelCount=0;
    Ort::Env* onnxruntimeEnv;
    Ort::SessionOptions* sessionOption;
    std::vector<Ort::Session> sessions;
    std::vector<ModelInfo> modelInfos;
    ModelInfo rawModelInfo;

    std::vector<Ort::Value> modelOutputs;   // model-0 output to modelOutputs[0] and can be used as input by model-1
    std::vector<Ort::Value> modelInputs;    // used by model-0
    int todo;
    std::string modelName;
};

#endif // __MODELEXECUTOR_H__
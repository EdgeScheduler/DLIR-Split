#ifndef __MODELEXECUTOR_H__
#define __MODELEXECUTOR_H__

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <map>
#include <queue>
#include "../Tensor/ModelTensorsInfo.h"
#include "Task.h"

class ModelExecutor
{
public:
    ModelExecutor(std::string model_name, Ort::SessionOptions *session_opt, Ort::Env *env, int token_id, int *token_manager);

    /// @brief add new task to executor
    void AddTask(std::map<std::string, TensorValue<float>> &datas);

    /// @brief record current task to the end.
    void ToNext();

    /// @brief Load task args, if not exist, it will come to block.
    void LoadTask();

    /// @brief Inference for once task.
    void RunOnce();

    /// @brief run all model automatically.
    void RunCycle();

    std::queue<Task> &GetResultQueue();
    std::queue<Task> &GetTaskQueue();

private:
    int modelCount = 0;
    Ort::Env *onnxruntimeEnv;
    Ort::SessionOptions *sessionOption;
    std::vector<Ort::Session> sessions;
    std::vector<ModelInfo> modelInfos;
    std::vector<std::vector<const char *>> inputLabels;
    std::vector<std::vector<const char *>> outputLabels;
    ModelInfo rawModelInfo;

    std::queue<Task> task_queue;
    std::queue<Task> finish_queue;

    int todo;
    std::string modelName;

    int tokenID;
    int *tokenManager;

    // runtime args
private:
    Task *current_task;
    // Ort::Session* _session;
    // std::vector<const char*>* _input_labels;
    // std::vector<const char*>* _output_labels;
    // std::vector<Ort::Value> _input_datas;
};

#endif // __MODELEXECUTOR_H__
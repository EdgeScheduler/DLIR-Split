#ifndef __EXECUTORMANAGER_H__
#define __EXECUTORMANAGER_H__

#include <onnxruntime_cxx_api.h>
#include <mutex>
#include <string>
#include <thread>
#include <memory>
#include <map>
#include <vector>
#include <condition_variable>
#include "TokenManager.h"
#include "ModelExecutor.h"

struct ExecutorDescribe
{
    std::shared_ptr<ModelExecutor> executor;
    int executorID;
    std::string modelName;
    std::thread threadHandle;
};

class ExecutorManager
{
public:
    ExecutorManager();
    ~ExecutorManager();

    /// @brief get all executor threads
    /// @return
    std::vector<std::thread *> GetAllThreads();

    /// @brief start an executor
    /// @param model_name
    void RunExecutor(std::string model_name);

    /// @brief add task to executor
    /// @param model_name executor/model name
    /// @param datas model inputs data
    void AddTask(std::string model_name, std::map<std::string, TensorValue<float>> &datas);

    /// @brief get executor describes
    /// @return
    std::map<std::string, std::shared_ptr<ExecutorDescribe>> &GetExecutorInformation();

    /// @brief give token to xx
    /// @param token ID, 0 means free
    /// @param block if token is still there, block or not.
    /// @return 
    bool Grant(int token, bool block=true);

    /// @brief join all thread to current-thread
    void Join();

private:
    Ort::SessionOptions sessionOption;
    Ort::Env environment;
    TokenManager tokenManager;
    int executorCount;
    std::mutex gpuMutex;
    std::condition_variable dealTask;

    std::map<std::string, std::shared_ptr<ExecutorDescribe>> executorMap;
};

#endif // __EXECUTORMANAGER_H__
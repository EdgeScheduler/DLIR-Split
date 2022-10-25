#ifndef __TASKDIGEST_H__
#define __TASKDIGEST_H__

#include <ctime>
#include <vector>
#include <memory>

class TaskDigest
{
public:
    /// @brief 
    /// @param executeTime each child-model execute-time(ms).
    /// @param requiredToken 
    /// @param requiredTokenCount how many child-models there is   
    /// @param modelExecuteTime 
    /// @param penaltyValue 
    TaskDigest(std::shared_ptr<std::vector<float>> executeTime, int requiredToken, int requiredTokenCount, float &modelExecuteTime,float penaltyValue=-0.3);

public:
    /// @brief calculate SLO-time(ms)
    /// @return 
    float GetSLO();

    /// @brief evaluate the score to the current wait-time, better choice face to higher score. 
    /// @param waitTime how long the task still need to in-queue. (ms)
    /// @return score<=1.0. If score>1, it means this task had been finished before.
    float Evaluate(float waitTime);

    float LeftRunTime();

    /// @brief Get new Token
    /// @param reduceTime used to return how long this task reduce
    /// @return -1 if there is no need.
    int GetToken(float& reduceTime);

public:
    int requiredToken;

private:
    clock_t startTime;
    std::shared_ptr<std::vector<float>> executeTime; // executeTime[0] is the last child-model run-time cost
    int requiredTokenCount;
    
    float leftRuntime;
    float &limitRuntime;
    float penaltyValue;
};

#endif // __TASKDIGEST_H__
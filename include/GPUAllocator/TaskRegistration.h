#ifndef __TASKREGISTRATION_H__
#define __TASKREGISTRATION_H__

#include<list>
#include"TaskDigest.h"

class TaskRegistration
{
public:
    TaskRegistration();
    void RegisteTask(std::shared_ptr<std::vector<float>> executeTime, int requiredToken, int requiredTokenCount, float &modelExecuteTime);

private:
    std::list<TaskDigest> tasks;
    float queueLength;              // how long the queue last.(ms)
};

#endif // __TASKREGISTRATION_H__
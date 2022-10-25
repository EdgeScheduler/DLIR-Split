#include "../../include/GPUAllocator/TaskRegistration.h"

TaskRegistration::TaskRegistration() : queueLength(0.0F)
{
}

void TaskRegistration::RegisteTask(std::shared_ptr<std::vector<float>> executeTime, int requiredToken, int requiredTokenCount, float &modelExecuteTime)
{
    TaskDigest task(executeTime, requiredToken, requiredTokenCount, modelExecuteTime);

#ifdef ALLOW_GPU_PARALLEL
    tasks.push_back(task);
    this->queueLength += task.LeftRunTime();
    return;
#else
    if (tasks.size() < 1)
    {
        tasks.push_back(task);
        this->queueLength += task.LeftRunTime();
        goto SCHEDULE;
    }
    else
    {
        float total_wait = queueLength;
        for (auto iter = tasks.begin(); iter != tasks.end(); iter++)
        {
            // it is not possible to insert before same type of task.
            if (iter->requiredToken == requiredToken)
            {
                tasks.insert(iter, std::move(task));
                goto SCHEDULE;
            }

            float new_task_back = task.Evaluate(total_wait);

            total_wait -= iter->LeftRunTime();
            float new_task_front = task.Evaluate(total_wait);
            float iter_front = iter->Evaluate(total_wait);
            float iter_back = iter->Evaluate(total_wait + task.LeftRunTime());

            if (new_task_back + iter_front > new_task_front + iter_back)
            {
                tasks.insert(iter, std::move(task));
                goto SCHEDULE;
            }
        }
        tasks.insert(tasks.end(), std::move(task));
    }

SCHEDULE:
    // to release lock;
    return;
#endif // ALLOW_GPU_PARALLEL
}

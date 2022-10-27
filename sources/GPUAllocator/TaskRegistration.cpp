#include "../../include/GPUAllocator/TaskRegistration.h"
#include <iostream>

TaskRegistration::TaskRegistration(TokenManager *tokenManager, std::condition_variable *dealTask) : queueLength(0.0F), tokenManager(tokenManager), dealTask(dealTask), currentTask(nullptr)
{
}

void TaskRegistration::RegisteTask(std::string name, std::shared_ptr<std::vector<float>> executeTime, int requiredToken, int requiredTokenCount, float &modelExecuteTime)
{
    TaskDigest task(name, executeTime, requiredToken, requiredTokenCount, modelExecuteTime);
    std::unique_lock<std::mutex> lock(mutex);

#ifdef ALLOW_GPU_PARALLEL
    tasks.push_front(task);
    this->queueLength += task.LeftRunTime();
#else
    if (tasks.size() < 1)
    {
        tasks.push_front(task);
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

        // update current_task (queue head)
        currentTask=&tasks.back();

        goto SCHEDULE;
    }

SCHEDULE:
#endif // ALLOW_GPU_PARALLEL
    // to release lock;
    this->queueLength += task.LeftRunTime();
    //std::cout<<"add: "<<task.LeftRunTime()<<std::endl;
    lock.unlock();
    m_notEmpty.notify_all();
    return;
}

void TaskRegistration::TokenDispense()
{
    float reduce_time = 0.0F;
    int next_token = 0;
    // TaskDigest* currentTaskPtr=nullptr;
    while (true)
    {
        // std::unique_lock<std::mutex> lock(mutex);
        // std::string discribe;
        
        if(this->currentTask==nullptr or this->currentTask->requiredTokenCount<1)
        {
            // to read valid task
            std::unique_lock<std::mutex> lock(mutex);
            while (true)
            {
                m_notEmpty.wait(lock, [this]() -> bool{ return tasks.size() > 0; });
                currentTask=&tasks.back();
                if(currentTask->requiredTokenCount<1)
                {
                    tasks.pop_back();
                    continue;
                }
                else
                {
                    break;
                }
            }
            lock.unlock();
        }

        tokenManager->WaitFree();
        queueLength -= reduce_time;

        next_token = this->currentTask->GetToken(reduce_time);          // currentTask is allowed to be update by TaskRegistration::RegisteTask

        if (tokenManager)
        {
            // std::cout << next_token << ": " << discribe << std::endl;
            tokenManager->Grant(next_token, true);
            if (tasks.size() < 1)
            {
                queueLength = 0.0F;
            }

            this->dealTask->notify_all();
        }
    }
}

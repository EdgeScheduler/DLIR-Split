#include "../../include/GPUAllocator/TokenManager.h"

TokenManager::TokenManager() : flag(0)
{
}

void TokenManager::Release()
{
    std::unique_lock<std::mutex> lock(mutex);
    this->flag = 0;
    lock.unlock();
    needNewToken.notify_all();
}

bool TokenManager::Grant(int token, bool block)
{
#ifndef ALLOW_GPU_PARALLEL
    std::unique_lock<std::mutex> lock(mutex);
    if (this->flag > 0)
    {
        if (block)
        {
            needNewToken.wait(lock, [this]() -> bool
                           { return this->flag < 1; });
            this->flag = token;
            lock.unlock();
            return true;
        }
        else
        {
            lock.unlock();
            return false;
        }
    }
    else
    {
        this->flag = token;
        lock.unlock();
        return true;
    }
#else
    return true;

#endif
}

TokenManager::operator int()
{
    return this->flag;
}

int TokenManager::GetFlag()
{
    return this->flag;
}

std::condition_variable& TokenManager::NeedNewToken()
{
    return this->needNewToken;
}

void TokenManager::WaitFree()
{
    std::unique_lock<std::mutex> lock(mutex);
    needNewToken.wait(lock, [this]() -> bool
                           { return this->flag < 1; });
}

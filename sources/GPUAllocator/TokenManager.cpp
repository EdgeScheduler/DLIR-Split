#include "../../include/GPUAllocator/TokenManager.h"

TokenManager::TokenManager():flag(0)
{

}

void TokenManager::Release()
{
    std::unique_lock<std::mutex> lock(mutex);
    this->flag=0;
    lock.unlock();
    needWrite.notify_all();
}

bool TokenManager::Grant(int token, bool block)
{

    std::unique_lock<std::mutex> lock(mutex);
    if(this->flag>0)
    {
        if(block)
        {
            needWrite.wait(lock,[this]()->bool{return this->flag<1;});
            this->flag=token;
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
        this->flag=token;
        lock.unlock();
        return true;
    }
}

TokenManager::operator int()
{
    return this->flag;
}

#include "../../include/GPUAllocator/ExecutorManager.h"
#include "../../include/Tensor/TensorValue.hpp"
#include <iostream>
using namespace std;

int main()
{
    ExecutorManager executorManager;

    executorManager.RunExecutor("resnet50");
    executorManager.RunExecutor("vgg19");
    executorManager.RunExecutor("googlenet");

    

    executorManager.Join();
    return 0;
}
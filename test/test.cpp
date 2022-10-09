#include<iostream>
#include"../headers/common/drivers.h"
#include"../headers/common/pathmanager.h"

int main()
{
    std::cout<<Drivers::CPUDriver<<std::endl;
    std::cout<<PathManager::GetProjectRootFold()<<std::endl;
    return 0;
}
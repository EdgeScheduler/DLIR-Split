#include <iostream>
#include "../include/common/Drivers.h"
#include "../include/common/PathManager.h"
#include"../include/common/JsonSerializer.h"
#include"../include/tensor/TensorValue.h"

using namespace std;

/// @brief test partial-function
/// @return
int main()
{
    // std::string model_name = "googlenet";
    // std::cout<<Drivers::CPUDriver<<std::endl;

    //test PathManager
    // cout << RootPathManager::GetRunRootFold() << endl;
    // cout << OnnxPathManager::GetOnnxRootFold() << endl;
    // cout << OnnxPathManager::GetModelSavePath(model_name) << endl;
    // cout << OnnxPathManager::GetModelParamsSavePath(model_name) << endl;
    // cout << OnnxPathManager::GetChildModelSavePath(model_name,0) << endl;
    // cout << OnnxPathManager::GetChildModelParamsSavePath(model_name,0) << endl;
    // cout << OnnxPathManager::GetChildModelSumParamsSavePath(model_name) << endl;
    // cout << OnnxPathManager::GetChildModelSumCacheSavePath(model_name) << endl;

    // auto a=JsonSerializer::LoadJson("test.json");
    // JsonSerializer::StoreJson(a,"aa/test_1.json",true);

    ValueInfo info("label",{1,2,20});
    TensorValue value=TensorValue(info,false);
    
    for(int i=0;i<10;i++)
    {
        value.Random();
        value.Print(10);
    }
    

    return 0;
}
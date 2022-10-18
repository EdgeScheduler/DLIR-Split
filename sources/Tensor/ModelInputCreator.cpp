#include "../../include/Tensor/ModelInputCreator.h"

ModelInputCreator::ModelInputCreator(const TensorsInfo& tensorsInfo): tensorsInfo(tensorsInfo)
{
    for(const ValueInfo& info: tensorsInfo.GetAllTensors())
    {
        dataTemplate.insert(std::pair<std::string, TensorValue<float>>(info.GetName(),TensorValue(info,false)));
    }
}

std::map<std::string, TensorValue<float>> ModelInputCreator::CreateInput()
{
    for(auto iter=dataTemplate.begin();iter!=dataTemplate.end();iter++)
    {
        iter->second.Random();
    }
    return this->dataTemplate;
}
